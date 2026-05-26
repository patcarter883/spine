"""Vector Indexer - background job for ingesting codebase into vector store.

Runs as a background job in RalphLoopWorker to chunk the codebase via AST
boundaries (using tree-sitter via ``spine.agents.tools.ast_extract``),
summarize with LLM, and embed for vector search.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import numpy as np

from spine.agents.tools.ast_extract import extract_symbols as ast_extract_symbols
from spine.config import SpineConfig
from spine.persistence.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Extensions the AST extractor knows how to parse.  Keep this narrower
# than ``plan_tools._CODE_EXTENSIONS`` — we only want extensions that
# yield symbols, not every text file (markdown, yaml, json) in the tree.
_INDEXABLE_EXTENSIONS: frozenset[str] = frozenset({".py", ".php", ".ts", ".tsx"})


class VectorIndexer:
    """Background job processor for vector store population.

    Uses mcp-codebase-index tools to discover functions and classes,
    then processes them concurrently with summarization and embedding.
    """

    def __init__(self, config: SpineConfig | None = None) -> None:
        self.config = config or SpineConfig.load()
        self.store = VectorStore(self.config.checkpoint_path)
        self._embedding_client = None
        self._embed_lock = asyncio.Lock()

    async def index_codebase(self, workspace_root: str | None = None) -> dict[str, Any]:
        """Index the entire codebase into the vector store.

        Args:
            workspace_root: Optional workspace root override.

        Returns:
            Dict with stats: total_processed, skipped, errors.
        """
        workspace_root = workspace_root or self.config.workspace_root
        self.store.ensure_schema()

        # Discover symbols using MCP tools
        symbols = await self._discover_symbols(workspace_root)
        logger.info("Discovered %d symbols for indexing", len(symbols))

        # Process in concurrent batches
        max_concurrent = self.config.vector_indexing.get("max_concurrent_chunks", 5)
        semaphore = asyncio.Semaphore(max_concurrent)

        results = await asyncio.gather(
            *[
                self._process_symbol(symbol, semaphore, workspace_root)
                for symbol in symbols
            ],
            return_exceptions=True,
        )

        # Summary
        success_count = sum(1 for r in results if r is True)
        error_count = sum(1 for r in results if isinstance(r, Exception))

        return {
            "total_processed": len(symbols),
            "success": success_count,
            "errors": error_count,
        }

    async def _discover_symbols(self, workspace_root: str) -> list[dict[str, Any]]:
        """Discover symbols using MCP for file listing, tree-sitter for parsing.

        Uses mcp-codebase-index for file discovery (fast, cached), then
        local tree-sitter parsing for symbol extraction.  Each symbol is
        sliced to its own byte range — ``raw_code`` is the function/class
        body, NOT the containing file.
        """
        from spine.mcp.client import get_mcp_tools

        try:
            mcp_tools = get_mcp_tools(
                self.config.mcp_servers,
                cache_key="indexing",
                workspace_root=workspace_root,
            )
            tool_by_name = {t.name: t for t in mcp_tools}

            list_files_tool = tool_by_name.get("mcp_codebase-index_list_files")
            if not list_files_tool:
                logger.warning("mcp_codebase-index_list_files tool not available")
                return []

            files_result = await list_files_tool.ainvoke({"root": workspace_root})
            all_files = self._parse_tool_result(files_result)

            # Filter to extensions the AST extractor knows how to parse.
            candidate = [
                f for f in all_files
                if isinstance(f, str)
                and os.path.splitext(f)[1].lower() in _INDEXABLE_EXTENSIONS
            ]

            target = [
                f for f in candidate
                if f.startswith("spine/") or f.startswith("tests/") or f.startswith("src/")
            ]
            if not target:
                target = candidate

            logger.info(
                "Found %d indexable files (%d after scope filter), parsing for symbols...",
                len(candidate), len(target),
            )

            symbols: list[dict[str, Any]] = []
            for file_path in target[:200]:
                if not isinstance(file_path, str):
                    continue
                full_path = os.path.join(workspace_root, file_path)
                symbols.extend(self._extract_symbols_from_file(full_path, file_path))

            logger.info("Discovered %d symbols across %d files", len(symbols), min(len(target), 200))
            return symbols

        except Exception as e:
            logger.error("MCP discovery failed: %s", e, exc_info=True)
            return []

    @staticmethod
    def _extract_symbols_from_file(
        full_path: str, rel_path: str
    ) -> list[dict[str, Any]]:
        """Extract symbols via tree-sitter, returning per-symbol byte slices."""
        extracted = ast_extract_symbols(full_path, rel_path)
        return [
            {
                "file_path": s.file_path,
                "symbol_name": s.symbol_name,
                "symbol_type": s.symbol_type,
                "raw_code": s.raw_code,
                "start_byte": s.start_byte,
                "end_byte": s.end_byte,
                "lang": s.lang,
            }
            for s in extracted
        ]

    @staticmethod
    def _parse_tool_result(result: Any) -> list[Any]:
        """Normalize MCP tool results to a list of entries.

        Handles LangChain tool response format:
        [{"type": "text", "text": "[...json...]"}] -> parsed list
        """
        import json

        if isinstance(result, list):
            # LangChain MCP tool response: [{"type": "text", "text": "..."}]
            if len(result) == 1 and isinstance(result[0], dict) and "text" in result[0]:
                text = result[0]["text"]
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return parsed
                    return [parsed]
                except (json.JSONDecodeError, TypeError):
                    return [text] if text else []
            return result
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, list):
                    return parsed
                return [parsed]
            except (json.JSONDecodeError, TypeError):
                return [result] if result else []
        if isinstance(result, dict):
            # Some tools return {"items": [...]} or {"results": [...]}
            for key in ("items", "results", "symbols", "functions", "classes"):
                if key in result:
                    return result[key]
            return [result]
        return []

    async def _process_symbol(
        self,
        symbol: dict[str, Any],
        semaphore: asyncio.Semaphore,
        workspace_root: str,
    ) -> bool:
        """Process a single symbol: summarize, embed, store.

        ``raw_code`` is the per-symbol byte slice produced by the
        tree-sitter extractor — NOT the containing file.  This means
        each row holds at most a single function/class body, keeping
        recall hits small and on-topic.
        """
        async with semaphore:
            try:
                raw_code = symbol.get("raw_code", "")
                if not raw_code:
                    return False

                # Run summarization and embedding concurrently
                summary_task = asyncio.create_task(
                    self._summarize_code(raw_code, symbol["symbol_name"])
                )
                embedding_task = asyncio.create_task(self._embed_text(raw_code))

                summary, raw_embed = await asyncio.gather(
                    summary_task, embedding_task, return_exceptions=True
                )

                # Handle failures
                needs_enrichment = False
                if isinstance(summary, Exception):
                    logger.warning(
                        "Summarization failed for %s: %s",
                        symbol["symbol_name"],
                        summary,
                    )
                    summary = raw_code[:500] or "Summary failed"
                    needs_enrichment = True

                if isinstance(raw_embed, Exception):
                    logger.warning(
                        "Embedding raw code failed for %s: %s",
                        symbol["symbol_name"],
                        raw_embed,
                    )
                    raw_embed = np.zeros(VectorStore.EMBEDDING_DIM, dtype=np.float32)

                # Re-embed the summary for better semantic retrieval
                summary_embed = await self._embed_text(str(summary))
                if isinstance(summary_embed, np.ndarray) and summary_embed.any():
                    embedding = summary_embed
                else:
                    embedding = raw_embed

                # Store in vector database
                self.store.insert(
                    file_path=symbol["file_path"],
                    symbol_name=symbol["symbol_name"],
                    symbol_type=symbol["symbol_type"],
                    enriched_summary=str(summary),
                    raw_code=raw_code,
                    embedding=embedding,
                    needs_enrichment=needs_enrichment,
                    lang=symbol.get("lang", "python"),
                )

                return True

            except Exception as e:
                logger.error(
                    "Processing failed for %s: %s",
                    symbol.get("symbol_name", "unknown"),
                    e,
                )
                return False

    async def _summarize_code(self, raw_code: str, symbol_name: str) -> str:
        """Summarize code using the configured summarization model."""
        from spine.agents.helpers import resolve_model

        model = resolve_model(None, phase="summarization")

        if isinstance(model, str):
            from langchain.chat_models import init_chat_model

            model = init_chat_model(model)

        prompt = (
            f"Write a 2-sentence summary of what this code does and what "
            f"dependencies it relies on.\n\n```\n{raw_code}\n```"
        )

        response = await model.ainvoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    async def _embed_text(self, text: str) -> np.ndarray:
        """Embed text using a shared embedding client with retry."""
        from langchain_openai import OpenAIEmbeddings

        # Lazily create shared client (thread-safe via lock)
        if self._embedding_client is None:
            async with self._embed_lock:
                if self._embedding_client is None:
                    provider_cfg = self.config.resolve_embedding_provider()
                    if not provider_cfg:
                        raise ValueError(
                            f"Embedding provider '{self.config.embedding_provider}' not found"
                        )
                    self._embedding_client = OpenAIEmbeddings(
                        model=provider_cfg.get("model", "text-embedding-3-large"),
                        api_key=provider_cfg.get("api_key") or "",
                        **(provider_cfg.get("base_url") and {"base_url": provider_cfg["base_url"]}) or {},
                    )

        # Retry transient embedding failures (vLLM overload)
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await self._embedding_client.aembed_query(text)
                return np.array(result, dtype=np.float32)
            except (ValueError, RuntimeError) as e:
                if "No embedding" in str(e) and attempt < max_retries - 1:
                    delay = 1.0 * (2 ** attempt)
                    logger.debug("Embedding retry %d/%d after %.1fs", attempt + 1, max_retries, delay)
                    await asyncio.sleep(delay)
                    continue
                raise


async def run_indexing_job(workspace_root: str | None = None) -> dict[str, Any]:
    """Run the vector indexing job.

    Entry point for dispatch from RalphLoopWorker.

    Args:
        workspace_root: Optional workspace root override.

    Returns:
        Dict with indexing stats.
    """
    indexer = VectorIndexer()
    return await indexer.index_codebase(workspace_root)