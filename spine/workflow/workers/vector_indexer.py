"""Vector Indexer - background job for ingesting codebase into vector store.

Runs as a background job in RalphLoopWorker to chunk the codebase via AST
boundaries (using mcp-codebase-index), summarize with LLM, and embed for
vector search.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import numpy as np

from spine.config import SpineConfig
from spine.persistence.vector_store import VectorStore

logger = logging.getLogger(__name__)


class VectorIndexer:
    """Background job processor for vector store population.

    Uses mcp-codebase-index tools to discover functions and classes,
    then processes them concurrently with summarization and embedding.
    """

    def __init__(self, config: SpineConfig | None = None) -> None:
        self.config = config or SpineConfig.load()
        self.store = VectorStore(self.config.checkpoint_path)

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
        """Discover symbols via mcp-codebase-index MCP tools.

        Uses list_files to find Python files, then get_functions/get_classes
        per file to enumerate symbols for indexing.
        """
        from spine.mcp.client import get_mcp_tools

        try:
            mcp_tools = get_mcp_tools(
                self.config.mcp_servers,
                cache_key="indexing",
                workspace_root=workspace_root,
            )

            # Build a lookup by name for fast access
            tool_by_name: dict[str, Any] = {}
            for t in mcp_tools:
                tool_by_name[t.name] = t

            # Step 1: List all Python files
            list_files_tool = tool_by_name.get("mcp_codebase-index_list_files")
            if not list_files_tool:
                logger.warning("mcp_codebase-index_list_files tool not available")
                return []

            files_result = await list_files_tool.ainvoke({
                "pattern": "*.py",
                "root": workspace_root,
            })
            py_files = self._parse_tool_result(files_result)

            if not py_files:
                logger.warning("No Python files found for indexing")
                return []

            logger.info("Found %d Python files to index", len(py_files))

            # Step 2: Get functions and classes from each file
            get_functions = tool_by_name.get("mcp_codebase-index_get_functions")
            get_classes = tool_by_name.get("mcp_codebase-index_get_classes")

            symbols: list[dict[str, Any]] = []
            files_processed = 0

            # Limit to internal packages (spine/, tests/)
            target_files = [
                f for f in py_files
                if isinstance(f, str) and (
                    f.startswith("spine/") or f.startswith("tests/")
                )
            ]
            if not target_files:
                target_files = [f for f in py_files if isinstance(f, str)]

            for file_path in target_files[:200]:  # Cap to prevent overloading
                if not isinstance(file_path, str):
                    continue
                files_processed += 1

                if get_functions:
                    try:
                        funcs = await get_functions.ainvoke(
                            {"file_path": file_path}
                        )
                        for entry in self._parse_tool_result(funcs):
                            if isinstance(entry, dict) and "name" in entry:
                                symbols.append({
                                    "file_path": file_path,
                                    "symbol_name": entry["name"],
                                    "symbol_type": "function",
                                })
                    except Exception as e:
                        logger.debug("Failed to get functions for %s: %s", file_path, e)

                if get_classes:
                    try:
                        classes = await get_classes.ainvoke(
                            {"file_path": file_path}
                        )
                        for entry in self._parse_tool_result(classes):
                            if isinstance(entry, dict) and "name" in entry:
                                symbols.append({
                                    "file_path": file_path,
                                    "symbol_name": entry["name"],
                                    "symbol_type": "class",
                                })
                    except Exception as e:
                        logger.debug("Failed to get classes for %s: %s", file_path, e)

            logger.info(
                "Discovered %d symbols in %d files",
                len(symbols),
                files_processed,
            )
            return symbols

        except Exception as e:
            logger.error("MCP discovery failed: %s", e, exc_info=True)
            return []

    @staticmethod
    def _parse_tool_result(result: Any) -> list[Any]:
        """Normalize tool results to a list of entries."""
        import json

        if isinstance(result, list):
            return result
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, list):
                    return parsed
                return [parsed]
            except (json.JSONDecodeError, TypeError):
                return []
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
        """Process a single symbol: fetch code, summarize, embed, store."""
        async with semaphore:
            try:
                # Fetch raw code
                raw_code = await self._fetch_raw_code(
                    symbol["file_path"], symbol["symbol_name"], workspace_root
                )
                if not raw_code:
                    return False

                # Run summarization and embedding concurrently
                summary_task = asyncio.create_task(
                    self._summarize_code(raw_code, symbol["symbol_name"])
                )
                embedding_task = asyncio.create_task(self._embed_text(raw_code))

                summary, embedding = await asyncio.gather(
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

                if isinstance(embedding, Exception):
                    logger.warning(
                        "Embedding failed for %s: %s",
                        symbol["symbol_name"],
                        embedding,
                    )
                    embedding = np.zeros(VectorStore.EMBEDDING_DIM, dtype=np.float32)

                # Store in vector database
                self.store.insert(
                    file_path=symbol["file_path"],
                    symbol_name=symbol["symbol_name"],
                    symbol_type=symbol["symbol_type"],
                    enriched_summary=str(summary),
                    raw_code=raw_code,
                    embedding=embedding,
                    needs_enrichment=needs_enrichment,
                )

                return True

            except Exception as e:
                logger.error(
                    "Processing failed for %s: %s",
                    symbol.get("symbol_name", "unknown"),
                    e,
                )
                return False

    async def _fetch_raw_code(
        self,
        file_path: str,
        symbol_name: str,
        workspace_root: str,
    ) -> str:
        """Fetch source code for a symbol via MCP tools."""
        from spine.mcp.client import get_mcp_tools

        try:
            mcp_tools = get_mcp_tools(
                self.config.mcp_servers,
                cache_key="indexing",
                workspace_root=workspace_root,
            )

            tool_by_name = {t.name: t for t in mcp_tools}

            for tool_name in (
                "mcp_codebase-index_get_function_source",
                "mcp_codebase-index_get_class_source",
            ):
                tool = tool_by_name.get(tool_name)
                if not tool:
                    continue
                try:
                    result = await tool.ainvoke({
                        "name": symbol_name,
                        "file_path": file_path,
                    })
                    if isinstance(result, str) and result.strip():
                        return result
                except Exception:
                    continue

            # Fallback: fetch whole file from disk
            import os

            full_path = os.path.join(workspace_root, file_path)
            with open(full_path, encoding="utf-8") as f:
                return f.read()

        except OSError as e:
            logger.warning("Could not read %s:%s — %s", file_path, symbol_name, e)
            return ""

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
        """Embed text using the configured embedding provider."""
        from langchain_openai import OpenAIEmbeddings

        provider_cfg = self.config.resolve_embedding_provider()
        if not provider_cfg:
            raise ValueError(f"Embedding provider '{self.config.embedding_provider}' not found")

        model_name = provider_cfg.get("model", "text-embedding-3-large")
        api_key = provider_cfg.get("api_key") or ""
        base_url = provider_cfg.get("base_url")

        embeddings = OpenAIEmbeddings(
            model=model_name,
            api_key=api_key,
            **(base_url and {"base_url": base_url}) or {},
        )

        result = await embeddings.aembed_query(text)
        return np.array(result, dtype=np.float32)


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