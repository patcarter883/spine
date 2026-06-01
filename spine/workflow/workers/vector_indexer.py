"""Vector Indexer - background job for ingesting codebase into vector store.

Runs as a background job in RalphLoopWorker to chunk the codebase via AST
boundaries (using tree-sitter via ``spine.agents.tools.ast_extract``),
summarize with LLM, and embed for vector search.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import numpy as np

from spine.agents.prompt_format import Tag, hostage_layout, xml_blocks
from spine.agents.tools.ast_extract import extract_symbols as ast_extract_symbols
from spine.config import SpineConfig
from spine.persistence.vector_store import VectorStore

logger = logging.getLogger(__name__)

_INDEXABLE_EXTENSIONS: frozenset[str] = frozenset({".py", ".php", ".ts", ".tsx"})

# Directories we never want to walk into — vendored deps, build output,
# and Python/Node caches. Match by path prefix or by any path component
# equal to one of these names.
_EXCLUDED_DIR_NAMES: frozenset[str] = frozenset({
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "site-packages",
    ".tox",
    ".next",
    ".nuxt",
})


def _is_excluded(file_path: str) -> bool:
    """True if any path component is in the excluded-dirs blacklist."""
    parts = file_path.split("/")
    return any(part in _EXCLUDED_DIR_NAMES for part in parts)


def _is_test_file(file_path: str) -> bool:
    """True for test files (pytest layout: tests/ dir or test_*/*_test names)."""
    base = file_path.rsplit("/", 1)[-1]
    return (
        file_path.startswith("tests/")
        or "/tests/" in file_path
        or base.startswith("test_")
        or base.endswith("_test.py")
        or base == "conftest.py"
    )


def _build_embed_text(
    *,
    symbol_name: str,
    qualified_name: str,
    file_path: str,
    summary: str,
    raw_code: str,
) -> str:
    """Concatenate the identifier-anchored signal for embedding.

    The lexical anchors (qualified name + file path) are the strongest
    overlap with code-search queries that mention the symbol or file
    directly. The summary supplies semantic signal; the raw-code
    signature (first non-blank line, typically ``def name(args):`` /
    ``class Name(Base):``) keeps language-level identifier tokens in
    play even when the summary is paraphrased away from them.
    """
    signature_line = ""
    for line in raw_code.splitlines():
        stripped = line.strip()
        if stripped:
            signature_line = stripped[:200]
            break
    header = qualified_name if qualified_name != symbol_name else symbol_name
    return f"{header} {file_path}\n{summary}\n{signature_line}"


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
        # Document-side prefix for asymmetric embedding models (e.g.
        # nomic-embed-text wants "search_document: " on indexed text and
        # "search_query: " on queries). Empty for symmetric models.
        provider_cfg = self.config.resolve_embedding_provider() or {}
        self._document_prefix = provider_cfg.get("document_prefix", "")

    async def index_codebase(self, workspace_root: str | None = None) -> dict[str, Any]:
        """Index the entire codebase into the vector store.

        Args:
            workspace_root: Optional workspace root override.

        Returns:
            Dict with stats: total_processed, skipped, errors.
        """
        workspace_root = workspace_root or self.config.workspace_root

        # Probe the live model's embedding dimension and size the vec0 table
        # to match (recreating it if the model — and thus the dimension —
        # changed, e.g. a swap from Qwen3-4096 to nomic-768). Done before
        # ensure_schema so a fresh/empty table is created at the right width.
        probe = await self._embed_text("dimension probe")
        self.store.set_embedding_dim(int(probe.shape[0]))
        logger.info("Embedding dimension probed: %d", self.store.embedding_dim)
        self.store.ensure_schema()

        symbols = await self._discover_symbols(workspace_root)
        logger.info("Discovered %d symbols for indexing", len(symbols))

        max_concurrent = self.config.vector_indexing.get("max_concurrent_chunks", 5)
        semaphore = asyncio.Semaphore(max_concurrent)

        results = await asyncio.gather(
            *[
                self._process_symbol(symbol, semaphore, workspace_root)
                for symbol in symbols
            ],
            return_exceptions=True,
        )

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

            index_tests = getattr(self.config, "index_tests", False)
            target = [
                f for f in all_files
                if isinstance(f, str)
                and os.path.splitext(f)[1].lower() in _INDEXABLE_EXTENSIONS
                and not _is_excluded(f)
                and (index_tests or not _is_test_file(f))
            ]

            logger.info(
                "Found %d indexable files (post-blacklist, index_tests=%s), parsing for symbols...",
                len(target), index_tests,
            )

            symbols: list[dict[str, Any]] = []
            for file_path in target:
                if not isinstance(file_path, str):
                    continue
                full_path = os.path.join(workspace_root, file_path)
                symbols.extend(self._extract_symbols_from_file(full_path, file_path))

            logger.info("Discovered %d symbols across %d files", len(symbols), len(target))
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
                "qualified_name": s.qualified_name,
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
        """Process a single symbol: summarize, build embed text, embed, store.

        ``raw_code`` is the per-symbol byte slice produced by the
        tree-sitter extractor — NOT the containing file.

        The embedded text is the lexical-anchored composite returned by
        :func:`_build_embed_text` (qualified name + file path + summary
        + signature line) rather than the raw code or the summary
        alone, so code-search queries that mention the symbol or file
        name lexically overlap with the indexed vector.
        """
        async with semaphore:
            try:
                raw_code = symbol.get("raw_code", "")
                if not raw_code:
                    return False

                symbol_name = symbol["symbol_name"]
                qualified_name = symbol.get("qualified_name", symbol_name)
                file_path = symbol["file_path"]

                try:
                    summary = await self._summarize_code(
                        raw_code, qualified_name, file_path
                    )
                    needs_enrichment = False
                except Exception as exc:
                    logger.warning(
                        "Summarization failed for %s: %s", qualified_name, exc
                    )
                    summary = raw_code[:500] or "Summary failed"
                    needs_enrichment = True

                embed_text = _build_embed_text(
                    symbol_name=symbol_name,
                    qualified_name=qualified_name,
                    file_path=file_path,
                    summary=str(summary),
                    raw_code=raw_code,
                )
                # Document-side embedding prefix (e.g. nomic's
                # "search_document: "). Indexed documents and queries use
                # different prefixes for asymmetric models; the store guards
                # the dimension, so a mismatch raises in insert().
                embedding = await self._embed_text(self._document_prefix + embed_text)

                self.store.insert(
                    file_path=file_path,
                    symbol_name=qualified_name,
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

    async def _summarize_code(
        self, raw_code: str, qualified_name: str, file_path: str
    ) -> str:
        """Summarize code in an extractive identifier-dense format.

        The output is consumed by the embedder, not a human — so we
        bias toward identifier-rich, terse extraction (public API,
        called names, side effects, one-line purpose). Generic prose
        summaries paraphrase the identifiers away, which is exactly
        what hurts recall on identifier-mentioning queries.
        """
        from spine.agents.helpers import resolve_model

        model = resolve_model(None, phase="summarization")

        if isinstance(model, str):
            from langchain.chat_models import init_chat_model

            model = init_chat_model(model)

        system = xml_blocks(
            (
                Tag.ROLE,
                "You produce identifier-dense extractive summaries of code "
                "symbols for a semantic-search index. The reader is an "
                "embedding model, not a human — bias toward terse, "
                "identifier-rich extraction over fluent prose.",
            ),
            (
                Tag.OUTPUT_SCHEMA,
                "Public API: <names of exported functions/classes/methods>\n"
                "Calls: <names of called functions or imported symbols, comma-separated>\n"
                "Side effects: <writes/network/mutation/none>\n"
                "Purpose: <one short sentence>",
            ),
            (
                Tag.CONSTRAINTS,
                "- Do not paraphrase identifier names; use them verbatim.\n"
                "- Omit lines that have no content (do not write 'none' as filler "
                "  except for Side effects).\n"
                "- No markdown, no code fences, no preamble.",
            ),
        )
        user = hostage_layout(
            xml_blocks(
                (Tag.OBJECTIVE, f"{qualified_name}  ({file_path})"),
                (Tag.RETRIEVED_CODE, raw_code),
            ),
            "Emit the four lines of the output schema for the code above.",
        )

        from langchain_core.messages import HumanMessage, SystemMessage

        response = await model.ainvoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        return response.content if hasattr(response, "content") else str(response)

    async def _embed_text(self, text: str) -> np.ndarray:
        """Embed text using a shared embedding client with retry."""
        from langchain_openai import OpenAIEmbeddings

        if self._embedding_client is None:
            async with self._embed_lock:
                if self._embedding_client is None:
                    provider_cfg = self.config.resolve_embedding_provider()
                    if not provider_cfg:
                        raise ValueError(
                            f"Embedding provider '{self.config.embedding_provider}' not found"
                        )
                    model_name = provider_cfg.get("model")
                    if not model_name:
                        raise ValueError(
                            f"Embedding provider {provider_cfg.get('name')!r} "
                            f"has no 'model' set"
                        )
                    self._embedding_client = OpenAIEmbeddings(
                        model=model_name,
                        api_key=provider_cfg.get("api_key") or "",
                        **(provider_cfg.get("base_url") and {"base_url": provider_cfg["base_url"]}) or {},
                    )

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
