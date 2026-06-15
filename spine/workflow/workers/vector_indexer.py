"""Vector Indexer - background job for ingesting codebase into vector store.

Runs as a background job in RalphLoopWorker to chunk the codebase via AST
boundaries (using tree-sitter via ``spine.agents.tools.ast_extract``),
summarize with LLM, and embed for vector search.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from typing import Any

import numpy as np

from spine.agents.prompt_format import Tag, hostage_layout, xml_blocks
from spine.agents.tools.ast_extract import extract_edges as ast_extract_edges
from spine.agents.tools.ast_extract import extract_symbols as ast_extract_symbols
from spine.config import SpineConfig
from spine.persistence.vector_store import VectorStore

logger = logging.getLogger(__name__)

_INDEXABLE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".php", ".ts", ".tsx",
    ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh", ".hxx",
})

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
    "vendor",
    "external",
    "third_party",
    "extern",
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
        or base.endswith(("_test.py", "_test.c", "_test.cpp", "_test.cc"))
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
        """Incrementally index the codebase into the vector store.

        Only files whose content hash differs from the last index are
        re-processed; files removed from the tree have their symbols
        pruned. A full re-index happens naturally after ``--wipe`` (which
        clears the ledger, so every file looks new).

        Args:
            workspace_root: Optional workspace root override.

        Returns:
            Dict with stats: files_total, files_changed, files_removed,
            files_skipped, symbols_indexed, errors.
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

        target_files = await self._discover_target_files(workspace_root)

        # Compute current content hashes; diff against the ledger.
        current: dict[str, tuple[float, str]] = {}
        for rel in target_files:
            state = self._file_state(os.path.join(workspace_root, rel))
            if state is not None:
                current[rel] = state
        prev = self.store.get_indexed_files()
        target_set = set(current)

        removed = [p for p in prev if p not in target_set]
        changed = [p for p in current if current[p][1] != prev.get(p)]
        skipped = len(current) - len(changed)

        # Prune files that vanished from the tree (or became excluded).
        for rel in removed:
            self.store.delete_file_symbols(rel)
        if removed:
            logger.info("Pruned %d removed file(s) from the index", len(removed))

        logger.info(
            "Incremental index: %d target, %d changed, %d skipped, %d removed",
            len(current), len(changed), skipped, len(removed),
        )

        # Process changed files concurrently, but commit each file's ledger row
        # the moment *its* symbols all finish — not after the whole run. The
        # ledger is what makes an interrupted run resumable: a file recorded at
        # its content hash is skipped next time. Writing it only at the very end
        # meant any crash (or a fresh `--wipe` restart) re-did every file from
        # scratch. Per-file commit turns the index into a checkpointed job.
        #
        # Global concurrency is unchanged: every file's symbols share one
        # semaphore, so up to ``max_concurrent_chunks`` symbols are in flight at
        # once across all files — slots fill from the whole corpus, not one file
        # at a time.
        max_concurrent = self.config.vector_indexing.get("max_concurrent_chunks", 5)
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _process_file(rel: str) -> tuple[int, int]:
            """Index one file's symbols; commit its ledger row iff all stored.

            Returns ``(success_count, error_count)`` for the file. A file is
            "indexed at this hash" only when every symbol stored cleanly; files
            with a failure are left out of the ledger and retried next run.
            Files with zero symbols count as clean so they aren't re-extracted
            every run.
            """
            self.store.delete_file_symbols(rel)
            full_path = os.path.join(workspace_root, rel)
            file_symbols = self._extract_symbols_from_file(full_path, rel)
            # Dependency edges (PHP only — other languages are covered by
            # mcp-codebase-index). Pure AST work, no LLM/embedding cost.
            self.store.replace_file_edges(rel, ast_extract_edges(full_path, rel))
            results = await asyncio.gather(
                *[self._process_symbol(s, semaphore, workspace_root) for s in file_symbols],
                return_exceptions=True,
            )
            succeeded = sum(1 for r in results if r is True)
            failed = len(results) - succeeded
            if failed == 0:
                mtime, content_hash = current[rel]
                self.store.upsert_indexed_file(rel, mtime, content_hash)
            return succeeded, failed

        per_file = await asyncio.gather(
            *[_process_file(rel) for rel in changed],
            return_exceptions=True,
        )

        # Aggregate stats; a file-level exception (rather than a per-symbol one)
        # counts as a wholly-failed file that simply isn't in the ledger.
        success_count = sum(r[0] for r in per_file if isinstance(r, tuple))
        error_count = sum(r[1] for r in per_file if isinstance(r, tuple))
        for r in per_file:
            if not isinstance(r, tuple):
                logger.error("File-level indexing failure: %r", r)

        return {
            "files_total": len(current),
            "files_changed": len(changed),
            "files_removed": len(removed),
            "files_skipped": skipped,
            "symbols_indexed": success_count,
            "errors": error_count,
        }

    @staticmethod
    def _file_state(full_path: str) -> tuple[float, str] | None:
        """Return ``(mtime, sha256_hex)`` for a file, or None if unreadable."""
        try:
            mtime = os.stat(full_path).st_mtime
            with open(full_path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
            return mtime, digest
        except OSError as exc:
            logger.debug("Could not stat/read %s: %s", full_path, exc)
            return None

    async def _discover_target_files(self, workspace_root: str) -> list[str]:
        """List indexable source files by walking the workspace directly.

        Prefers ``git ls-files`` (tracked + untracked-but-not-ignored, so
        .gitignore is respected); falls back to an ``os.walk`` with
        directory pruning outside a git repo. Applies the extension
        allow-list, the excluded-dirs blacklist, and the test-file
        exclusion (unless ``index_tests``). Returns workspace-relative
        paths.
        """
        all_files = await asyncio.to_thread(self._list_workspace_files, workspace_root)

        index_tests = getattr(self.config, "index_tests", False)
        target = [
            f for f in all_files
            if os.path.splitext(f)[1].lower() in _INDEXABLE_EXTENSIONS
            and not _is_excluded(f)
            and (index_tests or not _is_test_file(f))
        ]
        logger.info(
            "Found %d indexable files (post-blacklist, index_tests=%s)",
            len(target), index_tests,
        )
        return target

    @staticmethod
    def _list_workspace_files(workspace_root: str) -> list[str]:
        """Workspace-relative file paths via git ls-files, else os.walk."""
        import subprocess

        try:
            result = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                cwd=workspace_root,
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
            return [line for line in result.stdout.splitlines() if line]
        except (OSError, subprocess.SubprocessError) as exc:
            logger.info("git ls-files unavailable (%s); falling back to os.walk", exc)

        files: list[str] = []
        for dirpath, dirnames, filenames in os.walk(workspace_root):
            dirnames[:] = [
                d for d in dirnames
                if d not in _EXCLUDED_DIR_NAMES and not d.startswith(".")
            ]
            for name in filenames:
                full = os.path.join(dirpath, name)
                files.append(os.path.relpath(full, workspace_root))
        return files

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
        from spine.agents.helpers import resolve_chat_model

        # resolve_chat_model centralizes the resolve_model + init_chat_model
        # coercion AND applies stream_usage/streaming so summary generation
        # reports token usage (trace 019ec965).
        model = resolve_chat_model(None, phase="summarization")

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
