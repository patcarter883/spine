"""Analysis backend — compiles a :class:`RepoManifest` from a repository.

This is slice 1 of the onboarding engine. It analyses a target repo using
*semantic* signals only — never raw line-by-line file reads at runtime — to
protect the context window:

- File discovery via ``mcp_codebase-index_list_files`` (fast, cached) with a
  filesystem ``os.walk`` fallback when the index is unavailable.
- Symbol extraction via :func:`spine.agents.tools.ast_extract.extract_symbols`,
  which byte-slices each function/class/method/interface body rather than
  reading whole files.
- Optional summary enrichment via
  :meth:`spine.persistence.vector_store.VectorStore.search_similar` (uses the
  pre-indexed ``enriched_summary`` rows; no embedding calls at analysis time).

The constraint is hard: this module MUST NOT ``open(...).readlines()`` source
files line-by-line. ``extract_symbols`` does the only file read, and it returns
byte-sliced symbol bodies, not raw text streamed to the model.

For greenfield mode the analyzer skips discovery entirely and returns a
near-empty manifest seeded with the caller-supplied ``tech_stack`` —
slice 3 synthesises best-practice defaults from that.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from spine.agents.tools.ast_extract import Symbol, extract_symbols
from spine.config import SpineConfig
from spine.work.onboarding.manifest import (
    DependencyEdge,
    ModuleBoundary,
    PatternFinding,
    RepoManifest,
    SymbolRef,
)

logger = logging.getLogger(__name__)

# Extensions the AST extractor can parse. Mirrors VectorIndexer — only
# extensions that yield symbols, not every text file in the tree.
_INDEXABLE_EXTENSIONS: frozenset[str] = frozenset({".py", ".php", ".ts", ".tsx"})

# Directories never worth walking — vendored deps, VCS, build output, caches.
_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".spine",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".tox",
        "vendor",
    }
)

# Cap on files parsed so a huge repo cannot blow the analysis budget.
_MAX_FILES: int = 400

# Extension → language tag (for tech-stack inference).
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".php": "php",
    ".ts": "typescript",
    ".tsx": "typescript",
}


class RepoAnalyzer:
    """Compiles a :class:`RepoManifest` from a repository via AST + index.

    Construction is cheap; all heavy work happens in :meth:`analyze`. The
    ``VectorStore`` is opened lazily and only used to enrich symbol summaries
    from rows a prior indexing job already produced — no embedding/LLM calls
    are made here.
    """

    def __init__(self, config: SpineConfig | None = None) -> None:
        self.config = config or SpineConfig.load()

    # ── Public API ──────────────────────────────────────────────────────

    async def analyze(
        self,
        workspace_root: str,
        mode: str = "brownfield",
        tech_stack: list[str] | None = None,
    ) -> RepoManifest:
        """Analyse *workspace_root* and return a compiled :class:`RepoManifest`.

        Args:
            workspace_root: Absolute path to the repository to analyse.
            mode: ``"brownfield"`` (analyse existing code) or ``"greenfield"``
                (return a seed manifest from ``tech_stack`` only).
            tech_stack: Caller-supplied stack tags. Required for greenfield;
                merged with inferred tags for brownfield.

        Returns:
            A :class:`RepoManifest`. For greenfield, boundaries/chains/patterns
            are empty and only ``tech_stack``/``core_domains`` are populated.
        """
        generated_at = datetime.now(timezone.utc).isoformat()
        seed_stack = list(tech_stack or [])

        if mode == "greenfield":
            return RepoManifest(
                workspace_root=workspace_root,
                mode="greenfield",
                tech_stack=seed_stack,
                core_domains=[],
                module_boundaries=[],
                dependency_chains=[],
                patterns=[],
                symbol_count=0,
                file_count=0,
                generated_at=generated_at,
                notes="greenfield seed — no existing code analysed",
            )

        files = await self._discover_files(workspace_root)
        workspace_packages = detect_workspace_packages(workspace_root)
        is_monorepo = len(workspace_packages) >= 2
        symbols, parsed_files = self._extract_symbols(workspace_root, files)
        notes_parts: list[str] = []
        if not files:
            notes_parts.append("file discovery returned nothing (index + walk both empty)")
        if is_monorepo:
            pkg_names = ", ".join(p["dotted_name"] for p in workspace_packages)
            notes_parts.append(f"monorepo detected ({len(workspace_packages)} packages: {pkg_names})")

        summaries = self._load_summaries()
        if summaries:
            notes_parts.append(f"enriched {len(summaries)} summaries from vector index")
        else:
            notes_parts.append("vector index unavailable — AST-only, no summaries")

        boundaries = self._build_boundaries(symbols, summaries, workspace_packages)
        dependency_chains = self._build_dependency_edges(symbols, workspace_packages)
        patterns = self._extract_patterns(symbols, summaries)
        tech = self._infer_tech_stack(files, seed_stack)
        core_domains = [b.name for b in boundaries]

        return RepoManifest(
            workspace_root=workspace_root,
            mode="brownfield",
            tech_stack=tech,
            core_domains=core_domains,
            module_boundaries=boundaries,
            dependency_chains=dependency_chains,
            patterns=patterns,
            symbol_count=len(symbols),
            file_count=parsed_files,
            generated_at=generated_at,
            notes="; ".join(notes_parts),
            is_monorepo=is_monorepo,
            workspace_packages=workspace_packages,
        )

    # ── File discovery (index first, walk fallback) ─────────────────────

    async def _discover_files(self, workspace_root: str) -> list[str]:
        """Return repo-relative source file paths, index-first with walk fallback."""
        files = await self._discover_files_via_index(workspace_root)
        if files:
            return files
        logger.info("Onboarding analyzer: index discovery empty, falling back to os.walk")
        return self._discover_files_via_walk(workspace_root)

    async def _discover_files_via_index(self, workspace_root: str) -> list[str]:
        """Discover files using ``mcp_codebase-index_list_files`` (cached, fast)."""
        try:
            from spine.mcp.client import get_mcp_tools

            mcp_tools = get_mcp_tools(
                self.config.mcp_servers,
                cache_key="onboarding",
                workspace_root=workspace_root,
            )
            tool_by_name = {t.name: t for t in mcp_tools}
            list_files_tool = tool_by_name.get("mcp_codebase-index_list_files")
            if not list_files_tool:
                logger.info("mcp_codebase-index_list_files unavailable for onboarding")
                return []

            files_result = await list_files_tool.ainvoke({"root": workspace_root})
            all_files = self._parse_tool_result(files_result)
            return self._filter_source_files(all_files)
        except Exception as exc:  # pragma: no cover — MCP env-dependent
            logger.info("Onboarding analyzer: index discovery failed: %s", exc)
            return []

    def _discover_files_via_walk(self, workspace_root: str) -> list[str]:
        """Fallback discovery: walk the tree for source files (no file *reads*).

        ``os.walk`` only lists paths — it does not read source content. The
        no-line-read constraint applies to source *content*, which is only ever
        read by ``extract_symbols`` (byte slices). Listing filenames is allowed.
        """
        results: list[str] = []
        for dirpath, dirnames, filenames in os.walk(workspace_root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
            for name in filenames:
                ext = os.path.splitext(name)[1].lower()
                if ext not in _INDEXABLE_EXTENSIONS:
                    continue
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, workspace_root)
                results.append(rel)
        return self._filter_source_files(results)

    @staticmethod
    def _filter_source_files(all_files: list[Any]) -> list[str]:
        """Keep parseable source files, drop non-strings/unsupported extensions."""
        return [
            f
            for f in all_files
            if isinstance(f, str)
            and os.path.splitext(f)[1].lower() in _INDEXABLE_EXTENSIONS
            and not any(
                part in _SKIP_DIRS or part.startswith(".")
                for part in f.replace("\\", "/").split("/")
            )
        ]

    @staticmethod
    def _parse_tool_result(result: Any) -> list[Any]:
        """Normalise an MCP tool result to a list of entries.

        Mirrors ``VectorIndexer._parse_tool_result`` — handles the LangChain
        ``[{"type": "text", "text": "[...json...]"}]`` envelope, plain JSON
        strings, and ``{"items": [...]}``-style dicts.
        """
        import json

        if isinstance(result, list):
            if len(result) == 1 and isinstance(result[0], dict) and "text" in result[0]:
                text = result[0]["text"]
                try:
                    parsed = json.loads(text)
                    return parsed if isinstance(parsed, list) else [parsed]
                except (json.JSONDecodeError, TypeError):
                    return [text] if text else []
            return result
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                return parsed if isinstance(parsed, list) else [parsed]
            except (json.JSONDecodeError, TypeError):
                return [result] if result else []
        if isinstance(result, dict):
            for key in ("items", "results", "symbols", "functions", "classes", "files"):
                if key in result:
                    return result[key]
            return [result]
        return []

    # ── Symbol extraction (byte-sliced, no line reads) ──────────────────

    def _extract_symbols(
        self, workspace_root: str, files: list[str]
    ) -> tuple[list[Symbol], int]:
        """Extract symbols across *files* via tree-sitter byte slicing.

        Returns the flat symbol list plus the count of files that yielded at
        least one symbol.
        """
        symbols: list[Symbol] = []
        parsed_files = 0
        for rel_path in files[:_MAX_FILES]:
            full_path = os.path.join(workspace_root, rel_path)
            extracted = extract_symbols(full_path, rel_path)
            if extracted:
                parsed_files += 1
                symbols.extend(extracted)
        logger.info(
            "Onboarding analyzer: extracted %d symbols from %d files",
            len(symbols),
            parsed_files,
        )
        return symbols, parsed_files

    # ── Summary enrichment (from pre-indexed vector store) ──────────────

    def _load_summaries(self) -> dict[tuple[str, str], str]:
        """Load ``(file_path, symbol_name) -> enriched_summary`` from the index.

        Reads existing rows produced by a prior indexing job. No embeddings or
        LLM calls are made here — if the store/table is missing we simply
        return an empty map and the manifest carries empty summaries.
        """
        try:
            from spine.persistence.vector_store import VectorStore

            store = VectorStore(self.config.checkpoint_path)
            conn = store._get_connection()  # noqa: SLF001 — read-only metadata access
            cursor = conn.execute(
                "SELECT file_path, symbol_name, enriched_summary FROM symbol_metadata"
            )
            summaries: dict[tuple[str, str], str] = {}
            for row in cursor:
                summaries[(row["file_path"], row["symbol_name"])] = row["enriched_summary"]
            store.close()
            return summaries
        except Exception as exc:  # pragma: no cover — store may not exist
            logger.info("Onboarding analyzer: summary enrichment unavailable: %s", exc)
            return {}

    def _symbol_ref(
        self, sym: Symbol, summaries: dict[tuple[str, str], str]
    ) -> SymbolRef:
        """Build a lightweight :class:`SymbolRef` (no raw code) for *sym*."""
        return SymbolRef(
            file_path=sym.file_path,
            symbol_name=sym.symbol_name,
            symbol_type=sym.symbol_type,
            lang=sym.lang,
            summary=summaries.get((sym.file_path, sym.symbol_name), ""),
        )

    # ── Module boundaries ───────────────────────────────────────────────

    def _build_boundaries(
        self,
        symbols: list[Symbol],
        summaries: dict[tuple[str, str], str],
        workspace_packages: list[dict] | None = None,
    ) -> list[ModuleBoundary]:
        """Group symbols into logical module boundaries by package directory.

        When *workspace_packages* is provided, symbols are grouped by their
        workspace package root; otherwise the two-segment prefix fallback is
        used. This is the monolithic path — it groups all symbols via
        :func:`group_symbols_by_module` and builds one boundary per group via
        :meth:`_build_boundary_for_unit`.
        """
        grouped = group_symbols_by_module(symbols, workspace_packages)
        boundaries: list[ModuleBoundary] = []
        for name in sorted(grouped):
            module_symbols, path = grouped[name]
            boundaries.append(
                self._build_boundary_for_unit(name, path, module_symbols, summaries)
            )
        return boundaries

    def _build_boundary_for_unit(
        self,
        name: str,
        path: str,
        unit_symbols: list[Symbol],
        summaries: dict[tuple[str, str], str],
    ) -> ModuleBoundary:
        """Build a single :class:`ModuleBoundary` for one module group.

        Extracted from :meth:`_build_boundaries` verbatim so the distributed
        analysis explorer can build exactly the same boundary for its assigned
        unit. ``name``/``path`` are the group's dotted module name and
        repo-relative directory (from :func:`group_symbols_by_module`).
        """
        # Prefer classes/interfaces as "key" symbols, then functions.
        ranked = sorted(
            unit_symbols,
            key=lambda s: (
                0 if s.symbol_type in ("class", "interface") else 1,
                s.symbol_name,
            ),
        )
        key_symbols = [self._symbol_ref(s, summaries) for s in ranked[:8]]
        return ModuleBoundary(
            name=name,
            path=path,
            role=self._describe_module(name, unit_symbols),
            key_symbols=key_symbols,
        )

    @staticmethod
    def _describe_module(name: str, symbols: list[Symbol]) -> str:
        """Produce a short prose role for a module from its symbol mix."""
        classes = sum(1 for s in symbols if s.symbol_type in ("class", "interface"))
        functions = sum(1 for s in symbols if s.symbol_type == "function")
        methods = sum(1 for s in symbols if s.symbol_type == "method")
        return (
            f"{name}: {len(symbols)} symbols "
            f"({classes} classes/interfaces, {functions} functions, {methods} methods)"
        )

    # ── Dependency edges ────────────────────────────────────────────────

    def _build_dependency_edges(
        self,
        symbols: list[Symbol],
        workspace_packages: list[dict] | None = None,
    ) -> list[DependencyEdge]:
        """Derive inter-module ``depends_on`` edges from import statements.

        Imports are read from the byte-sliced symbol bodies already in memory
        (no extra file reads). We only record cross-module edges between
        package prefixes that actually appear as boundaries, deduplicated.
        """
        pkg_index = _build_pkg_index(workspace_packages) if workspace_packages else []
        modules = self._module_names(symbols, pkg_index)
        edges: set[tuple[str, str]] = set()
        for sym in symbols:
            src_module = _module_of_with_packages(sym.file_path, pkg_index) if pkg_index else self._module_of(sym.file_path)
            for imported in self._iter_imports(sym.raw_code):
                dst = self._match_module(imported, modules)
                if dst and dst != src_module:
                    edges.add((src_module, dst))
        return [
            DependencyEdge(src=src, dst=dst, kind="depends_on")
            for src, dst in sorted(edges)
        ]

    @staticmethod
    def _module_names(symbols: list[Symbol], pkg_index: list[tuple[str, str]] | None = None) -> set[str]:
        """Set of module names present across *symbols*, package-aware when *pkg_index* is given."""
        names: set[str] = set()
        for sym in symbols:
            if pkg_index:
                names.add(_module_of_with_packages(sym.file_path, pkg_index))
            else:
                names.add(RepoAnalyzer._module_of(sym.file_path))
        return names

    @staticmethod
    def _module_of(file_path: str) -> str:
        """Two-segment dotted module name for a repo-relative file path."""
        parts = file_path.replace("\\", "/").split("/")
        return ".".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else "")

    @staticmethod
    def _iter_imports(raw_code: str) -> list[str]:
        """Extract dotted import targets from a Python symbol body.

        Scans the in-memory byte-slice for ``import X`` / ``from X import``
        statements. Operates on the already-extracted symbol code, not a file.
        """
        targets: list[str] = []
        for line in raw_code.splitlines():
            stripped = line.strip()
            if stripped.startswith("from ") and " import " in stripped:
                mod = stripped[len("from ") :].split(" import ", 1)[0].strip()
                if mod and not mod.startswith("."):
                    targets.append(mod)
            elif stripped.startswith("import "):
                rest = stripped[len("import ") :].split(" as ", 1)[0].strip()
                for part in rest.split(","):
                    mod = part.strip()
                    if mod:
                        targets.append(mod)
        return targets

    @staticmethod
    def _match_module(imported: str, modules: set[str]) -> str | None:
        """Map a dotted import target to a known two-segment module name."""
        parts = imported.split(".")
        if len(parts) >= 2:
            candidate = ".".join(parts[:2])
            if candidate in modules:
                return candidate
        if parts and parts[0] in modules:
            return parts[0]
        return None

    # ── Pattern findings ────────────────────────────────────────────────

    def _extract_patterns(
        self, symbols: list[Symbol], summaries: dict[tuple[str, str], str]
    ) -> list[PatternFinding]:
        """Extract coding conventions over the full symbol set (monolithic path).

        Thin wrapper around :meth:`_extract_patterns_for_unit` so the monolithic
        analyzer and each distributed explorer share one extraction body. The
        distributed explorer calls :meth:`_extract_patterns_for_unit` with only
        its unit's symbol subset; the aggregator dedupes + re-caps the union.
        """
        return self._extract_patterns_for_unit(symbols, summaries)

    def _extract_patterns_for_unit(
        self,
        symbols: list[Symbol],
        summaries: dict[tuple[str, str], str],
    ) -> list[PatternFinding]:
        """Extract coding conventions from a subset of byte-sliced symbol bodies.

        Each finding records representative ``SymbolRef`` evidence — never raw
        source text — so the manifest stays compact. The evidence cap (3) is
        applied per call; when called per-unit, the aggregator re-caps the
        merged union so the global manifest matches the monolithic output.
        """
        findings: list[PatternFinding] = []

        def evidence_for(predicate: Any, cap: int = 3) -> list[SymbolRef]:
            refs: list[SymbolRef] = []
            for sym in symbols:
                if predicate(sym):
                    refs.append(self._symbol_ref(sym, summaries))
                    if len(refs) >= cap:
                        break
            return refs

        # Logging: module-level logging.getLogger(__name__) usage.
        logging_ev = evidence_for(
            lambda s: "logging.getLogger" in s.raw_code or "getLogger(__name__)" in s.raw_code
        )
        if logging_ev:
            findings.append(
                PatternFinding(
                    category="logging",
                    description="module-level logging.getLogger(__name__)",
                    evidence=logging_ev,
                )
            )

        # Data model: frozen dataclasses.
        dataclass_ev = evidence_for(
            lambda s: "@dataclass" in s.raw_code and s.symbol_type == "class"
        )
        if dataclass_ev:
            frozen = any("frozen=True" in s.raw_code for s in symbols if "@dataclass" in s.raw_code)
            findings.append(
                PatternFinding(
                    category="data_model",
                    description=(
                        "frozen dataclasses for internal data models"
                        if frozen
                        else "dataclasses for internal data models"
                    ),
                    evidence=dataclass_ev,
                )
            )

        # Typing: from __future__ import annotations + type hints.
        typing_ev = evidence_for(lambda s: "->" in s.raw_code and "def " in s.raw_code)
        if typing_ev:
            findings.append(
                PatternFinding(
                    category="naming",
                    description="full type hints on function/method signatures",
                    evidence=typing_ev,
                )
            )

        # Error handling: try/except with logging or raise-from.
        error_ev = evidence_for(
            lambda s: "try:" in s.raw_code and ("except" in s.raw_code)
        )
        if error_ev:
            findings.append(
                PatternFinding(
                    category="error_handling",
                    description="try/except guards around fallible operations",
                    evidence=error_ev,
                )
            )

        # Testing: pytest-style test_* functions / Test* classes.
        testing_ev = evidence_for(
            lambda s: (s.symbol_name.startswith("test_") and s.symbol_type == "function")
            or (s.symbol_name.startswith("Test") and s.symbol_type == "class")
        )
        if testing_ev:
            findings.append(
                PatternFinding(
                    category="testing",
                    description="pytest test_* functions / Test* classes under tests/",
                    evidence=testing_ev,
                )
            )

        # Config: SpineConfig.load() usage.
        config_ev = evidence_for(lambda s: "SpineConfig" in s.raw_code or "config.yaml" in s.raw_code)
        if config_ev:
            findings.append(
                PatternFinding(
                    category="config",
                    description="centralised SpineConfig.load() configuration access",
                    evidence=config_ev,
                )
            )

        return findings

    # ── Tech-stack inference ────────────────────────────────────────────

    @staticmethod
    def _infer_tech_stack(files: list[str], seed: list[str]) -> list[str]:
        """Merge caller-seeded stack tags with languages inferred from files."""
        stack: list[str] = list(seed)
        seen = {s.lower() for s in stack}
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            lang = _EXT_TO_LANG.get(ext)
            if lang and lang not in seen:
                stack.append(lang)
                seen.add(lang)
        return stack


_PACKAGE_MARKERS: tuple[str, ...] = (
    "package.json",
    "pyproject.toml",
    "setup.py",
    "Cargo.toml",
)

_LIB_DIR_HINTS: frozenset[str] = frozenset({"libs", "lib", "packages", "shared", "common"})


def _pkg_kind(name: str, parent: str | None) -> str:
    """Heuristic: packages under a lib-hint dir, or named lib-like → 'lib'."""
    if parent and parent.lower() in _LIB_DIR_HINTS:
        return "lib"
    if name.lower() in _LIB_DIR_HINTS or name.lower().startswith("lib"):
        return "lib"
    return "app"


def detect_workspace_packages(workspace_root: str) -> list[dict]:
    """Scan depth-1 and depth-2 for workspace package marker files.

    Returns a list of package dicts with keys ``name``, ``dotted_name``,
    ``path``, ``kind`` (``"app"`` or ``"lib"``), and ``marker`` (the
    triggering file name). Returns an empty list for single-app repos or on
    any filesystem error.

    Depth-2 packages are only scanned under directories that are NOT
    themselves packages at depth 1, preventing false sub-packages like
    ``admin.src`` when ``admin/package.json`` already exists.
    """
    packages: list[dict] = []
    found_at_depth1: set[str] = set()
    try:
        depth1 = sorted(os.listdir(workspace_root))
    except OSError:
        return []

    for entry in depth1:
        if entry.startswith(".") or entry in _SKIP_DIRS:
            continue
        entry_abs = os.path.join(workspace_root, entry)
        if not os.path.isdir(entry_abs):
            continue
        for marker in _PACKAGE_MARKERS:
            if os.path.isfile(os.path.join(entry_abs, marker)):
                found_at_depth1.add(entry)
                packages.append({
                    "name": entry,
                    "dotted_name": entry,
                    "path": entry,
                    "kind": _pkg_kind(entry, parent=None),
                    "marker": marker,
                })
                break

    for entry in depth1:
        if entry.startswith(".") or entry in _SKIP_DIRS or entry in found_at_depth1:
            continue
        entry_abs = os.path.join(workspace_root, entry)
        if not os.path.isdir(entry_abs):
            continue
        try:
            subs = sorted(os.listdir(entry_abs))
        except OSError:
            continue
        for sub in subs:
            if sub.startswith(".") or sub in _SKIP_DIRS:
                continue
            sub_abs = os.path.join(entry_abs, sub)
            if not os.path.isdir(sub_abs):
                continue
            for marker in _PACKAGE_MARKERS:
                if os.path.isfile(os.path.join(sub_abs, marker)):
                    packages.append({
                        "name": sub,
                        "dotted_name": f"{entry}.{sub}",
                        "path": f"{entry}/{sub}",
                        "kind": _pkg_kind(sub, parent=entry),
                        "marker": marker,
                    })
                    break

    return packages


def _build_pkg_index(workspace_packages: list[dict]) -> list[tuple[str, str]]:
    """Return ``(path_prefix, dotted_name)`` pairs sorted longest-prefix first."""
    return sorted(
        [(p["path"].rstrip("/"), p["dotted_name"]) for p in workspace_packages],
        key=lambda x: -len(x[0]),
    )


def _module_of_with_packages(file_path: str, pkg_index: list[tuple[str, str]]) -> str:
    """Resolve *file_path* to its module name using a pre-built *pkg_index*.

    Tries each (prefix, dotted_name) pair longest-first. Falls back to the
    two-segment logic when no package matches.
    """
    norm = file_path.replace("\\", "/")
    for prefix, dotted_name in pkg_index:
        if norm == prefix or norm.startswith(prefix + "/"):
            return dotted_name
    parts = norm.split("/")
    return ".".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else "")


def group_symbols_by_module(
    symbols: list[Symbol],
    workspace_packages: list[dict] | None = None,
) -> dict[str, tuple[list[Symbol], str]]:
    """Group symbols by package root (monorepo) or two-segment prefix (single-app).

    Module-level so the distributed analysis manager can split symbols into
    per-unit work the same way the monolithic :meth:`RepoAnalyzer._build_boundaries`
    does.

    When *workspace_packages* is provided and non-empty, each file is matched
    against the longest package path prefix (greedy); files outside any detected
    package fall back to the two-segment prefix logic. When *workspace_packages*
    is empty or ``None``, the original two-segment behaviour is used unchanged.

    Args:
        symbols: The flat symbol list from :meth:`RepoAnalyzer._extract_symbols`.
        workspace_packages: Optional list from :func:`detect_workspace_packages`.

    Returns:
        ``{module_name: (symbols_in_module, module_path)}`` keyed by dotted
        module name with its repo-relative directory path.
    """
    pkg_index = _build_pkg_index(workspace_packages) if workspace_packages else []
    by_module: dict[str, list[Symbol]] = defaultdict(list)
    module_path: dict[str, str] = {}
    for sym in symbols:
        norm = sym.file_path.replace("\\", "/")
        if pkg_index:
            name = _module_of_with_packages(norm, pkg_index)
            # Derive path from the matched package prefix or fallback segments
            matched_prefix = next(
                (prefix for prefix, dn in pkg_index if dn == name), None
            )
            path = matched_prefix if matched_prefix is not None else "/".join(norm.split("/")[:2])
        else:
            parts = norm.split("/")
            pkg_parts = parts[:2] if len(parts) >= 2 else parts[:1]
            path = "/".join(pkg_parts)
            name = ".".join(pkg_parts)
        by_module[name].append(sym)
        module_path[name] = path
    return {name: (by_module[name], module_path[name]) for name in by_module}


async def build_repo_manifest(
    workspace_root: str,
    mode: str = "brownfield",
    tech_stack: list[str] | None = None,
    config: SpineConfig | None = None,
) -> RepoManifest:
    """Module-level convenience wrapper around :meth:`RepoAnalyzer.analyze`.

    Args:
        workspace_root: Absolute path to the repository to analyse.
        mode: ``"brownfield"`` or ``"greenfield"``.
        tech_stack: Caller-supplied stack tags (required for greenfield).
        config: Optional :class:`SpineConfig`; defaults to ``SpineConfig.load()``.

    Returns:
        The compiled :class:`RepoManifest`.
    """
    analyzer = RepoAnalyzer(config=config)
    return await analyzer.analyze(workspace_root, mode=mode, tech_stack=tech_stack)
