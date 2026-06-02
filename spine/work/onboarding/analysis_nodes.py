"""Analysis map-reduce nodes for the distributed onboarding engine.

This module implements Phase A of the onboarding graph (design Revision 2,
§2.4, §4.2, §8) — a *single-round deterministic map-reduce* that produces a
:class:`spine.work.onboarding.manifest.RepoManifest` **byte-identical** (after a
name-sort) to the monolithic :meth:`spine.work.onboarding.analyzer.RepoAnalyzer.analyze`
path:

- **:func:`_analysis_manager_node`** (the map's prelude): runs the deterministic
  discovery + symbol extraction *once*, groups the symbols into per-package
  ``analysis_units`` via :func:`group_symbols_by_module`, and seeds the summary
  cache + tech/notes signals into state. Greenfield short-circuits to **zero**
  units and seeds the greenfield manifest directly. When
  ``onboarding_distributed_analysis`` is ``False`` the manager runs the
  monolithic :meth:`RepoAnalyzer.analyze` inline and emits zero units (the
  router then routes straight to the aggregator, which simply passes the
  pre-built manifest through).
- **:func:`_analysis_router`**: emits one ``Send("analysis_explorer", ...)`` per
  unit, or routes to ``aggregate_analysis`` when there are no units.
- **:func:`_analysis_explorer_node`** (the map): **deterministic by default**.
  For its one assigned unit it builds the :class:`ModuleBoundary` (via
  :meth:`RepoAnalyzer._build_boundary_for_unit`), extracts the unit's pattern
  findings (via :meth:`RepoAnalyzer._extract_patterns_for_unit`), and emits the
  unit's ``raw_imports`` so the aggregator can resolve cross-module edges
  globally. It appends exactly ``[one slice]`` to ``repo_slices``
  (``operator.add``). The opt-in LLM-enrich mode (``onboarding_explorer_llm``)
  is NOT implemented in this PR — the deterministic branch is the only path.
- **:func:`_aggregate_analysis_node`** (the reduce): looks up slices BY KEY
  (never by index, so completion order is irrelevant), resolves the GLOBAL
  cross-module dependency edges over the union of every unit's ``raw_imports``,
  dedupes + re-caps the merged pattern findings to <=3 evidence each, sorts the
  boundaries by name, assembles the :class:`RepoManifest`, persists it ONCE, and
  sets ``manifest`` / ``manifest_path``.

The MEMORY rule is respected: explorers make no tool/LLM calls in the default
deterministic mode, so there is no tool-error text to leak. A future LLM-enrich
explorer (PR-5, out of scope) would carry only ``{error, module_name}`` — never
raw exception text.

:func:`build_analysis_graph` wires manager → (Send per unit) → explorer →
aggregate into an uncompiled ``StateGraph`` for the parity tests.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from spine.agents.tools.ast_extract import Symbol
from spine.config import SpineConfig
from spine.work.onboarding.analyzer import RepoAnalyzer, group_symbols_by_module
from spine.work.onboarding.manifest import (
    DependencyEdge,
    ModuleBoundary,
    PatternFinding,
    RepoManifest,
    SymbolRef,
)
from spine.work.onboarding.onboarding_state import OnboardingGraphState

logger = logging.getLogger(__name__)

# Evidence cap per pattern finding — mirrors the monolithic
# ``RepoAnalyzer._extract_patterns`` ``evidence_for`` cap so the re-capped union
# matches byte-for-byte.
_EVIDENCE_CAP = 3


# ── Config / state helpers ───────────────────────────────────────────────────


def _spine_config(config: RunnableConfig | None) -> SpineConfig:
    """Resolve the :class:`SpineConfig` from the runnable config (or load it)."""
    if config:
        configurable = config.get("configurable", {}) or {}
        spine_config = configurable.get("spine_config")
        if isinstance(spine_config, SpineConfig):
            return spine_config
    return SpineConfig.load()


def _distributed_analysis_enabled(config: RunnableConfig | None) -> bool:
    """Whether the distributed map-reduce analysis path is enabled."""
    return bool(getattr(_spine_config(config), "onboarding_distributed_analysis", True))


def _symbol_to_dict(sym: Symbol) -> dict[str, Any]:
    """Serialise a :class:`Symbol` to the plain dict the state carries."""
    return {
        "file_path": sym.file_path,
        "symbol_name": sym.symbol_name,
        "symbol_type": sym.symbol_type,
        "raw_code": sym.raw_code,
        "start_byte": sym.start_byte,
        "end_byte": sym.end_byte,
        "lang": sym.lang,
    }


def _symbol_from_dict(data: dict[str, Any]) -> Symbol:
    """Reconstruct a :class:`Symbol` from its serialised dict form."""
    return Symbol(
        file_path=data["file_path"],
        symbol_name=data["symbol_name"],
        symbol_type=data["symbol_type"],
        raw_code=data.get("raw_code", ""),
        start_byte=int(data.get("start_byte", 0) or 0),
        end_byte=int(data.get("end_byte", 0) or 0),
        lang=data.get("lang", ""),
    )


def _summaries_from_entries(raw: Any) -> dict[tuple[str, str], str]:
    """Rebuild the ``(file_path, symbol_name) -> summary`` map from a triple list.

    Summaries are serialised as a list of ``[file_path, symbol_name, summary]``
    triples (tuple keys are not JSON/state friendly). Used for both the per-unit
    slice carried on ``active_unit`` (finding #9) and the legacy global channel.
    """
    out: dict[tuple[str, str], str] = {}
    for entry in raw or []:
        if isinstance(entry, (list, tuple)) and len(entry) == 3:
            out[(entry[0], entry[1])] = entry[2]
    return out


def _summaries_from_state(state: OnboardingGraphState) -> dict[tuple[str, str], str]:
    """Rebuild the global summaries map from the legacy ``state["summaries"]``."""
    return _summaries_from_entries(state.get("summaries"))


# ── Manager (map prelude) ────────────────────────────────────────────────────


async def _analysis_manager_node(
    state: OnboardingGraphState,
    config: RunnableConfig = None,  # noqa: RUF013 - LangGraph injects; None for direct calls
) -> dict[str, Any]:
    """Deterministic prelude: discover files, extract symbols, group into units.

    Greenfield short-circuits to a seeded greenfield manifest and **zero**
    units. When ``onboarding_distributed_analysis`` is ``False`` the manager runs
    the monolithic :meth:`RepoAnalyzer.analyze` inline and emits zero units (the
    aggregator then passes the manifest through unchanged).

    Otherwise it runs discovery + symbol extraction once, groups symbols into
    per-package ``analysis_units`` (each unit carries its serialised symbol
    subset + name + path), and seeds the summary cache, tech stack, notes, file
    count, and a generated-at timestamp into state for the aggregator.

    Returns the state delta. ``analysis_units`` is a single-writer channel.
    """
    work_id = state.get("work_id", "unknown")
    workspace_root = state.get("workspace_root", "")
    mode = state.get("mode", "brownfield")
    seed_stack = list(state.get("tech_stack", []) or [])
    generated_at = datetime.now(timezone.utc).isoformat()

    spine_config = _spine_config(config)
    analyzer = RepoAnalyzer(config=spine_config)

    # Greenfield: no discovery, no units — seed the greenfield manifest and let
    # the aggregator pass it through.
    if mode == "greenfield":
        manifest = await analyzer.analyze(
            workspace_root, mode="greenfield", tech_stack=seed_stack
        )
        logger.info("[%s] analysis_manager: greenfield seed (0 units)", work_id)
        return {
            "analysis_units": [],
            "prebuilt_manifest": manifest.to_dict(),
        }

    # Monolithic fallback (tiny repos / flag off): build the whole manifest here
    # and emit zero units so the aggregator just passes it through.
    if not _distributed_analysis_enabled(config):
        manifest = await analyzer.analyze(
            workspace_root, mode="brownfield", tech_stack=seed_stack
        )
        logger.info(
            "[%s] analysis_manager: distributed analysis disabled — monolithic "
            "manifest (0 units)",
            work_id,
        )
        return {
            "analysis_units": [],
            "prebuilt_manifest": manifest.to_dict(),
        }

    # Distributed deterministic path: discover + extract once, then fan out.
    files = await analyzer._discover_files(workspace_root)  # noqa: SLF001
    symbols, parsed_files = analyzer._extract_symbols(workspace_root, files)  # noqa: SLF001

    notes_parts: list[str] = []
    if not files:
        notes_parts.append("file discovery returned nothing (index + walk both empty)")
    summaries = analyzer._load_summaries()  # noqa: SLF001
    if summaries:
        notes_parts.append(f"enriched {len(summaries)} summaries from vector index")
    else:
        notes_parts.append("vector index unavailable — AST-only, no summaries")

    tech = analyzer._infer_tech_stack(files, seed_stack)  # noqa: SLF001
    from spine.work.onboarding.analyzer import detect_workspace_packages  # noqa: PLC0415
    workspace_packages = detect_workspace_packages(workspace_root)
    is_monorepo = len(workspace_packages) >= 2
    grouped = group_symbols_by_module(symbols, workspace_packages)

    units: list[dict[str, Any]] = []
    for name in sorted(grouped):
        unit_symbols, path = grouped[name]
        # Slice the global summaries map down to ONLY this unit's symbols and
        # attach it to the unit (finding #9). Each ``Send`` then carries just the
        # summaries the explorer can actually use — ``_build_boundary_for_unit``
        # / ``_extract_patterns_for_unit`` only ever look up ``(file_path,
        # symbol_name)`` pairs from this unit — instead of re-hydrating the whole
        # global map per fan-out. The merged manifest is unchanged because the
        # union of every unit's slice equals the global map for the keys any
        # explorer reads.
        unit_summaries = [
            [s.file_path, s.symbol_name, summaries[(s.file_path, s.symbol_name)]]
            for s in unit_symbols
            if (s.file_path, s.symbol_name) in summaries
        ]
        units.append(
            {
                "name": name,
                "path": path,
                "symbols": [_symbol_to_dict(s) for s in unit_symbols],
                "summaries": unit_summaries,
                "workspace_packages": workspace_packages,
            }
        )

    # Global symbol scan order (discovery order) so the aggregator can re-cap
    # pattern evidence in exactly the order the monolithic ``_extract_patterns``
    # would scan — guaranteeing byte-parity regardless of per-unit fan-out.
    symbol_order = [[s.file_path, s.symbol_name] for s in symbols]

    logger.info(
        "[%s] analysis_manager: %d units from %d symbols / %d files",
        work_id,
        len(units),
        len(symbols),
        parsed_files,
    )
    return {
        # ``summaries`` is no longer seeded into shared state: each unit carries
        # its own sliced summaries (finding #9), so a giant global map is never
        # serialised once per explorer Send.
        "analysis_units": units,
        "symbol_order": symbol_order,
        "tech_stack": tech,
        "file_count": parsed_files,
        "symbol_count": len(symbols),
        "analysis_notes": notes_parts,
        "generated_at": generated_at,
        "workspace_packages": workspace_packages,
        "is_monorepo": is_monorepo,
    }


# ── Router (map fan-out) ─────────────────────────────────────────────────────


def _analysis_router(state: OnboardingGraphState) -> list[Send] | str:
    """Fan out one ``Send("analysis_explorer", ...)`` per unit, else aggregate.

    With zero units (greenfield or monolithic fallback) the function returns the
    plain node name ``"aggregate_analysis"`` (NOT a ``Send``): a ``Send`` payload
    *replaces* the node's input state, which would hide the manager's
    ``prebuilt_manifest`` from the aggregator. Returning the node name routes
    with the full graph state intact, so the aggregator sees ``prebuilt_manifest``
    and passes it through.
    """
    units = list(state.get("analysis_units", []) or [])
    if not units:
        return "aggregate_analysis"
    return [Send("analysis_explorer", {"active_unit": unit}) for unit in units]


# ── Explorer (map) ───────────────────────────────────────────────────────────


def _analysis_explorer_node(
    state: OnboardingGraphState,
    config: RunnableConfig = None,  # noqa: RUF013 - LangGraph injects; None for direct calls
) -> dict[str, Any]:
    """Deterministic map over ONE unit: boundary + patterns + raw imports.

    For its assigned ``active_unit`` it:

    - builds the :class:`ModuleBoundary` (:meth:`RepoAnalyzer._build_boundary_for_unit`),
    - extracts the unit's pattern findings
      (:meth:`RepoAnalyzer._extract_patterns_for_unit`), and
    - emits the unit's ``raw_imports`` — one ``{src, imports}`` per symbol — so
      the aggregator can resolve cross-module edges over the GLOBAL module set.

    Appends exactly ``[one slice]`` to ``repo_slices`` (``operator.add``). Makes
    no LLM/tool calls in the default deterministic mode (no error text to leak).
    """
    active = state.get("active_unit", {}) or {}
    name = active.get("name", "")
    path = active.get("path", "")
    unit_symbols = [_symbol_from_dict(s) for s in active.get("symbols", []) or []]
    # Summaries are sliced per unit at fan-out (finding #9): read the unit's own
    # slice from ``active_unit`` rather than re-parsing a global map. Fall back to
    # the legacy global ``state["summaries"]`` channel for back-compat callers
    # that still seed it.
    summaries = _summaries_from_entries(active.get("summaries"))
    if not summaries:
        summaries = _summaries_from_state(state)

    spine_config = _spine_config(config)
    analyzer = RepoAnalyzer(config=spine_config)

    boundary = analyzer._build_boundary_for_unit(  # noqa: SLF001
        name, path, unit_symbols, summaries
    )
    patterns = analyzer._extract_patterns_for_unit(unit_symbols, summaries)  # noqa: SLF001

    from spine.work.onboarding.analyzer import _build_pkg_index, _module_of_with_packages  # noqa: PLC0415
    unit_wp = active.get("workspace_packages") or []
    pkg_index = _build_pkg_index(unit_wp) if unit_wp else []
    raw_imports: list[dict[str, Any]] = []
    for sym in unit_symbols:
        if pkg_index:
            src_module = _module_of_with_packages(sym.file_path, pkg_index)
        else:
            src_module = RepoAnalyzer._module_of(sym.file_path)  # noqa: SLF001
        imports = RepoAnalyzer._iter_imports(sym.raw_code)  # noqa: SLF001
        if imports:
            raw_imports.append({"src": src_module, "imports": imports})

    slice_payload = {
        "module_name": name,
        "boundary": _boundary_to_dict(boundary),
        "patterns": [_pattern_to_dict(p) for p in patterns],
        "raw_imports": raw_imports,
    }
    return {"repo_slices": [slice_payload]}


# ── Aggregator (reduce) ──────────────────────────────────────────────────────


def _aggregate_analysis_node(
    state: OnboardingGraphState,
    config: RunnableConfig = None,  # noqa: RUF013 - LangGraph injects; None for direct calls
) -> dict[str, Any]:
    """Reduce: assemble + persist the manifest from the per-unit slices.

    Looks slices up BY KEY (``module_name``) so completion order is irrelevant
    (design Risk #7). Resolves GLOBAL cross-module dependency edges over the
    union of every unit's ``raw_imports`` (so an import that crosses into a
    module owned by a *different* explorer is still recorded — exactly the
    monolithic behaviour). Dedupes + re-caps merged patterns to <=3 evidence
    each, sorts boundaries by name, assembles the :class:`RepoManifest`, persists
    it ONCE, and sets ``manifest`` / ``manifest_path``.

    When the manager pre-built a manifest (greenfield / monolithic fallback) it
    is passed through unchanged (still persisted once).
    """
    work_id = state.get("work_id", "unknown")
    workspace_root = state.get("workspace_root", "")

    prebuilt = state.get("prebuilt_manifest")
    if prebuilt:
        manifest = RepoManifest.from_dict(prebuilt)
        manifest_path = _persist(manifest, workspace_root, work_id, config)
        logger.info("[%s] aggregate_analysis: passed pre-built manifest through", work_id)
        return {"manifest": manifest.to_dict(), "manifest_path": manifest_path}

    slices = list(state.get("repo_slices", []) or [])

    # Boundaries, looked up by key then sorted by name (order-independent).
    boundaries = sorted(
        (_boundary_from_dict(s["boundary"]) for s in slices if s.get("boundary")),
        key=lambda b: b.name,
    )
    module_names = {b.name for b in boundaries}

    # Global dependency edges over the union of raw imports. Mirrors
    # ``RepoAnalyzer._build_dependency_edges`` exactly: ``_match_module`` against
    # the GLOBAL module set, drop self-edges, dedupe, sort.
    edges_set: set[tuple[str, str]] = set()
    for sl in slices:
        for entry in sl.get("raw_imports", []) or []:
            src_module = entry.get("src", "")
            for imported in entry.get("imports", []) or []:
                dst = RepoAnalyzer._match_module(imported, module_names)  # noqa: SLF001
                if dst and dst != src_module:
                    edges_set.add((src_module, dst))
    dependency_chains = [
        DependencyEdge(src=src, dst=dst, kind="depends_on")
        for src, dst in sorted(edges_set)
    ]

    symbol_order = [tuple(x) for x in (state.get("symbol_order") or [])]
    patterns = _merge_patterns(slices, symbol_order)

    tech = list(state.get("tech_stack", []) or [])
    notes_parts = list(state.get("analysis_notes", []) or [])
    core_domains = [b.name for b in boundaries]

    manifest = RepoManifest(
        workspace_root=workspace_root,
        mode="brownfield",
        tech_stack=tech,
        core_domains=core_domains,
        module_boundaries=boundaries,
        dependency_chains=dependency_chains,
        patterns=patterns,
        symbol_count=int(state.get("symbol_count", 0) or 0),
        file_count=int(state.get("file_count", 0) or 0),
        generated_at=state.get("generated_at", "")
        or datetime.now(timezone.utc).isoformat(),
        notes="; ".join(notes_parts),
        is_monorepo=bool(state.get("is_monorepo", False)),
        workspace_packages=list(state.get("workspace_packages", []) or []),
    )
    manifest_path = _persist(manifest, workspace_root, work_id, config)
    logger.info(
        "[%s] aggregate_analysis: %d boundaries, %d edges, %d patterns",
        work_id,
        len(boundaries),
        len(dependency_chains),
        len(patterns),
    )
    return {"manifest": manifest.to_dict(), "manifest_path": manifest_path}


def _merge_patterns(
    slices: list[dict[str, Any]],
    symbol_order: list[tuple[str, str]],
) -> list[PatternFinding]:
    """Merge per-unit pattern findings into the global, deduped, re-capped set.

    The monolithic ``_extract_patterns`` produces at most one finding per
    category, in a FIXED category order, each carrying the first <=3 matching
    symbols **in global discovery (scan) order**. Per-unit explorers each emit
    their unit's findings (already capped to the first 3 matches per category
    within the unit, which is a superset of the global top-3 for that category).
    Here we merge by category, re-order the collected evidence into the global
    ``symbol_order``, re-cap to 3, and re-emit in the monolithic category order —
    so the result is byte-identical to the monolith regardless of fan-out /
    completion order.
    """
    # Stable category order matching ``RepoAnalyzer._extract_patterns``.
    category_order = ["logging", "data_model", "naming", "error_handling", "testing", "config"]
    descriptions: dict[str, str] = {}
    all_descriptions: dict[str, set[str]] = {}
    candidates: dict[str, list[SymbolRef]] = {}
    extra_order: list[str] = []

    for sl in slices:
        for p in sl.get("patterns", []) or []:
            cat = p.get("category", "")
            if not cat:
                continue
            desc = p.get("description", "")
            if cat not in descriptions:
                descriptions[cat] = desc
                candidates[cat] = []
                all_descriptions[cat] = set()
                if cat not in category_order:
                    extra_order.append(cat)
            all_descriptions[cat].add(desc)
            for ev in p.get("evidence", []) or []:
                candidates[cat].append(_symbol_ref_from_dict(ev))

    # ``data_model`` carries a GLOBAL "frozen" signal in the monolith: the
    # frozen variant is used if ANY dataclass anywhere is frozen. A unit that
    # saw no frozen dataclass reports the plain variant; prefer the frozen
    # variant whenever any unit observed it so the merged description matches.
    if "data_model" in descriptions:
        for desc in all_descriptions["data_model"]:
            if "frozen" in desc:
                descriptions["data_model"] = desc
                break

    # Global scan-order rank for deterministic, monolith-matching re-cap.
    rank = {key: i for i, key in enumerate(symbol_order)}

    def _evidence_rank(ref: SymbolRef) -> int:
        return rank.get((ref.file_path, ref.symbol_name), len(rank))

    findings: list[PatternFinding] = []
    for cat in [*category_order, *extra_order]:
        if cat not in descriptions:
            continue
        ordered = sorted(candidates[cat], key=_evidence_rank)
        findings.append(
            PatternFinding(
                category=cat,
                description=descriptions[cat],
                evidence=ordered[:_EVIDENCE_CAP],
            )
        )
    return findings


# ── (De)serialisation helpers for the in-state slices ────────────────────────


def _symbol_ref_to_dict(ref: SymbolRef) -> dict[str, Any]:
    """Serialise a :class:`SymbolRef`."""
    return {
        "file_path": ref.file_path,
        "symbol_name": ref.symbol_name,
        "symbol_type": ref.symbol_type,
        "lang": ref.lang,
        "summary": ref.summary,
    }


def _symbol_ref_from_dict(data: dict[str, Any]) -> SymbolRef:
    """Reconstruct a :class:`SymbolRef`."""
    return SymbolRef(
        file_path=data["file_path"],
        symbol_name=data["symbol_name"],
        symbol_type=data["symbol_type"],
        lang=data["lang"],
        summary=data.get("summary", ""),
    )


def _boundary_to_dict(boundary: ModuleBoundary) -> dict[str, Any]:
    """Serialise a :class:`ModuleBoundary`."""
    return {
        "name": boundary.name,
        "path": boundary.path,
        "role": boundary.role,
        "key_symbols": [_symbol_ref_to_dict(s) for s in boundary.key_symbols],
    }


def _boundary_from_dict(data: dict[str, Any]) -> ModuleBoundary:
    """Reconstruct a :class:`ModuleBoundary`."""
    return ModuleBoundary(
        name=data["name"],
        path=data["path"],
        role=data["role"],
        key_symbols=[_symbol_ref_from_dict(s) for s in data.get("key_symbols", [])],
    )


def _pattern_to_dict(pattern: PatternFinding) -> dict[str, Any]:
    """Serialise a :class:`PatternFinding`."""
    return {
        "category": pattern.category,
        "description": pattern.description,
        "evidence": [_symbol_ref_to_dict(s) for s in pattern.evidence],
    }


def _persist(
    manifest: RepoManifest,
    workspace_root: str,
    work_id: str,
    config: RunnableConfig | None,
) -> str:
    """Persist the manifest ONCE via the engine's persistence helper.

    Delegates to :func:`spine.work.onboarding.engine._persist_manifest` so the
    distributed path writes ``repo_manifest.json`` to exactly the same location
    (and idempotently) as the monolithic engine.
    """
    from spine.work.onboarding.engine import _persist_manifest

    return _persist_manifest(manifest, workspace_root, work_id)


# ── Graph builder ────────────────────────────────────────────────────────────


#: Destinations of the analysis manager's ``_analysis_router`` (Send per unit
#: → ``analysis_explorer``, or ``aggregate_analysis`` when there are no units).
#: Hoisted so the half-builder and the composed onboarding graph share ONE
#: route-map literal.
ANALYSIS_ROUTE_MAP: list[str] = ["analysis_explorer", "aggregate_analysis"]


def add_analysis_nodes_and_edges(graph: StateGraph) -> None:
    """Add the analysis-phase nodes + INTERIOR edges to ``graph`` in place.

    Adds ``analysis_manager`` / ``analysis_explorer`` / ``aggregate_analysis``
    and the interior wiring (manager ``Send`` fan-out → explorer →
    ``aggregate_analysis``), but NOT the ``START`` entry edge or any terminal
    edge — the caller owns those so the same wiring can serve both the
    standalone half-graph (``START`` → … → ``END``) and the composed onboarding
    graph (``aggregate_analysis`` → ``doc_manager``). This is the single source
    of truth for Phase A's topology, shared with
    :func:`spine.work.onboarding.onboarding_graph.build_onboarding_graph`.
    """
    graph.add_node("analysis_manager", _analysis_manager_node)
    graph.add_node("analysis_explorer", _analysis_explorer_node)
    graph.add_node("aggregate_analysis", _aggregate_analysis_node)

    graph.add_conditional_edges(
        "analysis_manager",
        _analysis_router,
        ANALYSIS_ROUTE_MAP,
    )
    graph.add_edge("analysis_explorer", "aggregate_analysis")


def build_analysis_graph() -> StateGraph:
    """Build the analysis half of the onboarding graph (manager → explorer → agg).

    The returned graph is uncompiled. It expects an initial state carrying at
    least ``work_id``, ``workspace_root``, ``mode``, and ``tech_stack``. It runs:

        analysis_manager → (Send per unit) → analysis_explorer
        → aggregate_analysis → END

    Used by :mod:`tests.unit.test_onboarding_analysis_parity`. The interior
    nodes/edges are shared with the composed graph via
    :func:`add_analysis_nodes_and_edges`; this builder only adds the ``START``
    and ``END`` boundary edges.
    """
    graph: StateGraph = StateGraph(OnboardingGraphState)
    add_analysis_nodes_and_edges(graph)
    graph.add_edge(START, "analysis_manager")
    graph.add_edge("aggregate_analysis", END)
    return graph
