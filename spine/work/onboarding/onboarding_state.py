"""Graph state for the distributed onboarding engine.

This module defines :class:`OnboardingGraphState`, the single ``TypedDict`` the
isolated, in-engine onboarding ``StateGraph`` threads through every node
(design Revision 2, §3). It is kept OUT of
:mod:`spine.workflow.subgraph_state` deliberately: onboarding runs its own
graph (no critic / rework / human-review / inter-phase merge) and registering
it in the shared subgraph machinery would force ``PhaseName`` enum churn
through every exhaustive map (the documented Risk #1).

The state spans both phases of the graph:

- **Phase A — analysis** (deterministic map-reduce; unused until PR-3 but
  defined now so the schema is stable): a manager seeds ``analysis_units``,
  one explorer per unit appends ``[one slice]`` to ``repo_slices``
  (``operator.add``), and an aggregator sets ``manifest`` / ``manifest_path``
  once.
- **Phase B — synthesis** (this PR): the documentation manager builds
  ``manifest_index`` and seeds ``sections`` once; one worker per section
  appends ``[one SectionResult]`` to ``section_results`` (``operator.add``);
  the assembler/aggregator set ``written`` once.

Reducer rationale (NO new reducers — reuse the three proven ones):

- :data:`operator.add` on ``repo_slices`` and ``section_results`` — each
  parallel ``Send`` branch emits exactly ``[one_dict]``; LangGraph applies
  reducer updates sequentially within a super-step, so N branches compose
  race-free.
- :func:`_slice_list_reducer` on ``sections`` — reused verbatim; seeds the
  list and keeps the door open for ``{"add": [...], "remove": [id...]}``
  directives (e.g. a manager that re-plans).
- :func:`_merge_read_cache` on ``read_cache`` — LLM-enriched explorer mode
  only (PR-5).
- **No reducer** on single-writer channels (``analysis_units``, ``manifest``,
  ``manifest_path``, ``manifest_index``, ``written``): a reducer there is the
  documented "aggregate appends/duplicates" footgun. Aggregate/assemble nodes
  *transform but do not re-emit* the ``operator.add`` channels.
"""

from __future__ import annotations

from operator import add as _op_add
from typing import Annotated

from typing_extensions import TypedDict

from spine.models.state import _merge_read_cache
from spine.workflow.subgraph_state import _slice_list_reducer


class OnboardingGraphState(TypedDict, total=False):
    """State threaded through the isolated onboarding ``StateGraph``."""

    # ── inputs (seeded by the engine before ainvoke) ──
    work_id: str
    workspace_root: str
    mode: str  # "brownfield" | "greenfield"
    tech_stack: list[str]

    # ── PHASE A: analysis (deterministic map-reduce, PR-3) ──
    analysis_units: list[dict]  # set ONCE by the manager; single writer → no reducer
    active_unit: dict  # transient, per-Send
    repo_slices: Annotated[list[dict], _op_add]  # each explorer appends [one slice]
    # Single-writer signals the manager seeds for the aggregator (no reducer):
    summaries: list  # [[file_path, symbol_name, summary], ...] enrichment cache
    symbol_order: list  # [[file_path, symbol_name], ...] global discovery (scan) order
    analysis_notes: list[str]  # caveats merged into the manifest's notes
    generated_at: str  # ISO timestamp captured once at analysis start
    symbol_count: int  # total extracted symbols
    file_count: int  # files that yielded symbols
    prebuilt_manifest: dict  # greenfield / monolithic-fallback manifest passthrough
    workspace_packages: list[dict]  # set ONCE by analysis_manager; single writer
    is_monorepo: bool  # set ONCE by analysis_manager; single writer
    manifest: dict  # RepoManifest.to_dict(); set ONCE by aggregator
    manifest_path: str  # set ONCE by aggregator

    # ── PHASE B: synthesis plan (TIER A) ──
    manifest_index: dict  # compact index; set ONCE by the plan prelude
    section_token_cap: int  # per-fragment ceiling; set ONCE by the manager (for the router)
    sections: Annotated[list[dict], _slice_list_reducer]  # SectionPlan items; seeded once
    active_section: dict  # transient, per-Send: {doc_id, order, title, fragment, instruction}

    # ── PHASE B: synthesis fan-out (TIER B/C) ──
    section_results: Annotated[list[dict], _op_add]  # each worker appends [one SectionResult]
    written: dict  # {doc_name: path}; set ONCE by aggregate_synthesis
    placeholder_docs: list  # docs written placeholder-only; set ONCE by assemble_docs

    # ── shared dedupe (LLM-enriched explorer mode ONLY) ──
    read_cache: Annotated[dict, _merge_read_cache]
