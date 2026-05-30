"""The composed onboarding ``StateGraph`` (design Revision 2, §2.1, §4.2, §5).

This module wires the two phases that already exist as standalone halves into a
single, isolated, in-engine graph:

- **Phase A — analysis** (deterministic map-reduce,
  :mod:`spine.work.onboarding.analysis_nodes`): ``analysis_manager`` →
  ``Send`` per unit → ``analysis_explorer`` → ``aggregate_analysis``. The
  aggregator assembles + persists the :class:`~spine.work.onboarding.manifest.RepoManifest`
  ONCE and sets ``manifest`` in state.
- **Phase B — synthesis** (two-tier manager/worker hierarchy,
  :mod:`spine.work.onboarding.synthesis_nodes`): ``doc_manager`` → ``Send``
  per section → ``section_worker`` → ``assemble_docs`` → ``aggregate_synthesis``.

The manifest flows **in-state** from Phase A to Phase B (single round): an
**unconditional edge** ``aggregate_analysis → doc_manager`` hands the manifest
straight to the documentation manager — no LLM ever receives the whole manifest
(the manager reads only the compact index, workers only bounded fragments).

``build_onboarding_graph`` returns an *uncompiled* ``StateGraph``; the engine
compiles it with a per-work :class:`AsyncSqliteSaver` and ``ainvoke``s it.

Progress reporting is threaded through ``RunnableConfig["configurable"]``: when
the engine seeds a ``progress`` callback, the wrapped ``analysis_manager`` and
``doc_manager`` nodes fire it with the SAME ``current_phase`` strings the UI
expects (``analyze`` at the start of analysis, ``synthesize`` at the start of
synthesis). ``scaffold`` is fired by the engine before the graph runs
(greenfield only); ``completed`` after the graph finishes.

The graph is deliberately NOT registered in ``compose.py`` /
``WORKFLOW_SEQUENCES`` / ``_SUBGRAPH_BUILDER_REGISTRY`` — onboarding needs no
critic/rework/human-review/inter-phase merge, and registering it would force
``PhaseName`` enum churn through every exhaustive map (the documented Risk #1).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from spine.work.onboarding.analysis_nodes import (
    _aggregate_analysis_node,
    _analysis_explorer_node,
    _analysis_manager_node,
    _analysis_router,
)
from spine.work.onboarding.onboarding_state import OnboardingGraphState
from spine.work.onboarding.synthesis_nodes import (
    _aggregate_synthesis_node,
    _assemble_docs_node,
    _doc_manager_node,
    _section_router,
    _section_worker_node,
)

logger = logging.getLogger(__name__)


# ── Module-level route maps ──────────────────────────────────────────────────
#
# Conditional-edge destination lists, hoisted to module level so they are a
# single source of truth (mirrors the analysis/synthesis half builders).

#: Destinations of the analysis manager's ``_analysis_router`` (Send per unit
#: → ``analysis_explorer``, or the plain node name when there are no units).
ANALYSIS_ROUTE_MAP: list[str] = ["analysis_explorer", "aggregate_analysis"]

#: Destinations of the doc manager's ``_section_router`` (one Send per section
#: → ``section_worker``).
SECTION_ROUTE_MAP: list[str] = ["section_worker"]


# ── Progress callback plumbing ───────────────────────────────────────────────


def _fire_progress(config: RunnableConfig | None, phase: str) -> None:
    """Invoke the engine-supplied progress callback for ``phase``, if present.

    The engine seeds ``configurable["progress"]`` with a callable that fires
    ``update_work_phase_started`` / ``_update_work_progress`` against the work
    DB. Node wrappers call this at phase boundaries with the SAME phase strings
    the UI expects (``analyze`` / ``synthesize``). A missing or failing callback
    never breaks the graph run.
    """
    if not config:
        return
    configurable = config.get("configurable", {}) or {}
    callback = configurable.get("progress")
    if not callable(callback):
        return
    try:
        callback(phase)
    except Exception:  # noqa: BLE001 - progress reporting is best-effort
        logger.warning("onboarding progress callback failed for phase %s", phase)


async def _analysis_manager_with_progress(
    state: OnboardingGraphState,
    config: RunnableConfig = None,  # noqa: RUF013 - LangGraph injects; None for direct calls
) -> dict[str, Any]:
    """``analysis_manager`` wrapper that fires the ``analyze`` phase progress."""
    _fire_progress(config, "analyze")
    return await _analysis_manager_node(state, config)


async def _doc_manager_with_progress(
    state: OnboardingGraphState,
    config: RunnableConfig = None,  # noqa: RUF013 - LangGraph injects; None for direct calls
) -> dict[str, Any]:
    """``doc_manager`` wrapper that fires the ``synthesize`` phase progress."""
    _fire_progress(config, "synthesize")
    return await _doc_manager_node(state, config)


# ── Graph builder ────────────────────────────────────────────────────────────


def build_onboarding_graph() -> StateGraph:
    """Build the composed onboarding graph (Phase A → Phase B), uncompiled.

    The returned :class:`StateGraph` is NOT compiled — the engine compiles it
    with a per-work :class:`AsyncSqliteSaver` (with a no-checkpointer fallback)
    and ``ainvoke``s it. It expects an initial state carrying at least
    ``work_id``, ``workspace_root``, ``mode``, and ``tech_stack``.

    Topology::

        START → analysis_manager
              → (Send per unit) → analysis_explorer
              → aggregate_analysis          # assembles + persists manifest ONCE
              → doc_manager                 # UNCONDITIONAL edge; manifest in-state
              → (Send per section) → section_worker
              → assemble_docs
              → aggregate_synthesis → END

    The ``analysis_manager`` and ``doc_manager`` nodes are wrapped so they fire
    the ``analyze`` / ``synthesize`` progress callbacks (when the engine seeds
    one) without changing their state-transformation behaviour.

    Returns:
        An uncompiled ``StateGraph`` over :class:`OnboardingGraphState`.
    """
    graph: StateGraph = StateGraph(OnboardingGraphState)

    # Phase A — analysis map-reduce.
    graph.add_node("analysis_manager", _analysis_manager_with_progress)
    graph.add_node("analysis_explorer", _analysis_explorer_node)
    graph.add_node("aggregate_analysis", _aggregate_analysis_node)

    # Phase B — synthesis hierarchy.
    graph.add_node("doc_manager", _doc_manager_with_progress)
    graph.add_node("section_worker", _section_worker_node)
    graph.add_node("assemble_docs", _assemble_docs_node)
    graph.add_node("aggregate_synthesis", _aggregate_synthesis_node)

    # Phase A wiring.
    graph.add_edge(START, "analysis_manager")
    graph.add_conditional_edges(
        "analysis_manager",
        _analysis_router,
        ANALYSIS_ROUTE_MAP,
    )
    graph.add_edge("analysis_explorer", "aggregate_analysis")

    # A → B hand-off: the manifest flows in-state via a single UNCONDITIONAL
    # edge (single round, no convergence loop, no whole-manifest LLM call).
    graph.add_edge("aggregate_analysis", "doc_manager")

    # Phase B wiring.
    graph.add_conditional_edges(
        "doc_manager",
        _section_router,
        SECTION_ROUTE_MAP,
    )
    graph.add_edge("section_worker", "assemble_docs")
    graph.add_edge("assemble_docs", "aggregate_synthesis")
    graph.add_edge("aggregate_synthesis", END)

    return graph


# Public re-export so callers can build a progress callback inline if desired.
ProgressCallback = Callable[[str], None]
