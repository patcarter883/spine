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

Progress reporting lives entirely in the engine: it streams node updates
(``astream(stream_mode="updates")``) and maps the ``analysis_manager`` /
``doc_manager`` node boundaries to the ``analyze`` / ``synthesize``
``current_phase`` strings the UI expects (see
:mod:`spine.work.onboarding.phases` for the shared vocabulary). ``scaffold`` is
fired by the engine before the graph runs (greenfield only); ``completed``
after the graph finishes. The graph nodes therefore stay pure — no progress
wrappers — so there is exactly ONE progress mechanism.

The graph is deliberately NOT registered in ``compose.py`` /
``WORKFLOW_SEQUENCES`` / ``_SUBGRAPH_BUILDER_REGISTRY`` — onboarding needs no
critic/rework/human-review/inter-phase merge, and registering it would force
``PhaseName`` enum churn through every exhaustive map (the documented Risk #1).
"""

from __future__ import annotations

from typing import Callable

from langgraph.graph import END, START, StateGraph

from spine.work.onboarding.analysis_nodes import add_analysis_nodes_and_edges
from spine.work.onboarding.onboarding_state import OnboardingGraphState
from spine.work.onboarding.synthesis_nodes import add_synthesis_nodes_and_edges


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

    The ``analysis_manager`` / ``doc_manager`` node boundaries drive the
    ``analyze`` / ``synthesize`` progress reporting, but that mapping lives in
    the engine's update stream — the nodes themselves are unwrapped and pure.

    Returns:
        An uncompiled ``StateGraph`` over :class:`OnboardingGraphState`.
    """
    graph: StateGraph = StateGraph(OnboardingGraphState)

    # Phase A + Phase B interior nodes/edges come from the SAME helpers the
    # standalone half-builders use (single source of truth for each phase's
    # topology) — this composed builder only supplies the START entry, the A→B
    # hand-off, and the END terminal so the two halves are stitched into one
    # graph without re-declaring their interior wiring.
    add_analysis_nodes_and_edges(graph)
    add_synthesis_nodes_and_edges(graph)

    graph.add_edge(START, "analysis_manager")

    # A → B hand-off: the manifest flows in-state via a single UNCONDITIONAL
    # edge (single round, no convergence loop, no whole-manifest LLM call).
    graph.add_edge("aggregate_analysis", "doc_manager")

    graph.add_edge("aggregate_synthesis", END)

    return graph


# Public re-export so callers can build a progress callback inline if desired.
ProgressCallback = Callable[[str], None]
