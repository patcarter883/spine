"""Shared onboarding ``current_phase`` string constants.

The onboarding engine records a small fixed vocabulary of ``current_phase``
values as a job progresses: ``"scaffold"`` (greenfield only, fired pre-graph),
``"analyze"`` (at the analysis manager), ``"synthesize"`` (at the documentation
manager), and finally ``"completed"``.

Three independent call sites depend on these EXACT strings and would silently
drift if each hard-coded its own copy:

- :mod:`spine.work.onboarding.engine` maps graph node names → these phases and
  records them on the work-entries row;
- :mod:`spine.work.onboarding.onboarding_graph` references them in its topology
  documentation;
- :mod:`spine.ui._pages.onboarding` renders the progress bar over the fixed
  per-mode sequence built from them.

Defining them once here makes the phase-bar contract a single source of truth.
This module is intentionally dependency-free so the thin UI page can import it
without pulling in the engine or the workflow graph.
"""

from __future__ import annotations

#: Greenfield-only deterministic scaffold step (fired before the graph runs).
PHASE_SCAFFOLD = "scaffold"

#: Deterministic analysis map-reduce (fired at ``analysis_manager``).
PHASE_ANALYZE = "analyze"

#: Two-tier documentation synthesis (fired at ``doc_manager``).
PHASE_SYNTHESIZE = "synthesize"

#: Terminal phase recorded after the graph finishes.
PHASE_COMPLETED = "completed"

#: Fixed phase sequences per onboarding mode (display/progress-bar order).
#: Greenfield scaffolds an empty project BEFORE analysing/synthesising defaults,
#: so ``scaffold`` comes first; the greenfield list is a strict superset of the
#: brownfield list, which the UI relies on when the mode is unknown.
PHASES_BY_MODE: dict[str, list[str]] = {
    "greenfield": [PHASE_SCAFFOLD, PHASE_ANALYZE, PHASE_SYNTHESIZE],
    "brownfield": [PHASE_ANALYZE, PHASE_SYNTHESIZE],
}
