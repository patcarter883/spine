"""SPINE artifact gate — structural pre-check before phase transitions.

An artifact gate ensures a phase doesn't run if its prerequisite phase
produced no artifacts. Currently only the tasks→implement transition is
gated: implement requires meaningful task artifacts before it can generate code.

The implement→verify transition is NOT gated. Verify always runs after
implement — if implement produced nothing, verify can detect and report that.
There is no reason for a human review gate between implement and verify.

The gate is wired as a **node** in the LangGraph StateGraph, not a conditional
edge function. This is critical: when the gate fails, it must write
``status = "needs_review"`` and a feedback entry to state so the dispatcher
can detect the human-review condition. A conditional edge function cannot
return state updates in LangGraph, so a pure-routing gate would silently
end the workflow with ``status = "running"`` → ``"completed"``.

When the gate passes, it returns ``status = "running"`` unchanged and routes
to the next phase node. When it fails, it sets ``status = "needs_review"``,
adds a feedback entry, and routes to END.
"""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.models.enums import PhaseName
from spine.models.state import WorkflowState

logger = logging.getLogger(__name__)

# ── Minimum artifact length (characters) to count as meaningful ────────
MIN_ARTIFACT_CHARS = 50


def _has_meaningful_artifacts(state: WorkflowState, required_phase: str) -> bool:
    """Return True if the required phase produced non-trivial artifacts."""
    artifacts = state.get("artifacts", {})
    phase_arts = artifacts.get(required_phase, {})

    if not isinstance(phase_arts, dict):
        return False

    for _name, content in phase_arts.items():
        if content is not None and len(str(content).strip()) >= MIN_ARTIFACT_CHARS:
            return True

    return False


def make_artifact_gate_node(required_phase: str, next_node: str) -> Any:
    """Create an artifact gate node function for the workflow graph.

    The returned function has the LangGraph node signature
    ``(state, config) -> partial_state_update``. It checks whether
    ``required_phase`` produced meaningful artifacts:

    - **Pass**: returns ``{"status": "running"}`` (unchanged) so the
      conditional edge routes to ``next_node``.
    - **Fail**: returns ``{"status": "needs_review", "feedback": [...]}``
      so the conditional edge routes to END and the dispatcher detects
      the human-review condition.

    Args:
        required_phase: The phase that must have artifacts (e.g. ``"implement"``).
        next_node: The target node if the gate passes (used for the edge map).

    Returns:
        A node function suitable for ``graph.add_node()``.
    """

    def gate_node(
        state: WorkflowState, config: RunnableConfig | None = None
    ) -> dict[str, Any]:
        work_id = state.get("work_id", "unknown")

        if _has_meaningful_artifacts(state, required_phase):
            logger.debug(
                "[%s] Artifact gate passed for %s → %s",
                work_id, required_phase, next_node,
            )
            return {
                "current_phase": required_phase,
                "status": "running",
                "prompt_request": None,
            }

        logger.warning(
            "[%s] Artifact gate: %s produced no meaningful artifacts, "
            "cannot proceed to %s. Flagging for human review.",
            work_id, required_phase, next_node,
        )
        return {
            "current_phase": required_phase,
            "status": "needs_review",
            "feedback": [
                {
                    "status": "needs_review",
                    "tier": "structural",
                    "reason": (
                        f"Artifact gate: {required_phase} produced no "
                        f"meaningful artifacts (≥{MIN_ARTIFACT_CHARS} chars), "
                        f"cannot proceed to {next_node}."
                    ),
                    "suggestions": [],
                }
            ],
            "prompt_request": None,
        }

    # Give the function a readable name for LangGraph Studio / debug
    gate_node.__name__ = f"gate_{required_phase}_to_{next_node}"
    return gate_node


def artifact_gate_router(state: WorkflowState) -> str:
    """Route based on the status set by the gate node.

    Intended as a conditional edge function after a gate node.
    Reads ``state["status"]``: ``"running"`` → proceed, anything else → END.
    """
    if state.get("status") == "running":
        return "proceed"
    return "needs_review"


# ── Legacy helper (kept for backward compat with older compose.py) ─────
def make_artifact_gate_fn(required_phase: str, next_node: str) -> Any:
    """Create a gate function for a conditional edge (legacy).

    .. deprecated::
        Use :func:`make_artifact_gate_node` instead. The legacy version
        cannot set state, so the dispatcher fails to detect needs_review.

    Returns a callable with the signature ``(state) -> str`` that can be
    used as a LangGraph conditional edge function.
    """

    def gate_fn(state: WorkflowState) -> str:
        if _has_meaningful_artifacts(state, required_phase):
            return "proceed"
        work_id = state.get("work_id", "unknown")
        logger.warning(
            f"[{work_id}] Artifact gate: {required_phase} produced no "
            f"meaningful artifacts, cannot proceed to {next_node}. "
            f"Flagging for human review."
        )
        return "needs_review"

    return gate_fn
