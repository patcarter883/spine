"""SPINE CRITIC phase — reviews the output of the previous phase.

The critic is wired into the workflow graph as a conditional node.
It performs two-tier review (structural + agent) and routes:
- PASSED → next phase
- NEEDS_REVISION → rework previous phase
- NEEDS_REVIEW → flag for human review

This module provides the node function for the LangGraph graph.
The actual review logic is in ``spine.workflow.critic_review``.
"""

from __future__ import annotations

import logging
from typing import Any

from typing import Optional

from langchain_core.runnables import RunnableConfig

from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.workflow.critic_review import (
    _get_reviewed_phase,
    structural_critic_check,
    agent_critic_check,
)
from spine.agents.artifacts import materialize_phase_artifacts
from spine.workflow.registry import get_registry

logger = logging.getLogger(__name__)


def call_critic(state: WorkflowState, config: Optional[RunnableConfig] = None) -> dict[str, Any]:
    """Execute the CRITIC phase node.

    Runs two-tier review and updates state with feedback and retry counts.
    The routing decision (passed/needs_revision/needs_review) is handled
    by the conditional edge in the composer, using ``critic_router()``.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        Partial state update with critic feedback and retry count increment.
    """
    work_id = state.get("work_id", "unknown")
    reviewed_phase = _get_reviewed_phase(state)
    workspace_root = state.get("workspace_root", ".")

    logger.info(f"[{work_id}] CRITIC reviewing {reviewed_phase}")

    # ── Tier 1: Structural check ──
    structural_result = structural_critic_check(state, reviewed_phase)

    if structural_result["status"] != "passed":
        logger.info(f"[{work_id}] Structural check failed for {reviewed_phase}")
        # Increment retry count for the reviewed phase
        retry_count = state.get("retry_count", {})
        current = retry_count.get(reviewed_phase, 0)
        phase_artifacts = {"review.md": structural_result["reason"]}
        materialize_phase_artifacts(PhaseName.CRITIC.value, phase_artifacts, workspace_root)
        return {
            "artifacts": {PhaseName.CRITIC.value: phase_artifacts},
            "feedback": [structural_result],
            "retry_count": {reviewed_phase: current + 1},
            "current_phase": PhaseName.CRITIC.value,
        }

    # ── Tier 2: Agent check ──
    agent_result = agent_critic_check(state, reviewed_phase, config=config)
    logger.info(f"[{work_id}] Agent check for {reviewed_phase}: {agent_result['status']}")

    # Increment retry count if needs revision
    retry_count = state.get("retry_count", {})
    current = retry_count.get(reviewed_phase, 0)
    new_retry = current + 1 if agent_result["status"] != "passed" else current

    # Materialize this phase's artifacts to disk immediately
    phase_artifacts = {"review.md": agent_result["reason"]}
    materialize_phase_artifacts(PhaseName.CRITIC.value, phase_artifacts, workspace_root)

    return {
        "artifacts": {PhaseName.CRITIC.value: phase_artifacts},
        "feedback": [agent_result],
        "retry_count": {reviewed_phase: new_retry},
        "current_phase": PhaseName.CRITIC.value,
    }


def _build_critic_agent(state: WorkflowState, config: Optional[RunnableConfig] = None) -> Any:
    """Build a critic Deep Agent. Used by the registry as build_agent_fn."""
    from spine.critic.agent import build_critic_agent

    return build_critic_agent(state, config)


# ── Self-register on import ──
_registry = get_registry()
_registry.register(
    name=PhaseName.CRITIC.value,
    call_fn=call_critic,
    build_agent_fn=_build_critic_agent,
    description="Review previous phase output (structural + agent tiers)",
)
