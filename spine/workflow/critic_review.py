"""SPINE critic review — two-tier critic with structural and agent checks.

The critic has two tiers:
1. **Structural** (fast, no LLM): checks artifacts exist, aren't empty, have
   basic structure. If structural fails, rework immediately — skip agent critic.
2. **Agent** (deep, LLM-based): quality review by the critic Deep Agent.
   Agent exceptions → rework, not crash.

Feedback is tagged by tier so the reworking phase knows what to address.
"""

from __future__ import annotations

import logging
from typing import Any

from spine.models.enums import PhaseName, ReviewStatus
from spine.models.state import WorkflowState

logger = logging.getLogger(__name__)


def _get_reviewed_phase(state: WorkflowState) -> str:
    """Determine which phase the critic is reviewing.

    First checks the explicit ``critic_reviewing`` field (set by the
    critic node function). Falls back to inspecting artifact keys.
    """
    # Explicit field takes priority
    critic_reviewing = state.get("critic_reviewing", "")
    if critic_reviewing:
        return critic_reviewing

    # Fallback: look at artifact keys
    artifacts = state.get("artifacts", {})
    non_critic_phases = [k for k in artifacts if k != PhaseName.CRITIC.value]
    if non_critic_phases:
        return non_critic_phases[-1]
    # Last resort
    feedback = state.get("feedback", [])
    if feedback:
        last = feedback[-1] if isinstance(feedback[-1], dict) else {}
        return last.get("phase", "unknown")
    return "unknown"


def structural_critic_check(state: WorkflowState, reviewed_phase: str) -> dict[str, Any]:
    """Fast, no-LLM structural check of a phase's output.

    Checks:
    - Artifacts exist for the reviewed phase
    - Artifact content is non-empty
    - Basic structure (has reasonable length for a document)

    Args:
        state: The current workflow state.
        reviewed_phase: The phase being reviewed.

    Returns:
        A dict with keys: ``status`` (ReviewStatus), ``tier``, ``reason``,
        ``suggestions``.
    """
    artifacts = state.get("artifacts", {})
    phase_artifacts = artifacts.get(reviewed_phase, {})

    if not phase_artifacts:
        return {
            "status": ReviewStatus.NEEDS_REVISION.value,
            "tier": "structural",
            "reason": f"No artifacts produced by {reviewed_phase} phase",
            "suggestions": [f"Ensure the {reviewed_phase} phase generates output documents"],
        }

    # Check each artifact has non-empty content
    for name, content in phase_artifacts.items():
        if not content or len(str(content).strip()) < 50:
            return {
                "status": ReviewStatus.NEEDS_REVISION.value,
                "tier": "structural",
                "reason": f"Artifact '{name}' from {reviewed_phase} is too short or empty",
                "suggestions": [
                    f"Expand the {name} artifact with more detail",
                    "Include specific technical decisions and rationale",
                ],
            }

    return {
        "status": ReviewStatus.PASSED.value,
        "tier": "structural",
        "reason": f"Structural check passed for {reviewed_phase}",
        "suggestions": [],
    }


def agent_critic_check(state: WorkflowState, reviewed_phase: str) -> dict[str, Any]:
    """Deep, LLM-based quality review by the critic agent.

    Delegates to the critic Deep Agent for a thorough review.
    If the agent raises an exception, returns NEEDS_REVISION rather than crashing.

    Args:
        state: The current workflow state.
        reviewed_phase: The phase being reviewed.

    Returns:
        A dict with keys: ``status``, ``tier``, ``reason``, ``suggestions``.
    """
    try:
        from spine.workflow.registry import get_registry

        registry = get_registry()
        critic_def = registry.get(PhaseName.CRITIC.value)
        if critic_def is None:
            logger.warning("Critic phase not registered, skipping agent review")
            return {
                "status": ReviewStatus.PASSED.value,
                "tier": "agent",
                "reason": "Critic agent not available, skipping",
                "suggestions": [],
            }

        # Build the critic agent and invoke it
        critic_agent = critic_def.build_agent_fn(state)
        artifacts = state.get("artifacts", {})
        phase_artifacts = artifacts.get(reviewed_phase, {})

        # Format the review request
        artifact_summary = "\n".join(
            f"--- {name} ---\n{content[:2000]}" for name, content in phase_artifacts.items()
        )

        result = critic_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Review the output of the {reviewed_phase} phase.\n\n"
                            f"Work description: {state.get('description', '')}\n\n"
                            f"Artifacts:\n{artifact_summary}\n\n"
                            f"Provide a review: PASSED, NEEDS_REVISION, or NEEDS_REVIEW.\n"
                            f"Include specific reasons and suggestions for improvement."
                        ),
                    }
                ]
            }
        )

        # Parse the agent's response for the review status
        return _parse_agent_review(result, reviewed_phase)

    except Exception as e:
        logger.error(f"Critic agent failed: {e}", exc_info=True)
        return {
            "status": ReviewStatus.NEEDS_REVISION.value,
            "tier": "agent",
            "reason": f"Critic agent error: {e}",
            "suggestions": ["Review the critic agent configuration and logs"],
        }


def _parse_agent_review(result: Any, reviewed_phase: str) -> dict[str, Any]:
    """Parse the critic agent's response into a structured review dict.

    Looks for PASSED, NEEDS_REVISION, or NEEDS_REVIEW keywords in the response.
    Falls back to NEEDS_REVISION if unclear.
    """
    messages = result.get("messages", [])
    last_message = messages[-1] if messages else None
    content = ""
    if last_message:
        content = getattr(last_message, "content", str(last_message))

    content_upper = content.upper()

    if "NEEDS_REVIEW" in content_upper:
        status = ReviewStatus.NEEDS_REVIEW.value
    elif "NEEDS_REVISION" in content_upper:
        status = ReviewStatus.NEEDS_REVISION.value
    elif "PASSED" in content_upper:
        status = ReviewStatus.PASSED.value
    else:
        # Unclear response → needs revision (conservative)
        status = ReviewStatus.NEEDS_REVISION.value

    return {
        "status": status,
        "tier": "agent",
        "reason": f"Agent review of {reviewed_phase}: {content[:500]}",
        "suggestions": [],
    }


def critic_router(state: WorkflowState) -> str:
    """Conditional edge function for critic nodes.

    Runs two-tier review (structural then agent) and returns the routing key:
    - ``"passed"`` → proceed to next phase
    - ``"needs_revision"`` → rework the previous phase (if retries remain)
    - ``"needs_review"`` → flag for human review (stop workflow)

    Also manages retry counting: increments per phase, caps at max_retries.

    Args:
        state: The current workflow state.

    Returns:
        A routing key string for the conditional edge.
    """
    reviewed_phase = _get_reviewed_phase(state)

    # ── Tier 1: Structural check ──
    structural_result = structural_critic_check(state, reviewed_phase)
    if structural_result["status"] != ReviewStatus.PASSED.value:
        return _handle_review_outcome(state, reviewed_phase, structural_result)

    # ── Tier 2: Agent check ──
    agent_result = agent_critic_check(state, reviewed_phase)
    return _handle_review_outcome(state, reviewed_phase, agent_result)


def _handle_review_outcome(
    state: WorkflowState,
    reviewed_phase: str,
    review: dict[str, Any],
) -> str:
    """Process a review result, managing retry counts and routing.

    If the review passed, return "passed".
    If needs revision and retries remain, return "needs_revision".
    If needs revision but retries exceeded, return "needs_review".
    If needs human review directly, return "needs_review".
    """
    status = review["status"]

    if status == ReviewStatus.PASSED.value:
        return "passed"

    if status == ReviewStatus.NEEDS_REVIEW.value:
        return "needs_review"

    # NEEDS_REVISION — check retry count
    retry_count = state.get("retry_count", {})
    phase_retries = retry_count.get(reviewed_phase, 0)
    max_retries = state.get("max_retries", 3)

    if phase_retries >= max_retries:
        logger.warning(
            f"Phase '{reviewed_phase}' exceeded max retries "
            f"({phase_retries}/{max_retries}), flagging for human review"
        )
        return "needs_review"

    # Increment retry count (the _merge_dicts reducer will handle the update)
    # Note: the actual state update happens in the phase node; here we just route
    return "needs_revision"
