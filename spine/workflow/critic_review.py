"""SPINE critic review — two-tier critic with structural and agent checks.

The critic has two tiers:
1. **Structural** (fast, no LLM): checks artifacts exist, aren't empty, have
   basic structure. If structural fails, rework immediately — skip agent critic.
2. **Agent** (deep, LLM-based): quality review by the critic Deep Agent.
   Agent exceptions → rework, not crash.

Feedback is tagged by tier so the reworking phase knows what to address.

Context engineering: artifact content is on disk. The critic gets a short
preview inline with paths to full files, rather than inlining everything.
SpineContext is passed at invoke time.

Agent critic check is async to avoid event-loop binding errors when
subagents inherit the parent checkpointer — the sync ``invoke_with_retry``
runs in a thread pool which breaks ``asyncio.Lock`` objects bound to the
original event loop.
"""

from __future__ import annotations

import logging
from typing import Any

from spine.exceptions import CriticalContractFailure
from spine.models.enums import PhaseName, ReviewStatus
from spine.models.state import WorkflowState

# Phase → state field carrying the raw structured JSON produced by the
# phase agent. Critics MUST consume these instead of artifact previews;
# absence is a CriticalContractFailure.
_STRUCTURED_STATE_FIELD: dict[str, str] = {
    PhaseName.SPECIFY.value: "specification_json",
    PhaseName.PLAN.value: "plan_json",
}

logger = logging.getLogger(__name__)


def _build_review_prompt(
    *,
    reviewed_phase: str,
    structured_payload: str,
    description: str,
) -> str:
    """Build the user-message prompt for the critic agent.

    Inlines the original user description (when present) so phase-specific
    critic augmentations such as :data:`_SPECIFY_REVIEW_INSTRUCTIONS` can
    check spec-to-description traceability without re-reading state.
    """
    description = (description or "").strip()
    description_section = (
        f"## Original User Description\n{description}\n\n"
        if description
        else ""
    )
    return (
        f"Review ONLY the structured output below for the {reviewed_phase} "
        "phase. Do not attempt to read files or run commands; everything "
        "you need is inlined.\n\n"
        f"{description_section}"
        "## Structured Output Under Review\n"
        f"```json\n{structured_payload}\n```\n\n"
        "Respond with PASSED, NEEDS_REVISION, or NEEDS_REVIEW and include "
        "concrete reasons and suggestions."
    )


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


async def agent_critic_check(
    state: WorkflowState,
    reviewed_phase: str,
    config: Any | None = None,
) -> dict[str, Any]:
    """Deep, LLM-based quality review by the critic agent.

    Delegates to the critic Deep Agent for a thorough review.
    If the agent raises an exception, returns NEEDS_REVISION rather than crashing.

    Context engineering: artifacts are materialized to disk and referenced
    by path. The critic gets a short preview inline (first 2000 chars) with
    the path to the full file, so it can read details on demand.

    Args:
        state: The current workflow state.
        reviewed_phase: The phase being reviewed.
        config: LangGraph runtime config (passed to the agent builder).

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

        # Build the critic agent and invoke it with retry
        from spine.agents.retry import ainvoke_with_retry
        from spine.agents.context import build_context
        from spine.agents.artifacts import materialize_artifacts

        critic_agent = critic_def.build_agent_fn(state, config)
        logger.info(
            "[%s] Critic agent built — invoking LLM review for phase '%s'",
            state.get("work_id", "?"), reviewed_phase,
        )

        # Materialize artifacts to disk so the critic can read them on demand.
        workspace_root = state.get("workspace_root", ".")
        work_id = state.get("work_id", "unknown")
        materialize_artifacts(state, workspace_root, work_id=work_id)

        # Critics review the structured JSON the phase agent produced — never
        # a truncated .md preview. Missing structured state is a contract
        # failure that must surface, not get papered over with a fallback.
        structured_field = _STRUCTURED_STATE_FIELD.get(reviewed_phase)
        if structured_field is None:
            raise CriticalContractFailure(
                phase=f"critic/{reviewed_phase}",
                reason=(
                    f"No structured-state field is registered for phase "
                    f"'{reviewed_phase}'. Critics require structured output; "
                    f"add an entry to _STRUCTURED_STATE_FIELD before enabling "
                    f"a critic for this phase."
                ),
            )
        structured_payload = state.get(structured_field)
        if not structured_payload or not isinstance(structured_payload, str):
            raise CriticalContractFailure(
                phase=f"critic/{reviewed_phase}",
                reason=(
                    f"State field '{structured_field}' is missing or empty — "
                    f"the {reviewed_phase} agent did not propagate structured "
                    f"output. Critics cannot review without it."
                ),
            )

        prompt = _build_review_prompt(
            reviewed_phase=reviewed_phase,
            structured_payload=structured_payload,
            description=state.get("description") or "",
        )

        # If the critic subgraph ran a plan-before-do step, prepend the
        # directive so the do node knows the planning context (e.g. which
        # tiers to weight, which areas the planner flagged).
        directive_raw = state.get("critic_directive")
        if directive_raw:
            from spine.agents.plan_do import format_directive_for_prompt
            block = format_directive_for_prompt(directive_raw)
            if block:
                prompt = block + "\n" + prompt

        ctx = build_context(state, PhaseName.CRITIC)

        result = await ainvoke_with_retry(
            critic_agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=f"critic/{reviewed_phase}",
            work_id=state.get("work_id", "unknown"),
            work_type=state.get("work_type", ""),
            context=ctx,
        )

        # Parse the agent's response for the review status
        parsed = _parse_agent_review(result, reviewed_phase)
        logger.info(
            "[%s] Critic agent review complete: status=%s reason=%s",
            state.get("work_id", "?"), parsed.get("status"), parsed.get("reason", "")[:120],
        )
        return parsed

    except CriticalContractFailure:
        # Contract failures must propagate — critics cannot review without
        # structured input, and silently downgrading to NEEDS_REVISION would
        # mask the real defect (e.g. a phase agent that didn't emit JSON).
        raise
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

    First checks for structured output (CriticReview via response_format).
    Falls back to keyword parsing for backwards compatibility.
    """
    # Try structured output first (from response_format=CriticReview)
    structured = result.get("structured_response")
    if structured is not None:
        # Handle CriticReview Pydantic model
        if isinstance(structured, dict):
            status_raw = structured.get("status", "NEEDS_REVISION")
            reason = structured.get("reason", "Structured review completed")
            suggestions = structured.get("suggestions", [])
        elif hasattr(structured, "model_dump"):
            data = structured.model_dump()
            status_raw = data.get("status", "NEEDS_REVISION")
            reason = data.get("reason", "Structured review completed")
            suggestions = data.get("suggestions", [])
        else:
            # Fallback for unexpected structured format
            return _parse_agent_review_fallback(result, reviewed_phase)

        # Normalize status to lowercase (CriticReview uses uppercase)
        status = status_raw.lower() if isinstance(status_raw, str) else status_raw

        return {
            "status": status,
            "tier": "agent",
            "reason": reason,
            "suggestions": suggestions,
        }

    # Fallback to keyword parsing for backwards compatibility
    return _parse_agent_review_fallback(result, reviewed_phase)


def _parse_agent_review_fallback(result: Any, reviewed_phase: str) -> dict[str, Any]:
    """Fallback keyword parser for unstructured critic agent responses.

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
        "reason": f"Agent review of {reviewed_phase}: {content}",
        "suggestions": [],
    }


def critic_router(state: WorkflowState) -> str:
    """Conditional edge function for critic nodes.

    Reads the feedback that ``call_critic`` already wrote into state and
    returns the routing key.  Does NOT re-run the review — that would
    duplicate every LLM call.

    Routing:
    - ``"passed"`` → proceed to next phase
    - ``"needs_revision"`` → rework the previous phase (if retries remain)
    - ``"needs_review"`` → flag for human review (stop workflow)
    - ``"failed"`` → stop workflow as failed

    Args:
        state: The current workflow state (already updated by call_critic).

    Returns:
        A routing key string for the conditional edge.
    """
    if state.get("status") == "failed":
        return "failed"

    # Single source of truth: the critic mapper writes last_critic_review with
    # the canonical phase_status from the subgraph. Reading feedback[-1] was
    # fragile because feedback is operator.add and any prior "passed" entry
    # from another tier or phase could shadow the current critic's verdict.
    lcr = state.get("last_critic_review") or {}
    if not lcr:
        logger.warning(
            "critic_router: last_critic_review missing — routing needs_revision"
        )
        return "needs_revision"

    reviewed_phase = lcr.get("phase") or _get_reviewed_phase(state)
    review_status = lcr.get("status", ReviewStatus.NEEDS_REVISION.value)

    review = {
        "status": review_status,
        "tier": lcr.get("tier", "unknown"),
        "reason": lcr.get("reason", ""),
        "suggestions": lcr.get("suggestions", []),
    }
    decision = _handle_review_outcome(state, reviewed_phase, review)
    retry_count = state.get("retry_count", {})
    logger.info(
        "[%s] critic_router: phase=%s status=%s retries=%d/%d → %s",
        state.get("work_id", "?"),
        reviewed_phase,
        review_status,
        retry_count.get(reviewed_phase, 0),
        state.get("max_retries", 3),
        decision,
    )
    return decision


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
