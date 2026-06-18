"""SPINE adversarial review — LLM red-team review of the approved plan.

A single-tier, agent-only review (the deterministic structure of the plan was
already validated by ``critic_plan``). Mirrors
:func:`spine.workflow.critic_review.agent_critic_check` but builds the
adversarial agent and always reviews the PLAN phase. The shared prompt
builder, response parser, and scope-refutation / terminal-corroboration
guards are reused so the adversarial verdict is held to the same anti-false-
escalation standard as the critic.
"""

from __future__ import annotations

import logging
from typing import Any

from spine.exceptions import CriticalContractFailure
from spine.models.enums import PhaseName, ReviewStatus
from spine.models.state import WorkflowState

logger = logging.getLogger(__name__)

# The adversarial stage always red-teams the PLAN.
_REVIEWED_PHASE = PhaseName.PLAN.value


async def agent_adversarial_check(
    state: WorkflowState,
    config: Any | None = None,
) -> dict[str, Any]:
    """Run the adversarial Deep Agent over the approved plan.

    Returns a dict with keys ``status`` / ``tier`` / ``reason`` /
    ``suggestions`` (and ``blocker_category`` / ``cited_exclusions``), exactly
    like :func:`agent_critic_check`, so :func:`_adversarial_result_mapper` can
    consume it the same way it consumes a critic verdict. The ``tier`` is
    labelled ``"adversarial"`` to distinguish it in feedback rendering.

    On any non-contract failure, returns NEEDS_REVISION rather than crashing
    (a failed red-team must not silently pass a critical plan).
    """
    try:
        # Imported lazily to avoid a circular import: this module is pulled in
        # by adversarial_subgraph → compose → spine.workflow package init, and
        # spine.workflow.critic_review triggers that same package init.
        from spine.workflow.critic_review import (
            _STRUCTURED_STATE_FIELD,
            _build_review_prompt,
            _corroborate_terminal_verdict,
            _parse_agent_review,
            _validate_scope_claim,
        )
        from spine.workflow.registry import get_registry

        registry = get_registry()
        adv_def = registry.get(PhaseName.ADVERSARIAL.value)
        if adv_def is None or adv_def.build_agent_fn is None:
            logger.warning("Adversarial phase not registered — skipping review")
            return {
                "status": ReviewStatus.PASSED.value,
                "tier": "adversarial",
                "reason": "Adversarial agent not available, skipping",
                "suggestions": [],
            }

        from spine.agents.retry import ainvoke_with_retry
        from spine.agents.context import build_context
        from spine.agents.artifacts import materialize_artifacts

        adv_agent = adv_def.build_agent_fn(state, config)
        work_id = state.get("work_id", "unknown")
        logger.info("[%s] Adversarial agent built — invoking red-team review", work_id)

        workspace_root = state.get("workspace_root", ".")
        materialize_artifacts(state, workspace_root, work_id=work_id)

        # The adversarial reviewer works from the structured plan.json (and the
        # spec for scope checks) — never a truncated preview. Missing structured
        # state is a contract failure, identical to the critic's guarantee.
        structured_field = _STRUCTURED_STATE_FIELD[_REVIEWED_PHASE]
        structured_payload = state.get(structured_field)
        if not structured_payload or not isinstance(structured_payload, str):
            raise CriticalContractFailure(
                phase=f"adversarial/{_REVIEWED_PHASE}",
                reason=(
                    f"State field '{structured_field}' is missing or empty — "
                    "the plan agent did not propagate structured output. The "
                    "adversarial reviewer cannot red-team a plan it cannot see."
                ),
            )

        spec_payload = state.get("specification_json")

        prompt = _build_review_prompt(
            reviewed_phase=_REVIEWED_PHASE,
            structured_payload=structured_payload,
            description=state.get("description") or "",
            spec_payload=spec_payload,
            # No prior-review injection: the adversarial loop tracks its own
            # round via adversarial_retry_count, and the red-team should attack
            # the reworked plan fresh each pass.
            prior_review=None,
        )

        # Prepend the plan-before-do directive when the subgraph produced one.
        directive_raw = state.get("adversarial_directive")
        if directive_raw:
            from spine.agents.plan_do import format_directive_for_prompt

            block = format_directive_for_prompt(directive_raw)
            if block:
                prompt = block + "\n" + prompt

        ctx = build_context(state, PhaseName.ADVERSARIAL)

        async def run_once(p: str) -> dict[str, Any]:
            res = await ainvoke_with_retry(
                adv_agent,
                {"messages": [{"role": "user", "content": p}]},
                phase_name=f"adversarial/{_REVIEWED_PHASE}",
                work_id=work_id,
                work_type=state.get("work_type", ""),
                context=ctx,
            )
            parsed = _parse_agent_review(res, _REVIEWED_PHASE)
            parsed["tier"] = "adversarial"
            return parsed

        parsed = await run_once(prompt)

        # Reuse the critic's anti-false-escalation guards: refute an uncited
        # scope-creep rejection, and require a second opinion before a single
        # red-team vote halts the run for human review.
        parsed = _validate_scope_claim(parsed, spec_payload, _REVIEWED_PHASE, work_id)
        parsed = await _corroborate_terminal_verdict(
            parsed,
            run_once,
            reviewed_phase=_REVIEWED_PHASE,
            structured_payload=structured_payload,
            spec_payload=spec_payload,
            description=state.get("description") or "",
            work_id=work_id,
        )
        parsed["tier"] = "adversarial"

        logger.info(
            "[%s] Adversarial review complete: status=%s reason=%s",
            work_id, parsed.get("status"), (parsed.get("reason") or "")[:120],
        )
        return parsed

    except CriticalContractFailure:
        raise
    except Exception as e:
        logger.error(f"Adversarial agent failed: {e}", exc_info=True)
        return {
            "status": ReviewStatus.NEEDS_REVISION.value,
            "tier": "adversarial",
            "reason": f"Adversarial agent error: {e}",
            "suggestions": ["Review the adversarial agent configuration and logs"],
        }
