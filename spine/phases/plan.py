"""SPINE PLAN phase — define the technical architecture.

This phase takes a specification and produces a technical plan document.
It delegates to the plan Deep Agent which designs the architecture,
identifies components, and defines interfaces.

Outputs:
    - Artifacts: technical plan document
    - Prompt Request: if human input is needed during planning
"""

from __future__ import annotations

import logging
from typing import Any

from typing import Optional

from langchain_core.runnables import RunnableConfig

from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.plan_agent import build_plan_agent
from spine.agents.helpers import extract_response
from spine.agents.retry import invoke_with_retry
from spine.workflow.registry import get_registry

logger = logging.getLogger(__name__)


def call_plan(state: WorkflowState, config: Optional[RunnableConfig] = None) -> dict[str, Any]:
    """Execute the PLAN phase.

    Delegates to the plan Deep Agent, which designs the technical architecture
    based on the specification. If reworking, includes prior feedback.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        Partial state update with plan artifacts.
    """
    description = state.get("description", "")
    work_id = state.get("work_id", "unknown")
    retry_count = state.get("retry_count", {}).get(PhaseName.PLAN.value, 0)
    feedback = state.get("feedback", [])
    artifacts = state.get("artifacts", {})
    spec = artifacts.get(PhaseName.SPECIFY.value, {}).get("specification.md", "")

    logger.info(f"[{work_id}] PLAN phase starting (retry={retry_count})")

    try:
        agent = build_plan_agent(state, config)

        prompt = (
            f"Create a detailed technical plan based on the following specification.\n\n"
            f"## Work Description\n{description}\n\n"
        )
        if spec:
            prompt += f"## Specification\n{spec}\n\n"
        if retry_count > 0 and feedback:
            feedback_text = "\n".join(
                f"- [{f.get('tier', 'unknown')}] {f.get('reason', '')}"
                for f in feedback
                if isinstance(f, dict)
            )
            prompt += f"## Previous Review Feedback\n{feedback_text}\n"

        result = invoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.PLAN.value,
            work_id=work_id,
        )

        plan_content = extract_response(result)

        return {
            "artifacts": {PhaseName.PLAN.value: {"plan.md": plan_content}},
            "current_phase": PhaseName.PLAN.value,
            "status": "running",
            "prompt_request": None,
        }

    except Exception as e:
        logger.error(f"[{work_id}] PLAN phase failed: {e}", exc_info=True)
        return {
            "artifacts": {PhaseName.PLAN.value: {}},
            "status": "failed",
            "prompt_request": {"message": f"PLAN phase failed: {e}", "phase": PhaseName.PLAN.value},
        }


# ── Self-register on import ──
_registry = get_registry()
_registry.register(
    name=PhaseName.PLAN.value,
    call_fn=call_plan,
    build_agent_fn=build_plan_agent,
    description="Define the technical architecture and plan",
)
