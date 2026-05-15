"""SPINE PLAN phase — define the technical architecture.

This phase takes a specification and produces a technical plan document.
Context engineering: specification is on disk (not inlined). SpineContext
passed at invoke time.

Phase node functions are async to avoid event-loop binding errors when
subagents inherit the parent checkpointer.
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
from spine.agents.retry import ainvoke_with_retry
from spine.agents.context import build_context
from spine.agents.artifacts import materialize_artifacts, materialize_phase_artifacts, _artifact_path
from spine.workflow.registry import get_registry

logger = logging.getLogger(__name__)


async def call_plan(state: WorkflowState, config: Optional[RunnableConfig] = None) -> dict[str, Any]:
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
    work_type = state.get("work_type", "")
    retry_count = state.get("retry_count", {}).get(PhaseName.PLAN.value, 0)
    feedback = state.get("feedback", [])
    workspace_root = state.get("workspace_root", ".")

    logger.info(f"[{work_id}] PLAN phase starting (retry={retry_count})")

    try:
        agent = build_plan_agent(state, config)

        # Materialize prior artifacts to disk
        materialize_artifacts(state, workspace_root, work_id=work_id)

        # Build prompt — specification is on disk at work_id-scoped path
        spec_path = _artifact_path(work_id, PhaseName.SPECIFY.value)
        prompt = (
            f"Create a detailed technical plan based on the specification.\n\n"
            f"## Work Description\n{description}\n\n"
            f"The full specification is available on disk at "
            f"`{spec_path}/specification.md` — read it with "
            f"`read_file` before planning.\n\n"
        )
        if retry_count > 0 and feedback:
            feedback_text = "\n".join(
                f"- [{f.get('tier', 'unknown')}] {f.get('reason', '')}"
                for f in feedback
                if isinstance(f, dict)
            )
            prompt += f"## Previous Review Feedback\n{feedback_text}\n"

        ctx = build_context(state, PhaseName.PLAN)

        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.PLAN.value,
            work_id=work_id,
            work_type=work_type,
            context=ctx,
        )

        plan_content = extract_response(result)

        # Materialize this phase's artifacts to disk immediately
        phase_artifacts = {"plan.md": plan_content}
        materialize_phase_artifacts(PhaseName.PLAN.value, phase_artifacts, workspace_root, work_id=work_id)

        return {
            "artifacts": {PhaseName.PLAN.value: phase_artifacts},
            "current_phase": PhaseName.PLAN.value,
            "status": "running",
            "prompt_request": None,
        }

    except Exception as e:
        logger.error(f"[{work_id}] PLAN phase failed: {e}", exc_info=True)
        return {
            "artifacts": {PhaseName.PLAN.value: {}},
            "current_phase": PhaseName.PLAN.value,
            "status": "running",
            "prompt_request": {
                "message": f"PLAN phase failed: {e}",
                "phase": PhaseName.PLAN.value,
            },
        }
_registry = get_registry()
_registry.register(
    name=PhaseName.PLAN.value,
    call_fn=call_plan,
    build_agent_fn=build_plan_agent,
    description="Define the technical architecture and plan",
)
