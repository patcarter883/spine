"""SPINE TASKS phase — break the plan into executable feature slices.

This is where decomposition occurs. The tasks Deep Agent reads the plan
and breaks it into smaller, independent feature slices that can be
implemented in parallel or sequentially based on dependencies.

Outputs:
    - Artifacts: tasks/feature slices document
    - Prompt Request: if human input is needed during decomposition
"""

from __future__ import annotations

import logging
from typing import Any

from typing import Optional

from langchain_core.runnables import RunnableConfig

from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.tasks_agent import build_tasks_agent
from spine.agents.helpers import extract_response
from spine.agents.retry import invoke_with_retry
from spine.workflow.registry import get_registry

logger = logging.getLogger(__name__)


def call_tasks(state: WorkflowState, config: Optional[RunnableConfig] = None) -> dict[str, Any]:
    """Execute the TASKS phase.

    Delegates to the tasks Deep Agent, which decomposes the plan into
    feature slices with dependencies. If reworking, includes prior feedback.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        Partial state update with task artifacts.
    """
    description = state.get("description", "")
    work_id = state.get("work_id", "unknown")
    retry_count = state.get("retry_count", {}).get(PhaseName.TASKS.value, 0)
    feedback = state.get("feedback", [])
    artifacts = state.get("artifacts", {})
    plan = artifacts.get(PhaseName.PLAN.value, {}).get("plan.md", "")
    spec = artifacts.get(PhaseName.SPECIFY.value, {}).get("specification.md", "")

    logger.info(f"[{work_id}] TASKS phase starting (retry={retry_count})")

    try:
        agent = build_tasks_agent(state, config)

        prompt = (
            f"Break the following plan into smaller, executable feature slices "
            f"with clear dependencies.\n\n"
            f"## Work Description\n{description}\n\n"
        )
        if spec:
            prompt += f"## Specification\n{spec}\n\n"
        if plan:
            prompt += f"## Plan\n{plan}\n\n"
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
            phase_name=PhaseName.TASKS.value,
            work_id=work_id,
        )

        tasks_content = extract_response(result)

        return {
            "artifacts": {PhaseName.TASKS.value: {"tasks.md": tasks_content}},
            "current_phase": PhaseName.TASKS.value,
            "status": "running",
            "prompt_request": None,
        }

    except Exception as e:
        logger.error(f"[{work_id}] TASKS phase failed: {e}", exc_info=True)
        return {
            "artifacts": {PhaseName.TASKS.value: {}},
            "status": "failed",
            "prompt_request": {
                "message": f"TASKS phase failed: {e}",
                "phase": PhaseName.TASKS.value,
            },
        }


# ── Self-register on import ──
_registry = get_registry()
_registry.register(
    name=PhaseName.TASKS.value,
    call_fn=call_tasks,
    build_agent_fn=build_tasks_agent,
    description="Decompose the plan into executable feature slices",
)
