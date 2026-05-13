"""SPINE IMPLEMENT phase — generate code to implement feature slices.

The implement Deep Agent reads the tasks/feature slices and generates
code to implement each one. It uses subagents for parallel implementation
of independent slices.

Outputs:
    - Artifacts: implemented code files
    - Prompt Request: if human input is needed during implementation
"""

from __future__ import annotations

import logging
from typing import Any

from typing import Optional

from langchain_core.runnables import RunnableConfig

from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.implement_agent import build_implement_agent
from spine.workflow.registry import get_registry

logger = logging.getLogger(__name__)


def call_implement(state: WorkflowState, config: Optional[RunnableConfig] = None) -> dict[str, Any]:
    """Execute the IMPLEMENT phase.

    Delegates to the implement Deep Agent, which writes code for each
    feature slice. If reworking, includes prior feedback.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        Partial state update with implementation artifacts.
    """
    description = state.get("description", "")
    work_id = state.get("work_id", "unknown")
    retry_count = state.get("retry_count", {}).get(PhaseName.IMPLEMENT.value, 0)
    feedback = state.get("feedback", [])
    artifacts = state.get("artifacts", {})
    tasks_doc = artifacts.get(PhaseName.TASKS.value, {}).get("tasks.md", "")
    plan = artifacts.get(PhaseName.PLAN.value, {}).get("plan.md", "")
    spec = artifacts.get(PhaseName.SPECIFY.value, {}).get("specification.md", "")

    logger.info(f"[{work_id}] IMPLEMENT phase starting (retry={retry_count})")

    try:
        agent = build_implement_agent(state, config)

        prompt = (
            f"Implement the feature slices described below. Write clean, "
            f"production-quality code for each slice.\n\n"
            f"## Work Description\n{description}\n\n"
        )
        if spec:
            prompt += f"## Specification\n{spec}\n\n"
        if plan:
            prompt += f"## Plan\n{plan}\n\n"
        if tasks_doc:
            prompt += f"## Feature Slices\n{tasks_doc}\n\n"
        if retry_count > 0 and feedback:
            feedback_text = "\n".join(
                f"- [{f.get('tier', 'unknown')}] {f.get('reason', '')}"
                for f in feedback
                if isinstance(f, dict)
            )
            prompt += f"## Previous Review Feedback\n{feedback_text}\n"

        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

        impl_content = _extract_response(result)

        return {
            "artifacts": {PhaseName.IMPLEMENT.value: {"implementation.md": impl_content}},
            "current_phase": PhaseName.IMPLEMENT.value,
            "status": "running",
            "prompt_request": None,
        }

    except Exception as e:
        logger.error(f"[{work_id}] IMPLEMENT phase failed: {e}", exc_info=True)
        return {
            "artifacts": {PhaseName.IMPLEMENT.value: {}},
            "status": "failed",
            "prompt_request": {
                "message": f"IMPLEMENT phase failed: {e}",
                "phase": PhaseName.IMPLEMENT.value,
            },
        }


def _extract_response(result: Any) -> str:
    """Extract the text content from the agent's last message."""
    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        return getattr(last, "content", str(last))
    return ""


# ── Self-register on import ──
_registry = get_registry()
_registry.register(
    name=PhaseName.IMPLEMENT.value,
    call_fn=call_implement,
    build_agent_fn=build_implement_agent,
    description="Generate code to implement feature slices",
)
