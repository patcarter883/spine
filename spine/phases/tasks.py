"""SPINE TASKS phase — break the plan into executable feature slices.

This is where decomposition occurs. The tasks Deep Agent reads the plan
(on disk, not inlined) and breaks it into smaller, independent feature
slices that can be implemented in parallel or sequentially.
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
from spine.agents.context import build_context
from spine.agents.artifacts import materialize_artifacts, materialize_phase_artifacts
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
    workspace_root = state.get("workspace_root", ".")

    logger.info(f"[{work_id}] TASKS phase starting (retry={retry_count})")

    try:
        agent = build_tasks_agent(state, config)

        # Materialize prior artifacts to disk
        materialize_artifacts(state, workspace_root)

        # Build prompt — plan and spec are on disk
        prompt = (
            f"Break the plan into smaller, executable feature slices "
            f"with clear dependencies.\n\n"
            f"## Work Description\n{description}\n\n"
            "Prior artifacts are available on disk:\n"
            "- Specification: `.spine/artifacts/specify/specification.md`\n"
            "- Plan: `.spine/artifacts/plan/plan.md`\n\n"
            "Read them with `read_file` before decomposing.\n\n"
        )
        if retry_count > 0 and feedback:
            feedback_text = "\n".join(
                f"- [{f.get('tier', 'unknown')}] {f.get('reason', '')}"
                for f in feedback
                if isinstance(f, dict)
            )
            prompt += f"## Previous Review Feedback\n{feedback_text}\n"

        ctx = build_context(state, PhaseName.TASKS)

        result = invoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.TASKS.value,
            work_id=work_id,
            context=ctx,
        )

        tasks_content = extract_response(result)

        # Materialize this phase's artifacts to disk immediately
        phase_artifacts = {"tasks.md": tasks_content}
        materialize_phase_artifacts(PhaseName.TASKS.value, phase_artifacts, workspace_root)

        return {
            "artifacts": {PhaseName.TASKS.value: phase_artifacts},
            "current_phase": PhaseName.TASKS.value,
            "status": "running",
            "prompt_request": None,
        }

    except Exception as e:
        logger.error(f"[{work_id}] TASKS phase failed: {e}", exc_info=True)
        return {
            "artifacts": {PhaseName.TASKS.value: {}},
            "current_phase": PhaseName.TASKS.value,
            "status": "running",
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
