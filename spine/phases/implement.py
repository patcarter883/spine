"""SPINE IMPLEMENT phase — generate code to implement feature slices.

The implement Deep Agent reads the tasks/feature slices (on disk) and
generates code to implement each one. Prior artifacts are NOT inlined —
the agent reads them on demand from the filesystem.

Context engineering: summarization middleware enabled for long-running
multi-slice implementation.
"""

from __future__ import annotations

import logging
from typing import Any

from typing import Optional

from langchain_core.runnables import RunnableConfig

from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.implement_agent import build_implement_agent
from spine.agents.helpers import extract_response
from spine.agents.retry import invoke_with_retry
from spine.agents.context import build_context
from spine.agents.artifacts import (
    materialize_artifacts,
    materialize_phase_artifacts,
    _artifact_path,
)
from spine.workflow.registry import get_registry

logger = logging.getLogger(__name__)


# Maximum characters of artifact content to store in WorkflowState.
_MAX_ARTIFACT_STATE_CHARS = 500


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
    work_type = state.get("work_type", "")
    retry_count = state.get("retry_count", {}).get(PhaseName.IMPLEMENT.value, 0)
    feedback = state.get("feedback", [])
    workspace_root = state.get("workspace_root", ".")

    logger.info(f"[{work_id}] IMPLEMENT phase starting (retry={retry_count})")

    try:
        agent = build_implement_agent(state, config)

        # Materialize prior artifacts to disk
        materialize_artifacts(state, workspace_root, work_id=work_id)

        # Build prompt — all prior artifacts are on disk, NOT inlined.
        # Skip spec/plan references for quick workflows that lack them.
        has_spec = "spec" in work_type
        spec_path = _artifact_path(work_id, PhaseName.SPECIFY.value)
        plan_path = _artifact_path(work_id, PhaseName.PLAN.value)
        tasks_path = _artifact_path(work_id, PhaseName.TASKS.value)

        prompt_lines = [
            "Implement the feature slices described below. Write clean, "
            "production-quality code for each slice.",
            "",
            "## Work Description",
            description,
            "",
        ]
        if has_spec:
            prompt_lines.extend([
                "Prior artifacts are available on disk — read them as needed:",
                f"- Specification: `{spec_path}/specification.md`",
                f"- Plan: `{plan_path}/plan.md`",
                f"- Feature Slices: `{tasks_path}/tasks.md`",
                "",
            ])
        else:
            prompt_lines.extend([
                "Prior artifacts are available on disk — read them as needed:",
                f"- Feature Slices: `{tasks_path}/tasks.md`",
                "",
            ])
        prompt_lines.extend([
            "Use `read_file` and `grep` to inspect them. Do NOT load "
            "everything into context at once.",
            "",
        ])
        prompt = "\n".join(prompt_lines)
        if retry_count > 0 and feedback:
            feedback_text = "\n".join(
                f"- [{f.get('tier', 'unknown')}] {f.get('reason', '')}"
                for f in feedback
                if isinstance(f, dict)
            )
            prompt += f"## Previous Review Feedback\n{feedback_text}\n"

        ctx = build_context(state, PhaseName.IMPLEMENT)

        result = invoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.IMPLEMENT.value,
            work_id=work_id,
            context=ctx,
        )

        impl_content = extract_response(result)

        # Materialize full content to disk immediately
        phase_artifacts = {"implementation.md": impl_content}
        materialize_phase_artifacts(PhaseName.IMPLEMENT.value, phase_artifacts, workspace_root, work_id=work_id)

        return {
            "artifacts": {PhaseName.IMPLEMENT.value: {
                "implementation.md": impl_content[:_MAX_ARTIFACT_STATE_CHARS]
            }},
            "current_phase": PhaseName.IMPLEMENT.value,
            "status": "running",
            "prompt_request": None,
        }

    except Exception as e:
        logger.error(f"[{work_id}] IMPLEMENT phase failed: {e}", exc_info=True)
        return {
            "artifacts": {PhaseName.IMPLEMENT.value: {}},
            "current_phase": PhaseName.IMPLEMENT.value,
            "status": "running",
            "prompt_request": {
                "message": f"IMPLEMENT phase failed: {e}",
                "phase": PhaseName.IMPLEMENT.value,
            },
        }


# ── Self-register on import ──
_registry = get_registry()
_registry.register(
    name=PhaseName.IMPLEMENT.value,
    call_fn=call_implement,
    build_agent_fn=build_implement_agent,
    description="Generate code to implement feature slices",
)
