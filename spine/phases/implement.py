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
from spine.agents.artifacts import materialize_artifacts, materialize_phase_artifacts
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
    workspace_root = state.get("workspace_root", ".")

    logger.info(f"[{work_id}] IMPLEMENT phase starting (retry={retry_count})")

    try:
        agent = build_implement_agent(state, config)

        # Materialize prior artifacts to disk
        materialize_artifacts(state, workspace_root)

        # Build prompt — all prior artifacts are on disk, NOT inlined
        prompt = (
            f"Implement the feature slices described below. Write clean, "
            f"production-quality code for each slice.\n\n"
            f"## Work Description\n{description}\n\n"
            "Prior artifacts are available on disk — read them as needed:\n"
            "- Specification: `.spine/artifacts/specify/specification.md`\n"
            "- Plan: `.spine/artifacts/plan/plan.md`\n"
            "- Feature Slices: `.spine/artifacts/tasks/tasks.md`\n\n"
            "Use `read_file` and `grep` to inspect them. Do NOT load "
            "everything into context at once.\n\n"
        )
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

        # Materialize this phase's artifacts to disk immediately
        phase_artifacts = {"implementation.md": impl_content}
        materialize_phase_artifacts(PhaseName.IMPLEMENT.value, phase_artifacts, workspace_root)

        return {
            "artifacts": {PhaseName.IMPLEMENT.value: phase_artifacts},
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
