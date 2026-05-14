"""SPINE VERIFY phase — confirm implementation meets requirements.

The verify Deep Agent reviews the implementation against the specification,
plan, and tasks. All prior artifacts are on disk (not inlined) — the agent
reads them on demand with filesystem tools.

Context engineering: summarization middleware enabled for long-running
multi-slice verification.
"""

from __future__ import annotations

import logging
from typing import Any

from typing import Optional

from langchain_core.runnables import RunnableConfig

from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.verify_agent import build_verify_agent
from spine.agents.helpers import extract_response
from spine.agents.retry import invoke_with_retry
from spine.agents.context import build_context
from spine.agents.artifacts import materialize_artifacts, materialize_phase_artifacts
from spine.workflow.registry import get_registry

logger = logging.getLogger(__name__)


def call_verify(state: WorkflowState, config: Optional[RunnableConfig] = None) -> dict[str, Any]:
    """Execute the VERIFY phase.

    Delegates to the verify Deep Agent, which reviews all artifacts
    against the original requirements and produces a verification report.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        Partial state update with verification artifacts and final status.
    """
    description = state.get("description", "")
    work_id = state.get("work_id", "unknown")
    workspace_root = state.get("workspace_root", ".")

    logger.info(f"[{work_id}] VERIFY phase starting")

    try:
        agent = build_verify_agent(state, config)

        # Materialize prior artifacts to disk
        materialize_artifacts(state, workspace_root)

        # Build prompt — all prior artifacts are on disk, NOT inlined
        prompt = (
            f"Verify that the implementation meets the requirements. "
            f"Check that all feature slices are implemented correctly, "
            f"the plan was followed, and the original task is complete.\n\n"
            f"## Original Requirements\n{description}\n\n"
            "Prior artifacts are available on disk — read them as needed:\n"
            "- Specification: `.spine/artifacts/specify/specification.md`\n"
            "- Plan: `.spine/artifacts/plan/plan.md`\n"
            "- Feature Slices: `.spine/artifacts/tasks/tasks.md`\n"
            "- Implementation: `.spine/artifacts/implement/implementation.md`\n\n"
            "Use `read_file` and `grep` to inspect them. Do NOT load "
            "everything into context at once.\n\n"
            "Also inspect the actual code files on disk using `ls` and "
            "`read_file` — the implementation summary may not reflect "
            "the actual state of the code."
        )

        ctx = build_context(state, PhaseName.VERIFY)

        result = invoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.VERIFY.value,
            work_id=work_id,
            context=ctx,
        )

        verify_content = extract_response(result)

        # Provide a fallback if the agent returned nothing useful
        if not verify_content or len(verify_content.strip()) < 20:
            verify_content = (
                "Verification could not produce a meaningful report. "
                "The agent returned insufficient output. "
                "Manual review is required."
            )

        # Determine final status from verification
        is_verified = "VERIFIED" in verify_content.upper() or "PASSED" in verify_content.upper()
        final_status = "completed" if is_verified else "needs_review"

        # Materialize this phase's artifacts to disk immediately
        phase_artifacts = {"verification.md": verify_content}
        materialize_phase_artifacts(PhaseName.VERIFY.value, phase_artifacts, workspace_root)

        return {
            "artifacts": {PhaseName.VERIFY.value: phase_artifacts},
            "current_phase": PhaseName.VERIFY.value,
            "status": final_status,
            "prompt_request": None,
            "feedback": [
                {
                    "status": "passed" if is_verified else "needs_review",
                    "tier": "verify",
                    "reason": verify_content[:500],
                    "suggestions": [],
                }
            ],
        }

    except Exception as e:
        logger.error(f"[{work_id}] VERIFY phase failed: {e}", exc_info=True)
        return {
            "artifacts": {PhaseName.VERIFY.value: {}},
            "current_phase": PhaseName.VERIFY.value,
            "status": "needs_review",
            "prompt_request": {
                "message": f"VERIFY phase failed: {e}",
                "phase": PhaseName.VERIFY.value,
            },
        }


# ── Self-register on import ──
_registry = get_registry()
_registry.register(
    name=PhaseName.VERIFY.value,
    call_fn=call_verify,
    build_agent_fn=build_verify_agent,
    description="Verify implementation meets requirements",
)
