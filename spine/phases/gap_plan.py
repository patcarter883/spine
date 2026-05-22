"""SPINE GAP_PLAN phase — create a targeted gap remediation plan.

The gap plan Deep Agent reads the verification report for failed
items, references the original plan and codebase map, and produces
a ``gap_plan.md`` with specific fix instructions for implement.

Does NOT re-explore the codebase — relies on existing artifacts.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.gap_plan_agent import build_gap_plan_agent
from spine.agents.helpers import extract_response
from spine.agents.retry import ainvoke_with_retry
from spine.agents.context import build_context
from spine.agents.artifacts import (
    materialize_artifacts,
    materialize_phase_artifacts,
    scan_artifact_dir,
    _artifact_path,
)
from spine.workflow.registry import get_registry

logger = logging.getLogger(__name__)

_MAX_ARTIFACT_STATE_CHARS = 500


async def call_gap_plan(
    state: WorkflowState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """Execute the GAP_PLAN phase.

    Reads the verification report, original plan, and codebase map.
    Produces a gap_plan.md with targeted fix instructions.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        Partial state update with gap plan artifacts and status.
    """
    work_id = state.get("work_id", "unknown")
    work_type = state.get("work_type", "")
    workspace_root = state.get("workspace_root", ".")

    logger.info(f"[{work_id}] GAP_PLAN phase starting")

    try:
        agent = build_gap_plan_agent(state, config)
        materialize_artifacts(state, workspace_root, work_id=work_id)

        verify_path = _artifact_path(work_id, PhaseName.VERIFY.value)
        plan_path = _artifact_path(work_id, PhaseName.PLAN.value)
        tasks_path = _artifact_path(work_id, PhaseName.TASKS.value)
        gap_plan_path = _artifact_path(work_id, PhaseName.GAP_PLAN.value)

        prompt = (
            f"Read the verification report at `{verify_path}/verification.md` "
            f"to understand what issues were found. Reference the codebase map "
            f"at `{tasks_path}/codebase-map.md` and the original plan at "
            f"`{plan_path}/plan.md` for context.\n\n"
            f"Write `gap_plan.md` to `{gap_plan_path}/gap_plan.md` using `write_file`.\n"
            "This file is REQUIRED — without it the phase is treated as failed."
        )

        ctx = build_context(state, PhaseName.GAP_PLAN)

        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.GAP_PLAN.value,
            work_id=work_id,
            work_type=work_type,
            context=ctx,
        )

        gap_plan_content = extract_response(result)

        disk_artifacts = scan_artifact_dir(
            workspace_root,
            work_id,
            PhaseName.GAP_PLAN.value,
            max_preview_chars=_MAX_ARTIFACT_STATE_CHARS,
        )

        if not disk_artifacts:
            if not gap_plan_content or len(gap_plan_content.strip()) < 20:
                gap_plan_content = (
                    "Gap plan could not produce a meaningful report. "
                    "The agent returned insufficient output. "
                    "Manual review is required."
                )
            materialize_phase_artifacts(
                PhaseName.GAP_PLAN.value,
                {"gap_plan.md": gap_plan_content},
                workspace_root,
                work_id=work_id,
            )
            disk_artifacts = {"gap_plan.md": gap_plan_content[:_MAX_ARTIFACT_STATE_CHARS]}

        return {
            "artifacts": {PhaseName.GAP_PLAN.value: disk_artifacts},
            "current_phase": PhaseName.GAP_PLAN.value,
            "status": "running",
            "prompt_request": None,
        }

    except Exception as e:
        logger.error(f"[{work_id}] GAP_PLAN phase failed: {e}", exc_info=True)
        return {
            "artifacts": {PhaseName.GAP_PLAN.value: {}},
            "current_phase": PhaseName.GAP_PLAN.value,
            "status": "needs_review",
            "prompt_request": {
                "message": f"GAP_PLAN phase failed: {e}",
                "phase": PhaseName.GAP_PLAN.value,
            },
        }


# ── Self-register on import ──
_registry = get_registry()
_registry.register(
    name=PhaseName.GAP_PLAN.value,
    call_fn=call_gap_plan,
    build_agent_fn=build_gap_plan_agent,
    description="Create targeted gap remediation plan from verify feedback",
)
