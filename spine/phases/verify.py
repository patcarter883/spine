"""SPINE VERIFY phase — confirm implementation meets requirements.

The verify Deep Agent reviews the implementation against the specification,
plan, and tasks. All prior artifacts are on disk (not inlined) — the agent
reads them on demand with filesystem tools.

Context engineering: summarization middleware enabled for long-running
multi-slice verification. RLM parallel dispatch via eval+PTC for per-slice
verification subagents.
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
from spine.agents.artifacts import (
    materialize_artifacts,
    materialize_phase_artifacts,
    _artifact_path,
)
from spine.workflow.registry import get_registry

logger = logging.getLogger(__name__)


# Maximum characters of artifact content to store in WorkflowState.
_MAX_ARTIFACT_STATE_CHARS = 500


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
    work_type = state.get("work_type", "")
    workspace_root = state.get("workspace_root", ".")

    logger.info(f"[{work_id}] VERIFY phase starting")

    try:
        agent = build_verify_agent(state, config)

        # Materialize prior artifacts to disk
        materialize_artifacts(state, workspace_root, work_id=work_id)

        # Build prompt — all prior artifacts are on disk, NOT inlined.
        # Skip spec/plan references for quick workflows that lack them.
        has_spec = "spec" in work_type
        spec_path = _artifact_path(work_id, PhaseName.SPECIFY.value)
        plan_path = _artifact_path(work_id, PhaseName.PLAN.value)
        tasks_path = _artifact_path(work_id, PhaseName.TASKS.value)
        impl_path = _artifact_path(work_id, PhaseName.IMPLEMENT.value)

        prompt_lines = [
            "Verify that the implementation meets the requirements. "
            "Check that all feature slices are implemented correctly, "
            "the plan was followed, and the original task is complete.",
            "",
            "## Original Requirements",
            description,
            "",
            "Prior artifacts are available on disk — read them as needed:",
        ]
        if has_spec:
            prompt_lines.extend([
                f"- Specification: `{spec_path}/specification.md`",
                f"- Plan: `{plan_path}/plan.md`",
            ])
        prompt_lines.extend([
            f"- Feature Slices: `{tasks_path}/tasks.md`",
            f"- Implementation: `{impl_path}/implementation.md`",
            "",
            "Use `read_file` and `grep` to inspect them. Do NOT load "
            "everything into context at once.",
            "",
            "Also inspect the actual code files on disk using `ls` and "
            "`read_file` — the implementation summary may not reflect "
            "the actual state of the code.",
            "",
            "**RLM parallel verify pattern:** Use `eval` to read the "
            "tasks artifact, extract the slice list, then dispatch a "
            "`slice-verifier` subagent per slice via "
            "`Promise.allSettled(tools.task(...))`. Synthesize the "
            "verification report from subagent results in code — do NOT "
            "re-read each slice file manually into conversation.",
        ])
        prompt = "\n".join(prompt_lines)

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

        # Determine final status from verification (use full content)
        is_verified = "VERIFIED" in verify_content.upper() or "PASSED" in verify_content.upper()
        final_status = "completed" if is_verified else "needs_review"

        # Materialize full content to disk immediately
        phase_artifacts = {"verification.md": verify_content}
        materialize_phase_artifacts(PhaseName.VERIFY.value, phase_artifacts, workspace_root, work_id=work_id)

        return {
            "artifacts": {PhaseName.VERIFY.value: {
                "verification.md": verify_content[:_MAX_ARTIFACT_STATE_CHARS]
            }},
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
