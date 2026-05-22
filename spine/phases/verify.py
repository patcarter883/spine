"""SPINE VERIFY phase — confirm implementation meets requirements.

The verify Deep Agent reviews the implementation against the specification,
plan, and tasks. All prior artifacts are on disk (not inlined) — the agent
reads them on demand with filesystem tools.

Context engineering: read cache prevents re-reading files across subagent turns.
multi-slice verification. RLM parallel dispatch via eval+PTC for per-slice
verification subagents.

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
from spine.agents.verify_agent import build_verify_agent
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


# Maximum characters of artifact content to store in WorkflowState.
_MAX_ARTIFACT_STATE_CHARS = 500


async def call_verify(
    state: WorkflowState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """Execute the VERIFY phase.

    Delegates to the verify Deep Agent, which reviews all artifacts
    against the original requirements and produces a verification report.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        Partial state update with verification artifacts and final status.
    """
    # description intentionally not needed for verify — work from artifacts
    work_id = state.get("work_id", "unknown")
    work_type = state.get("work_type", "")
    workspace_root = state.get("workspace_root", ".")

    logger.info(f"[{work_id}] VERIFY phase starting")

    try:
        agent = build_verify_agent(state, config)

        # Materialize prior artifacts to disk
        materialize_artifacts(state, workspace_root, work_id=work_id)

        # Build prompt — work from the feature slices, not the original
        # description (already captured and expanded in the upstream artifacts).
        verify_dir = f".spine/artifacts/{work_id}/verify"
        tasks_dir = f".spine/artifacts/{work_id}/tasks"
        impl_dir = f".spine/artifacts/{work_id}/implement"
        context_seed = f"globalThis.context = {{work_id: '{work_id}', phase: 'verify', tasks_dir: '{tasks_dir}', verify_dir: '{verify_dir}', impl_dir: '{impl_dir}'}};\n\n"

        from spine.agents.artifacts import list_slice_files

        slice_files = list_slice_files(workspace_root, work_id)
        slice_count = len(slice_files)

        if slice_count == 0:
            slice_inventory = (
                "⚠ No slice-*.md files found in tasks/ directory. "
                "Use `ls` + `glob` to locate slice files before proceeding."
            )
        else:
            slice_inventory = f"{slice_count} slice file(s) found in `{tasks_dir}/`:\n" + "\n".join(
                f"  - `{tasks_dir}/{name}`" for name in slice_files
            )

        if slice_count == 1:
            dispatch_note = (
                "There is 1 slice. Dispatch a single `slice-verifier` for "
                "consistency — same context-management benefit."
            )
        elif slice_count >= 2:
            dispatch_note = (
                f"There are {slice_count} slices. Dispatch all of them in "
                "parallel via `Promise.allSettled(tools.task(...))` inside one "
                "`eval` call."
            )
        else:
            dispatch_note = (
                "Slice files not yet discovered. Use `glob` to list "
                f"`{tasks_dir}/slice-*.md`, then dispatch one subagent per slice."
            )

        spec_path = _artifact_path(work_id, PhaseName.SPECIFY.value)
        plan_path = _artifact_path(work_id, PhaseName.PLAN.value)
        tasks_path = _artifact_path(work_id, PhaseName.TASKS.value)
        impl_path = _artifact_path(work_id, PhaseName.IMPLEMENT.value)

        prompt_lines = [
            "Verify that the implementation meets the requirements. "
            "Check that all feature slices are implemented correctly, "
            "the plan was followed, and the original task is complete.",
            "",
            "Prior artifacts are available on disk — read them as needed:",
            f"- Specification: `{spec_path}/specification.md`",
            f"- Plan: `{plan_path}/plan.md`",
            f"- Feature Slices: `{tasks_path}/tasks.md`",
            f"- Codebase map: `{tasks_path}/codebase-map.md`",
            f"- Implementation: `{impl_path}/implementation.md`",
            "",
            "Use `read_file` and `grep` to inspect them. Do NOT load "
            "everything into context at once.",
            "",
            "Read the codebase map FIRST — it contains file paths, key functions, and conventions "
            "discovered during the tasks phase. Use it instead of re-exploring the codebase.",
            "",
            "Also inspect the actual code files on disk using `ls` and "
            "`read_file` — the implementation summary may not reflect "
            "the actual state of the code.",
            "",
            "## Slice Inventory",
            slice_inventory,
            "",
            "## Step 2 Guidelines",
            dispatch_note,
            "",
            "**RLM parallel verify pattern:** Use `eval` to read the "
            "tasks artifact, extract the slice list, then dispatch a "
            "`slice-verifier` subagent per slice via "
            "`Promise.allSettled(tools.task(...))`. Synthesize the "
            "verification report from subagent results in code — do NOT "
            "re-read each slice file manually into conversation.",
        ]
        prompt = context_seed + "\n".join(prompt_lines)

        ctx = build_context(state, PhaseName.VERIFY)

        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.VERIFY.value,
            work_id=work_id,
            work_type=work_type,
            context=ctx,
        )

        verify_content = extract_response(result)

        # ── Collect artifacts from disk (agent writes via write_file) ─────
        disk_artifacts = scan_artifact_dir(
            workspace_root,
            work_id,
            PhaseName.VERIFY.value,
            max_preview_chars=_MAX_ARTIFACT_STATE_CHARS,
        )

        # Fallback: if agent wrote nothing, use the extracted response
        if not disk_artifacts:
            # Provide a fallback if the agent returned nothing useful
            if not verify_content or len(verify_content.strip()) < 20:
                verify_content = (
                    "Verification could not produce a meaningful report. "
                    "The agent returned insufficient output. "
                    "Manual review is required."
                )
            materialize_phase_artifacts(
                PhaseName.VERIFY.value,
                {"verification.md": verify_content},
                workspace_root,
                work_id=work_id,
            )
            disk_artifacts = {"verification.md": verify_content[:_MAX_ARTIFACT_STATE_CHARS]}

        # Determine final status from verification content
        # Check the verification artifact (use first one found on disk, or the response)
        verify_text = ""
        if disk_artifacts:
            verify_text = next(iter(disk_artifacts.values()), "")
        is_verified = "VERIFIED" in verify_text.upper() or "PASSED" in verify_text.upper()
        final_status = "completed" if is_verified else "needs_review"

        return {
            "artifacts": {PhaseName.VERIFY.value: disk_artifacts},
            "current_phase": PhaseName.VERIFY.value,
            "status": final_status,
            "prompt_request": None,
            "feedback": [
                {
                    "status": "passed" if is_verified else "needs_review",
                    "tier": "verify",
                    "reason": verify_text[:500],
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
