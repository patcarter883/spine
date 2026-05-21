"""SPINE IMPLEMENT phase — generate code to implement feature slices.

The implement Deep Agent reads the tasks/feature slices (on disk) and
generates code to implement each one. Prior artifacts are NOT inlined —
the agent reads them on demand from the filesystem.

Context engineering: dispatch-only orchestrator patterns for long-running
multi-slice implementation.

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
from spine.agents.implement_agent import build_implement_agent
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


async def call_implement(
    state: WorkflowState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """Execute the IMPLEMENT phase.

    Delegates to the implement Deep Agent, which writes code for each
    feature slice. If reworking, includes prior feedback.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        Partial state update with implementation artifacts.
    """
    # NOTE: The original work description is NOT needed here — IMPLEMENT works
    # from the feature slices and codebase map produced by TASKS (on disk),
    # not from the raw description.  The only additional input beyond prior
    # artifacts should be review feedback (critic gates, verify agent, human).
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
        impl_dir = f".spine/artifacts/{work_id}/implement"
        tasks_dir = f".spine/artifacts/{work_id}/tasks"
        context_seed = f"globalThis.context = {{work_id: '{work_id}', phase: 'implement', tasks_dir: '{tasks_dir}', impl_dir: '{impl_dir}'}};\n\n"

        rework_prefix = ""
        if retry_count > 0:
            rework_prefix = "⚠ **REWORK PASS**: Your primary objective is to revise the prior implementation. Address all points from the critic feedback.\n\n"

        from spine.agents.artifacts import list_slice_files
        slice_files = list_slice_files(workspace_root, work_id)
        slice_count = len(slice_files)

        if slice_count == 1:
            dispatch_note = (
                "There is 1 slice. Dispatch a single `slice-implementer` for "
                "consistency — same context-management benefit, no orchestrator "
                "drift between work items."
            )
        elif slice_count >= 2:
            dispatch_note = (
                f"There are {slice_count} slices. Dispatch all of them in "
                "parallel via `Promise.allSettled(tools.task(...))` inside a "
                "single `eval` call. Do NOT dispatch sequentially — the whole "
                "point of subagent dispatch is parallel isolated contexts."
            )
        else:
            dispatch_note = (
                "No slices were pre-discovered. Call `read_slice_files` — it will "
                "return whatever slice files exist. If it returns none, write a "
                "`write_implementation_report` with an empty slice_results list "
                "and a summary explaining the tasks phase produced no slices."
            )

        has_spec = "spec" in work_type
        spec_path = _artifact_path(work_id, PhaseName.SPECIFY.value)
        plan_path = _artifact_path(work_id, PhaseName.PLAN.value)
        tasks_path = _artifact_path(work_id, PhaseName.TASKS.value)

        prompt_lines = [
            "Implement the feature slices described below. Write clean, "
            "production-quality code for each slice.",
            "",
            "## Task Input",
            "Work from the feature slice files and codebase map produced by the TASKS phase.",
        ]
        if has_spec:
            prompt_lines.extend(
                [
                    f"- Specification: `{spec_path}/specification.md`",
                    f"- Plan: `{plan_path}/plan.md`",
                ]
            )
        prompt_lines.extend(
            [
                f"- Feature Slices: `{tasks_path}/tasks.md` and each `slice-*.md`",
                f"- Codebase map: `{tasks_path}/codebase-map.md`",
                "",
                "Read the codebase map FIRST — it contains file paths, key functions, and conventions "
                "discovered during the tasks phase. Use it instead of re-exploring the codebase.",
                "",
                "## Step 2 Guidelines",
                dispatch_note,
                "",
            ]
        )
        prompt = context_seed + rework_prefix + "\n".join(prompt_lines)
        if retry_count > 0 and feedback:
            feedback_text = "\n".join(
                f"- [{f.get('tier', 'unknown')}] {f.get('reason', '')}"
                for f in feedback
                if isinstance(f, dict)
            )
            prompt += f"## Previous Review Feedback\n{feedback_text}\n"

        ctx = build_context(state, PhaseName.IMPLEMENT)

        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.IMPLEMENT.value,
            work_id=work_id,
            work_type=work_type,
            context=ctx,
        )

        impl_content = extract_response(result)

        # ── Collect artifacts from disk (agent writes via write_file) ─────
        # The authoritative artifacts are the files the agent wrote to disk,
        # NOT the extracted LLM response.  For thinking models the response
        # is chain-of-thought reasoning.
        disk_artifacts = scan_artifact_dir(
            workspace_root,
            work_id,
            PhaseName.IMPLEMENT.value,
            max_preview_chars=_MAX_ARTIFACT_STATE_CHARS,
        )

        # Fallback: if agent wrote nothing, materialize from response
        if not disk_artifacts and impl_content.strip():
            materialize_phase_artifacts(
                PhaseName.IMPLEMENT.value,
                {"implementation.md": impl_content},
                workspace_root,
                work_id=work_id,
            )
            disk_artifacts = {"implementation.md": impl_content[:_MAX_ARTIFACT_STATE_CHARS]}

        return {
            "artifacts": {PhaseName.IMPLEMENT.value: disk_artifacts},
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
