"""SPINE IMPLEMENT phase — generate code to implement feature slices.

The IMPLEMENT phase is now dispatched via the Send API subgraph
(spine/workflow/subgraphs/implement_subgraph.py).  This module is
kept as a fallback for when the ``_SUBGRAPH_ENABLED`` feature flag
is turned off for IMPLEMENT.

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
    artifact_path,
)
from spine.workflow.registry import get_registry

logger = logging.getLogger(__name__)

_MAX_ARTIFACT_STATE_CHARS = 500


async def call_implement(
    state: WorkflowState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """Execute the IMPLEMENT phase (fallback path).

    Delegates to the implement Deep Agent, which writes code for each
    feature slice.  When the Send API subgraph is enabled (default),
    this function is not called — the subgraph handles all dispatch.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        Partial state update with implementation artifacts.
    """
    work_id = state.get("work_id", "unknown")
    work_type = state.get("work_type", "")
    retry_count = state.get("retry_count", {}).get(PhaseName.IMPLEMENT.value, 0)
    feedback = state.get("feedback", [])
    workspace_root = state.get("workspace_root", ".")

    logger.info(f"[{work_id}] IMPLEMENT phase starting (retry={retry_count})")

    try:
        agent = build_implement_agent(state, config)
        materialize_artifacts(state, workspace_root, work_id=work_id)

        impl_dir = artifact_path(work_id, "implement")
        plan_dir = artifact_path(work_id, PhaseName.PLAN.value)

        execution_waves = state.get("execution_waves", [])
        total_slices = sum(len(wave) for wave in execution_waves)

        rework_prefix = ""
        if retry_count > 0:
            rework_prefix = (
                "REWORK PASS: Revise the prior implementation. "
                "Address all points from the critic feedback.\n\n"
            )

        prompt = (
            f"{rework_prefix}"
            f"Implement {total_slices} feature slice(s) from the plan. "
            f"Write clean, production-quality code for each slice.\n\n"
            f"Read the plan artifacts — the plan.json file contains "
            f"structured feature_slices with id, title, target_files, "
            f"execution_requirements, dependencies, and acceptance_criteria:\n"
            f"- Plan: {plan_dir}/plan.md\n"
            f"- Structured plan: {plan_dir}/plan.json\n\n"
            f"Use `read_slice_files` to load all slice definitions in one call. "
            f"Dispatch a slice-implementer subagent per slice via `task` "
            f"inside `eval`, then synthesize results with "
            f"`write_implementation_report`.\n\n"
            f"Write implementation artifacts to `{impl_dir}/`."
        )

        if retry_count > 0 and feedback:
            feedback_text = "\n".join(
                f"- [{f.get('tier', 'unknown')}] {f.get('reason', '')}"
                for f in feedback
                if isinstance(f, dict)
            )
            prompt += f"\n\n## Previous Review Feedback\n{feedback_text}\n"

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

        disk_artifacts = scan_artifact_dir(
            workspace_root,
            work_id,
            PhaseName.IMPLEMENT.value,
            max_preview_chars=_MAX_ARTIFACT_STATE_CHARS,
        )

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


_registry = get_registry()
_registry.register(
    name=PhaseName.IMPLEMENT.value,
    call_fn=call_implement,
    build_agent_fn=build_implement_agent,
    description="Generate code to implement feature slices",
)