"""GAP_PLAN phase as a LangGraph subgraph.

The subgraph has two internal nodes:
1. ``run_agent`` — builds and invokes the gap_plan Deep Agent.
2. ``save_artifacts`` — saves the gap_plan.md artifact produced by the agent.

State schema: ``GapPlanSubgraphState`` — isolated from parent ``WorkflowState``.
"""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from spine.models.enums import PhaseName
from spine.workflow.subgraph_state import GapPlanSubgraphState
from spine.agents.gap_plan_agent import build_gap_plan_agent
from spine.agents.helpers import extract_response
from spine.agents.retry import ainvoke_with_retry
from spine.agents.context import build_context
from spine.agents.artifacts import (
    materialize_artifacts,
    materialize_phase_artifacts,
    scan_artifact_dir,
    artifact_path,
)

logger = logging.getLogger(__name__)
_MAX_ARTIFACT_STATE_CHARS = 500


async def _run_gap_plan_agent(
    state: GapPlanSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Run the gap_plan Deep Agent within the subgraph.

    Reads the verification report for failed items, references the
    original plan and codebase map, and produces gap_plan.md.
    """
    work_id = state.get("work_id", "unknown")
    work_type = state.get("work_type", "")
    workspace_root = state.get("workspace_root", ".")

    logger.info(f"[{work_id}] GAP_PLAN subgraph: run_agent starting")

    try:
        agent = build_gap_plan_agent(dict(state), config)
        materialize_artifacts(dict(state), workspace_root, work_id=work_id)

        verify_path = artifact_path(work_id, PhaseName.VERIFY.value)
        plan_path = artifact_path(work_id, PhaseName.PLAN.value)
        tasks_path = artifact_path(work_id, PhaseName.TASKS.value)
        impl_path = artifact_path(work_id, PhaseName.IMPLEMENT.value)
        gap_plan_path = artifact_path(work_id, PhaseName.GAP_PLAN.value)

        prompt = (
            "Read the verification report to understand what issues were "
            "found during verification. Then produce a gap_plan.md with "
            "specific, targeted fix instructions.\n\n"
            "## Artifacts Available on Disk\n"
            f"- Verification report: `{verify_path}/verification.md`\n"
            f"- Original plan: `{plan_path}/plan.md`\n"
            f"- Codebase map: `{tasks_path}/codebase-map.md`\n"
            f"- Tasks/slices: `{tasks_path}/tasks.md`\n"
            f"- Implementation summary: `{impl_path}/implementation.md`\n\n"
            "## Instructions\n"
            "1. Read the verification report first (`{verify_path}/verification.md`).\n"
            "2. For each failing slice, identify what needs to change by referencing "
            "the codebase map (`{tasks_path}/codebase-map.md`).\n"
            "3. Read the original plan (`{plan_path}/plan.md`) for context.\n"
            "4. Produce a concise gap_plan.md with per-file fix instructions.\n\n"
            "Do NOT re-explore the codebase or produce a full new plan.\n\n"
            f"Write `gap_plan.md` to `{gap_plan_path}/gap_plan.md` using `write_file`.\n"
            "This file is REQUIRED — without it the phase is treated as failed."
        ).format(
            verify_path=verify_path,
            tasks_path=tasks_path,
            plan_path=plan_path,
            gap_plan_path=gap_plan_path,
        )

        ctx = build_context(dict(state), PhaseName.GAP_PLAN)

        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.GAP_PLAN.value,
            work_id=work_id,
            work_type=work_type,
            context=ctx,
        )

        return {
            "messages": result.get("messages", []),
            "agent_response": extract_response(result),
        }

    except Exception as e:
        logger.error(f"[{work_id}] GAP_PLAN subgraph agent failed: {e}", exc_info=True)
        return {
            "messages": [],
            "agent_response": f"Agent error: {e}",
            "phase_status": "error",
        }


async def _save_gap_plan_artifacts(
    state: GapPlanSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Save artifacts from the gap_plan agent to disk and state."""
    workspace_root = state.get("workspace_root", ".")
    work_id = state.get("work_id", "unknown")
    agent_response = state.get("agent_response", "")
    existing_phase_status = state.get("phase_status", "")

    if existing_phase_status in ("error", "needs_review"):
        return {
            "artifacts_output": {},
            "phase_status": existing_phase_status,
        }

    disk_artifacts = scan_artifact_dir(
        workspace_root,
        work_id,
        PhaseName.GAP_PLAN.value,
        max_preview_chars=_MAX_ARTIFACT_STATE_CHARS,
    )

    if not disk_artifacts and agent_response.strip():
        materialize_phase_artifacts(
            PhaseName.GAP_PLAN.value,
            {"gap_plan.md": agent_response},
            workspace_root,
            work_id=work_id,
        )
        disk_artifacts = {"gap_plan.md": agent_response[:_MAX_ARTIFACT_STATE_CHARS]}

    return {
        "artifacts_output": disk_artifacts,
        "phase_status": "success" if disk_artifacts else "needs_review",
    }


def build_gap_plan_subgraph() -> Any:
    """Build the GAP_PLAN phase subgraph."""
    builder = StateGraph(GapPlanSubgraphState)
    builder.add_node("run_agent", _run_gap_plan_agent)
    builder.add_node("save_artifacts", _save_gap_plan_artifacts)
    builder.add_edge(START, "run_agent")
    builder.add_edge("run_agent", "save_artifacts")
    builder.add_edge("save_artifacts", END)
    return builder
