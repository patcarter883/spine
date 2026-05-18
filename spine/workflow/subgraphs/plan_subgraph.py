"""PLAN phase as a LangGraph subgraph.

The subgraph has two internal nodes:
1. ``run_agent`` — builds and invokes the plan Deep Agent.
2. ``save_artifacts`` — scans disk for artifacts.
"""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from spine.models.enums import PhaseName
from spine.workflow.subgraph_state import PlanSubgraphState
from spine.agents.plan_agent import build_plan_agent
from spine.agents.helpers import extract_response
from spine.agents.retry import ainvoke_with_retry
from spine.agents.context import build_context
from spine.agents.artifacts import (
    materialize_artifacts,
    materialize_phase_artifacts,
    scan_artifact_dir,
    _artifact_path,
)

logger = logging.getLogger(__name__)
_MAX_ARTIFACT_STATE_CHARS = 500


async def _run_plan_agent(
    state: PlanSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Run the plan Deep Agent within the subgraph.

    The original work description is NOT included in the prompt — the
    specification artifact from SPECIFY already captures and expands on
    it.  PLAN works from the spec on disk, not the raw description.
    """
    work_id = state.get("work_id", "unknown")
    work_type = state.get("work_type", "")
    workspace_root = state.get("workspace_root", ".")

    logger.info(f"[{work_id}] PLAN subgraph: run_agent starting")

    try:
        agent = build_plan_agent(dict(state), config)
        materialize_artifacts(dict(state), workspace_root, work_id=work_id)

        spec_path = _artifact_path(work_id, PhaseName.SPECIFY.value)

        plan_path = _artifact_path(work_id, PhaseName.PLAN.value)
        prompt = (
            "Create a detailed implementation plan based on the specification.\n\n"
            f"Read the specification from `{spec_path}/specification.md`.\n\n"
            f"Write the plan to `{plan_path}/plan.md` using `write_file`."
        )

        ctx = build_context(dict(state), PhaseName.PLAN)

        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.PLAN.value,
            work_id=work_id,
            work_type=work_type,
            context=ctx,
        )

        return {
            "messages": result.get("messages", []),
            "agent_response": extract_response(result),
        }

    except Exception as e:
        logger.error(f"[{work_id}] PLAN subgraph agent failed: {e}", exc_info=True)
        return {
            "messages": [],
            "agent_response": f"Agent error: {e}",
            "phase_status": "error",
        }


async def _save_plan_artifacts(
    state: PlanSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Save artifacts from the plan agent."""
    workspace_root = state.get("workspace_root", ".")
    work_id = state.get("work_id", "unknown")
    agent_response = state.get("agent_response", "")
    existing_phase_status = state.get("phase_status", "")

    if existing_phase_status in ("error", "needs_review"):
        return {"artifacts_output": {}, "phase_status": existing_phase_status}

    disk_artifacts = scan_artifact_dir(
        workspace_root, work_id, PhaseName.PLAN.value,
        max_preview_chars=_MAX_ARTIFACT_STATE_CHARS,
    )

    if not disk_artifacts and agent_response.strip():
        materialize_phase_artifacts(
            PhaseName.PLAN.value,
            {"plan.md": agent_response},
            workspace_root,
            work_id=work_id,
        )
        disk_artifacts = {"plan.md": agent_response[:_MAX_ARTIFACT_STATE_CHARS]}

    return {
        "artifacts_output": disk_artifacts,
        "phase_status": "success" if disk_artifacts else "needs_review",
    }


def build_plan_subgraph() -> Any:
    """Build the PLAN phase subgraph."""
    builder = StateGraph(PlanSubgraphState)
    builder.add_node("run_agent", _run_plan_agent)
    builder.add_node("save_artifacts", _save_plan_artifacts)
    builder.add_edge(START, "run_agent")
    builder.add_edge("run_agent", "save_artifacts")
    builder.add_edge("save_artifacts", END)
    return builder
