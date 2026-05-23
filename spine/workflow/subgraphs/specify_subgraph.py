"""SPECIFY phase as a LangGraph subgraph.

The subgraph has two internal nodes:
1. ``run_agent`` — builds and invokes the specify Deep Agent.
2. ``save_artifacts`` — scans disk for artifacts.
"""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from spine.models.enums import PhaseName
from spine.workflow.subgraph_state import SpecifySubgraphState
from spine.agents.specify_agent import build_specify_agent
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


async def _run_specify_agent(
    state: SpecifySubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Run the specify Deep Agent within the subgraph."""
    description = state.get("description", "")
    work_id = state.get("work_id", "unknown")
    work_type = state.get("work_type", "")
    workspace_root = state.get("workspace_root", ".")

    logger.info(f"[{work_id}] SPECIFY subgraph: run_agent starting")

    try:
        agent = build_specify_agent(dict(state), config)
        materialize_artifacts(dict(state), workspace_root, work_id=work_id)

        prompt = (
            "Write a formal specification for the work described below.\n\n"
            f"## Work Description\n{description}\n\n"
            f"Write the specification to `{artifact_path(work_id, PhaseName.SPECIFY.value)}/specification.md` "
            "using `write_file`. "
            "The spec must include: scope, requirements, constraints, "
            "and acceptance criteria."
        )

        ctx = build_context(dict(state), PhaseName.SPECIFY)

        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.SPECIFY.value,
            work_id=work_id,
            work_type=work_type,
            context=ctx,
        )

        return {
            "messages": result.get("messages", []),
            "agent_response": extract_response(result),
        }

    except Exception as e:
        logger.error(f"[{work_id}] SPECIFY subgraph agent failed: {e}", exc_info=True)
        return {
            "messages": [],
            "agent_response": f"Agent error: {e}",
            "phase_status": "error",
        }


async def _save_specify_artifacts(
    state: SpecifySubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Save artifacts from the specify agent."""
    workspace_root = state.get("workspace_root", ".")
    work_id = state.get("work_id", "unknown")
    agent_response = state.get("agent_response", "")
    existing_phase_status = state.get("phase_status", "")

    if existing_phase_status in ("error", "needs_review"):
        return {"artifacts_output": {}, "phase_status": existing_phase_status}

    disk_artifacts = scan_artifact_dir(
        workspace_root,
        work_id,
        PhaseName.SPECIFY.value,
        max_preview_chars=_MAX_ARTIFACT_STATE_CHARS,
    )

    if not disk_artifacts and agent_response.strip():
        materialize_phase_artifacts(
            PhaseName.SPECIFY.value,
            {"specification.md": agent_response},
            workspace_root,
            work_id=work_id,
        )
        disk_artifacts = {"specification.md": agent_response[:_MAX_ARTIFACT_STATE_CHARS]}

    return {
        "artifacts_output": disk_artifacts,
        "phase_status": "success" if disk_artifacts else "needs_review",
    }


def build_specify_subgraph() -> Any:
    """Build the SPECIFY phase subgraph.

    Returns:
        Uncompiled StateGraph builder. Call .compile() to get a runnable graph,
        or .compile(checkpointer=...) for per-phase checkpoint isolation.
    """
    builder = StateGraph(SpecifySubgraphState)
    builder.add_node("run_agent", _run_specify_agent)
    builder.add_node("save_artifacts", _save_specify_artifacts)
    builder.add_edge(START, "run_agent")
    builder.add_edge("run_agent", "save_artifacts")
    builder.add_edge("save_artifacts", END)
    return builder
