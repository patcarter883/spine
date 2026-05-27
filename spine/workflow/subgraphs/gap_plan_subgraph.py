"""GAP_PLAN phase as a LangGraph subgraph.

The subgraph has two internal nodes:
1. ``run_agent`` — builds and invokes the gap_plan Deep Agent.
2. ``save_artifacts`` — saves the gap_plan.md artifact produced by the agent.

State schema: ``GapPlanSubgraphState`` — isolated from parent ``WorkflowState``.
"""

import logging
from pathlib import Path
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
    original plan and codebase map, and produces gap_plan.md + gap_plan.json.
    """
    work_id = state.get("work_id", "unknown")
    work_type = state.get("work_type", "")
    workspace_root = state.get("workspace_root", ".")

    logger.info(f"[{work_id}] GAP_PLAN subgraph: run_agent starting")

    try:
        agent = build_gap_plan_agent(dict(state), config)
        materialize_artifacts(dict(state), workspace_root, work_id=work_id)

        # Get artifact paths for context
        verify_path = artifact_path(work_id, PhaseName.VERIFY.value)
        plan_path = artifact_path(work_id, PhaseName.PLAN.value)
        gap_plan_path = artifact_path(work_id, PhaseName.GAP_PLAN.value)

        prompt = (
            "Your tools are ready. Call ``read_verification_findings`` FIRST "
            "to load all verification inputs in one call. Then analyze the failures "
            "and call ``write_structured_gap_plan`` with your remediation items.\n\n"
            "## Artifact Paths for Reference\n"
            f"- Verification report: `{verify_path}/verification.md`\n"
            f"- Original plan: `{plan_path}/plan.md`\n"
            f"- Codebase map: `{artifact_path(work_id, 'tasks')}/codebase-map.md`\n"
            f"- Output: `{gap_plan_path}/gap_plan.md` and `{gap_plan_path}/gap_plan.json`\n"
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

        # Load gap_plan.json content for state propagation
        gap_plan_json_path = Path(workspace_root) / gap_plan_path / "gap_plan.json"
        gap_plan_json_content = ""
        if gap_plan_json_path.exists():
            try:
                gap_plan_json_content = gap_plan_json_path.read_text(encoding="utf-8")
            except OSError:
                pass

        return {
            "messages": result.get("messages", []),
            "agent_response": extract_response(result),
            "gap_plan_json": gap_plan_json_content,
            "read_cache": result.get("read_cache") or {},
        }

    except Exception as e:
        logger.error(f"[{work_id}] GAP_PLAN subgraph agent failed: {e}", exc_info=True)
        return {
            "messages": [],
            "agent_response": f"Agent error: {e}",
            "gap_plan_json": "",
            "phase_status": "error",
        }


async def _save_gap_plan_artifacts(
    state: GapPlanSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Save artifacts from the gap_plan agent to disk and state."""
    workspace_root = state.get("workspace_root", ".")
    work_id = state.get("work_id", "unknown")
    existing_phase_status = state.get("phase_status", "")

    if existing_phase_status in ("error", "needs_review"):
        return {
            "artifacts_output": {},
            "phase_status": existing_phase_status,
        }

    # Scan the gap_plan directory for artifacts (gap_plan.md and gap_plan.json)
    disk_artifacts = scan_artifact_dir(
        workspace_root,
        work_id,
        PhaseName.GAP_PLAN.value,
        max_preview_chars=_MAX_ARTIFACT_STATE_CHARS,
    )

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
