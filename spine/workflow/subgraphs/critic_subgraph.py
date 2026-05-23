"""CRITIC phase as a LangGraph subgraph.

The subgraph has two internal nodes:
1. ``structural_check`` — fast, no-LLM structural check.
2. ``agent_check`` — deep LLM-based quality review.

Additionally, for the PLAN phase, a ``plan_validation`` node validates
plan.json structure (feature_slices, dependencies, cycles).

Parameterized by ``reviewed_phase`` so the same subgraph builder can be
used for critic_specify, critic_plan, critic_tasks.
"""

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from spine.models.enums import PhaseName, ReviewStatus
from spine.workflow.subgraph_state import CriticSubgraphState
from spine.workflow.critic_review import structural_critic_check, agent_critic_check

logger = logging.getLogger(__name__)


async def _structural_check_node(
    state: CriticSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Run the structural critic check within the subgraph."""
    reviewed_phase = state.get("reviewed_phase", "unknown")
    # structural_critic_check expects a dict-like state with "artifacts" key
    pseudo_state = {"artifacts": state.get("artifacts", {})}
    result = structural_critic_check(pseudo_state, reviewed_phase)
    return {
        "structural_result": result,
        "phase_status": "success"
        if result["status"] == ReviewStatus.PASSED.value
        else result["status"],
    }


async def _plan_validation_node(
    state: CriticSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Validate plan.json structure when reviewing the PLAN phase.

    Checks plan.json for:
    - Non-empty feature_slices array
    - Every slice has acceptance_criteria and target_files
    - All dependency IDs reference existing slice IDs
    - No dependency cycles (simple DFS detection)

    Returns phase_status=NeedsRevision on failure, NeedsReview on error.
    """
    reviewed_phase = state.get("reviewed_phase", "")
    if PhaseName(reviewed_phase) != PhaseName.PLAN:
        # Not reviewing PLAN phase — skip validation
        return {"phase_status": "passed"}

    workspace_root = state.get("workspace_root", ".")
    work_id = state.get("work_id", "unknown")

    # Load plan.json from disk
    plan_path = Path(workspace_root) / ".spine" / "artifacts" / work_id / "plan" / "plan.json"
    if not plan_path.exists():
        # No plan.json — not a structured plan, skip validation
        return {"phase_status": "passed"}

    try:
        plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"[{work_id}] Could not load plan.json for validation: {e}")
        return {
            "phase_status": ReviewStatus.NEEDS_REVISION.value,
            "validation_result": {
                "status": ReviewStatus.NEEDS_REVISION.value,
                "tier": "structural",
                "reason": f"Could not load plan.json: {e}",
                "suggestions": ["Ensure plan.json is valid JSON"],
            },
        }

    # Validate feature_slices structure
    feature_slices = plan_data.get("feature_slices")
    if not isinstance(feature_slices, list) or len(feature_slices) == 0:
        return {
            "phase_status": ReviewStatus.NEEDS_REVISION.value,
            "validation_result": {
                "status": ReviewStatus.NEEDS_REVISION.value,
                "tier": "structural",
                "reason": "plan.json must contain a non-empty 'feature_slices' array",
                "suggestions": ["Add at least one feature slice with id, title, target_files, acceptance_criteria, and dependencies"],
            },
        }

    # Build ID set and validate each slice
    slice_ids: set[str] = set()
    errors: list[str] = []
    for i, s in enumerate(feature_slices):
        if not isinstance(s, dict):
            errors.append(f"Slice at index {i} is not a valid object")
            continue

        sid = s.get("id", "")
        if sid:
            slice_ids.add(sid)
        else:
            errors.append(f"Slice at index {i} is missing 'id'")

        if not s.get("acceptance_criteria"):
            errors.append(f"Slice '{sid or i}' is missing 'acceptance_criteria'")
        if not s.get("target_files"):
            errors.append(f"Slice '{sid or i}' is missing 'target_files'")

    if errors:
        return {
            "phase_status": ReviewStatus.NEEDS_REVISION.value,
            "validation_result": {
                "status": ReviewStatus.NEEDS_REVISION.value,
                "tier": "structural",
                "reason": "; ".join(errors),
                "suggestions": ["Ensure every slice has: id, target_files, acceptance_criteria"],
            },
        }

    # Dependency integrity: all referenced IDs must exist
    dep_errors: list[str] = []
    dep_graph: dict[str, list[str]] = {}
    for s in feature_slices:
        sid = s.get("id", "")
        deps = s.get("dependencies") or []
        dep_graph[sid] = deps
        for dep_id in deps:
            if dep_id not in slice_ids:
                dep_errors.append(f"Slice '{sid}' depends on unknown slice '{dep_id}'")

    if dep_errors:
        return {
            "phase_status": ReviewStatus.NEEDS_REVISION.value,
            "validation_result": {
                "status": ReviewStatus.NEEDS_REVISION.value,
                "tier": "structural",
                "reason": "Dependency integrity check failed: " + "; ".join(dep_errors),
                "suggestions": ["Ensure all dependency IDs reference existing slice IDs", "Check for typos in dependency references"],
            },
        }

    # Cycle detection via DFS
    def has_cycle(graph: dict[str, list[str]]) -> bool:
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {node: WHITE for node in graph}

        def dfs(node: str) -> bool:
            color[node] = GRAY
            for neighbor in graph.get(node, []):
                if neighbor not in color:
                    continue
                if color[neighbor] == GRAY:
                    return True
                if color[neighbor] == WHITE and dfs(neighbor):
                    return True
            color[node] = BLACK
            return False

        return any(color[n] == WHITE and dfs(n) for n in color)

    if has_cycle(dep_graph):
        return {
            "phase_status": ReviewStatus.NEEDS_REVISION.value,
            "validation_result": {
                "status": ReviewStatus.NEEDS_REVISION.value,
                "tier": "structural",
                "reason": "Dependency cycle detected among feature slices",
                "suggestions": ["Remove circular dependencies between slices", "Reorder slices into a proper DAG"],
            },
        }

    return {"phase_status": "passed"}


async def _agent_check_node(
    state: CriticSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Run the agent critic check within the subgraph."""
    reviewed_phase = state.get("reviewed_phase", "unknown")
    # agent_critic_check expects a dict-like state
    pseudo_state = {
        "artifacts": state.get("artifacts", {}),
        "description": state.get("description", ""),
        "workspace_root": state.get("workspace_root", "."),
        "work_id": state.get("work_id", "unknown"),
        "work_type": state.get("work_type", ""),
        "feedback": state.get("feedback", []),
        "retry_count": {reviewed_phase: state.get("retry_count", 0)},
        "max_retries": 3,
    }
    result = await agent_critic_check(pseudo_state, reviewed_phase, config)
    return {
        "agent_result": result,
        "phase_status": result["status"],
    }


def _critic_subgraph_router(state: CriticSubgraphState) -> str:
    """Route based on structural check result."""
    structural = state.get("structural_result", {})
    status = structural.get("status", ReviewStatus.NEEDS_REVISION.value)
    if status == ReviewStatus.PASSED.value:
        return "passed"
    if status == ReviewStatus.NEEDS_REVIEW.value:
        return "needs_review"
    return "needs_revision"


def _plan_validation_router(state: CriticSubgraphState) -> str:
    """Route after plan validation node."""
    phase_status = state.get("phase_status", "passed")
    if phase_status == "passed":
        return "passed"
    if phase_status == ReviewStatus.NEEDS_REVIEW.value:
        return "needs_review"
    return "needs_revision"


def build_critic_subgraph(reviewed_phase: str) -> Any:
    """Build a CRITIC phase subgraph for a specific reviewed phase.

    Args:
        reviewed_phase: The phase being reviewed (e.g. "plan", "tasks").
    """
    builder = StateGraph(CriticSubgraphState)

    builder.add_node("structural_check", _structural_check_node)
    builder.add_node("agent_check", _agent_check_node)

    # Add plan validation node only when reviewing PLAN phase
    if PhaseName(reviewed_phase) == PhaseName.PLAN:
        builder.add_node("plan_validation", _plan_validation_node)

    builder.add_edge(START, "structural_check")
    builder.add_conditional_edges(
        "structural_check",
        _critic_subgraph_router,
        {
            "passed": "agent_check" if PhaseName(reviewed_phase) != PhaseName.PLAN else "plan_validation",
            "needs_revision": END,
            "needs_review": END,
        },
    )

    # Plan validation node routing (only wired when reviewing PLAN)
    if PhaseName(reviewed_phase) == PhaseName.PLAN:
        builder.add_conditional_edges(
            "plan_validation",
            _plan_validation_router,
            {
                "passed": "agent_check",
                "needs_revision": END,
                "needs_review": END,
            },
        )

    builder.add_edge("agent_check", END)

    return builder
