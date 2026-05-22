"""CRITIC phase as a LangGraph subgraph.

The subgraph has two internal nodes:
1. ``structural_check`` — fast, no-LLM structural check.
2. ``agent_check`` — deep LLM-based quality review.

Parameterized by ``reviewed_phase`` so the same subgraph builder can be
used for critic_specify, critic_plan, critic_tasks.
"""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from spine.models.enums import ReviewStatus
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


def build_critic_subgraph(reviewed_phase: str) -> Any:
    """Build a CRITIC phase subgraph for a specific reviewed phase.

    Args:
        reviewed_phase: The phase being reviewed (e.g. "plan", "tasks").
    """
    builder = StateGraph(CriticSubgraphState)

    builder.add_node("structural_check", _structural_check_node)
    builder.add_node("agent_check", _agent_check_node)

    builder.add_edge(START, "structural_check")
    builder.add_conditional_edges(
        "structural_check",
        _critic_subgraph_router,
        {
            "passed": "agent_check",
            "needs_revision": END,
            "needs_review": END,
        },
    )
    builder.add_edge("agent_check", END)

    return builder
