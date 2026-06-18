"""ADVERSARIAL phase as a LangGraph subgraph.

Single agent tier: the deterministic plan structure was already validated by
``critic_plan``, so there is no structural / plan-validation node here. A
no-tool ``plan_directive`` step (plan→do split) precedes the red-team
``agent_check``, mirroring the critic subgraph's quality pattern.

    START → plan_directive → agent_check → END
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from spine.models.enums import PhaseName
from spine.workflow.subgraph_state import AdversarialSubgraphState
from spine.adversarial.review import agent_adversarial_check
from spine.agents.plan_do import run_plan_node
from spine.agents.prompt_format import Tag, hostage_layout, xml_blocks

logger = logging.getLogger(__name__)


async def _adversarial_directive_node(
    state: AdversarialSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """No-tool planning step before the adversarial agent_check.

    Lets the reviewer plan its attack (which slices / failure modes to press
    on) before the do node runs the full red-team agent.
    """
    work_id = state.get("work_id", "unknown")
    description = state.get("description", "")
    task = hostage_layout(
        xml_blocks((Tag.OBJECTIVE, description)),
        (
            "Plan a red-team review of the approved implementation plan. The "
            "do node will inspect plan.json (and the spec) and return a verdict "
            "(PASSED / NEEDS_REVISION / NEEDS_REVIEW). Identify the failure "
            "modes, hidden assumptions, and risks most worth attacking."
        ),
    )
    directive = await run_plan_node(
        state=dict(state),
        config=config,
        phase_path=PhaseName.ADVERSARIAL.value,
        task_description=task,
        role_hint="adversarial reviewer for the approved plan",
        workspace_root=state.get("workspace_root", "."),
    )
    logger.info(
        "[%s] ADVERSARIAL plan-directive: approach=%r",
        work_id, directive.approach[:80],
    )
    return {"adversarial_directive": directive.model_dump()}


async def _agent_check_node(
    state: AdversarialSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Run the adversarial agent review within the subgraph."""
    work_id = state.get("work_id", "unknown")
    logger.info("[%s] Adversarial agent check starting", work_id)
    pseudo_state = {
        "artifacts": state.get("artifacts", {}),
        "description": state.get("description", ""),
        "workspace_root": state.get("workspace_root", "."),
        "work_id": work_id,
        "work_type": state.get("work_type", ""),
        "specification_json": state.get("specification_json"),
        "plan_json": state.get("plan_json"),
        # Forward the directive so agent_adversarial_check can prepend it.
        "adversarial_directive": state.get("adversarial_directive"),
    }
    result = await agent_adversarial_check(pseudo_state, config)
    logger.info(
        "[%s] Adversarial agent check complete: status=%s",
        work_id, result.get("status"),
    )
    return {
        "agent_result": result,
        "phase_status": result["status"],
    }


def build_adversarial_subgraph() -> Any:
    """Build the ADVERSARIAL phase subgraph (always reviews PLAN)."""
    builder = StateGraph(AdversarialSubgraphState)

    builder.add_node("plan_directive", _adversarial_directive_node)
    builder.add_node("agent_check", _agent_check_node)

    builder.add_edge(START, "plan_directive")
    builder.add_edge("plan_directive", "agent_check")
    builder.add_edge("agent_check", END)

    return builder
