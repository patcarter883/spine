"""SPINE ADVERSARIAL phase — full red-team review of the approved plan.

Runs after ``critic_plan`` for critical work types (critical_task /
critical_reviewed_task). Wired into the workflow graph as a conditional
subgraph node (see ``spine.workflow.subgraphs.adversarial_subgraph`` and the
adversarial branches in ``spine.workflow.compose``). Routes:
- PASSED → next phase (implement gate, or END → awaiting_approval)
- NEEDS_REVISION → loop the plan back to PLAN (bounded by the adversarial
  retry budget, separate from the critic's)
- NEEDS_REVIEW → escalate to human review / flag the run

The review logic lives in ``spine.adversarial.review``; the agent in
``spine.adversarial.agent``.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.runnables import RunnableConfig

from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.workflow.registry import get_registry


def _build_adversarial_agent(
    state: WorkflowState, config: Optional[RunnableConfig] = None
) -> Any:
    """Build the adversarial Deep Agent. Used by the registry as build_agent_fn."""
    from spine.adversarial.agent import build_adversarial_agent

    return build_adversarial_agent(state, config)


# ── Self-register on import ──
_registry = get_registry()
_registry.register(
    name=PhaseName.ADVERSARIAL.value,
    build_agent_fn=_build_adversarial_agent,
    description="Red-team adversarial review of the approved plan",
)
