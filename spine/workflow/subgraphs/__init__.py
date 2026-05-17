"""SPINE phase subgraphs package.

Each phase (SPECIFY, PLAN, TASKS, IMPLEMENT, VERIFY, CRITIC) is implemented
as a compiled LangGraph StateGraph that runs as a node in the parent
orchestrator graph. Subgraphs have isolated state schemas and checkpointers.
"""

from spine.workflow.subgraphs.verify_subgraph import build_verify_subgraph
from spine.workflow.subgraphs.implement_subgraph import build_implement_subgraph
from spine.workflow.subgraphs.tasks_subgraph import build_tasks_subgraph
from spine.workflow.subgraphs.specify_subgraph import build_specify_subgraph
from spine.workflow.subgraphs.plan_subgraph import build_plan_subgraph
from spine.workflow.subgraphs.critic_subgraph import build_critic_subgraph

__all__ = [
    "build_verify_subgraph",
    "build_implement_subgraph",
    "build_tasks_subgraph",
    "build_specify_subgraph",
    "build_plan_subgraph",
    "build_critic_subgraph",
]
