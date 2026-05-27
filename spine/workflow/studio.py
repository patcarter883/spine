"""SPINE Studio entry points — compiled LangGraph graphs for LangSmith Studio.

Each function returns a compiled StateGraph that Studio can visualize,
interact with, and debug.  These are thin wrappers around
``build_workflow_graph()`` that pre-select the work type so Studio knows
which graph to render.

Usage::

    langgraph dev
    # Then open https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
"""

from __future__ import annotations

from spine.workflow.compose import build_workflow_graph


def task_graph():
    """TASK workflow: specify → plan → critic_plan → implement → verify."""
    return build_workflow_graph("task")


def critical_task_graph():
    """CRITICAL_TASK: specify → critic_specify → plan → critic_plan → implement → verify."""
    return build_workflow_graph("critical_task")


def reviewed_task_graph():
    """REVIEWED_TASK: specify → plan → critic_plan, then ENDs for human approval."""
    return build_workflow_graph("reviewed_task")


def critical_reviewed_task_graph():
    """CRITICAL_REVIEWED_TASK: specify → critic_specify → plan → critic_plan, then ENDs for human approval."""
    return build_workflow_graph("critical_reviewed_task")
