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


def spec_graph():
    """SPEC workflow: specify → plan → critic_plan → tasks → implement → verify."""
    return build_workflow_graph("spec")


def critical_spec_graph():
    """CRITICAL_SPEC: specify → critic_specify → plan → critic_plan → tasks →
    critic_tasks → implement → verify."""
    return build_workflow_graph("critical_spec")


def quick_graph():
    """QUICK workflow: tasks → implement → verify."""
    return build_workflow_graph("quick")


def critical_quick_graph():
    """CRITICAL_QUICK workflow: tasks → critic → implement → verify."""
    return build_workflow_graph("critical_quick")


def plan_graph():
    """PLAN workflow: specify → plan → critic_plan (no tasks/execution)."""
    return build_workflow_graph("plan")


def plan_spec_graph():
    """PLAN_SPEC workflow: specify → critic_specify → plan → critic_plan."""
    return build_workflow_graph("plan_spec")


def plan_only_graph():
    """PLAN_ONLY workflow: specify → plan → critic_plan (no spec critic)."""
    return build_workflow_graph("plan_only")


def critical_plan_only_graph():
    """CRITICAL_PLAN_ONLY: specify → critic_specify → plan → critic_plan."""
    return build_workflow_graph("critical_plan_only")
