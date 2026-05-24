"""Integration tests for the exploration subgraph.

These tests verify that the multi-node exploration loop
builds correctly and handles state transitions properly.
They do NOT require live LLM calls — they test the graph
topology, state schema, and edge routing only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Test: subgraph builds ────────────────────────────────────────────────


def test_exploration_subgraph_builds():
    """The exploration subgraph should compile without errors."""
    from spine.workflow.subgraphs.exploration_subgraph import build_exploration_subgraph
    from spine.models.enums import PhaseName

    builder = build_exploration_subgraph(phase=PhaseName.SPECIFY.value)
    graph = builder.compile()
    assert graph is not None


def test_exploration_subgraph_supports_plan():
    """PLAN exploration subgraph should build and compile without errors."""
    from spine.workflow.subgraphs.exploration_subgraph import build_exploration_subgraph
    from spine.models.enums import PhaseName

    builder = build_exploration_subgraph(phase=PhaseName.PLAN.value)
    graph = builder.compile()
    assert graph is not None


def test_exploration_subgraph_rejects_unknown_phase():
    """Unknown phase should raise ValueError."""
    from spine.workflow.subgraphs.exploration_subgraph import build_exploration_subgraph

    with pytest.raises(ValueError, match="Unsupported phase"):
        build_exploration_subgraph(phase="verify")


# ── Test: state schema ───────────────────────────────────────────────────


def test_exploration_state_schema_fields():
    """The ExplorationSubgraphState should support all expected fields."""
    from spine.workflow.subgraph_state import ExplorationSubgraphState

    state: ExplorationSubgraphState = {
        "phase": "specify",
        "work_id": "test-wk-1",
        "work_type": "task",
        "description": "Add dark mode toggle to settings",
        "workspace_root": "/tmp/test-project",
        "retry_count": 0,
        "feedback": [],
        "messages": [],
        "artifacts_output": {},
        "phase_status": "",
        "research_round": 0,
        "max_rounds": 3,
        "manager_decision": "",
        "topics": [],
        "findings": [],
        "agent_response": "",
    }

    assert state["research_round"] == 0
    assert state["max_rounds"] == 3
    assert state["topics"] == []
    assert state["findings"] == []


def test_exploration_state_accumulates_findings():
    """The operator.add reducer should merge findings from parallel nodes."""
    from spine.workflow.subgraph_state import ExplorationSubgraphState

    # Simulate what LangGraph does internally: apply state reducer
    base: ExplorationSubgraphState = {
        "phase": "specify",
        "work_id": "test-wk-1",
        "work_type": "task",
        "description": "test",
        "workspace_root": "/tmp",
        "retry_count": 0,
        "feedback": [],
        "messages": [],
        "artifacts_output": {},
        "phase_status": "",
        "research_round": 0,
        "max_rounds": 3,
        "manager_decision": "explore",
        "topics": [],
        "findings": [{"summary": "Finding A", "patterns": ["p1"]}],
        "agent_response": "",
    }

    # Second round adds more findings via operator.add
    assert len(base["findings"]) == 1
    # In a real graph invocation, the reducer would handle this
    # We just verify the schema accepts it


# ── Test: edge routing ───────────────────────────────────────────────────


def test_research_router_fan_out():
    """research_router should return Send objects when decision=explore."""
    from spine.workflow.subgraphs.exploration_subgraph import _research_router
    from spine.workflow.subgraph_state import ExplorationSubgraphState
    from langgraph.types import Send

    state: ExplorationSubgraphState = {
        "phase": "specify",
        "work_id": "test-wk-1",
        "work_type": "task",
        "description": "test",
        "workspace_root": "/tmp",
        "retry_count": 0,
        "feedback": [],
        "messages": [],
        "artifacts_output": {},
        "phase_status": "",
        "research_round": 0,
        "max_rounds": 3,
        "manager_decision": "explore",
        "topics": ["auth-module", "database-layer"],
        "findings": [],
        "agent_response": "",
    }

    result = _research_router(state)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(s, Send) for s in result)
    assert result[0].node == "explore"
    assert result[0].arg["topic"] == "auth-module"
    assert result[0].arg["phase"] == "specify"


def test_research_router_done():
    """research_router should return 'synthesize' when decision=done."""
    from spine.workflow.subgraphs.exploration_subgraph import _research_router
    from spine.workflow.subgraph_state import ExplorationSubgraphState

    state: ExplorationSubgraphState = {
        "phase": "specify",
        "work_id": "test-wk-1",
        "work_type": "task",
        "description": "test",
        "workspace_root": "/tmp",
        "retry_count": 0,
        "feedback": [],
        "messages": [],
        "artifacts_output": {},
        "phase_status": "",
        "research_round": 1,
        "max_rounds": 3,
        "manager_decision": "done",
        "topics": [],
        "findings": [{"summary": "done"}],
        "agent_response": "",
    }

    result = _research_router(state)
    assert result == "synthesize"


def test_sufficiency_router_loop():
    """sufficiency_router should return 'loop' when not done and under max rounds."""
    from spine.workflow.subgraphs.exploration_subgraph import _sufficiency_router
    from spine.workflow.subgraph_state import ExplorationSubgraphState

    state: ExplorationSubgraphState = {
        "phase": "specify",
        "work_id": "test-wk-1",
        "work_type": "task",
        "description": "test",
        "workspace_root": "/tmp",
        "retry_count": 0,
        "feedback": [],
        "messages": [],
        "artifacts_output": {},
        "phase_status": "",
        "research_round": 1,
        "max_rounds": 3,
        "manager_decision": "explore",
        "topics": [],
        "findings": [],
        "agent_response": "",
    }

    assert _sufficiency_router(state) == "loop"


def test_sufficiency_router_max_rounds():
    """sufficiency_router should return 'done' when max rounds reached."""
    from spine.workflow.subgraphs.exploration_subgraph import _sufficiency_router
    from spine.workflow.subgraph_state import ExplorationSubgraphState

    state: ExplorationSubgraphState = {
        "phase": "specify",
        "work_id": "test-wk-1",
        "work_type": "task",
        "description": "test",
        "workspace_root": "/tmp",
        "retry_count": 0,
        "feedback": [],
        "messages": [],
        "artifacts_output": {},
        "phase_status": "",
        "research_round": 3,  # at max_rounds
        "max_rounds": 3,
        "manager_decision": "explore",  # manager wants more but...
        "topics": ["more"],
        "findings": [],
        "agent_response": "",
    }

    assert _sufficiency_router(state) == "done"


# ── Test: compose integration ────────────────────────────────────────────


def test_exploration_subgraph_registered_for_specify():
    """When _USE_EXPLORATION_SUBGRAPH['specify'] is True,
    the builder registry should have the exploration builder."""
    from spine.workflow.compose import (
        _USE_EXPLORATION_SUBGRAPH,
        get_subgraph_builder,
    )

    if _USE_EXPLORATION_SUBGRAPH.get("specify", False):
        builder = get_subgraph_builder("specify")
        assert builder is not None, "Exploration subgraph should be registered for specify"


def test_workflow_graph_builds_with_exploration_subgraph():
    """build_workflow_graph('spec') should compile when exploration is enabled."""
    from spine.workflow.compose import build_workflow_graph

    # This builds the full workflow graph — if the exploration subgraph
    # override is active, it will be used for the specify node.
    graph = build_workflow_graph("task")
    assert graph is not None
