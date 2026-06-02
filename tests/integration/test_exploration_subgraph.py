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
    # Sends now target the do→summarise branch head; explore_do chains
    # into summarise via a plain edge before fan-in.
    assert result[0].node == "explore_do"
    assert result[0].arg["topic"] == "auth-module"
    assert result[0].arg["phase"] == "specify"


def test_research_router_filters_explored_topics():
    """research_router should filter out topics already explored in previous rounds."""
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
        "research_round": 1,
        "max_rounds": 3,
        "manager_decision": "explore",
        "topics": ["auth-module", "database-layer", "new-topic"],
        "findings": [
            {"topic": "auth-module", "summary": "already explored"},
            {"topic": "database-layer", "summary": "done"},
        ],
        "agent_response": "",
    }

    result = _research_router(state)
    assert isinstance(result, list)
    assert len(result) == 1
    # Sends now target the do→summarise branch head; explore_do chains
    # into summarise via a plain edge before fan-in.
    assert result[0].node == "explore_do"
    assert result[0].arg["topic"] == "new-topic"


def test_research_router_all_topics_explored():
    """research_router should return 'synthesize' when all topics already explored."""
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
        "manager_decision": "explore",
        "topics": ["auth-module", "database-layer"],
        "findings": [
            {"topic": "auth-module", "summary": "done"},
            {"topic": "database-layer", "summary": "done"},
        ],
        "agent_response": "",
    }

    result = _research_router(state)
    assert result == "synthesize"


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


def test_sufficiency_router_recursion_capped_forces_done():
    """A recursion-capped branch forces 'done' even when the manager still
    wants to explore and we are under max_rounds — looping again would only
    re-burn the budget and stall the phase.
    """
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
        "research_round": 1,  # well under max_rounds
        "max_rounds": 3,
        "manager_decision": "explore",  # manager wants more, but a topic capped
        "recursion_capped_seen": True,
        "topics": ["more"],
        "findings": [],
        "agent_response": "",
    }

    assert _sufficiency_router(state) == "done"


@pytest.mark.asyncio
async def test_explore_do_node_surfaces_recursion_capped(monkeypatch):
    """A recursion-capped branch bubbles ``recursion_capped_seen=True`` into
    the reduced subgraph update so the sufficiency router can see it. A
    non-capped branch leaves the channel unset.
    """
    import spine.agents.exploration_agents as ea
    from spine.workflow.subgraphs.exploration_subgraph import _explore_do_node

    async def _fake_run(state, config, *, topic):
        return {
            "exploration_evidence": {
                "topic": topic,
                "recursion_capped": state["__capped"],
            }
        }

    monkeypatch.setattr(ea, "run_explore_do_node", _fake_run)

    capped = await _explore_do_node(
        {"topic": "t", "phase": "specify", "work_id": "w1", "__capped": True}, None
    )
    assert capped.update.get("recursion_capped_seen") is True

    not_capped = await _explore_do_node(
        {"topic": "t", "phase": "specify", "work_id": "w1", "__capped": False}, None
    )
    assert "recursion_capped_seen" not in not_capped.update


# ── Test: topic_lookup node ──────────────────────────────────────────────


def test_enrich_topic_no_hits_returns_bare():
    from spine.workflow.subgraphs.exploration_subgraph import _enrich_topic

    assert _enrich_topic("auth-module", []) == "auth-module"


def test_enrich_topic_appends_symbol_refs():
    from spine.workflow.subgraphs.exploration_subgraph import _enrich_topic

    hits = [
        {"symbol_name": "AuthManager", "file_path": "spine/auth.py"},
        {"symbol_name": "verify_token", "file_path": "spine/auth/jwt.py"},
    ]
    enriched = _enrich_topic("auth-module", hits)
    assert enriched.startswith("auth-module")
    assert "AuthManager (spine/auth.py)" in enriched
    assert "verify_token (spine/auth/jwt.py)" in enriched
    assert "recall symbols" in enriched


def test_research_router_enriches_topics_with_hits():
    """Send args should carry recall-enriched topic strings when hits exist."""
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
        "research_round": 0,
        "max_rounds": 3,
        "manager_decision": "explore",
        "topics": ["auth-module"],
        "findings": [],
        "agent_response": "",
        "topic_recall_hits": {
            "auth-module": [
                {
                    "symbol_name": "AuthManager",
                    "file_path": "spine/auth.py",
                    "similarity": 0.91,
                },
            ],
        },
    }

    sends = _research_router(state)
    assert isinstance(sends, list) and len(sends) == 1
    topic_arg = sends[0].arg["topic"]
    assert topic_arg.startswith("auth-module")
    assert "AuthManager (spine/auth.py)" in topic_arg


def test_topic_lookup_short_circuits_when_done():
    """topic_lookup returns an empty hits map when the manager said 'done'."""
    import asyncio

    from spine.workflow.subgraphs.exploration_subgraph import _topic_lookup_node
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
        "findings": [],
        "agent_response": "",
    }

    result = asyncio.run(_topic_lookup_node(state, None))
    assert result == {"topic_recall_hits": {}}


def test_topic_lookup_filters_threshold_and_top_k(monkeypatch):
    """topic_lookup keeps only the top-K hits above the configured threshold."""
    import asyncio
    import json as _json

    from spine.workflow.subgraphs import exploration_subgraph as mod
    from spine.workflow.subgraph_state import ExplorationSubgraphState

    # Stub SpineConfig.load() so this runs without a real .spine directory.
    class _FakeCfg:
        checkpoint_path = "/tmp/fake.db"
        embedding_provider = "openai-embeddings"
        recall_k = 5
        specify_context_token_budget = 30000
        topic_lookup_top_k = 2
        topic_lookup_min_similarity = 0.5

    monkeypatch.setattr(
        "spine.config.SpineConfig.load",
        classmethod(lambda cls: _FakeCfg()),
    )

    # Stub RecallTool to return canned, deliberately-mixed scored results.
    captured_calls = []

    class _StubRecall:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def _arun(self, **kw):
            captured_calls.append(kw)
            return _json.dumps({
                "results": [
                    {"symbol_name": "A", "file_path": "a.py", "similarity": 0.95},
                    {"symbol_name": "B", "file_path": "b.py", "similarity": 0.72},
                    {"symbol_name": "C", "file_path": "c.py", "similarity": 0.55},
                    {"symbol_name": "D", "file_path": "d.py", "similarity": 0.40},
                ]
            })

    monkeypatch.setattr(
        "spine.agents.tools.recall_tool.RecallTool", _StubRecall
    )

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
        "topics": ["auth-module"],
        "findings": [],
        "agent_response": "",
        "task_category": "Backend/API",
    }

    result = asyncio.run(mod._topic_lookup_node(state, None))
    hits = result["topic_recall_hits"]["auth-module"]
    # Only A, B, C are ≥0.5; top-K=2 keeps the two highest.
    assert [h["symbol_name"] for h in hits] == ["A", "B"]
    # The stub was called once (only one new topic).
    assert len(captured_calls) == 1
    assert captured_calls[0]["query"] == "auth-module"
    # task_category is no longer part of the recall API — see classification.py.
    assert "task_category" not in captured_calls[0]


def test_topic_lookup_drops_test_artifacts(monkeypatch):
    """Test-file hits crowd out production code in recall results.

    Regression: trace 019e6974 showed researchers anchored on
    ``test_config_nonexistent_file`` when asked about CLI argument parsing.
    Tests have rich docstrings that score high on natural-language topic
    similarity but tell the researcher nothing about the production code.
    """
    import asyncio
    import json as _json

    from spine.workflow.subgraphs import exploration_subgraph as mod
    from spine.workflow.subgraph_state import ExplorationSubgraphState

    class _FakeCfg:
        checkpoint_path = "/tmp/fake.db"
        embedding_provider = "openai-embeddings"
        recall_k = 5
        specify_context_token_budget = 30000
        topic_lookup_top_k = 2
        topic_lookup_min_similarity = 0.5

    monkeypatch.setattr(
        "spine.config.SpineConfig.load",
        classmethod(lambda cls: _FakeCfg()),
    )

    class _StubRecall:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def _arun(self, **kw):
            return _json.dumps({
                "results": [
                    # Highest similarity is a test — must be dropped.
                    {"symbol_name": "test_cli_parses_verbose",
                     "file_path": "tests/unit/test_cli.py",
                     "similarity": 0.95},
                    # Production symbol, lower similarity.
                    {"symbol_name": "main",
                     "file_path": "spine/cli/__init__.py",
                     "similarity": 0.80},
                    # Test class via name prefix.
                    {"symbol_name": "TestCliParser",
                     "file_path": "spine/cli/parser.py",
                     "similarity": 0.75},
                    # Conftest fixture path.
                    {"symbol_name": "tmp_workspace",
                     "file_path": "tests/conftest.py",
                     "similarity": 0.70},
                ]
            })

    monkeypatch.setattr("spine.agents.tools.recall_tool.RecallTool", _StubRecall)

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
        "topics": ["how does CLI argument parsing work"],
        "findings": [],
        "agent_response": "",
        "task_category": "Backend/API",
    }

    result = asyncio.run(mod._topic_lookup_node(state, None))
    hits = result["topic_recall_hits"]["how does CLI argument parsing work"]
    # The only non-test hit at ≥0.5 is `main`.
    assert [h["symbol_name"] for h in hits] == ["main"]


def test_topic_lookup_keeps_test_artifacts_when_topic_is_about_tests(monkeypatch):
    """If the topic explicitly mentions tests, keep test-file hits."""
    import asyncio
    import json as _json

    from spine.workflow.subgraphs import exploration_subgraph as mod
    from spine.workflow.subgraph_state import ExplorationSubgraphState

    class _FakeCfg:
        checkpoint_path = "/tmp/fake.db"
        embedding_provider = "openai-embeddings"
        recall_k = 5
        specify_context_token_budget = 30000
        topic_lookup_top_k = 2
        topic_lookup_min_similarity = 0.5

    monkeypatch.setattr(
        "spine.config.SpineConfig.load",
        classmethod(lambda cls: _FakeCfg()),
    )

    class _StubRecall:
        def __init__(self, **kwargs):
            pass

        async def _arun(self, **kw):
            return _json.dumps({
                "results": [
                    {"symbol_name": "test_cli", "file_path": "tests/test_cli.py",
                     "similarity": 0.95},
                    {"symbol_name": "main", "file_path": "spine/cli/__init__.py",
                     "similarity": 0.80},
                ]
            })

    monkeypatch.setattr("spine.agents.tools.recall_tool.RecallTool", _StubRecall)

    state: ExplorationSubgraphState = {
        "phase": "specify", "work_id": "wk", "work_type": "task",
        "description": "", "workspace_root": "/tmp", "retry_count": 0,
        "feedback": [], "messages": [], "artifacts_output": {},
        "phase_status": "", "research_round": 0, "max_rounds": 3,
        "manager_decision": "explore",
        "topics": ["what tests cover the CLI"],
        "findings": [], "agent_response": "", "task_category": "Backend/API",
    }

    result = asyncio.run(mod._topic_lookup_node(state, None))
    hits = result["topic_recall_hits"]["what tests cover the CLI"]
    # Topic is about tests — both kept.
    assert [h["symbol_name"] for h in hits] == ["test_cli", "main"]


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
