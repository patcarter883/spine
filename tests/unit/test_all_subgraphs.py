"""Integration tests for all SPINE phase subgraphs.

Verifies that each phase subgraph compiles, has the correct nodes,
and that the parent workflow graph can be built for all work types
with subgraph nodes enabled.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest


class TestAllSubgraphsCompile:
    """Smoke tests: every subgraph compiles without error."""

    def test_verify_subgraph_compiles(self):
        from spine.workflow.subgraphs.verify_subgraph import build_verify_subgraph
        graph = build_verify_subgraph().compile()
        assert graph is not None
        nodes = set(graph.get_graph().nodes.keys())
        assert "run_agent" in nodes
        assert "save_artifacts" in nodes

    def test_implement_subgraph_compiles(self):
        from spine.workflow.subgraphs.implement_subgraph import build_implement_subgraph
        graph = build_implement_subgraph().compile()
        assert graph is not None
        nodes = set(graph.get_graph().nodes.keys())
        assert "run_agent" in nodes
        assert "save_artifacts" in nodes

    def test_tasks_subgraph_compiles(self):
        from spine.workflow.subgraphs.tasks_subgraph import build_tasks_subgraph
        graph = build_tasks_subgraph().compile()
        assert graph is not None
        nodes = set(graph.get_graph().nodes.keys())
        assert "run_agent" in nodes
        assert "save_artifacts" in nodes

    def test_specify_subgraph_compiles(self):
        from spine.workflow.subgraphs.specify_subgraph import build_specify_subgraph
        graph = build_specify_subgraph().compile()
        assert graph is not None
        nodes = set(graph.get_graph().nodes.keys())
        assert "run_agent" in nodes
        assert "save_artifacts" in nodes

    def test_plan_subgraph_compiles(self):
        from spine.workflow.subgraphs.plan_subgraph import build_plan_subgraph
        graph = build_plan_subgraph().compile()
        assert graph is not None
        nodes = set(graph.get_graph().nodes.keys())
        assert "run_agent" in nodes
        assert "save_artifacts" in nodes

    def test_critic_subgraph_compiles(self):
        from spine.workflow.subgraphs.critic_subgraph import build_critic_subgraph
        graph = build_critic_subgraph("plan").compile()
        assert graph is not None
        nodes = set(graph.get_graph().nodes.keys())
        assert "structural_check" in nodes
        assert "agent_check" in nodes


class TestParentGraphWithSubgraphs:
    """The parent orchestrator graph builds correctly with all subgraphs."""

    def test_quick_workflow_graph_compiles(self):
        from spine.workflow.compose import build_workflow_graph
        graph = build_workflow_graph("quick")
        assert graph is not None
        nodes = set(graph.get_graph().nodes.keys())
        assert "plan" in nodes
        assert "implement" in nodes
        assert "verify" in nodes
        assert "gate_plan_to_implement" in nodes
        # All phases should be subgraph nodes now
        assert any(n.startswith("gate_") for n in nodes)

    def test_critical_quick_workflow_graph_compiles(self):
        from spine.workflow.compose import build_workflow_graph
        graph = build_workflow_graph("critical_quick")
        assert graph is not None
        nodes = set(graph.get_graph().nodes.keys())
        assert "plan" in nodes
        assert "critic_plan" in nodes
        assert "implement" in nodes
        assert "verify" in nodes

    def test_spec_workflow_graph_compiles(self):
        from spine.workflow.compose import build_workflow_graph
        graph = build_workflow_graph("spec")
        assert graph is not None
        nodes = set(graph.get_graph().nodes.keys())
        assert "specify" in nodes
        assert "plan" in nodes
        assert "critic_plan" in nodes
        assert "implement" in nodes
        assert "verify" in nodes

    def test_critical_spec_workflow_graph_compiles(self):
        from spine.workflow.compose import build_workflow_graph
        graph = build_workflow_graph("critical_spec")
        assert graph is not None
        nodes = set(graph.get_graph().nodes.keys())
        assert "specify" in nodes
        assert "critic_specify" in nodes
        assert "plan" in nodes
        assert "critic_plan" in nodes
        assert "implement" in nodes
        assert "verify" in nodes

    def test_subgraph_nodes_have_readable_names(self):
        from spine.workflow.compose import build_workflow_graph
        graph = build_workflow_graph("quick")
        mermaid = graph.get_graph().draw_mermaid()
        # All nodes should appear in the mermaid output
        assert "plan" in mermaid
        assert "implement" in mermaid
        assert "verify" in mermaid


class TestSubgraphStateMappers:
    """State mapping between parent and subgraph states."""

    def test_base_state_mapper(self):
        from spine.workflow.compose import _base_state_mapper
        parent = {
            "work_id": "abc",
            "work_type": "quick",
            "description": "test",
            "workspace_root": "/tmp",
        }
        result = _base_state_mapper(parent, None)
        assert result["work_id"] == "abc"
        assert result["messages"] == []
        assert result["artifacts_output"] == {}

    def test_specify_state_mapper(self):
        from spine.workflow.compose import _specify_state_mapper
        parent = {
            "work_id": "abc",
            "work_type": "quick",
            "description": "test",
            "workspace_root": "/tmp",
            "retry_count": {"specify": 1},
        }
        result = _specify_state_mapper(parent, None)
        assert result["phase"] == "specify"
        assert result["retry_count"] == 1

    def test_plan_state_mapper(self):
        from spine.workflow.compose import _plan_state_mapper
        parent = {"work_id": "def", "work_type": "spec", "description": "d"}
        result = _plan_state_mapper(parent, None)
        assert result["phase"] == "plan"
        assert result["spec_path"] == ".spine/artifacts/def/specify"
        assert result["has_spec"] is True

    def test_plan_state_mapper_quick(self):
        from spine.workflow.compose import _plan_state_mapper
        parent = {"work_id": "def", "work_type": "quick", "description": "d"}
        result = _plan_state_mapper(parent, None)
        assert result["phase"] == "plan"
        assert result["spec_path"] == ""
        assert result["has_spec"] is False

    def test_tasks_state_mapper_quick(self):
        from spine.workflow.compose import _tasks_state_mapper
        parent = {"work_id": "abc", "work_type": "quick", "description": "d"}
        result = _tasks_state_mapper(parent, None)
        assert result["phase"] == "tasks"
        assert result["plan_path"] is None
        assert result["spec_path"] is None

    def test_tasks_state_mapper_spec(self):
        from spine.workflow.compose import _tasks_state_mapper
        parent = {"work_id": "abc", "work_type": "spec", "description": "d"}
        result = _tasks_state_mapper(parent, None)
        assert result["plan_path"] == ".spine/artifacts/abc/plan"
        assert result["spec_path"] == ".spine/artifacts/abc/specify"

    def test_implement_state_mapper(self):
        from spine.workflow.compose import _implement_state_mapper
        parent = {"work_id": "abc", "work_type": "quick", "description": "d"}
        result = _implement_state_mapper(parent, None)
        assert result["phase"] == "implement"
        assert result["plan_path"] == ".spine/artifacts/abc/plan"

    def test_critic_state_mapper(self):
        from spine.workflow.compose import _critic_state_mapper
        mapper = _critic_state_mapper("plan")
        parent = {"work_id": "abc", "work_type": "quick", "description": "d"}
        result = mapper(parent, None)
        assert result["phase"] == "critic"
        assert result["reviewed_phase"] == "plan"
        assert result["reviewed_phase_path"] == ".spine/artifacts/abc/plan"


class TestSubgraphResultMappers:
    """Result mapping from subgraph output back to parent state."""

    def test_specify_result_mapper_success(self):
        from spine.workflow.compose import _specify_result_mapper
        subgraph_result = {
            "artifacts_output": {"specification.md": "spec content"},
            "phase_status": "success",
        }
        result = _specify_result_mapper(subgraph_result, {"work_id": "test"})
        assert result["status"] == "running"
        assert result["phase_results"]["specify"]["artifact_count"] == 1

    def test_plan_result_mapper_needs_review(self):
        from spine.workflow.compose import _plan_result_mapper
        subgraph_result = {
            "artifacts_output": {},
            "phase_status": "needs_review",
        }
        result = _plan_result_mapper(subgraph_result, {"work_id": "test"})
        assert result["status"] == "needs_review"
        assert result["needs_review_phase"] == "plan"

    def test_tasks_result_mapper_error(self):
        from spine.workflow.compose import _tasks_result_mapper
        subgraph_result = {
            "artifacts_output": {},
            "phase_status": "error",
        }
        result = _tasks_result_mapper(subgraph_result, {"work_id": "test"})
        assert result["status"] == "failed"

    def test_implement_result_mapper_success(self):
        from spine.workflow.compose import _implement_result_mapper
        subgraph_result = {
            "artifacts_output": {"implementation.md": "code"},
            "phase_status": "success",
        }
        result = _implement_result_mapper(subgraph_result, {"work_id": "test"})
        assert result["status"] == "running"

    def test_critic_result_mapper_passed(self):
        from spine.workflow.compose import _critic_result_mapper
        mapper = _critic_result_mapper("plan")
        subgraph_result = {
            "structural_result": {"status": "passed", "tier": "structural", "reason": "ok", "suggestions": []},
            "agent_result": {"status": "passed", "tier": "agent", "reason": "ok", "suggestions": []},
            "phase_status": "passed",
        }
        result = mapper(subgraph_result, {"work_id": "test"})
        assert result["status"] == "running"
        assert result["current_phase"] == "critic"
        assert len(result["feedback"]) == 1

    def test_critic_result_mapper_needs_review(self):
        from spine.workflow.compose import _critic_result_mapper
        mapper = _critic_result_mapper("plan")
        subgraph_result = {
            "structural_result": {"status": "needs_review", "tier": "structural", "reason": "bad", "suggestions": []},
            "phase_status": "needs_review",
        }
        result = mapper(subgraph_result, {"work_id": "test"})
        assert result["status"] == "needs_review"
        assert result["needs_review_phase"] == "plan"


class TestGraphBackwardsCompatibility:
    """Verify that the parent graph can still use legacy call_fns."""

    def test_legacy_mode_compiles(self):
        from spine.workflow.compose import build_workflow_graph, _SUBGRAPH_ENABLED
        # Temporarily disable all subgraphs
        original = dict(_SUBGRAPH_ENABLED)
        try:
            for key in _SUBGRAPH_ENABLED:
                _SUBGRAPH_ENABLED[key] = False
            graph = build_workflow_graph("quick")
            assert graph is not None
            nodes = set(graph.get_graph().nodes.keys())
            assert "plan" in nodes
            assert "implement" in nodes
            assert "verify" in nodes
        finally:
            for key, val in original.items():
                _SUBGRAPH_ENABLED[key] = val
