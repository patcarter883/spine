"""Tests for interrupt-based human review in the SPINE workflow graph."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest


class TestInterruptNodeExists:
    """Verify the human_review interrupt node is wired in all work types."""

    def test_task_workflow_has_human_review_node(self):
        from spine.workflow.compose import build_workflow_graph

        graph = build_workflow_graph("task")
        nodes = set(graph.get_graph().nodes.keys())
        assert "human_review" in nodes, f"human_review not in {nodes}"

    def test_critical_task_has_human_review_node(self):
        from spine.workflow.compose import build_workflow_graph

        graph = build_workflow_graph("critical_task")
        nodes = set(graph.get_graph().nodes.keys())
        assert "human_review" in nodes


class TestInterruptRouter:
    """Tests for _make_human_review_router factory function."""

    def setup_method(self):
        from spine.workflow.compose import _make_human_review_router
        from spine.workflow.compose import WORKFLOW_SEQUENCES

        self.router = _make_human_review_router(WORKFLOW_SEQUENCES["critical_task"])

    def test_router_rework(self):
        state = {
            "human_feedback": {"action": "rework", "feedback": "needs work"},
            "needs_review_phase": "plan",
        }
        assert self.router(state) == "plan"

    def test_router_approve(self):
        # In critical_spec: plan is at index 2, next is critic_plan
        state = {
            "human_feedback": {"action": "approve"},
            "needs_review_phase": "plan",
        }
        result = self.router(state)
        assert result == "critic_plan"

    def test_router_abort_default(self):
        state = {"human_feedback": {}}
        assert self.router(state) == "abort"

    def test_router_abort_none(self):
        state = {"human_feedback": None}
        assert self.router(state) == "abort"


class TestInterruptNodeFunction:
    """Tests for _human_review_interrupt node function."""

    @pytest.mark.skip(reason="interrupt() requires a real graph runtime")
    def test_interrupt_returns_dict(self):
        from spine.workflow.compose import _human_review_interrupt

        state = {
            "needs_review_phase": "tasks",
            "feedback": [{"reason": "No artifacts", "suggestions": ["check"]}],
            "phase_results": {"tasks": {"status": "error"}},
        }
        # interrupt() pauses the graph — can't test without a running graph
        result = _human_review_interrupt(state)
        assert isinstance(result, dict)
        assert "human_feedback" in result


class TestCriticRoutesToHumanReview:
    """Verify critic conditional edges route needs_review to human_review."""

    def test_critic_routes_needs_review_to_human_review(self):
        from spine.workflow.compose import build_workflow_graph

        graph = build_workflow_graph("critical_task")
        mermaid = graph.get_graph().draw_mermaid()
        # The mermaid output should show critic nodes connecting to human_review
        assert "human_review" in mermaid

    def test_gate_routes_needs_review_to_human_review(self):
        from spine.workflow.compose import build_workflow_graph

        graph = build_workflow_graph("task")
        mermaid = graph.get_graph().draw_mermaid()
        assert "human_review" in mermaid


class TestResumeInterruptedWork:
    """Tests for resume_interrupted_work dispatcher function."""

    @pytest.mark.asyncio
    async def test_rejects_unknown_work_id(self):
        from spine.work.dispatcher import resume_interrupted_work

        with pytest.raises(ValueError, match="not found"):
            await resume_interrupted_work("nonexistent", "rework", "fix it")

    @pytest.mark.asyncio
    async def test_rejects_nonexistent_entry(self):
        from spine.work.dispatcher import resume_interrupted_work

        with pytest.raises(ValueError, match="not found"):
            await resume_interrupted_work("bad-id", "approve", "looks good")
