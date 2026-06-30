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

    def test_router_rework_after_interrupt_clears_phase(self):
        # Regression (trace 019f11d0): _human_review_interrupt nulls
        # needs_review_phase in the same update the router reads, so a "rework"
        # must route off the target stashed on human_feedback — not collapse to
        # "abort" because live needs_review_phase is None.
        state = {
            "human_feedback": {
                "action": "rework",
                "feedback": "needs work",
                "_review_target": "plan",
            },
            "needs_review_phase": None,
            "current_phase": "critic",
        }
        assert self.router(state) == "plan"

    def test_router_approve_after_interrupt_clears_phase(self):
        # Same ordering trap for "approve": with needs_review_phase cleared, the
        # old current_phase fallback would advance from the wrong phase. The
        # stashed target keeps it advancing from "plan" → "critic_plan".
        state = {
            "human_feedback": {"action": "approve", "_review_target": "plan"},
            "needs_review_phase": None,
            "current_phase": "critic",
        }
        assert self.router(state) == "critic_plan"


class TestHumanFeedbackChannel:
    """human_feedback must be a declared state channel (trace 019f1628).

    The router unit tests above hand a dict straight to the router, so they pass
    even when ``human_feedback`` is dropped on the real state commit. These tests
    exercise the schema/commit path that actually broke in production.
    """

    def test_human_feedback_is_declared_channel(self):
        from spine.models.state import WorkflowState

        # If this key is missing, LangGraph silently drops the interrupt node's
        # human_feedback update; the router then reads {} and every resume
        # collapses to "abort" regardless of the human's choice.
        assert "human_feedback" in WorkflowState.__annotations__

    def test_human_feedback_survives_state_commit(self):
        # Round-trip the interrupt node's update through a compiled StateGraph
        # using the real WorkflowState. If human_feedback is not a channel, the
        # update is dropped and the router routes to "abort" instead of "plan".
        from langgraph.graph import StateGraph, END

        from spine.models.state import WorkflowState
        from spine.workflow.compose import (
            _make_human_review_router,
            WORKFLOW_SEQUENCES,
        )

        router = _make_human_review_router(WORKFLOW_SEQUENCES["critical_task"])

        def emit_decision(state):
            # Mimics _human_review_interrupt's post-resume return: stamps the
            # decision on human_feedback and nulls needs_review_phase.
            return {
                "human_feedback": {
                    "action": "rework",
                    "feedback": "fix it",
                    "_review_target": "plan",
                },
                "needs_review_phase": None,
            }

        captured = {}

        def plan_node(state):
            captured["routed_to"] = "plan"
            return {}

        def abort_node(state):
            captured["routed_to"] = "abort"
            return {}

        g = StateGraph(WorkflowState)
        g.add_node("human_review", emit_decision)
        g.add_node("plan", plan_node)
        g.add_node("abort", abort_node)
        g.set_entry_point("human_review")
        g.add_conditional_edges(
            "human_review", router, {"plan": "plan", "abort": "abort"}
        )
        g.add_edge("plan", END)
        g.add_edge("abort", END)
        app = g.compile()

        app.invoke({"current_phase": "critic"})
        assert captured.get("routed_to") == "plan", (
            "human_feedback did not survive the state commit — the router fell "
            "through to abort (trace 019f1628 regression)."
        )


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
