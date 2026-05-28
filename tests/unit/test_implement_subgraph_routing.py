"""Unit tests for ``_route_slices`` in the IMPLEMENT subgraph."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from langgraph.types import Command, Send

from spine.workflow.subgraphs.implement_subgraph import (
    _plan_slice_implementer_node,
    _route_slices,
)


def _slice(slice_id: str) -> dict:
    return {"id": slice_id, "title": f"slice {slice_id}"}


def _base_state() -> dict:
    return {
        "phase": "implement",
        "work_id": "test-work",
        "work_type": "feature",
        "workspace_root": "/tmp/test",
        "plan_path": ".spine/artifacts/test-work/plan",
    }


class TestRouteSlices:
    def test_only_pending_returns_slice_implementer_sends(self):
        state = {**_base_state(), "pending_slices": [_slice("a"), _slice("b")], "failed_slices": []}
        result = _route_slices(state)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(s, Send) for s in result)
        # Each pending slice is dispatched to the plan-before-do branch
        # head; plan_slice_implementer chains into slice_implementer.
        assert all(s.node == "plan_slice_implementer" for s in result)
        ids = [s.arg["active_slice"]["id"] for s in result]
        assert ids == ["a", "b"]

    def test_only_failed_returns_fallback_decomposer_sends(self):
        state = {**_base_state(), "pending_slices": [], "failed_slices": [_slice("x")]}
        result = _route_slices(state)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].node == "fallback_decomposer"
        assert result[0].arg["active_slice"]["id"] == "x"

    def test_both_lists_empty_routes_to_synthesis(self):
        state = {**_base_state(), "pending_slices": [], "failed_slices": []}
        result = _route_slices(state)
        assert result == "synthesize_implementation"

    def test_mixed_pending_and_failed_dispatches_both(self):
        state = {
            **_base_state(),
            "pending_slices": [_slice("p1")],
            "failed_slices": [_slice("f1"), _slice("f2")],
        }
        result = _route_slices(state)
        assert isinstance(result, list)
        assert len(result) == 3
        by_node: dict[str, list[str]] = {}
        for s in result:
            by_node.setdefault(s.node, []).append(s.arg["active_slice"]["id"])
        assert by_node["plan_slice_implementer"] == ["p1"]
        assert sorted(by_node["fallback_decomposer"]) == ["f1", "f2"]

    def test_send_payload_carries_base_context(self):
        state = {**_base_state(), "pending_slices": [_slice("a")], "failed_slices": []}
        result = _route_slices(state)
        payload = result[0].arg
        assert payload["work_id"] == "test-work"
        assert payload["workspace_root"] == "/tmp/test"
        assert payload["plan_path"] == ".spine/artifacts/test-work/plan"
        assert "pending_slices" not in payload
        assert "failed_slices" not in payload


class TestPlanSliceImplementerCommand:
    """Regression for the InvalidUpdateError crash.

    Parallel ``Send("plan_slice_implementer", ...)`` branches must hand
    off to slice_implementer via per-branch ``Command(goto=Send(...))``,
    not by writing the directive to a shared LastValue channel.
    """

    @pytest.mark.asyncio
    async def test_returns_command_with_send_to_slice_implementer(self, monkeypatch):
        from spine.agents import plan_do
        from spine.agents.plan_do import SubagentDirective

        async def _fake_plan(*, state, config, phase_path, task_description, role_hint=""):
            return SubagentDirective(
                approach="edit auth.py",
                target_files=["spine/auth.py"],
            )

        monkeypatch.setattr(
            "spine.workflow.subgraphs.implement_subgraph.run_plan_node", _fake_plan
        )

        state = {
            **_base_state(),
            "active_slice": _slice("s1"),
        }
        out = await _plan_slice_implementer_node(state, None)
        assert isinstance(out, Command)
        assert isinstance(out.goto, Send)
        assert out.goto.node == "slice_implementer"
        # The directive rides on the Send payload, NOT on update — putting
        # it on update would crash apply_writes with N parallel branches.
        assert "active_slice_directive" in out.goto.arg
        assert out.goto.arg["active_slice_directive"]["target_files"] == ["spine/auth.py"]
        assert out.goto.arg["active_slice"]["id"] == "s1"
        # Base payload carried through.
        assert out.goto.arg["work_id"] == "test-work"
        # Nothing problematic on update.
        assert "active_slice_directive" not in (out.update or {})
