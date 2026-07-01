"""Unit tests for ``_route_slices`` in the IMPLEMENT subgraph."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from spine.workflow.subgraphs.implement_subgraph import (
    _dispatch_gate_node,
    _fallback_decomposer_node,
    _placement_feedback,
    _plan_slice_implementer_node,
    _route_slices,
    _slice_result_to_update,
    build_implement_subgraph,
)
from spine.workflow.subgraph_state import ImplementSubgraphState


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
        # Each pending slice is dispatched to the editor head — either the
        # tool path (plan_slice_implementer) or the synthesis editor, depending
        # on the implement_synthesis_placement flag in the ambient config.
        assert all(
            s.node in ("plan_slice_implementer", "synthesis_implementer")
            for s in result
        )
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
        # Pending → editor head (tool or synthesis, per the flag); failed →
        # fallback_decomposer.
        editor_node = next(
            n for n in by_node if n in ("plan_slice_implementer", "synthesis_implementer")
        )
        assert by_node[editor_node] == ["p1"]
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


class TestDispatchGateBarrier:
    """Regression for the ``artifacts_output`` InvalidUpdateError (trace 019f0193).

    A diamond DAG (root → {api, ui}) makes two slices become dependency-ready in
    the same super-step. The router must be evaluated exactly ONCE per super-step
    — on the fully-merged state — or sibling branches route divergently
    (duplicate dispatch; ``synthesize_implementation`` + ``save_artifacts``
    co-scheduled, both writing the un-reduced ``artifacts_output`` channel).
    The single ``dispatch_gate`` barrier with STATIC fan-in guarantees this.
    """

    def test_router_wired_only_on_dispatch_gate(self):
        """Structural guard: ``_route_slices`` must hang off ``dispatch_gate``
        alone, and every fan-out node must reach the gate by a plain edge."""
        graph = build_implement_subgraph().compile().get_graph()
        edges = graph.edges

        # Static edges from each fan-out node into the gate.
        static_into_gate = {
            e.source for e in edges if e.target == "dispatch_gate" and not e.conditional
        }
        assert {
            "split_slices",
            "slice_implementer",
            "synthesis_implementer",
            "fallback_decomposer",
        } <= static_into_gate

        # The only node with conditional (router) out-edges is the gate. None of
        # the fan-out nodes may carry the router any more.
        conditional_sources = {e.source for e in edges if e.conditional}
        assert conditional_sources == {"dispatch_gate"}

    @pytest.mark.asyncio
    async def test_diamond_dag_routes_once_no_collision(self):
        """Drive the real router + gate over the trace's diamond DAG with a stub
        implementer. Asserts the loop terminates with no InvalidUpdateError and
        ``synthesize`` runs exactly once."""
        synth_runs: list[int] = []

        async def _noop(state):  # split_slices stand-in
            return {}

        async def _stub_implementer(state):
            active = state.get("active_slice") or {}
            result = {
                "slice_name": active.get("id"),
                "status": "implemented",
                "files_modified": [f"{active.get('id')}.py"],
                "files_created": [],
                "test_results": "",
                "issues": [],
            }
            return {**_slice_result_to_update(active, result), "slice_dispatch_count": 1}

        async def _synth(state):
            synth_runs.append(1)
            return {"artifacts_output": {"implementation.md": "ok"}, "phase_status": "success"}

        async def _save(state):
            # Mirrors the real node: a second writer of the plain artifacts_output
            # channel. Co-scheduling it with _synth is exactly what crashed.
            return {"artifacts_output": dict(state.get("artifacts_output") or {})}

        b = StateGraph(ImplementSubgraphState)
        b.add_node("split_slices", _noop)
        b.add_node("dispatch_gate", _dispatch_gate_node)
        # Map every editor target the router may emit (flag-dependent) to the stub.
        b.add_node("plan_slice_implementer", _stub_implementer)
        b.add_node("synthesis_implementer", _stub_implementer)
        b.add_node("fallback_decomposer", _stub_implementer)
        b.add_node("synthesize_implementation", _synth)
        b.add_node("save_artifacts", _save)
        b.add_edge(START, "split_slices")
        b.add_edge("split_slices", "dispatch_gate")
        b.add_edge("plan_slice_implementer", "dispatch_gate")
        b.add_edge("synthesis_implementer", "dispatch_gate")
        b.add_edge("fallback_decomposer", "dispatch_gate")
        b.add_conditional_edges("dispatch_gate", _route_slices, {
            "plan_slice_implementer": "plan_slice_implementer",
            "synthesis_implementer": "synthesis_implementer",
            "fallback_decomposer": "fallback_decomposer",
            "synthesize_implementation": "synthesize_implementation",
        })
        b.add_edge("synthesize_implementation", "save_artifacts")
        b.add_edge("save_artifacts", END)
        graph = b.compile()

        init = {
            **_base_state(),
            "pending_slices": [
                {"id": "root", "title": "root", "dependencies": []},
                {"id": "api", "title": "api", "dependencies": ["root"]},
                {"id": "ui", "title": "ui", "dependencies": ["root"]},
            ],
            "completed_slices": [],
            "failed_slices": [],
        }
        # No InvalidUpdateError, and synthesis runs exactly once.
        result = await graph.ainvoke(init, {"recursion_limit": 60})
        assert synth_runs == [1]
        completed_ids = sorted(c["id"] for c in result.get("completed_slices", []))
        assert completed_ids == ["api", "root", "ui"]
        assert result.get("artifacts_output") == {"implementation.md": "ok"}


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


class TestFallbackDecomposerNode:
    """Termination + honesty + duplicate-read mitigation for the fallback loop."""

    @pytest.mark.asyncio
    async def test_depth_cap_removes_slice_and_reports_blocked(self, monkeypatch):
        """At the depth cap the slice MUST leave failed_slices (so the router
        can reach synthesis instead of re-dispatching forever) and be recorded
        as blocked (so the failure stays visible)."""
        monkeypatch.setattr(
            "spine.workflow.subgraphs.implement_subgraph._max_decompose_depth",
            lambda: 1,
        )
        state = {
            **_base_state(),
            "active_slice": {**_slice("stuck"), "_decompose_depth": 1},
        }
        out = await _fallback_decomposer_node(state, None)
        assert out["failed_slices"] == {"remove": ["stuck"]}
        blocked = out["completed_slices"]["add"]
        assert len(blocked) == 1
        assert blocked[0]["slice_name"] == "stuck"
        assert blocked[0]["status"] == "blocked"
        assert "exceeded fallback depth" in blocked[0]["issues"]

    @pytest.mark.asyncio
    async def test_decomposer_failure_reports_blocked_not_silent_drop(self, monkeypatch):
        """A raising decomposer (e.g. LengthFinishReasonError) must surface the
        slice as blocked rather than vanishing silently."""
        monkeypatch.setattr(
            "spine.workflow.subgraphs.implement_subgraph._max_decompose_depth",
            lambda: 2,
        )

        async def _boom(**kwargs):
            raise RuntimeError("LengthFinishReasonError")

        monkeypatch.setattr(
            "spine.agents.decomposer.run_decomposer", _boom
        )
        state = {**_base_state(), "active_slice": _slice("f1")}
        out = await _fallback_decomposer_node(state, None)
        assert out["failed_slices"] == {"remove": ["f1"]}
        blocked = out["completed_slices"]["add"][0]
        assert blocked["status"] == "blocked"
        assert "LengthFinishReasonError" in blocked["test_results"]

    @pytest.mark.asyncio
    async def test_same_file_micros_chained_sequentially(self, monkeypatch):
        """Micro-slices that all touch one file are chained on a sibling queue
        (one head in pending) instead of fanned out in parallel — avoiding N
        fresh-context re-reads of the same file."""
        monkeypatch.setattr(
            "spine.workflow.subgraphs.implement_subgraph._max_decompose_depth",
            lambda: 2,
        )

        async def _two_micros(**kwargs):
            return [
                {"id": "p-micro-1", "target_files": ["api.py"], "acceptance_criteria": ["a"]},
                {"id": "p-micro-2", "target_files": ["api.py"], "acceptance_criteria": ["b"]},
            ]

        monkeypatch.setattr("spine.agents.decomposer.run_decomposer", _two_micros)
        state = {
            **_base_state(),
            "active_slice": {**_slice("p"), "target_files": ["api.py"], "_decompose_depth": 0},
        }
        out = await _fallback_decomposer_node(state, None)
        adds = out["pending_slices"]["add"]
        assert len(adds) == 1  # one chain head, not two parallel slices
        head = adds[0]
        assert head["_decompose_depth"] == 1
        queued = head["_sibling_queue"]
        assert [q["id"] for q in queued] == ["p-micro-2"]
        # Depth is stamped on queued siblings too so the cap still advances.
        assert all(q["_decompose_depth"] == 1 for q in queued)

    @pytest.mark.asyncio
    async def test_multi_file_micros_stay_parallel(self, monkeypatch):
        """Micro-slices spanning different files keep fanning out in parallel —
        the sequential chain is only for the same-file duplicate-read case."""
        monkeypatch.setattr(
            "spine.workflow.subgraphs.implement_subgraph._max_decompose_depth",
            lambda: 2,
        )

        async def _two_files(**kwargs):
            return [
                {"id": "p-micro-1", "target_files": ["a.py"], "acceptance_criteria": ["a"]},
                {"id": "p-micro-2", "target_files": ["b.py"], "acceptance_criteria": ["b"]},
            ]

        monkeypatch.setattr("spine.agents.decomposer.run_decomposer", _two_files)
        state = {
            **_base_state(),
            "active_slice": {**_slice("p"), "target_files": ["a.py", "b.py"], "_decompose_depth": 0},
        }
        out = await _fallback_decomposer_node(state, None)
        adds = out["pending_slices"]["add"]
        assert len(adds) == 2
        assert all("_sibling_queue" not in s for s in adds)


class TestPlacementFeedback:
    """019f1c10: the creation-anchor guard's target/next_action must reach
    the synthesis retry, or a broken self-anchor repeats across cycles."""

    def _placement(self, failures):
        class _FakePlacement:
            pass

        p = _FakePlacement()
        p.failures = failures
        return p

    def test_includes_target_and_next_action_when_present(self):
        placement = self._placement([{
            "file": "svc.py",
            "symbol": "UIApi.add_embedding_provider",
            "status": "no_match",
            "detail": "symbol does not exist yet — you are CREATING it.",
            "target": "UIApi.add_llm_provider",
            "next_action": "Call ast_edit with action='insert_after' anchored to 'UIApi.add_llm_provider'.",
        }])
        text = _placement_feedback(placement)
        assert "UIApi.add_llm_provider" in text
        assert "insert_after" in text
        assert "[DO THIS:" in text

    def test_omits_target_next_action_fields_when_absent(self):
        placement = self._placement([{
            "file": "svc.py", "symbol": "x", "status": "syntax_error", "detail": "bad code",
        }])
        text = _placement_feedback(placement)
        assert "[DO THIS:" not in text
        assert "[target:" not in text
