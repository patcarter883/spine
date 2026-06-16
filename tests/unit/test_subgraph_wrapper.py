"""Tests for subgraph state schemas and wrapper factory."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from unittest.mock import AsyncMock
import asyncio


# ── Subgraph state schema tests ──


class TestSubgraphStateSchemas:
    """Tests for per-phase subgraph state schemas."""

    def test_base_subgraph_state_fields(self):
        from spine.workflow.subgraph_state import BaseSubgraphState

        state: BaseSubgraphState = {
            "phase": "verify",
            "work_id": "abc123",
            "work_type": "task",
            "description": "test",
            "workspace_root": "/tmp",
            "retry_count": 0,
            "feedback": [],
            "messages": [],
            "artifacts_output": {"verification.md": "pass"},
            "phase_status": "success",
        }
        assert state["phase"] == "verify"
        assert state["artifacts_output"]["verification.md"] == "pass"

    def test_verify_subgraph_state(self):
        from spine.workflow.subgraph_state import VerifySubgraphState

        state: VerifySubgraphState = {
            "phase": "verify",
            "work_id": "abc123",
            "tasks_path": ".spine/artifacts/abc123/tasks",
            "spec_path": ".spine/artifacts/abc123/specify",
            "plan_path": None,
            "phase_status": "success",
        }
        assert state["spec_path"] == ".spine/artifacts/abc123/specify"
        assert state["plan_path"] is None

    def test_verify_subgraph_state_optional_paths(self):
        from spine.workflow.subgraph_state import VerifySubgraphState

        state: VerifySubgraphState = {
            "phase": "verify",
            "work_id": "abc123",
            "tasks_path": ".spine/artifacts/abc123/tasks",
            # spec_path and plan_path omitted — total=False allows this
        }
        assert "spec_path" not in state

    def test_tasks_subgraph_state(self):
        from spine.workflow.subgraph_state import TasksSubgraphState

        state: TasksSubgraphState = {
            "phase": "tasks",
            "work_id": "abc123",
            "plan_path": ".spine/artifacts/abc123/plan/plan.md",
            "spec_path": ".spine/artifacts/abc123/specify/specification.md",
        }
        assert state["plan_path"].endswith("plan.md")

    def test_critic_subgraph_state(self):
        from spine.workflow.subgraph_state import CriticSubgraphState

        state: CriticSubgraphState = {
            "phase": "critic",
            "work_id": "abc123",
            "reviewed_phase": "plan",
            "reviewed_phase_path": ".spine/artifacts/abc123/plan",
        }
        assert state["reviewed_phase"] == "plan"


# ── Wrapper factory tests ──


class TestMakeSubgraphNode:
    """Tests for make_subgraph_node wrapper factory."""

    @pytest.mark.asyncio
    async def test_successful_subgraph_invocation(self):
        from spine.workflow.subgraph_wrapper import make_subgraph_node

        mock_subgraph = AsyncMock()
        mock_subgraph.ainvoke.return_value = {
            "artifacts_output": {"verification.md": "VERIFIED"},
            "phase_status": "success",
        }

        def state_mapper(parent, config):
            return {"phase": "verify", "work_id": parent.get("work_id", "")}

        def result_mapper(subgraph_result, parent):
            return {
                "current_phase": "verify",
                "status": "running",
                "artifacts": {"verify": {"verification.md": "VERIFIED"[:500]}},
            }

        node_fn = make_subgraph_node(mock_subgraph, "verify", state_mapper, result_mapper)

        parent_state = {"work_id": "test123", "status": "running"}
        result = await node_fn(parent_state, None)

        assert result["current_phase"] == "verify"
        assert result["status"] == "running"
        mock_subgraph.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancelled_error_returns_needs_review(self):
        from spine.workflow.subgraph_wrapper import make_subgraph_node

        mock_subgraph = AsyncMock()
        mock_subgraph.ainvoke.side_effect = asyncio.CancelledError()

        node_fn = make_subgraph_node(mock_subgraph, "verify", lambda p, c: {}, lambda r, p: {})

        parent_state = {"work_id": "test123", "status": "running"}
        result = await node_fn(parent_state, None)

        assert result["status"] == "needs_review"
        assert result["needs_review_phase"] == "verify"
        assert any("Cancelled" in f.get("reason", "") for f in result["feedback"])

    @pytest.mark.asyncio
    async def test_timeout_returns_needs_review(self):
        from spine.workflow.subgraph_wrapper import make_subgraph_node

        mock_subgraph = AsyncMock()
        mock_subgraph.ainvoke.side_effect = asyncio.TimeoutError()

        node_fn = make_subgraph_node(mock_subgraph, "verify", lambda p, c: {}, lambda r, p: {})

        parent_state = {"work_id": "test123", "status": "running"}
        result = await node_fn(parent_state, None)

        assert result["status"] == "needs_review"
        assert any("Timed out" in f.get("reason", "") for f in result["feedback"])

    @pytest.mark.asyncio
    async def test_generic_exception_returns_error(self):
        from spine.workflow.subgraph_wrapper import make_subgraph_node

        mock_subgraph = AsyncMock()
        mock_subgraph.ainvoke.side_effect = RuntimeError("boom")

        node_fn = make_subgraph_node(mock_subgraph, "verify", lambda p, c: {}, lambda r, p: {})

        parent_state = {"work_id": "test123", "status": "running"}
        result = await node_fn(parent_state, None)

        assert result["status"] == "needs_review"
        assert any("boom" in f.get("reason", "") for f in result["feedback"])

    def test_node_function_name(self):
        from spine.workflow.subgraph_wrapper import make_subgraph_node

        node_fn = make_subgraph_node(None, "verify", lambda p, c: {}, lambda r, p: {})
        assert node_fn.__name__ == "verify_subgraph"


class TestMakeSuccessResultMapper:
    """Tests for make_success_result_mapper factory."""

    def test_maps_artifacts_to_parent_state(self):
        from spine.workflow.subgraph_wrapper import make_success_result_mapper

        mapper = make_success_result_mapper("verify")
        subgraph_result = {
            "artifacts_output": {
                "verification.md": "VERIFIED all slices",
                "test-results.md": "18/18 passed",
            },
            "phase_status": "success",
        }
        parent_state = {"work_id": "test123"}
        result = mapper(subgraph_result, parent_state)

        assert result["current_phase"] == "verify"
        assert result["status"] == "running"
        assert result["phase_results"]["verify"]["status"] == "success"
        assert result["phase_results"]["verify"]["artifact_count"] == 2
        assert "verification.md" in result["artifacts"]["verify"]

    def test_truncates_long_artifacts(self):
        from spine.workflow.subgraph_wrapper import make_success_result_mapper

        mapper = make_success_result_mapper("implement")
        long_content = "x" * 10000
        subgraph_result = {
            "artifacts_output": {"implementation.md": long_content},
            "phase_status": "success",
        }
        result = mapper(subgraph_result, {})

        preview = result["artifacts"]["implement"]["implementation.md"]
        assert len(preview) == 500

    def test_handles_empty_artifacts(self):
        from spine.workflow.subgraph_wrapper import make_success_result_mapper

        mapper = make_success_result_mapper("tasks")
        subgraph_result = {
            "artifacts_output": {},
            "phase_status": "success",
        }
        result = mapper(subgraph_result, {})

        assert result["phase_results"]["tasks"]["artifact_count"] == 0

    def test_error_phase_status_not_labeled_success(self):
        """A swallowed phase error must not be recorded as a successful phase.

        Regression for trace 019ec90d: synthesis failed (phase_status=error,
        0 artifacts) but phase_results said status=success / error=None while
        the run as a whole failed.
        """
        from spine.workflow.subgraph_wrapper import make_success_result_mapper

        mapper = make_success_result_mapper("specify")
        subgraph_result = {
            "artifacts_output": {},
            "phase_status": "error",
            "agent_response": "Synthesis error: No endpoints found",
        }
        result = mapper(subgraph_result, {})

        entry = result["phase_results"]["specify"]
        assert entry["status"] == "error"
        assert entry["artifact_count"] == 0
        assert entry["error"] == "Synthesis error: No endpoints found"

    def test_needs_review_phase_status_propagated(self):
        from spine.workflow.subgraph_wrapper import make_success_result_mapper

        mapper = make_success_result_mapper("specify")
        subgraph_result = {
            "artifacts_output": {},
            "phase_status": "needs_review",
        }
        result = mapper(subgraph_result, {})

        entry = result["phase_results"]["specify"]
        assert entry["status"] == "needs_review"
        # Falls back to a descriptive error when no agent_response is present.
        assert entry["error"] and "needs_review" in entry["error"]


class TestErrorUpdate:
    """Tests for _error_update and _needs_review_update helpers."""

    def test_error_update_structure(self):
        from spine.workflow.subgraph_wrapper import _error_update

        result = _error_update({"work_id": "test123"}, "implement", "something broke")
        assert result["status"] == "needs_review"
        assert result["current_phase"] == "implement"
        assert result["phase_results"]["implement"]["status"] == "error"
        assert result["phase_results"]["implement"]["error"] == "something broke"

    def test_needs_review_update_structure(self):
        from spine.workflow.subgraph_wrapper import _needs_review_update

        result = _needs_review_update(
            {"work_id": "test123"},
            "tasks",
            "gate failed",
            suggestions=["check logs"],
        )
        assert result["status"] == "needs_review"
        assert result["needs_review_phase"] == "tasks"
        assert result["phase_results"]["tasks"]["status"] == "needs_review"
        assert "check logs" in result["feedback"][0]["suggestions"]


class TestStructuralRetryCarryover:
    """Contract-failure retries must reuse what the failed attempt salvaged
    (trace 019eb940: the plan retry re-ran ~15 min of exploration whose
    findings were intact)."""

    @pytest.mark.asyncio
    async def test_carryover_seeds_retry_input(self):
        from spine.exceptions import CriticalContractFailure
        from spine.workflow.subgraph_wrapper import make_subgraph_node

        findings = [{"topic": "t", "summary": "s"}]
        inputs: list[dict] = []

        class _Subgraph:
            calls = 0

            async def ainvoke(self, subgraph_input, config):
                inputs.append(dict(subgraph_input))
                _Subgraph.calls += 1
                if _Subgraph.calls == 1:
                    raise CriticalContractFailure(
                        phase="plan",
                        reason="plan.json missing",
                        carryover={
                            "findings": findings,
                            "findings_carried_over": True,
                            "synthesis_cap_escalated": True,
                        },
                    )
                return {"phase_status": "success", "artifacts_output": {}}

        node_fn = make_subgraph_node(
            _Subgraph(), "plan", lambda p, c: {"phase": "plan"}, lambda r, p: {"ok": True}
        )

        result = await node_fn({"work_id": "wk1", "status": "running"}, None)

        assert result == {"ok": True}
        assert len(inputs) == 2
        assert "findings" not in inputs[0]
        assert inputs[1]["findings"] == findings
        assert inputs[1]["findings_carried_over"] is True
        assert inputs[1]["synthesis_cap_escalated"] is True

    @pytest.mark.asyncio
    async def test_no_carryover_keeps_original_input(self):
        from spine.exceptions import CriticalContractFailure
        from spine.workflow.subgraph_wrapper import make_subgraph_node

        inputs: list[dict] = []

        class _Subgraph:
            calls = 0

            async def ainvoke(self, subgraph_input, config):
                inputs.append(dict(subgraph_input))
                _Subgraph.calls += 1
                if _Subgraph.calls == 1:
                    raise CriticalContractFailure(phase="plan", reason="nope")
                return {"phase_status": "success", "artifacts_output": {}}

        node_fn = make_subgraph_node(
            _Subgraph(), "plan", lambda p, c: {"phase": "plan"}, lambda r, p: {"ok": True}
        )

        await node_fn({"work_id": "wk1", "status": "running"}, None)

        assert inputs[0] == inputs[1]


class TestStructuralRetryTimeBudget:
    """A configured phase timeout is a budget for the WHOLE phase. A structural
    retry that cannot finish in the time left must be skipped (and the prior
    phases salvaged) rather than launched and cancelled mid-synthesis with zero
    artifacts (trace 019ec997)."""

    @staticmethod
    def _config_with_timeout(seconds: int):
        from types import SimpleNamespace

        return {
            "configurable": {
                "spine_config": SimpleNamespace(
                    phase_timeouts={"plan": seconds}, default_timeout=seconds
                )
            }
        }

    @pytest.mark.asyncio
    async def test_retry_skipped_when_budget_exhausted(self):
        from spine.exceptions import CriticalContractFailure
        from spine.workflow.subgraph_wrapper import make_subgraph_node

        calls = {"n": 0}

        class _Subgraph:
            async def ainvoke(self, subgraph_input, config):
                calls["n"] += 1
                raise CriticalContractFailure(phase="plan", reason="plan.json missing")

        node_fn = make_subgraph_node(
            _Subgraph(), "plan", lambda p, c: {"phase": "plan"}, lambda r, p: {"ok": True}
        )

        # 5s budget < the 30s retry floor → the retry is never launched.
        result = await node_fn(
            {"work_id": "wk1", "status": "running"}, self._config_with_timeout(5)
        )

        assert calls["n"] == 1  # first attempt only; retry skipped
        assert result["status"] == "needs_review"
        assert any(
            "Structural retry skipped" in f.get("reason", "")
            for f in result["feedback"]
        )

    @pytest.mark.asyncio
    async def test_retry_proceeds_with_generous_budget(self):
        from spine.exceptions import CriticalContractFailure
        from spine.workflow.subgraph_wrapper import make_subgraph_node

        calls = {"n": 0}

        class _Subgraph:
            async def ainvoke(self, subgraph_input, config):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise CriticalContractFailure(phase="plan", reason="missing")
                return {"phase_status": "success", "artifacts_output": {}}

        node_fn = make_subgraph_node(
            _Subgraph(), "plan", lambda p, c: {"phase": "plan"}, lambda r, p: {"ok": True}
        )

        # Generous budget leaves ample room above the floor → retry proceeds.
        result = await node_fn(
            {"work_id": "wk1", "status": "running"}, self._config_with_timeout(3600)
        )

        assert calls["n"] == 2
        assert result == {"ok": True}


# ── Salvage already-written artifacts on abort (trace 019ec997) ──


class TestAbortArtifactSalvage:
    """Cancellation/timeout after the structured write must not claim zero
    artifacts: the write tools persist to disk before the save node runs, so
    a phase aborted after the write still has valid, reusable output."""

    @staticmethod
    def _write_artifact(workspace_root, work_id, phase, name, content):
        from spine.agents.artifacts import artifact_path

        d = Path(workspace_root) / artifact_path(work_id, phase)
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(content, encoding="utf-8")

    def test_needs_review_reports_disk_artifacts(self, tmp_path):
        from spine.workflow.subgraph_wrapper import _needs_review_update

        self._write_artifact(str(tmp_path), "wk1", "plan", "plan.json", '{"x": 1}')
        self._write_artifact(str(tmp_path), "wk1", "plan", "plan.md", "# Plan")

        update = _needs_review_update(
            {"work_id": "wk1", "workspace_root": str(tmp_path)},
            "plan",
            "Cancelled — subgraph did not complete. Prior phases preserved.",
        )

        pr = update["phase_results"]["plan"]
        assert pr["artifact_count"] == 2
        assert pr["artifact_names"] == ["plan.json", "plan.md"]
        assert "Artifacts preserved on disk" in pr["error"]
        assert "plan.json" in update["feedback"][0]["reason"]

    def test_needs_review_reports_zero_when_nothing_on_disk(self, tmp_path):
        from spine.workflow.subgraph_wrapper import _needs_review_update

        update = _needs_review_update(
            {"work_id": "wk1", "workspace_root": str(tmp_path)},
            "plan",
            "Cancelled — subgraph did not complete.",
        )

        pr = update["phase_results"]["plan"]
        assert pr["artifact_count"] == 0
        assert pr["artifact_names"] == []
        assert "preserved on disk" not in pr["error"]

    @pytest.mark.asyncio
    async def test_cancelled_subgraph_salvages_disk_artifacts(self, tmp_path):
        from spine.workflow.subgraph_wrapper import make_subgraph_node

        self._write_artifact(str(tmp_path), "wk1", "plan", "plan.json", '{"x": 1}')

        class _Subgraph:
            async def ainvoke(self, subgraph_input, config):
                raise asyncio.CancelledError()

        node_fn = make_subgraph_node(
            _Subgraph(), "plan", lambda p, c: {"phase": "plan"}, lambda r, p: {"ok": True}
        )

        result = await node_fn(
            {"work_id": "wk1", "workspace_root": str(tmp_path), "status": "running"}, None
        )

        pr = result["phase_results"]["plan"]
        assert pr["status"] == "needs_review"
        assert pr["artifact_count"] == 1
        assert pr["artifact_names"] == ["plan.json"]
