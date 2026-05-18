"""Tests for subgraph state schemas and wrapper factory."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio


# ── Subgraph state schema tests ──

class TestSubgraphStateSchemas:
    """Tests for per-phase subgraph state schemas."""

    def test_base_subgraph_state_fields(self):
        from spine.workflow.subgraph_state import BaseSubgraphState

        state: BaseSubgraphState = {
            "phase": "verify",
            "work_id": "abc123",
            "work_type": "quick",
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

        node_fn = make_subgraph_node(
            mock_subgraph, "verify", lambda p, c: {}, lambda r, p: {}
        )

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

        node_fn = make_subgraph_node(
            mock_subgraph, "verify", lambda p, c: {}, lambda r, p: {}
        )

        parent_state = {"work_id": "test123", "status": "running"}
        result = await node_fn(parent_state, None)

        assert result["status"] == "needs_review"
        assert any("Timed out" in f.get("reason", "") for f in result["feedback"])

    @pytest.mark.asyncio
    async def test_generic_exception_returns_error(self):
        from spine.workflow.subgraph_wrapper import make_subgraph_node

        mock_subgraph = AsyncMock()
        mock_subgraph.ainvoke.side_effect = RuntimeError("boom")

        node_fn = make_subgraph_node(
            mock_subgraph, "verify", lambda p, c: {}, lambda r, p: {}
        )

        parent_state = {"work_id": "test123", "status": "running"}
        result = await node_fn(parent_state, None)

        assert result["status"] == "needs_review"
        assert any("boom" in f.get("reason", "") for f in result["feedback"])

    def test_node_function_name(self):
        from spine.workflow.subgraph_wrapper import make_subgraph_node

        node_fn = make_subgraph_node(
            None, "verify", lambda p, c: {}, lambda r, p: {}
        )
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


class TestErrorUpdate:
    """Tests for _error_update and _needs_review_update helpers."""

    def test_error_update_structure(self):
        from spine.workflow.subgraph_wrapper import _error_update

        result = _error_update(
            {"work_id": "test123"}, "implement", "something broke"
        )
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
