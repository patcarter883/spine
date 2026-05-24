"""Tests for the VERIFY phase subgraph.

Updated for Send API dispatch pattern — the verify subgraph now uses
``verify_router`` → ``Send("run_slice_verifier", ...)`` → ``aggregate_verification``
→ ``synthesize_verification`` → ``save_artifacts``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestVerifySubgraphCompilation:
    """Tests that the verify subgraph compiles correctly."""

    def test_verify_subgraph_compiles(self):
        from spine.workflow.subgraphs.verify_subgraph import build_verify_subgraph

        graph = build_verify_subgraph().compile()
        assert graph is not None

    def test_verify_subgraph_has_correct_nodes(self):
        from spine.workflow.subgraphs.verify_subgraph import build_verify_subgraph

        graph = build_verify_subgraph().compile()
        nodes = set(graph.get_graph().nodes.keys())
        assert "run_slice_verifier" in nodes
        assert "aggregate_verification" in nodes
        assert "synthesize_verification" in nodes
        assert "save_artifacts" in nodes

    def test_verify_subgraph_edges(self):
        from spine.workflow.subgraphs.verify_subgraph import build_verify_subgraph

        graph = build_verify_subgraph().compile()
        mermaid = graph.get_graph().draw_mermaid()
        assert "run_slice_verifier" in mermaid
        assert "aggregate_verification" in mermaid
        assert "synthesize_verification" in mermaid
        assert "save_artifacts" in mermaid
        assert "__end__" in mermaid


class TestVerifyRouter:
    """Tests for the _verify_router conditional edge function."""

    def test_verify_router_raises_on_missing_execution_waves(self):
        from spine.workflow.subgraphs.verify_subgraph import _verify_router
        from spine.workflow.subgraph_state import VerifySubgraphState
        from spine.exceptions import CriticalContractFailure

        with pytest.raises(CriticalContractFailure, match="execution_waves"):
            _verify_router(VerifySubgraphState(
                work_id="test", phase="verify", workspace_root=".",
            ))

    def test_verify_router_raises_on_empty_execution_waves(self):
        from spine.workflow.subgraphs.verify_subgraph import _verify_router
        from spine.workflow.subgraph_state import VerifySubgraphState
        from spine.exceptions import CriticalContractFailure

        with pytest.raises(CriticalContractFailure, match="execution_waves"):
            _verify_router(VerifySubgraphState(
                work_id="test", phase="verify", workspace_root=".",
                execution_waves=[],
            ))

    def test_verify_router_dispatches_sends(self):
        from spine.workflow.subgraphs.verify_subgraph import _verify_router
        from spine.workflow.subgraph_state import VerifySubgraphState
        from langgraph.types import Send

        result = _verify_router(VerifySubgraphState(
            work_id="test", phase="verify", workspace_root="/tmp",
            execution_waves=[[{"id": "s1", "title": "Slice 1"}]],
        ))
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Send)
        assert result[0].node == "run_slice_verifier"
        assert result[0].arg["slice"]["id"] == "s1"


class TestRunSliceVerifier:
    """Tests for the _run_slice_verifier_node."""

    @pytest.mark.asyncio
    async def test_run_verifier_node_success(self):
        from spine.workflow.subgraphs.verify_subgraph import _run_slice_verifier_node

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "messages": [
                MagicMock(
                    content='{"verdict": "VERIFIED", "checklist": [{"criterion":"test","passed":true,"detail":"ok"}], "gaps": [], "recommendations": []}'
                ),
            ]
        }

        with patch(
            "spine.agents.subagents.build_subagent_spec",
            return_value={"system_prompt": "test prompt", "tools": [], "response_format": None},
        ):
            with patch(
                "spine.agents.factory.build_phase_agent",
                return_value=mock_agent,
            ):
                state = {
                    "work_id": "test123",
                    "work_type": "task",
                    "phase": "verify",
                    "workspace_root": "/tmp",
                    "slice": {"id": "s1", "title": "Test Slice"},
                    "messages": [],
                }
                result = await _run_slice_verifier_node(state)
                assert "verification_results" in result
                assert len(result["verification_results"]) == 1
                assert result["verification_results"][0]["verdict"] == "VERIFIED"
                assert result["verification_results"][0]["slice_name"] == "s1"

    @pytest.mark.asyncio
    async def test_run_verifier_node_error_returns_not_verified(self):
        from spine.workflow.subgraphs.verify_subgraph import _run_slice_verifier_node

        with patch(
            "spine.agents.subagents.build_subagent_spec",
            side_effect=RuntimeError("boom"),
        ):
            state = {
                "work_id": "test",
                "work_type": "task",
                "phase": "verify",
                "workspace_root": ".",
                "slice": {"id": "s-err"},
            }
            result = await _run_slice_verifier_node(state)
            assert "verification_results" in result
            assert result["verification_results"][0]["verdict"] == "NOT_VERIFIED"
            assert result["verification_results"][0]["slice_name"] == "s-err"


class TestSaveVerifyArtifacts:
    """Tests for the _save_verify_artifacts node within the subgraph."""

    @pytest.mark.asyncio
    async def test_save_artifacts_with_disk_files(self, tmp_path):
        from spine.workflow.subgraphs.verify_subgraph import _save_verify_artifacts

        art_dir = tmp_path / ".spine" / "artifacts" / "test123" / "verify"
        art_dir.mkdir(parents=True)
        (art_dir / "verification.md").write_text("VERIFIED all slices")
        (art_dir / "verification.json").write_text(
            '{"overall_status": "VERIFIED", "summary": "All good"}'
        )

        state = {
            "work_id": "test123",
            "workspace_root": str(tmp_path),
            "agent_response": "",
            "phase_status": "",
        }
        result = await _save_verify_artifacts(state, None)
        assert result["phase_status"] == "success"
        assert "verification.md" in result["artifacts_output"]

    @pytest.mark.asyncio
    async def test_save_artifacts_not_verified_sets_needs_review(self, tmp_path):
        from spine.workflow.subgraphs.verify_subgraph import _save_verify_artifacts

        art_dir = tmp_path / ".spine" / "artifacts" / "test456" / "verify"
        art_dir.mkdir(parents=True)
        (art_dir / "verification.md").write_text("Some issues found, not complete")
        (art_dir / "verification.json").write_text(
            '{"overall_status": "FAILED", "summary": "Issues found"}'
        )

        state = {
            "work_id": "test456",
            "workspace_root": str(tmp_path),
            "agent_response": "",
            "phase_status": "",
        }
        result = await _save_verify_artifacts(state, None)
        assert result["phase_status"] == "needs_review"

    @pytest.mark.asyncio
    async def test_save_artifacts_falls_back_to_agent_response(self, tmp_path):
        from spine.workflow.subgraphs.verify_subgraph import _save_verify_artifacts

        state = {
            "work_id": "test789",
            "workspace_root": str(tmp_path),
            "agent_response": "VERIFIED everything looks good",
            "phase_status": "",
        }
        result = await _save_verify_artifacts(state, None)
        # Without verification.json on disk, authoritative status defaults
        # to unverified; phase_status becomes needs_review.
        assert result["phase_status"] == "needs_review"
        assert "verification.md" in result["artifacts_output"]

    @pytest.mark.asyncio
    async def test_save_artifacts_preserves_error_status(self):
        from spine.workflow.subgraphs.verify_subgraph import _save_verify_artifacts

        state = {
            "work_id": "test000",
            "workspace_root": "/tmp",
            "agent_response": "",
            "phase_status": "error",
        }
        result = await _save_verify_artifacts(state, None)
        assert result["phase_status"] == "error"

    @pytest.mark.asyncio
    async def test_save_artifacts_empty_response_fallback(self, tmp_path):
        from spine.workflow.subgraphs.verify_subgraph import _save_verify_artifacts

        state = {
            "work_id": "test111",
            "workspace_root": str(tmp_path),
            "agent_response": "  ",
            "phase_status": "",
        }
        result = await _save_verify_artifacts(state, None)
        assert result["phase_status"] == "needs_review"
        assert "insufficient output" in result["artifacts_output"]["verification.md"].lower()


class TestVerifyStateAndResult:
    """Tests state mapping between parent and verify subgraph."""

    def test_verify_state_mapper_includes_execution_waves(self):
        from spine.workflow.compose import _verify_state_mapper

        parent = {
            "work_id": "abc",
            "work_type": "task",
            "description": "fix bug",
            "workspace_root": "/projects/spine",
            "retry_count": {"verify": 0},
            "feedback": [],
        }
        result = _verify_state_mapper(parent, None)
        assert result["work_id"] == "abc"
        assert result["work_type"] == "task"
        assert result["spec_path"] == ".spine/artifacts/abc/specify"
        assert result["plan_path"] == ".spine/artifacts/abc/plan"
        assert result["execution_waves"] == []

    def test_verify_state_mapper_passes_execution_waves(self):
        from spine.workflow.compose import _verify_state_mapper

        parent = {
            "work_id": "def",
            "work_type": "task",
            "description": "build feature",
            "workspace_root": "/projects/spine",
            "execution_waves": [[{"id": "s1"}, {"id": "s2"}]],
        }
        result = _verify_state_mapper(parent, None)
        assert result["execution_waves"] == [[{"id": "s1"}, {"id": "s2"}]]

    def test_verify_result_mapper_success(self):
        from spine.workflow.compose import _verify_result_mapper

        subgraph_result = {
            "artifacts_output": {"verification.md": "VERIFIED"},
            "phase_status": "success",
        }
        result = _verify_result_mapper(subgraph_result, {"work_id": "test"})
        assert result["status"] == "running"
        assert result["current_phase"] == "verify"
        assert result["phase_results"]["verify"]["status"] == "success"

    def test_verify_result_mapper_needs_gap_fix(self):
        from spine.workflow.compose import _verify_result_mapper

        subgraph_result = {
            "artifacts_output": {},
            "phase_status": "needs_review",
        }
        result = _verify_result_mapper(subgraph_result, {"work_id": "test"})
        assert result["status"] == "needs_gap_fix"
        assert result["verify_attempts"] == 1
        assert any(f.get("status") == "needs_review" for f in result["feedback"])

    def test_verify_result_mapper_needs_review_after_max_gaps(self):
        from spine.workflow.compose import _verify_result_mapper

        subgraph_result = {
            "artifacts_output": {},
            "phase_status": "needs_review",
        }
        result = _verify_result_mapper(subgraph_result, {"work_id": "test", "verify_attempts": 2})
        assert result["status"] == "needs_review"
        assert result["needs_review_phase"] == "verify"
        assert any(f.get("status") == "needs_review" for f in result["feedback"])

    def test_verify_result_mapper_second_gap_fix(self):
        from spine.workflow.compose import _verify_result_mapper

        subgraph_result = {
            "artifacts_output": {},
            "phase_status": "needs_review",
        }
        result = _verify_result_mapper(subgraph_result, {"work_id": "test", "verify_attempts": 1})
        assert result["status"] == "needs_gap_fix"
        assert result["verify_attempts"] == 2

    def test_verify_result_mapper_error(self):
        from spine.workflow.compose import _verify_result_mapper

        subgraph_result = {
            "artifacts_output": {},
            "phase_status": "error",
        }
        result = _verify_result_mapper(subgraph_result, {"work_id": "test"})
        assert result["status"] == "failed"