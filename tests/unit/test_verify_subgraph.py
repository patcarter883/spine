"""Tests for the VERIFY phase subgraph."""

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
        assert "run_agent" in nodes
        assert "save_artifacts" in nodes

    def test_verify_subgraph_edges(self):
        from spine.workflow.subgraphs.verify_subgraph import build_verify_subgraph

        graph = build_verify_subgraph().compile()
        mermaid = graph.get_graph().draw_mermaid()
        assert "run_agent" in mermaid
        assert "save_artifacts" in mermaid
        assert "__end__" in mermaid


class TestRunVerifyAgent:
    """Tests for the _run_verify_agent node within the subgraph."""

    @pytest.mark.asyncio
    async def test_run_agent_success(self):
        from spine.workflow.subgraphs.verify_subgraph import _run_verify_agent

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "messages": [
                MagicMock(content="VERIFIED", usage_metadata={"input_tokens": 100, "output_tokens": 50})
            ]
        }

        with patch("spine.workflow.subgraphs.verify_subgraph.build_verify_agent", return_value=mock_agent):
            with patch("spine.workflow.subgraphs.verify_subgraph.materialize_artifacts"):
                state = {
                    "work_id": "test123",
                    "work_type": "quick",
                    "description": "test desc",
                    "workspace_root": "/tmp",
                    "messages": [],
                }
                result = await _run_verify_agent(state)
                assert "agent_response" in result
                assert result["agent_response"] == "VERIFIED"

    @pytest.mark.asyncio
    async def test_run_agent_with_spec_workflow(self):
        from spine.workflow.subgraphs.verify_subgraph import _run_verify_agent

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "messages": [MagicMock(content="VERIFIED")]
        }

        with patch("spine.workflow.subgraphs.verify_subgraph.build_verify_agent", return_value=mock_agent):
            with patch("spine.workflow.subgraphs.verify_subgraph.materialize_artifacts"):
                state = {
                    "work_id": "test456",
                    "work_type": "spec",
                    "description": "spec test",
                    "workspace_root": "/tmp",
                }
                result = await _run_verify_agent(state)
                assert "agent_response" in result

    @pytest.mark.asyncio
    async def test_run_agent_se_error_sets_error_status(self):
        from spine.workflow.subgraphs.verify_subgraph import _run_verify_agent

        with patch("spine.workflow.subgraphs.verify_subgraph.build_verify_agent", side_effect=RuntimeError("boom")):
            state = {"work_id": "test", "work_type": "quick", "description": "d", "workspace_root": "."}
            result = await _run_verify_agent(state)
            assert result.get("phase_status") == "error"
            assert "boom" in result["agent_response"]



class TestSaveVerifyArtifacts:
    """Tests for the _save_verify_artifacts node within the subgraph."""

    @pytest.mark.asyncio
    async def test_save_artifacts_with_disk_files(self, tmp_path):
        import asyncio
        from spine.workflow.subgraphs.verify_subgraph import _save_verify_artifacts

        # Create a fake artifact on disk
        art_dir = tmp_path / ".spine" / "artifacts" / "test123" / "verify"
        art_dir.mkdir(parents=True)
        (art_dir / "verification.md").write_text("VERIFIED all slices")

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
        assert result["phase_status"] == "success"
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

    def test_verify_state_mapper_quick_workflow(self):
        from spine.workflow.compose import _verify_state_mapper

        parent = {
            "work_id": "abc",
            "work_type": "quick",
            "description": "fix bug",
            "workspace_root": "/projects/spine",
            "retry_count": {"verify": 0},
            "feedback": [],
        }
        result = _verify_state_mapper(parent, None)
        assert result["work_id"] == "abc"
        assert result["work_type"] == "quick"
        assert result["spec_path"] is None
        assert result["plan_path"] == ".spine/artifacts/abc/plan"

    def test_verify_state_mapper_spec_workflow(self):
        from spine.workflow.compose import _verify_state_mapper

        parent = {
            "work_id": "def",
            "work_type": "spec",
            "description": "build feature",
            "workspace_root": "/projects/spine",
        }
        result = _verify_state_mapper(parent, None)
        assert result["spec_path"] == ".spine/artifacts/def/specify"
        assert result["plan_path"] == ".spine/artifacts/def/plan"

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

    def test_verify_result_mapper_needs_review(self):
        from spine.workflow.compose import _verify_result_mapper

        subgraph_result = {
            "artifacts_output": {},
            "phase_status": "needs_review",
        }
        result = _verify_result_mapper(subgraph_result, {"work_id": "test"})
        assert result["status"] == "needs_review"
        assert result["needs_review_phase"] == "verify"
        assert any(f.get("status") == "needs_review" for f in result["feedback"])

    def test_verify_result_mapper_error(self):
        from spine.workflow.compose import _verify_result_mapper

        subgraph_result = {
            "artifacts_output": {},
            "phase_status": "error",
        }
        result = _verify_result_mapper(subgraph_result, {"work_id": "test"})
        assert result["status"] == "failed"
