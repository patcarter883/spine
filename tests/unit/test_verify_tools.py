"""Unit tests for verify_tools — ReadVerifyContextTool and WriteVerificationReportTool."""

from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spine.agents.verify_tools import (
    ReadVerifyContextTool,
    WriteVerificationReportTool,
    build_verify_orchestrator_tools,
)


# ── WriteVerificationReportTool ────────────────────────────────────────────


class TestWriteVerificationReportTool:
    def _make_tool(self, tmp_path: Path, work_id: str = "wk-01") -> WriteVerificationReportTool:
        return WriteVerificationReportTool(
            workspace_root=str(tmp_path),
            verify_dir=f".spine/artifacts/{work_id}/verify",
        )

    def test_writes_verification_json(self, tmp_path):
        tool = self._make_tool(tmp_path)
        result = tool._run(
            verification_results=[
                {
                    "slice_name": "s1",
                    "verdict": "VERIFIED",
                    "checklist": [
                        {"criterion": "c1", "passed": True, "detail": "ok"}
                    ],
                    "gaps": [],
                    "recommendations": [],
                },
            ],
            summary="All verified",
        )
        assert "verification.json" in result
        json_path = (
            tmp_path / ".spine" / "artifacts" / "wk-01" / "verify" / "verification.json"
        )
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["summary"] == "All verified"
        assert data["overall_status"] == "VERIFIED"
        assert len(data["verification_results"]) == 1

    def test_overall_status_failed_when_not_verified(self, tmp_path):
        tool = self._make_tool(tmp_path)
        tool._run(
            verification_results=[
                {
                    "slice_name": "s1",
                    "verdict": "VERIFIED",
                    "checklist": [],
                    "gaps": [],
                    "recommendations": [],
                },
                {
                    "slice_name": "s2",
                    "verdict": "NOT_VERIFIED",
                    "checklist": [],
                    "gaps": [],
                    "recommendations": [],
                },
            ],
            summary="Mixed results.",
        )
        json_path = (
            tmp_path / ".spine" / "artifacts" / "wk-01" / "verify" / "verification.json"
        )
        data = json.loads(json_path.read_text())
        assert data["overall_status"] == "FAILED"


# ── ReadVerifyContextTool ──────────────────────────────────────────────────


class TestReadVerifyContextTool:
    def _make_tool(
        self, tmp_path: Path,
        plan_dir: str = ".spine/artifacts/wk-01/plan",
        impl_dir: str = ".spine/artifacts/wk-01/implement",
        verify_dir: str = ".spine/artifacts/wk-01/verify",
    ) -> ReadVerifyContextTool:
        return ReadVerifyContextTool(
            workspace_root=str(tmp_path),
            plan_dir=plan_dir,
            impl_dir=impl_dir,
            verify_dir=verify_dir,
        )

    def test_reads_plan_json_for_structured_slices(self, tmp_path):
        # Set up plan.json
        plan_dir = tmp_path / ".spine" / "artifacts" / "wk-01" / "plan"
        plan_dir.mkdir(parents=True)
        plan_dir.joinpath("plan.json").write_text(
            json.dumps({
                "feature_slices": [
                    {
                        "id": "s1",
                        "title": "Test Slice",
                        "description": "desc",
                        "target_files": ["src/s1.py"],
                        "execution_requirements": "do it",
                        "dependencies": [],
                        "acceptance_criteria": ["it works"],
                        "complexity": "small",
                    },
                ],
                "codebase_map": "mapping here",
            })
        )

        # Set up implementation.json
        impl_dir = tmp_path / ".spine" / "artifacts" / "wk-01" / "implement"
        impl_dir.mkdir(parents=True)
        impl_dir.joinpath("implementation.json").write_text(
            json.dumps({
                "slice_results": [
                    {"slice_name": "s1", "status": "implemented"}
                ],
            })
        )

        tool = self._make_tool(tmp_path)
        result = json.loads(tool._run())

        assert "s1" in result["slices"]
        assert isinstance(result["slices"]["s1"], dict)
        assert result["slices"]["s1"]["title"] == "Test Slice"
        assert result["codebase_map"] == "mapping here"
        assert result["implementation"]["slice_results"][0]["slice_name"] == "s1"

    def test_missing_plan_dir_returns_plan_error(self, tmp_path):
        tool = self._make_tool(tmp_path, plan_dir=".spine/artifacts/nonexistent/plan")
        result = json.loads(tool._run())

        assert "plan_error" in result
        assert "not found" in result["plan_error"].lower()

    def test_missing_impl_json_returns_impl_error(self, tmp_path):
        # Set up plan.json so that part succeeds
        plan_dir = tmp_path / ".spine" / "artifacts" / "wk-01" / "plan"
        plan_dir.mkdir(parents=True)
        plan_dir.joinpath("plan.json").write_text(
            json.dumps({"feature_slices": [], "codebase_map": ""})
        )

        tool = self._make_tool(tmp_path, impl_dir=".spine/artifacts/wk-01/nonexistent")
        result = json.loads(tool._run())

        assert "impl_error" in result

    def test_no_slices_returns_warning(self, tmp_path):
        plan_dir = tmp_path / ".spine" / "artifacts" / "wk-01" / "plan"
        plan_dir.mkdir(parents=True)
        plan_dir.joinpath("plan.json").write_text(
            json.dumps({"feature_slices": [], "codebase_map": ""})
        )

        tool = self._make_tool(tmp_path)
        result = json.loads(tool._run())

        assert result["slice_count"] == 0
        assert "warning" in result


# ── build_verify_orchestrator_tools ────────────────────────────────────────


class TestBuildVerifyOrchestratorTools:
    def test_returns_two_tools(self, tmp_path):
        tools = build_verify_orchestrator_tools(
            workspace_root=str(tmp_path),
            work_id="abc123",
        )
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "read_verify_context" in names
        assert "write_verification_report" in names

    def test_tools_have_correct_paths(self, tmp_path):
        tools = build_verify_orchestrator_tools(
            workspace_root=str(tmp_path),
            work_id="xyz",
        )
        read_tool = next(t for t in tools if t.name == "read_verify_context")
        write_tool = next(t for t in tools if t.name == "write_verification_report")

        assert isinstance(read_tool, ReadVerifyContextTool)
        assert isinstance(write_tool, WriteVerificationReportTool)
        assert read_tool.plan_dir == ".spine/artifacts/xyz/plan"
        assert read_tool.impl_dir == ".spine/artifacts/xyz/implement"
        assert read_tool.verify_dir == ".spine/artifacts/xyz/verify"
        assert write_tool.verify_dir == ".spine/artifacts/xyz/verify"
        assert read_tool.workspace_root == str(tmp_path)
        assert write_tool.workspace_root == str(tmp_path)