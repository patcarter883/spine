"""Unit tests for implement_tools — ReadSliceFilesTool and WriteImplementationReportTool."""

from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spine.agents.implement_tools import (
    ReadSliceFilesTool,
    WriteImplementationReportTool,
    build_implement_orchestrator_tools,
)


# ── ReadSliceFilesTool ────────────────────────────────────────────────────


class TestReadSliceFilesTool:
    def _make_tasks_dir(self, tmp_path: Path) -> tuple[Path, str]:
        work_id = "test-work-01"
        tasks = tmp_path / ".spine" / "artifacts" / work_id / "tasks"
        tasks.mkdir(parents=True)
        return tasks, work_id

    def test_returns_slices_and_codebase_map(self, tmp_path):
        work_id = "test-work-01"
        plan = tmp_path / ".spine" / "artifacts" / work_id / "plan"
        plan.mkdir(parents=True)
        (plan / "plan.json").write_text(
            json.dumps({
                "architecture_overview": "test",
                "feature_slices": [
                    {
                        "id": "slice-foo",
                        "title": "Foo",
                        "target_files": ["src/foo.py"],
                        "execution_requirements": "Do the foo thing.",
                        "dependencies": [],
                        "acceptance_criteria": ["foo works"],
                        "complexity": "small",
                    },
                    {
                        "id": "slice-bar",
                        "title": "Bar",
                        "target_files": ["src/bar.py"],
                        "execution_requirements": "Do the bar thing.",
                        "dependencies": ["slice-foo"],
                        "acceptance_criteria": ["bar works"],
                        "complexity": "medium",
                    },
                ],
                "codebase_map": "# Map\n- src/main.py: entry point",
            })
        )

        tool = ReadSliceFilesTool(
            workspace_root=str(tmp_path),
            plan_dir=f".spine/artifacts/{work_id}/plan",
        )
        result = json.loads(tool._run())

        assert result["slice_count"] == 2
        assert "slice-foo" in result["slices"]
        assert "slice-bar" in result["slices"]
        assert "Do the foo thing." in result["slices"]["slice-foo"]["execution_requirements"]
        assert "Do the bar thing." in result["slices"]["slice-bar"]["execution_requirements"]
        assert "entry point" in result["codebase_map"]
        assert result["plan_dir"].endswith("plan")
        # slice-foo should have all expected keys
        slice_foo = result["slices"]["slice-foo"]
        assert slice_foo["title"] == "Foo"
        assert slice_foo["target_files"] == ["src/foo.py"]
        assert slice_foo["dependencies"] == []

    def test_missing_codebase_map_returns_empty_string(self, tmp_path):
        work_id = "test-work-01"
        plan = tmp_path / ".spine" / "artifacts" / work_id / "plan"
        plan.mkdir(parents=True)
        (plan / "plan.json").write_text(
            json.dumps({
                "feature_slices": [{"id": "alpha", "title": "A", "target_files": [], "execution_requirements": "", "dependencies": [], "acceptance_criteria": [], "complexity": "small"}],
                "codebase_map": "",
            })
        )

        tool = ReadSliceFilesTool(
            workspace_root=str(tmp_path),
            plan_dir=f".spine/artifacts/{work_id}/plan",
        )
        result = json.loads(tool._run())

        assert result["codebase_map"] == ""
        assert result["slice_count"] == 1

    def test_missing_plan_dir_returns_error(self, tmp_path):
        tool = ReadSliceFilesTool(
            workspace_root=str(tmp_path),
            plan_dir=".spine/artifacts/nonexistent/plan",
        )
        result = json.loads(tool._run())

        assert "error" in result
        assert "not found" in result["error"].lower()

    def test_no_slices_returns_warning(self, tmp_path):
        work_id = "test-work-01"
        plan = tmp_path / ".spine" / "artifacts" / work_id / "plan"
        plan.mkdir(parents=True)
        (plan / "plan.json").write_text(
            json.dumps({"feature_slices": [], "codebase_map": "# Map"})
        )

        tool = ReadSliceFilesTool(
            workspace_root=str(tmp_path),
            plan_dir=f".spine/artifacts/{work_id}/plan",
        )
        result = json.loads(tool._run())

        assert result["slice_count"] == 0
        assert "warning" in result

    def test_slices_ordered_by_plan_json(self, tmp_path):
        work_id = "test-work-01"
        plan = tmp_path / ".spine" / "artifacts" / work_id / "plan"
        plan.mkdir(parents=True)
        # Input order: a, m, z — dict keys should reflect plan.json order
        (plan / "plan.json").write_text(
            json.dumps({
                "feature_slices": [
                    {"id": "a-slice", "title": "A", "target_files": [], "execution_requirements": "", "dependencies": [], "acceptance_criteria": [], "complexity": "small"},
                    {"id": "m-slice", "title": "M", "target_files": [], "execution_requirements": "", "dependencies": [], "acceptance_criteria": [], "complexity": "small"},
                    {"id": "z-slice", "title": "Z", "target_files": [], "execution_requirements": "", "dependencies": [], "acceptance_criteria": [], "complexity": "small"},
                ],
            })
        )

        tool = ReadSliceFilesTool(
            workspace_root=str(tmp_path),
            plan_dir=f".spine/artifacts/{work_id}/plan",
        )
        result = json.loads(tool._run())

        keys = list(result["slices"].keys())
        # Keys preserve plan.json array order (JSON order)
        assert keys == ["a-slice", "m-slice", "z-slice"]


# ── WriteImplementationReportTool ─────────────────────────────────────────


class TestWriteImplementationReportTool:
    def _make_tool(self, tmp_path: Path, work_id: str = "wk-01") -> WriteImplementationReportTool:
        return WriteImplementationReportTool(
            workspace_root=str(tmp_path),
            impl_dir=f".spine/artifacts/{work_id}/implement",
        )

    def _slice_result(self, name: str, status: str = "implemented") -> dict:
        return {
            "slice_name": name,
            "status": status,
            "files_modified": [f"src/{name}.py"],
            "files_created": [],
            "test_results": "All tests passed",
            "issues": [],
        }

    def test_writes_implementation_md(self, tmp_path):
        tool = self._make_tool(tmp_path)
        result = tool._run(
            slice_results=[self._slice_result("foo"), self._slice_result("bar")],
            summary="Implemented foo and bar slices successfully.",
        )

        assert "implementation.md written" in result
        impl_path = tmp_path / ".spine" / "artifacts" / "wk-01" / "implement" / "implementation.md"
        assert impl_path.exists()
        content = impl_path.read_text()
        assert "# Implementation Report" in content
        assert "foo" in content
        assert "bar" in content
        assert "Implemented foo and bar" in content

    def test_creates_impl_dir_if_missing(self, tmp_path):
        tool = self._make_tool(tmp_path, work_id="new-work")
        tool._run(
            slice_results=[self._slice_result("alpha")],
            summary="Done.",
        )
        impl_path = (
            tmp_path / ".spine" / "artifacts" / "new-work" / "implement" / "implementation.md"
        )
        assert impl_path.exists()

    def test_blocked_slice_flagged(self, tmp_path):
        tool = self._make_tool(tmp_path)
        sr = self._slice_result("blocker", status="blocked")
        sr["issues"] = ["Missing dependency: auth_service"]
        tool._run(slice_results=[sr], summary="One slice blocked.")

        content = (
            tmp_path / ".spine" / "artifacts" / "wk-01" / "implement" / "implementation.md"
        ).read_text()
        assert "❌" in content
        assert "blocked" in content
        assert "Missing dependency" in content

    def test_status_summary_counts(self, tmp_path):
        tool = self._make_tool(tmp_path)
        slices = [
            self._slice_result("s1", "implemented"),
            self._slice_result("s2", "partial"),
            self._slice_result("s3", "blocked"),
        ]
        tool._run(slice_results=slices, summary="Mixed results.")

        content = (
            tmp_path / ".spine" / "artifacts" / "wk-01" / "implement" / "implementation.md"
        ).read_text()
        assert "Total slices: 3" in content
        assert "Implemented: 1" in content
        assert "Partial: 1" in content
        assert "Blocked: 1" in content

    def test_files_changed_section(self, tmp_path):
        tool = self._make_tool(tmp_path)
        sr = {
            "slice_name": "feat",
            "status": "implemented",
            "files_modified": ["src/a.py", "src/b.py"],
            "files_created": ["src/new.py"],
            "test_results": "ok",
            "issues": [],
        }
        tool._run(slice_results=[sr], summary="Done.")

        content = (
            tmp_path / ".spine" / "artifacts" / "wk-01" / "implement" / "implementation.md"
        ).read_text()
        assert "src/a.py" in content
        assert "src/new.py" in content
        assert "Files Changed" in content

    def test_empty_slice_results(self, tmp_path):
        tool = self._make_tool(tmp_path)
        result = tool._run(slice_results=[], summary="No slices.")
        assert "implementation.md written" in result
        content = (
            tmp_path / ".spine" / "artifacts" / "wk-01" / "implement" / "implementation.md"
        ).read_text()
        assert "Total slices: 0" in content

    def test_writes_implementation_json(self, tmp_path):
        """Verify WriteImplementationReportTool writes implementation.json."""
        tool = self._make_tool(tmp_path)
        result = tool._run(
            slice_results=[self._slice_result("s1")],
            summary="JSON output test.",
        )
        assert "implementation.json" in result
        json_path = (
            tmp_path / ".spine" / "artifacts" / "wk-01" / "implement" / "implementation.json"
        )
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["summary"] == "JSON output test."
        assert len(data["slice_results"]) == 1
        assert data["slice_results"][0]["slice_name"] == "s1"
        assert data["slice_results"][0]["status"] == "implemented"


# ── build_implement_orchestrator_tools ────────────────────────────────────


class TestBuildImplementOrchestratorTools:
    def test_returns_two_tools(self, tmp_path):
        tools = build_implement_orchestrator_tools(
            workspace_root=str(tmp_path),
            work_id="abc123",
        )
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "read_slice_files" in names
        assert "write_implementation_report" in names

    def test_tools_have_correct_paths(self, tmp_path):
        tools = build_implement_orchestrator_tools(
            workspace_root=str(tmp_path),
            work_id="xyz",
        )
        read_tool = next(t for t in tools if t.name == "read_slice_files")
        write_tool = next(t for t in tools if t.name == "write_implementation_report")

        assert isinstance(read_tool, ReadSliceFilesTool)
        assert isinstance(write_tool, WriteImplementationReportTool)
        assert read_tool.plan_dir == ".spine/artifacts/xyz/plan"
        assert write_tool.impl_dir == ".spine/artifacts/xyz/implement"
        assert read_tool.workspace_root == str(tmp_path)
        assert write_tool.workspace_root == str(tmp_path)
