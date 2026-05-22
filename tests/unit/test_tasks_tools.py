"""Unit tests for tasks_tools — WriteTasksArtifactsTool and factory."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spine.agents.tasks_tools import (
    WriteTasksArtifactsTool,
    build_tasks_agent_tools,
)
from spine.agents.plan_tools import ReadPriorArtifactsTool, SearchCodebaseTool


# ── WriteTasksArtifactsTool ───────────────────────────────────────────────


class TestWriteTasksArtifactsTool:
    def _tool(self, tmp_path: Path, work_id: str = "wk-t1") -> WriteTasksArtifactsTool:
        return WriteTasksArtifactsTool(
            workspace_root=str(tmp_path),
            tasks_dir=f".spine/artifacts/{work_id}/tasks",
        )

    def _slice(self, name: str, deps: list[str] | None = None) -> dict:
        return {
            "name": name,
            "description": f"Implement {name}.",
            "files_to_modify": [f"spine/ui/{name}.py"],
            "files_to_create": [],
            "dependencies": deps or [],
            "acceptance_criteria": [f"{name} works correctly"],
            "complexity": "small",
            "modification_targets": f"# spine/ui/{name}.py [L10-15]\nsome_code()",
        }

    def test_writes_all_three_artifact_types(self, tmp_path):
        tool = self._tool(tmp_path)
        result = tool._run(
            slices=[self._slice("foo"), self._slice("bar", deps=["foo"])],
            overview="Two slices: foo and bar.",
            dependency_waves="Wave 1: foo. Wave 2: bar.",
            codebase_map="## Files\n- spine/ui/foo.py: foo module\n",
        )
        assert "PHASE COMPLETE" in result

        base = tmp_path / ".spine/artifacts/wk-t1/tasks"
        assert (base / "slice-foo.md").exists()
        assert (base / "slice-bar.md").exists()
        assert (base / "tasks.md").exists()
        assert (base / "codebase-map.md").exists()

    def test_slice_file_content(self, tmp_path):
        tool = self._tool(tmp_path)
        tool._run(
            slices=[self._slice("alpha")],
            overview="One slice.",
            dependency_waves="Wave 1: alpha.",
            codebase_map="map content",
        )
        content = (tmp_path / ".spine/artifacts/wk-t1/tasks/slice-alpha.md").read_text()
        assert "# Slice: alpha" in content
        assert "Implement alpha." in content
        assert "spine/ui/alpha.py" in content
        assert "alpha works correctly" in content
        assert "small" in content
        assert "L10-15" in content

    def test_dependencies_listed(self, tmp_path):
        tool = self._tool(tmp_path)
        tool._run(
            slices=[self._slice("b", deps=["a"])],
            overview="B depends on A.",
            dependency_waves="Wave 1: a. Wave 2: b.",
            codebase_map="",
        )
        content = (tmp_path / ".spine/artifacts/wk-t1/tasks/slice-b.md").read_text()
        assert "a" in content  # dependency listed

    def test_no_dependencies_section(self, tmp_path):
        tool = self._tool(tmp_path)
        tool._run(
            slices=[self._slice("standalone")],
            overview="",
            dependency_waves="",
            codebase_map="",
        )
        content = (tmp_path / ".spine/artifacts/wk-t1/tasks/slice-standalone.md").read_text()
        assert "None" in content  # no deps

    def test_tasks_md_has_overview_and_waves(self, tmp_path):
        tool = self._tool(tmp_path)
        tool._run(
            slices=[self._slice("x"), self._slice("y")],
            overview="X and Y together.",
            dependency_waves="Wave 1: x, y (parallel).",
            codebase_map="",
        )
        content = (tmp_path / ".spine/artifacts/wk-t1/tasks/tasks.md").read_text()
        assert "X and Y together." in content
        assert "Wave 1: x, y (parallel)." in content
        assert "slice-x.md" not in content  # tasks.md doesn't reference filenames directly
        assert "x" in content
        assert "y" in content

    def test_tasks_md_file_change_matrix(self, tmp_path):
        tool = self._tool(tmp_path)
        s1 = self._slice("s1")
        s2 = {**self._slice("s2"), "files_to_modify": ["spine/ui/s1.py"]}  # same file as s1
        tool._run(
            slices=[s1, s2],
            overview="",
            dependency_waves="",
            codebase_map="",
        )
        content = (tmp_path / ".spine/artifacts/wk-t1/tasks/tasks.md").read_text()
        assert "File Change Matrix" in content

    def test_codebase_map_content(self, tmp_path):
        tool = self._tool(tmp_path)
        tool._run(
            slices=[self._slice("z")],
            overview="",
            dependency_waves="",
            codebase_map="## Files\n- spine/work/dispatcher.py: work dispatcher\n",
        )
        content = (tmp_path / ".spine/artifacts/wk-t1/tasks/codebase-map.md").read_text()
        assert "# Codebase Map" in content
        assert "dispatcher" in content

    def test_creates_directory(self, tmp_path):
        tool = self._tool(tmp_path, work_id="brand-new")
        tool._run(
            slices=[self._slice("z")],
            overview="",
            dependency_waves="",
            codebase_map="",
        )
        assert (tmp_path / ".spine/artifacts/brand-new/tasks").is_dir()

    def test_result_says_phase_complete(self, tmp_path):
        tool = self._tool(tmp_path)
        result = tool._run(
            slices=[self._slice("done")],
            overview="Done.",
            dependency_waves="Wave 1: done.",
            codebase_map="map",
        )
        assert "PHASE COMPLETE" in result
        assert "no further tool calls" in result.lower() or "no further" in result.lower()

    def test_accepts_dict_slices(self, tmp_path):
        """Pydantic may pass slice dicts rather than SliceDefinition objects."""
        tool = self._tool(tmp_path)
        result = tool._run(
            slices=[self._slice("dict-slice")],
            overview="dict input",
            dependency_waves="Wave 1: dict-slice.",
            codebase_map="",
        )
        assert "PHASE COMPLETE" in result
        assert (tmp_path / ".spine/artifacts/wk-t1/tasks/slice-dict-slice.md").exists()


# ── build_tasks_agent_tools ───────────────────────────────────────────────


class TestBuildTasksAgentTools:
    def test_returns_three_tools(self, tmp_path):
        tools = build_tasks_agent_tools(
            workspace_root=str(tmp_path),
            work_id="abc",
        )
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert "read_prior_artifacts" in names
        assert "search_codebase" in names
        assert "write_tasks_artifacts" in names

    def test_tool_types(self, tmp_path):
        tools = build_tasks_agent_tools(
            workspace_root=str(tmp_path),
            work_id="xyz",
            prior_phase_dirs={"specify": ".spine/artifacts/xyz/specify"},
        )
        read_tool = next(t for t in tools if t.name == "read_prior_artifacts")
        search_tool = next(t for t in tools if t.name == "search_codebase")
        write_tool = next(t for t in tools if t.name == "write_tasks_artifacts")

        assert isinstance(read_tool, ReadPriorArtifactsTool)
        assert isinstance(search_tool, SearchCodebaseTool)
        assert isinstance(write_tool, WriteTasksArtifactsTool)

    def test_write_tool_has_correct_tasks_dir(self, tmp_path):
        tools = build_tasks_agent_tools(
            workspace_root=str(tmp_path),
            work_id="wk-99",
        )
        write_tool = next(t for t in tools if t.name == "write_tasks_artifacts")
        assert isinstance(write_tool, WriteTasksArtifactsTool)
        assert write_tool.tasks_dir == ".spine/artifacts/wk-99/tasks"

    def test_feedback_passed_to_read_tool(self, tmp_path):
        tools = build_tasks_agent_tools(
            workspace_root=str(tmp_path),
            work_id="wk-fb",
            feedback=["Slices too coarse."],
        )
        read_tool = next(t for t in tools if t.name == "read_prior_artifacts")
        assert isinstance(read_tool, ReadPriorArtifactsTool)
        assert "Slices too coarse." in read_tool.feedback
