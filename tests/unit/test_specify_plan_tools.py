"""Unit tests for specify_tools and plan_tools."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spine.agents.specify_tools import (
    ReadWorkContextTool,
    WriteSpecificationTool,
    build_specify_orchestrator_tools,
)
from spine.agents.plan_tools import (
    ReadPriorArtifactsTool,
    SearchCodebaseTool,
    WritePlanTool,
    build_plan_agent_tools,
)


# ── ReadWorkContextTool ───────────────────────────────────────────────────


class TestReadWorkContextTool:
    def _tool(self, tmp_path: Path, feedback=None, has_prior_spec=False) -> ReadWorkContextTool:
        work_id = "wk-spec"
        spec_dir = f".spine/artifacts/{work_id}/specify"
        if has_prior_spec:
            p = tmp_path / spec_dir
            p.mkdir(parents=True)
            (p / "specification.md").write_text("# Prior Spec\nOld content.")
        return ReadWorkContextTool(
            workspace_root=str(tmp_path),
            work_id=work_id,
            work_type="spec",
            description="Build a widget factory.",
            feedback=feedback or [],
            spec_dir=spec_dir,
        )

    def test_returns_basic_context(self, tmp_path):
        tool = self._tool(tmp_path)
        result = json.loads(tool._run())
        assert result["description"] == "Build a widget factory."
        assert result["work_id"] == "wk-spec"
        assert result["work_type"] == "spec"
        assert result["feedback"] == []
        assert result["prior_spec"] == ""

    def test_includes_feedback(self, tmp_path):
        tool = self._tool(tmp_path, feedback=["Missing error handling.", "Add retry logic."])
        result = json.loads(tool._run())
        assert len(result["feedback"]) == 2
        assert "Missing error handling." in result["feedback"]

    def test_loads_prior_spec_on_rework(self, tmp_path):
        tool = self._tool(tmp_path, has_prior_spec=True)
        result = json.loads(tool._run())
        assert "Prior Spec" in result["prior_spec"]
        assert "Old content." in result["prior_spec"]

    def test_prior_spec_empty_when_no_rework(self, tmp_path):
        tool = self._tool(tmp_path, has_prior_spec=False)
        result = json.loads(tool._run())
        assert result["prior_spec"] == ""

    def test_spec_dir_in_result(self, tmp_path):
        tool = self._tool(tmp_path)
        result = json.loads(tool._run())
        assert "specify" in result["spec_dir"]


# ── WriteSpecificationTool ────────────────────────────────────────────────


class TestWriteSpecificationTool:
    def _tool(self, tmp_path: Path, work_id: str = "wk-s1") -> WriteSpecificationTool:
        return WriteSpecificationTool(
            workspace_root=str(tmp_path),
            spec_dir=f".spine/artifacts/{work_id}/specify",
        )

    def _full_args(self) -> dict:
        return {
            "overview": "Build a widget factory system.",
            "requirements": "- FR1: Create widgets\n- NFR1: <100ms latency",
            "architecture": "Three-layer: API, service, storage.",
            "interfaces": "POST /widgets, GET /widgets/{id}",
            "success_criteria": "- All tests pass\n- P99 <100ms",
        }

    def test_writes_specification_md(self, tmp_path):
        tool = self._tool(tmp_path)
        result = tool._run(**self._full_args())
        assert "specification.md written" in result
        spec = tmp_path / ".spine/artifacts/wk-s1/specify/specification.md"
        assert spec.exists()
        content = spec.read_text()
        assert "# Specification" in content
        assert "Build a widget factory" in content

    def test_all_sections_present(self, tmp_path):
        tool = self._tool(tmp_path)
        tool._run(**self._full_args())
        content = (tmp_path / ".spine/artifacts/wk-s1/specify/specification.md").read_text()
        for section in ["Overview", "Requirements", "Architecture", "Interfaces", "Success Criteria"]:
            assert section in content

    def test_open_questions_optional(self, tmp_path):
        tool = self._tool(tmp_path)
        args = self._full_args()
        args["open_questions"] = "What about rate limiting?"
        tool._run(**args)
        content = (tmp_path / ".spine/artifacts/wk-s1/specify/specification.md").read_text()
        assert "Open Questions" in content
        assert "rate limiting" in content

    def test_no_open_questions_section_when_empty(self, tmp_path):
        tool = self._tool(tmp_path)
        tool._run(**self._full_args())
        content = (tmp_path / ".spine/artifacts/wk-s1/specify/specification.md").read_text()
        assert "Open Questions" not in content

    def test_creates_directory(self, tmp_path):
        tool = self._tool(tmp_path, work_id="brand-new")
        tool._run(**self._full_args())
        assert (tmp_path / ".spine/artifacts/brand-new/specify/specification.md").exists()


# ── build_specify_orchestrator_tools ─────────────────────────────────────


class TestBuildSpecifyOrchestratorTools:
    def test_returns_two_tools(self, tmp_path):
        tools = build_specify_orchestrator_tools(
            workspace_root=str(tmp_path),
            work_id="abc",
            description="desc",
            work_type="spec",
        )
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "read_work_context" in names
        assert "write_specification" in names

    def test_feedback_injected(self, tmp_path):
        tools = build_specify_orchestrator_tools(
            workspace_root=str(tmp_path),
            work_id="x",
            description="d",
            work_type="quick",
            feedback=["fix this"],
        )
        read_tool = next(t for t in tools if t.name == "read_work_context")
        assert isinstance(read_tool, ReadWorkContextTool)
        assert read_tool.feedback == ["fix this"]


# ── ReadPriorArtifactsTool ────────────────────────────────────────────────


class TestReadPriorArtifactsTool:
    def _setup(self, tmp_path: Path, work_id: str = "wk-p") -> tuple[str, dict[str, str]]:
        spec_dir = f".spine/artifacts/{work_id}/specify"
        p = tmp_path / spec_dir
        p.mkdir(parents=True)
        (p / "specification.md").write_text("# Spec\nContent here.")
        return work_id, {PhaseName_SPECIFY: spec_dir}

    def _tool(self, tmp_path, work_id, prior_dirs) -> ReadPriorArtifactsTool:
        return ReadPriorArtifactsTool(
            workspace_root=str(tmp_path),
            work_id=work_id,
            work_type="spec",
            description="Plan a widget.",
            feedback=[],
            plan_dir=f".spine/artifacts/{work_id}/plan",
            prior_phase_dirs=prior_dirs,
        )

    def test_loads_prior_spec(self, tmp_path):
        work_id, prior_dirs = self._setup(tmp_path)
        tool = self._tool(tmp_path, work_id, prior_dirs)
        result = json.loads(tool._run())
        assert "specify" in result["artifacts"]
        assert "specification.md" in result["artifacts"]["specify"]
        assert "Content here." in result["artifacts"]["specify"]["specification.md"]

    def test_missing_phase_dir_omitted(self, tmp_path):
        tool = self._tool(tmp_path, "wk-p2", {})
        result = json.loads(tool._run())
        assert result["artifacts"] == {}
        assert "warning" in result

    def test_basic_context_fields(self, tmp_path):
        work_id, prior_dirs = self._setup(tmp_path)
        tool = self._tool(tmp_path, work_id, prior_dirs)
        result = json.loads(tool._run())
        assert result["work_id"] == work_id
        assert result["description"] == "Plan a widget."
        assert "plan_dir" in result


# Constant alias for test readability
PhaseName_SPECIFY = "specify"


# ── SearchCodebaseTool ────────────────────────────────────────────────────


class TestSearchCodebaseTool:
    def _tool(self, tmp_path: Path) -> SearchCodebaseTool:
        return SearchCodebaseTool(workspace_root=str(tmp_path))

    def _setup_files(self, tmp_path: Path) -> None:
        src = tmp_path / "spine" / "agents"
        src.mkdir(parents=True)
        (src / "factory.py").write_text("def build_phase_agent(state, config):\n    pass\n")
        (src / "helpers.py").write_text("def resolve_model(config):\n    return 'gpt-4'\n")
        (tmp_path / "spine").mkdir(exist_ok=True)
        (tmp_path / "spine" / "models.py").write_text("class WorkflowState:\n    pass\n")

    def test_finds_matching_files(self, tmp_path):
        self._setup_files(tmp_path)
        tool = self._tool(tmp_path)
        result = json.loads(tool._run(queries=["build_phase_agent"]))
        files = [r["file"] for r in result["results"]]
        assert any("factory.py" in f for f in files)

    def test_multi_query_scoring(self, tmp_path):
        self._setup_files(tmp_path)
        tool = self._tool(tmp_path)
        # factory.py matches one query, helpers.py matches another
        result = json.loads(tool._run(queries=["build_phase_agent", "resolve_model"]))
        assert result["total_files_found"] >= 1
        # File matching most queries should be ranked first (or at least present)
        all_files = [r["file"] for r in result["results"]]
        assert len(all_files) >= 1

    def test_empty_workspace_returns_empty(self, tmp_path):
        tool = self._tool(tmp_path)
        result = json.loads(tool._run(queries=["nonexistent_symbol_xyz"]))
        assert result["total_files_found"] == 0
        assert result["results"] == []

    def test_file_patterns_restrict_scope(self, tmp_path):
        self._setup_files(tmp_path)
        tool = self._tool(tmp_path)
        result = json.loads(tool._run(
            queries=["pass"],
            file_patterns=["spine/agents/*.py"],
        ))
        # Should only find files matching the pattern
        for r in result["results"]:
            assert "spine/agents/" in r["file"]

    def test_result_includes_preview(self, tmp_path):
        self._setup_files(tmp_path)
        tool = self._tool(tmp_path)
        result = json.loads(tool._run(queries=["WorkflowState"]))
        if result["results"]:
            assert "preview" in result["results"][0]
            assert len(result["results"][0]["preview"]) > 0


# ── WritePlanTool ─────────────────────────────────────────────────────────


class TestWritePlanTool:
    def _tool(self, tmp_path: Path, work_id: str = "wk-pl") -> WritePlanTool:
        return WritePlanTool(
            workspace_root=str(tmp_path),
            plan_dir=f".spine/artifacts/{work_id}/plan",
        )

    def _full_args(self) -> dict:
        return {
            "architecture_overview": "Three services: API, worker, DB.",
            "technology_choices": "Python 3.12, FastAPI, SQLite.",
            "module_structure": "- spine/api.py\n- spine/worker.py",
            "api_designs": "POST /work, GET /work/{id}",
            "implementation_order": "1. DB layer\n2. API layer\n3. Worker",
            "testing_strategy": "pytest tests/unit/, pytest tests/integration/",
        }

    def test_writes_plan_md(self, tmp_path):
        tool = self._tool(tmp_path)
        result = tool._run(**self._full_args())
        assert "plan.md written" in result
        plan = tmp_path / ".spine/artifacts/wk-pl/plan/plan.md"
        assert plan.exists()
        content = plan.read_text()
        assert "# Technical Plan" in content

    def test_all_sections_present(self, tmp_path):
        tool = self._tool(tmp_path)
        tool._run(**self._full_args())
        content = (tmp_path / ".spine/artifacts/wk-pl/plan/plan.md").read_text()
        for section in ["Architecture Overview", "Technology Choices",
                        "Module Structure", "API Designs", "Implementation Order",
                        "Testing Strategy"]:
            assert section in content

    def test_risks_section_optional(self, tmp_path):
        tool = self._tool(tmp_path)
        args = self._full_args()
        args["risks"] = "Risk: tight deadline."
        tool._run(**args)
        content = (tmp_path / ".spine/artifacts/wk-pl/plan/plan.md").read_text()
        assert "Risks" in content
        assert "tight deadline" in content

    def test_no_risks_section_when_empty(self, tmp_path):
        tool = self._tool(tmp_path)
        tool._run(**self._full_args())
        content = (tmp_path / ".spine/artifacts/wk-pl/plan/plan.md").read_text()
        assert "Risks" not in content

    def test_creates_directory(self, tmp_path):
        tool = self._tool(tmp_path, work_id="fresh")
        tool._run(**self._full_args())
        assert (tmp_path / ".spine/artifacts/fresh/plan/plan.md").exists()


# ── build_plan_agent_tools ────────────────────────────────────────────────


class TestBuildPlanAgentTools:
    def test_returns_three_tools(self, tmp_path):
        tools = build_plan_agent_tools(
            workspace_root=str(tmp_path),
            work_id="abc",
            description="desc",
            work_type="spec",
            prior_phase_dirs={},
        )
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert "read_prior_artifacts" in names
        assert "search_codebase" in names
        assert "write_plan" in names

    def test_prior_phase_dirs_passed_through(self, tmp_path):
        prior = {"specify": ".spine/artifacts/x/specify"}
        tools = build_plan_agent_tools(
            workspace_root=str(tmp_path),
            work_id="x",
            description="d",
            work_type="spec",
            prior_phase_dirs=prior,
        )
        read_tool = next(t for t in tools if t.name == "read_prior_artifacts")
        assert isinstance(read_tool, ReadPriorArtifactsTool)
        assert read_tool.prior_phase_dirs == prior

    def test_search_tool_has_workspace(self, tmp_path):
        tools = build_plan_agent_tools(
            workspace_root=str(tmp_path),
            work_id="y",
            description="d",
            work_type="quick",
            prior_phase_dirs={},
        )
        search_tool = next(t for t in tools if t.name == "search_codebase")
        assert isinstance(search_tool, SearchCodebaseTool)
        assert search_tool.workspace_root == str(tmp_path)
