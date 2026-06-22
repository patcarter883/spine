"""Unit tests for specify_tools and plan_tools."""

from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spine.agents.specify_tools import (
    ReadWorkContextTool,
    WriteSpecificationTool,
    build_specify_orchestrator_tools,
)
from spine.agents.plan_tools import (
    ReadPriorArtifactsTool,
    SearchCodebaseTool,
    build_plan_agent_tools,
)
from spine.agents.artifacts import artifact_path
from spine.agents.plan_agent import _resolve_prior_phase_dirs


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
            work_type="task",
            description="Build a widget factory.",
            feedback=feedback or [],
            spec_dir=spec_dir,
        )

    def test_returns_basic_context(self, tmp_path):
        tool = self._tool(tmp_path)
        result = json.loads(tool._run())
        assert result["description"] == "Build a widget factory."
        assert result["work_id"] == "wk-spec"
        assert result["work_type"] == "task"
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
            "title": "Widget Factory System",
            "summary": "Build a widget factory system that creates widgets quickly.",
            "objectives": ["Ship widget creation API", "Hit P99 <100ms"],
            "requirements": [
                "FR1: Create widgets via POST /widgets",
                "NFR1: P99 latency <100ms",
            ],
            "constraints": ["Must run on existing infra"],
            "scope_inclusions": ["API service", "Storage layer"],
            "scope_exclusions": ["UI work", "Billing"],
            "known_risks": ["Cold-start latency on cold storage"],
        }

    def test_writes_specification_md(self, tmp_path):
        tool = self._tool(tmp_path)
        result = tool._run(**self._full_args())
        assert "specification.md" in result
        assert "specification.json" in result
        spec = tmp_path / ".spine/artifacts/wk-s1/specify/specification.md"
        assert spec.exists()
        content = spec.read_text()
        assert "# Widget Factory System" in content
        assert "factory system" in content

    def test_writes_specification_json(self, tmp_path):
        tool = self._tool(tmp_path)
        tool._run(**self._full_args())
        json_path = tmp_path / ".spine/artifacts/wk-s1/specify/specification.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["title"] == "Widget Factory System"
        assert data["summary"].startswith("Build a widget")
        assert "FR1: Create widgets via POST /widgets" in data["requirements"]
        assert "API service" in data["scope_inclusions"]

    def test_all_sections_present(self, tmp_path):
        tool = self._tool(tmp_path)
        tool._run(**self._full_args())
        content = (tmp_path / ".spine/artifacts/wk-s1/specify/specification.md").read_text()
        for section in [
            "Summary",
            "Objectives",
            "Requirements",
            "Constraints",
            "Scope — Inclusions",
            "Scope — Exclusions",
            "Known Risks",
        ]:
            assert section in content

    def test_optional_sections_omitted_when_empty(self, tmp_path):
        tool = self._tool(tmp_path)
        # Only title, summary, requirements — the rest left empty
        tool._run(
            title="Minimal Spec",
            summary="A minimal spec.",
            requirements=["FR1: Do the thing"],
        )
        content = (tmp_path / ".spine/artifacts/wk-s1/specify/specification.md").read_text()
        assert "# Minimal Spec" in content
        assert "## Summary" in content
        assert "## Requirements" in content
        # Optional sections must not appear when their lists are empty
        assert "## Objectives" not in content
        assert "## Constraints" not in content
        assert "Scope — Inclusions" not in content
        assert "Known Risks" not in content

    def test_requirements_field_is_required(self, tmp_path):
        """Pydantic schema must reject missing requirements via the tool's input model."""
        from spine.agents.specify_tools import _WriteSpecificationInput
        import pytest

        with pytest.raises(Exception):
            _WriteSpecificationInput(title="t", summary="s")  # type: ignore[call-arg]
        with pytest.raises(Exception):
            _WriteSpecificationInput(title="t", summary="s", requirements=[])

    def test_creates_directory(self, tmp_path):
        tool = self._tool(tmp_path, work_id="brand-new")
        tool._run(**self._full_args())
        assert (tmp_path / ".spine/artifacts/brand-new/specify/specification.md").exists()
        assert (tmp_path / ".spine/artifacts/brand-new/specify/specification.json").exists()

    # ── Lever D: empty-scope pre-check for non-trivial work ───────────────

    def _tool_with_description(
        self, tmp_path: Path, description: str, work_id: str = "wk-d"
    ) -> WriteSpecificationTool:
        return WriteSpecificationTool(
            workspace_root=str(tmp_path),
            spec_dir=f".spine/artifacts/{work_id}/specify",
            work_description=description,
        )

    def test_rejects_empty_scope_for_nontrivial_description(self, tmp_path):
        """Long descriptions or descriptions containing implementation keywords
        ('implement', 'refactor', etc.) trigger the empty-scope guard."""
        tool = self._tool_with_description(
            tmp_path,
            description="Implement a project onboarding engine that ingests "
            "repository metadata, builds a knowledge map, and routes new "
            "engineers through a guided tour of the codebase.",
            work_id="wk-nontrivial",
        )
        result = tool._run(
            title="Onboarding",
            summary="Build it.",
            requirements=["FR1: ingest repo"],
            scope_inclusions=[],
            scope_exclusions=[],
        )
        assert result.startswith("VALIDATION_ERROR")
        assert "scope_inclusions" in result
        assert "scope_exclusions" in result
        assert not (
            tmp_path / ".spine/artifacts/wk-nontrivial/specify/specification.md"
        ).exists()

    def test_rejects_when_only_one_scope_field_is_empty(self, tmp_path):
        tool = self._tool_with_description(
            tmp_path,
            description="Refactor the auth middleware to meet new compliance.",
            work_id="wk-half",
        )
        result = tool._run(
            title="t",
            summary="s",
            requirements=["FR1"],
            scope_inclusions=["spine/auth"],
            scope_exclusions=[],
        )
        assert result.startswith("VALIDATION_ERROR")
        assert "scope_exclusions" in result
        assert "scope_inclusions" not in result.split("MUST be non-empty: ")[1]

    def test_allows_empty_scope_for_trivial_description(self, tmp_path):
        """Short descriptions with no implementation verbs are exempt from
        the empty-scope guard (proportionality rule)."""
        tool = self._tool_with_description(
            tmp_path,
            description="Add a --verbose flag to spine CLI",
            work_id="wk-trivial",
        )
        result = tool._run(
            title="Verbose Flag",
            summary="Add a --verbose flag.",
            requirements=["FR1: --verbose echoes config path"],
            scope_inclusions=[],
            scope_exclusions=[],
        )
        assert not result.startswith("VALIDATION_ERROR")
        assert (
            tmp_path / ".spine/artifacts/wk-trivial/specify/specification.md"
        ).exists()

    def test_no_description_does_not_trigger_guard(self, tmp_path):
        """A WriteSpecificationTool built without an injected work_description
        (e.g. legacy callers) must not spuriously reject empty scope."""
        tool = WriteSpecificationTool(
            workspace_root=str(tmp_path),
            spec_dir=".spine/artifacts/wk-bare/specify",
        )
        result = tool._run(
            title="t",
            summary="s",
            requirements=["FR1"],
            scope_inclusions=[],
            scope_exclusions=[],
        )
        assert not result.startswith("VALIDATION_ERROR")

    def test_factory_threads_description_into_write_tool(self, tmp_path):
        tools = build_specify_orchestrator_tools(
            workspace_root=str(tmp_path),
            work_id="wf",
            description="Implement onboarding engine",
            work_type="task",
        )
        write_tool = next(t for t in tools if t.name == "write_specification")
        assert isinstance(write_tool, WriteSpecificationTool)
        assert write_tool.work_description == "Implement onboarding engine"


# ── build_specify_orchestrator_tools ─────────────────────────────────────


class TestBuildSpecifyOrchestratorTools:
    def test_returns_two_tools(self, tmp_path):
        tools = build_specify_orchestrator_tools(
            workspace_root=str(tmp_path),
            work_id="abc",
            description="desc",
            work_type="task",
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
            work_type="task",
            feedback=["fix this"],
        )
        read_tool = next(t for t in tools if t.name == "read_work_context")
        assert isinstance(read_tool, ReadWorkContextTool)
        assert read_tool.feedback == ["fix this"]

    def test_excludes_read_work_context_when_flag_false(self, tmp_path):
        # Eager-injection mode (trace 019ec965): the work context is inlined
        # into the prompt, so read_work_context is dropped from the surface.
        tools = build_specify_orchestrator_tools(
            workspace_root=str(tmp_path),
            work_id="x",
            description="d",
            work_type="task",
            include_read_work_context=False,
        )
        names = {t.name for t in tools}
        assert names == {"write_specification"}


# ── Eager work-context injection (load_prior_spec / build_work_context_block) ─


class TestEagerWorkContext:
    def test_load_prior_spec_absent(self, tmp_path):
        from spine.agents.specify_tools import load_prior_spec

        assert load_prior_spec(str(tmp_path), "no-such-work") == ""

    def test_load_prior_spec_present(self, tmp_path):
        from spine.agents.specify_tools import load_prior_spec

        work_id = "wk-rework"
        spec_dir = tmp_path / ".spine" / "artifacts" / work_id / "specify"
        spec_dir.mkdir(parents=True)
        (spec_dir / "specification.md").write_text("# Prior\nbody")
        assert "Prior" in load_prior_spec(str(tmp_path), work_id)

    def test_block_empty_elides(self):
        from spine.agents.specify_tools import build_work_context_block

        assert build_work_context_block("") == ""
        assert build_work_context_block("   \n  ") == ""

    def test_block_wraps_prior_spec(self):
        from spine.agents.specify_tools import build_work_context_block

        block = build_work_context_block("# Prior Spec\nstuff")
        assert "Prior Specification" in block
        assert "# Prior Spec" in block


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
            work_type="task",
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
        result = json.loads(
            tool._run(
                queries=["pass"],
                file_patterns=["spine/agents/*.py"],
            )
        )
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

    def test_relative_workspace_root_still_finds_matches(self, tmp_path, monkeypatch):
        """Regression for trace 019e6974: when workspace_root='.' (the default
        in state.get('workspace_root', '.')), rglob and rg returned paths
        with different leading prefixes ('spine/...' vs './spine/...') and
        the membership check dropped every hit silently."""
        self._setup_files(tmp_path)
        # Simulate the workflow's default: workspace_root="." with cwd set
        # to the project root.
        monkeypatch.chdir(tmp_path)
        tool = SearchCodebaseTool(workspace_root=".")
        result = json.loads(tool._run(queries=["build_phase_agent"]))
        assert result["total_files_found"] >= 1, (
            f"Relative workspace_root should still find matches, got "
            f"total_files_found={result['total_files_found']}"
        )

    def test_empty_workspace_root_falls_back_to_cwd(self, tmp_path, monkeypatch):
        """workspace_root='' (also seen on bare SearchCodebaseTool()) should
        not silently search nothing."""
        self._setup_files(tmp_path)
        monkeypatch.chdir(tmp_path)
        tool = SearchCodebaseTool(workspace_root="")
        result = json.loads(tool._run(queries=["build_phase_agent"]))
        assert result["total_files_found"] >= 1


# ── SearchCodebaseTool arg coercion (trace 019e72f5) ──────────────────────


class TestSearchCodebaseArgCoercion:
    """Regression for trace 019e72f5: local models serialise `queries` as a
    JSON-string or spill their tool-call XML into the value, and the bare
    `list[str]` schema 400'd with an unhelpful `list_type` error. The
    before-validator now coerces recoverable strings and rejects spilled
    markup with a teaching message — mirroring the codebase_query hardening."""

    def test_json_string_queries_coerced_to_list(self):
        from spine.agents.plan_tools import _SearchCodebaseInput

        m = _SearchCodebaseInput.model_validate({"queries": '["WorkflowState", "submit_work"]'})
        assert m.queries == ["WorkflowState", "submit_work"]

    def test_bare_string_query_wrapped_in_list(self):
        from spine.agents.plan_tools import _SearchCodebaseInput

        # Not valid JSON → treated as a single search term, not rejected.
        m = _SearchCodebaseInput.model_validate({"queries": "build_phase_agent"})
        assert m.queries == ["build_phase_agent"]

    def test_clean_list_passes_through_unchanged(self):
        from spine.agents.plan_tools import _SearchCodebaseInput

        m = _SearchCodebaseInput.model_validate({"queries": ["a", "b"], "file_patterns": ["*.py"]})
        assert m.queries == ["a", "b"]
        assert m.file_patterns == ["*.py"]

    def test_file_patterns_json_string_coerced(self):
        from spine.agents.plan_tools import _SearchCodebaseInput

        m = _SearchCodebaseInput.model_validate(
            {"queries": ["x"], "file_patterns": '["spine/agents/*.py"]'}
        )
        assert m.file_patterns == ["spine/agents/*.py"]

    def test_markup_spill_rejected_with_teaching_message(self):
        import pytest
        from pydantic import ValidationError

        from spine.agents.plan_tools import _SearchCodebaseInput

        # The exact shape from trace 019e72f5: the model fused the
        # file_patterns arg into the queries string and spilled its
        # <arg_value> tool-call envelope.
        bad = '["Click command group", "initialization\n<arg_key>file_patterns</arg_key>\n<arg_value>["*.py"]'
        with pytest.raises(ValidationError, match="tool-call markup"):
            _SearchCodebaseInput.model_validate({"queries": bad})

    def test_markup_in_list_element_rejected(self):
        import pytest
        from pydantic import ValidationError

        from spine.agents.plan_tools import _SearchCodebaseInput

        with pytest.raises(ValidationError, match="tool-call markup"):
            _SearchCodebaseInput.model_validate({"queries": ["ok", "bad</arg_value>"]})

    def test_invoke_with_json_string_queries_succeeds(self, tmp_path):
        """End-to-end via the langchain tool path that 400'd in the trace:
        a JSON-string `queries` arg now validates and runs."""
        (tmp_path / "mod.py").write_text("def build_phase_agent():\n    pass\n")
        tool = SearchCodebaseTool(workspace_root=str(tmp_path))
        result = json.loads(tool.invoke({"queries": '["build_phase_agent"]'}))
        assert result["queries_run"] == ["build_phase_agent"]


# ── WritePlanTool ─────────────────────────────────────────────────────────


class TestStructuredWritePlanTool:
    def _tool(self, tmp_path: Path, work_id: str = "wk-pl"):
        from spine.agents.plan_tools import StructuredWritePlanTool

        return StructuredWritePlanTool(
            workspace_root=str(tmp_path),
            plan_dir=f".spine/artifacts/{work_id}/plan",
        )

    def _full_args(self) -> dict:
        return {
            "architecture_overview": "Three services: API, worker, DB.",
            "technology_choices": ["Python 3.12", "FastAPI", "SQLite"],
            "feature_slices": [
                {
                    "id": "db-layer",
                    "title": "Database layer",
                    "target_files": ["spine/db.py"],
                    "execution_requirements": "Create SQLAlchemy models and migrations.",
                    "dependencies": [],
                    "acceptance_criteria": ["Tables created", "Migrations apply cleanly"],
                    "complexity": "small",
                },
                {
                    "id": "api-layer",
                    "title": "API layer",
                    "target_files": ["spine/api.py"],
                    "execution_requirements": "FastAPI routes for CRUD.",
                    "dependencies": ["db-layer"],
                    "acceptance_criteria": ["POST /work returns 201"],
                    "complexity": "medium",
                },
            ],
            "testing_strategy": "pytest tests/unit/, pytest tests/integration/",
            "risks": ["Tight deadline", "Cold-start latency"],
        }

    def test_writes_plan_md_and_json(self, tmp_path):
        tool = self._tool(tmp_path)
        result = tool._run(**self._full_args())
        assert "plan.md" in result
        assert "plan.json" in result
        plan_md = tmp_path / ".spine/artifacts/wk-pl/plan/plan.md"
        plan_json = tmp_path / ".spine/artifacts/wk-pl/plan/plan.json"
        assert plan_md.exists()
        assert plan_json.exists()
        content = plan_md.read_text()
        assert "# Technical Plan" in content

    def test_lists_render_as_bullets(self, tmp_path):
        tool = self._tool(tmp_path)
        tool._run(**self._full_args())
        content = (tmp_path / ".spine/artifacts/wk-pl/plan/plan.md").read_text()
        # technology_choices and risks render as bullets, one per item
        assert "- Python 3.12" in content
        assert "- FastAPI" in content
        assert "- SQLite" in content
        assert "- Tight deadline" in content
        assert "- Cold-start latency" in content

    def test_json_lists_are_arrays_not_strings(self, tmp_path):
        tool = self._tool(tmp_path)
        tool._run(**self._full_args())
        data = json.loads(
            (tmp_path / ".spine/artifacts/wk-pl/plan/plan.json").read_text()
        )
        # The downstream consumer expects technology_choices and risks as arrays
        assert isinstance(data["technology_choices"], list)
        assert data["technology_choices"] == ["Python 3.12", "FastAPI", "SQLite"]
        assert isinstance(data["risks"], list)
        assert data["risks"] == ["Tight deadline", "Cold-start latency"]

    def test_missing_acceptance_criteria_rejected_in_run(self, tmp_path):
        """Completeness is validated in _run (not the Pydantic schema) so a
        partial retry can pass the schema and receive a clear, actionable
        message instead of a raw nested ValidationError. The schema is kept
        lenient (fields optional) to enable incremental patching."""
        tool = self._tool(tmp_path)
        out = tool._run(
            architecture_overview="x",
            feature_slices=[
                {
                    "id": "s",
                    "title": "t",
                    "execution_requirements": "do it",
                    "acceptance_criteria": [],
                }
            ],
            testing_strategy="y",
        )
        assert out.startswith("VALIDATION_ERROR")
        assert "acceptance_criteria" in out

    def test_optional_sections_omitted_when_empty(self, tmp_path):
        tool = self._tool(tmp_path)
        args = self._full_args()
        args["technology_choices"] = []
        args["risks"] = []
        tool._run(**args)
        content = (tmp_path / ".spine/artifacts/wk-pl/plan/plan.md").read_text()
        assert "## Technology Choices" not in content
        assert "Risks & Open Questions" not in content

    def test_creates_directory(self, tmp_path):
        tool = self._tool(tmp_path, work_id="fresh")
        tool._run(**self._full_args())
        assert (tmp_path / ".spine/artifacts/fresh/plan/plan.md").exists()
        assert (tmp_path / ".spine/artifacts/fresh/plan/plan.json").exists()

    # ── Lever B: structural pre-validation inside the write tool ──────────

    def test_rejects_unknown_dependency(self, tmp_path):
        """A slice depending on a non-existent ID must be rejected BEFORE
        the file write, so the synthesizer's tool loop can self-correct
        without round-tripping through the critic."""
        tool = self._tool(tmp_path, work_id="dep-bad")
        args = self._full_args()
        args["feature_slices"] = [
            {
                "id": "only-slice",
                "title": "Only",
                "target_files": ["x.py"],
                "execution_requirements": "do",
                "dependencies": ["does-not-exist"],
                "acceptance_criteria": ["it works"],
                "complexity": "small",
            }
        ]
        result = tool._run(**args)
        assert result.startswith("VALIDATION_ERROR")
        assert "does-not-exist" in result
        assert not (tmp_path / ".spine/artifacts/dep-bad/plan/plan.md").exists()
        assert not (tmp_path / ".spine/artifacts/dep-bad/plan/plan.json").exists()

    def test_rejects_cycle(self, tmp_path):
        tool = self._tool(tmp_path, work_id="cyc")
        args = self._full_args()
        args["feature_slices"] = [
            {
                "id": "a",
                "title": "A",
                "target_files": ["a.py"],
                "execution_requirements": "do",
                "dependencies": ["b"],
                "acceptance_criteria": ["ok"],
                "complexity": "small",
            },
            {
                "id": "b",
                "title": "B",
                "target_files": ["b.py"],
                "execution_requirements": "do",
                "dependencies": ["a"],
                "acceptance_criteria": ["ok"],
                "complexity": "small",
            },
        ]
        result = tool._run(**args)
        assert result.startswith("VALIDATION_ERROR")
        assert "cycle" in result.lower()
        assert not (tmp_path / ".spine/artifacts/cyc/plan/plan.md").exists()

    def test_rejects_duplicate_ids(self, tmp_path):
        tool = self._tool(tmp_path, work_id="dup")
        args = self._full_args()
        args["feature_slices"] = [
            {
                "id": "same",
                "title": "First",
                "target_files": ["a.py"],
                "execution_requirements": "do",
                "dependencies": [],
                "acceptance_criteria": ["ok"],
                "complexity": "small",
            },
            {
                "id": "same",
                "title": "Second",
                "target_files": ["b.py"],
                "execution_requirements": "do",
                "dependencies": [],
                "acceptance_criteria": ["ok"],
                "complexity": "small",
            },
        ]
        result = tool._run(**args)
        assert result.startswith("VALIDATION_ERROR")
        assert "same" in result
        assert not (tmp_path / ".spine/artifacts/dup/plan/plan.md").exists()

    def test_writes_when_valid_dag(self, tmp_path):
        """Happy path: a clean DAG passes pre-validation and writes."""
        tool = self._tool(tmp_path, work_id="ok")
        result = tool._run(**self._full_args())
        assert not result.startswith("VALIDATION_ERROR")
        assert (tmp_path / ".spine/artifacts/ok/plan/plan.md").exists()
        assert (tmp_path / ".spine/artifacts/ok/plan/plan.json").exists()


# ── build_plan_agent_tools ────────────────────────────────────────────────


class TestBuildPlanAgentTools:
    def test_returns_three_tools(self, tmp_path):
        tools = build_plan_agent_tools(
            workspace_root=str(tmp_path),
            work_id="abc",
            description="desc",
            work_type="task",
            prior_phase_dirs={},
        )
        assert len(tools) == 3
        names = {t.name for t in tools}
        assert "read_prior_artifacts" in names
        assert "search_codebase" in names
        assert "write_structured_plan" in names
        assert "write_plan" not in names

    def test_prior_phase_dirs_passed_through(self, tmp_path):
        prior = {"specify": ".spine/artifacts/x/specify"}
        tools = build_plan_agent_tools(
            workspace_root=str(tmp_path),
            work_id="x",
            description="d",
            work_type="task",
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
            work_type="task",
            prior_phase_dirs={},
        )
        search_tool = next(t for t in tools if t.name == "search_codebase")
        assert isinstance(search_tool, SearchCodebaseTool)
        assert search_tool.workspace_root == str(tmp_path)


# ── _resolve_prior_phase_dirs (disk is source of truth) ───────────────────


class TestResolvePriorPhaseDirs:
    """prior-phase dirs must be discovered from disk, not just state['artifacts'].

    Regression: PlanSubgraphState drops the 'artifacts' channel and the
    standalone plan path never sets it, so read_prior_artifacts answered
    "No prior artifacts found" while the spec sat on disk.
    """

    def _spec_on_disk(self, tmp_path: Path, work_id: str) -> None:
        specify = tmp_path / artifact_path(work_id, "specify")
        specify.mkdir(parents=True)
        (specify / "specification.md").write_text("# spec", encoding="utf-8")
        (specify / "specification.md.meta.json").write_text("{}", encoding="utf-8")

    def test_discovers_specify_from_disk_when_state_empty(self, tmp_path):
        work_id = "wk-r1"
        self._spec_on_disk(tmp_path, work_id)
        state = {
            "workspace_root": str(tmp_path),
            "phase": "plan",
            "artifacts": {},  # the dropped/empty channel
        }
        dirs = _resolve_prior_phase_dirs(state, work_id)
        assert dirs.get("specify") == artifact_path(work_id, "specify")

    def test_excludes_current_phase(self, tmp_path):
        work_id = "wk-r2"
        plan = tmp_path / artifact_path(work_id, "plan")
        plan.mkdir(parents=True)
        (plan / "plan.json").write_text("{}", encoding="utf-8")
        state = {"workspace_root": str(tmp_path), "phase": "plan", "artifacts": {}}
        assert "plan" not in _resolve_prior_phase_dirs(state, work_id)

    def test_ignores_meta_only_dir(self, tmp_path):
        work_id = "wk-r3"
        d = tmp_path / artifact_path(work_id, "specify")
        d.mkdir(parents=True)
        (d / "specification.md.meta.json").write_text("{}", encoding="utf-8")
        state = {"workspace_root": str(tmp_path), "phase": "plan", "artifacts": {}}
        assert "specify" not in _resolve_prior_phase_dirs(state, work_id)

    def test_state_declared_phase_still_honoured(self, tmp_path):
        work_id = "wk-r4"
        state = {
            "workspace_root": str(tmp_path),
            "phase": "plan",
            "artifacts": {"specify": {"specification.md": "# spec"}},
        }
        dirs = _resolve_prior_phase_dirs(state, work_id)
        assert dirs.get("specify") == artifact_path(work_id, "specify")


# ── SearchCodebaseTool: forgiving term matching (meet the model's phrasing) ──


class TestSearchCodebaseTokenization:
    """A query is an OR of its whitespace-separated terms, so a model that
    crams several symbols into one query string still gets results instead of
    grepping the whole phrase as one regex and matching nothing.
    """

    def _tool(self, tmp_path):
        return SearchCodebaseTool(workspace_root=str(tmp_path))

    def test_crammed_multiword_query_matches_any_term(self, tmp_path):
        (tmp_path / "a.py").write_text(
            "class UIApi:\n    def get_providers(self):\n        return []\n"
        )
        (tmp_path / "b.py").write_text("def add_llm_provider(name):\n    return name\n")
        (tmp_path / "c.py").write_text("x = 1\n")
        out = json.loads(
            self._tool(tmp_path)._run(queries=["UIApi get_providers add_llm_provider"])
        )
        files = {r["file"] for r in out["results"]}
        assert "a.py" in files
        assert "b.py" in files
        assert "c.py" not in files

    def test_ranks_files_by_number_of_queries_matched(self, tmp_path):
        (tmp_path / "hi.py").write_text("UIApi and set_phase_provider both here\n")
        (tmp_path / "lo.py").write_text("only UIApi here\n")
        out = json.loads(
            self._tool(tmp_path)._run(queries=["UIApi", "set_phase_provider"])
        )
        scores = {r["file"]: r["score"] for r in out["results"]}
        assert scores["hi.py"] > scores["lo.py"]

    def test_dotted_filename_term_matched_as_fixed_string(self, tmp_path):
        (tmp_path / "uses.py").write_text("# see config.reference.yaml for an example\n")
        out = json.loads(self._tool(tmp_path)._run(queries=["config.reference.yaml"]))
        assert "uses.py" in {r["file"] for r in out["results"]}
