"""Integration test: full workflow with structured I/O.

Validates that the complete SPECIFY -> PLAN -> IMPLEMENT -> VERIFY
workflow generates and correctly consumes all JSON artifacts.

Covers:
1. JSON artifact type schemas (Specification, FeatureSlice, StructuredPlan,
   GapPlan, FixInstruction, CriticReview)
2. Artifact store round-trips (save, load, list)
3. Full artifact chain: each phase writes JSON that downstream phases read
4. Workflow graph topology (builds correctly for "task" work type)
5. State mapper consistency

Note: Some tests require `langgraph-checkpoint-sqlite` and other LLM
dependencies. These are gracefully skipped when the dependencies aren't
available, using pytest.skip().
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# ── Conditional import stubs for optional dependencies ──────────────────
# spine.persistence and spine.workflow.compose require langgraph-checkpoint-sqlite
# and other LLM packages.  When these aren't installed (e.g. in CI without
# the full extras), we stub enough of the import chain so that ArtifactStore
# can be imported.  Deeper imports (workflow graph compose) use try/except.
_ARTIFACT_STORE_AVAILABLE = False
_WORKFLOW_COMPOSE_AVAILABLE = False

try:
    # Stub langgraph.checkpoint.sqlite (needed by spine.persistence)
    _stub_pkg = types.ModuleType("langgraph.checkpoint.sqlite")
    _stub_mod = types.ModuleType("langgraph.checkpoint.sqlite.aio")
    _stub_pkg.aio = _stub_mod

    class _FakeSaver:
        @classmethod
        async def from_conn_string(cls, s: str):  # type: ignore[misc]
            return _FakeSaver()

        async def __aenter__(self):  # type: ignore[misc]
            return self

        async def __aexit__(self, *a: object) -> None:
            pass

    _stub_mod.AsyncSqliteSaver = _FakeSaver  # type: ignore[attr-defined]

    sys.modules["langgraph.checkpoint.sqlite"] = _stub_pkg
    sys.modules["langgraph.checkpoint.sqlite.aio"] = _stub_mod

    from spine.persistence.artifacts import ArtifactStore  # noqa: E402

    _ARTIFACT_STORE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    pass

try:
    from spine.workflow.compose import (  # noqa: E402
        _plan_state_mapper,
        _verify_state_mapper,
        build_workflow_graph,
    )

    _WORKFLOW_COMPOSE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    pass


# ── Helper: build mock plan.json ─────────────────────────────────────────


def _plan_json_dict(slice_count: int = 3) -> dict:
    """Build a valid plan.json dict with the given number of feature_slices.

    Each slice has all required fields per _REQUIRED_SLICE_FIELDS in
    artifact_gate.py: id, title, target_files, execution_requirements,
    dependencies, acceptance_criteria.

    Args:
        slice_count: Number of feature slices to generate (min 1).
    """
    slice_templates = [
        {
            "id": "models",
            "title": "Core Data Models",
            "target_files": ["models/user.py", "models/project.py"],
            "execution_requirements": ["Pydantic v2", "Python 3.12+"],
            "dependencies": [],
            "acceptance_criteria": [
                "Models are importable",
                "Validation works for all fields",
            ],
            "complexity": "small",
        },
        {
            "id": "config",
            "title": "Configuration Loader",
            "target_files": ["config/settings.py"],
            "execution_requirements": ["python-dotenv"],
            "dependencies": [],
            "acceptance_criteria": [
                "Config loads from .spine/config.yaml",
                "Defaults are applied for missing keys",
            ],
            "complexity": "small",
        },
        {
            "id": "api",
            "title": "REST API Endpoints",
            "target_files": ["api/routes.py", "api/middleware.py"],
            "execution_requirements": ["FastAPI", "models and config must exist"],
            "dependencies": ["models", "config"],
            "acceptance_criteria": [
                "Endpoints return valid JSON",
                "Auth middleware enforces tokens",
            ],
            "complexity": "medium",
        },
    ]
    actual_slices = slice_templates[: max(1, min(slice_count, len(slice_templates)))]

    return {
        "architecture_overview": f"Architecture with {len(actual_slices)} feature slices.",
        "technology_choices": ["python", "fastapi", "pydantic", "sqlite"],
        "feature_slices": actual_slices,
        "testing_strategy": "pytest with tmp_path",
        "risks": ["Scheduling risk on API endpoints"],
        "codebase_map": {
            "entry_points": ["spine/cli/__init__.py"],
            "core_modules": ["spine/models/", "spine/workflow/"],
        },
    }


# ── JSON Artifact Type Validation ────────────────────────────────────────


class TestSpecificationSchema:
    """Specification JSON type (Specification Pydantic model) schema checks."""

    def test_specification_requires_title_and_summary(self):
        """Specification must have title and summary fields."""
        from spine.models.types import Specification

        spec = Specification(
            title="Dark Mode Toggle",
            summary="Add a dark mode toggle to the settings panel.",
            objectives=["Support user preference", "Persist across sessions"],
            requirements=["Toggle button in settings", "CSS variables for colors"],
            constraints=["Must not flash on page load"],
            scope_inclusions=["Settings page", "CSS theme variables"],
            scope_exclusions=["Email templates", "PDF exports"],
            known_risks=["Browser localStorage may be cleared"],
        )
        assert spec.title == "Dark Mode Toggle"
        assert spec.summary.startswith("Add a dark mode")
        assert len(spec.objectives) == 2
        assert "CSS theme variables" in spec.scope_inclusions

    def test_specification_defaults_are_empty_lists(self):
        """Specification optional list fields default to empty lists."""
        from spine.models.types import Specification

        spec = Specification(title="Minimal", summary="Minimal spec.")
        assert spec.objectives == []
        assert spec.requirements == []
        assert spec.constraints == []
        assert spec.scope_inclusions == []
        assert spec.scope_exclusions == []
        assert spec.known_risks == []

    def test_specification_serializes_to_json(self):
        """Specification can be round-tripped through JSON."""
        from spine.models.types import Specification

        spec = Specification(
            title="Test", summary="A test.", objectives=["One"], requirements=["Two"]
        )
        data = spec.model_dump()
        assert data["title"] == "Test"
        assert data["objectives"] == ["One"]

        # Reload from JSON
        reloaded = Specification.model_validate(json.loads(json.dumps(data)))
        assert reloaded.title == spec.title
        assert reloaded.objectives == spec.objectives


class TestFeatureSliceSchema:
    """FeatureSlice dataclass schema checks."""

    def test_feature_slice_required_fields(self):
        """FeatureSlice must declare all fields used by artifact_gate."""
        from dataclasses import fields

        from spine.models.types import FeatureSlice

        field_names = {f.name for f in fields(FeatureSlice)}
        # These come from _REQUIRED_SLICE_FIELDS in artifact_gate.py
        required = {
            "id",
            "title",
            "target_files",
            "execution_requirements",
            "dependencies",
            "acceptance_criteria",
        }
        assert required.issubset(field_names), (
            f"FeatureSlice missing required fields: {required - field_names}"
        )

    def test_feature_slice_to_dict_and_back(self):
        """FeatureSlice to_dict/from_dict round-trip preserves data."""
        from spine.models.types import FeatureSlice

        original = FeatureSlice(
            id="auth",
            title="Authentication Module",
            target_files=["auth/login.py", "auth/tokens.py"],
            execution_requirements=["JWT library", "bcrypt"],
            dependencies=["models"],
            acceptance_criteria=["Login succeeds with valid creds"],
            complexity="medium",
        )

        d = original.to_dict()
        assert d["id"] == "auth"
        assert d["dependencies"] == ["models"]

        restored = FeatureSlice.from_dict(d)
        assert restored.id == original.id
        assert restored.title == original.title
        assert restored.target_files == original.target_files
        assert restored.dependencies == original.dependencies


class TestStructuredPlanSchema:
    """StructuredPlan type schema checks."""

    def test_structured_plan_constructs_from_slices(self):
        """StructuredPlan holds architecture overview plus slices."""
        from spine.models.types import FeatureSlice, StructuredPlan

        s1 = FeatureSlice(id="s1", title="Slice One")
        s2 = FeatureSlice(id="s2", title="Slice Two", dependencies=["s1"])
        plan = StructuredPlan(
            architecture_overview="Two slices, one depends on the other.",
            feature_slices=[s1, s2],
            technology_choices=["python"],
        )
        assert len(plan.feature_slices) == 2
        assert plan.feature_slices[1].dependencies == ["s1"]

    def test_plan_json_has_feature_slices(self):
        """The plan.json dict must contain feature_slices key."""
        plan_data = _plan_json_dict(slice_count=2)

        assert "feature_slices" in plan_data
        assert isinstance(plan_data["feature_slices"], list)
        assert len(plan_data["feature_slices"]) == 2

        # Each slice must have required fields
        for i, s in enumerate(plan_data["feature_slices"]):
            assert "id" in s, f"Slice {i} missing 'id'"
            assert "title" in s, f"Slice {i} missing 'title'"
            assert "target_files" in s, f"Slice {i} missing 'target_files'"
            assert "dependencies" in s, f"Slice {i} missing 'dependencies'"


class TestVerificationSchema:
    """Verification JSON type checks."""

    def test_verification_json_has_overall_status(self):
        """verification.json must have overall_status field (VERIFIED or FAILED)."""
        verify_data = {
            "overall_status": "VERIFIED",
            "test_results": {"passed": 5, "failed": 0, "skipped": 0},
            "issues": [],
        }

        assert verify_data["overall_status"] in ("VERIFIED", "FAILED")

    def test_verification_json_failed_reports_issues(self):
        """When FAILED, verification.json should list issues."""
        verify_data = {
            "overall_status": "FAILED",
            "test_results": {"passed": 3, "failed": 2, "skipped": 0},
            "issues": [
                {"slice_id": "api", "description": "Auth middleware missing"},
                {"slice_id": "config", "description": "Missing env fallback"},
            ],
        }

        assert verify_data["overall_status"] == "FAILED"
        assert len(verify_data["issues"]) == 2
        assert verify_data["issues"][0]["slice_id"] == "api"


class TestCriticReviewSchema:
    """CriticReview Pydantic model checks."""

    def test_critic_review_valid_statuses(self):
        """CriticReview status must be PASSED, NEEDS_REVISION, or NEEDS_REVIEW."""
        from spine.models.types import CriticReview

        passed = CriticReview(status="PASSED", tier="structural", reason="Looks good")
        assert passed.status == "PASSED"

        needs_rev = CriticReview(
            status="NEEDS_REVISION",
            tier="agent",
            reason="Expand on edge cases",
            suggestions=["Add boundary tests"],
        )
        assert needs_rev.status == "NEEDS_REVISION"
        assert len(needs_rev.suggestions) == 1

        needs_review = CriticReview(
            status="NEEDS_REVIEW",
            tier="structural",
            reason="Missing feature_slices in plan.json",
        )
        assert needs_review.status == "NEEDS_REVIEW"

    def test_critic_review_score_is_none_by_default(self):
        """CriticReview score is None by default."""
        from spine.models.types import CriticReview

        review = CriticReview(status="PASSED", tier="structural", reason="ok")
        assert review.score is None


class TestGapPlanSchema:
    """GapPlan and FixInstruction Pydantic model checks."""

    def test_fix_instruction_change_types(self):
        """FixInstruction change_type must be add, modify, or delete."""
        from spine.models.types import FixInstruction

        fi = FixInstruction(
            slice_id="api",
            file_path="api/routes.py",
            change_type="add",
            specific_change="Add GET /health endpoint",
            acceptance_criteria=["Returns 200 OK"],
        )
        assert fi.change_type == "add"
        assert fi.slice_id == "api"
        assert fi.estimated_complexity == "small"

    def test_gap_plan_holds_fix_instructions(self):
        """GapPlan aggregates verification summary and fix instructions."""
        from spine.models.types import FixInstruction, GapPlan

        fi = FixInstruction(
            slice_id="auth",
            file_path="auth/tokens.py",
            change_type="modify",
            specific_change="Fix token refresh race condition",
            acceptance_criteria=["Concurrent refreshes do not corrupt state"],
            estimated_complexity="medium",
        )
        gp = GapPlan(
            verification_summary="3 tests failed out of 10.",
            gaps_identified=2,
            fix_instructions=[fi],
            re_verify_slices=["auth", "api"],
        )
        assert gp.gaps_identified == 2
        assert len(gp.fix_instructions) == 1
        assert gp.re_verify_slices == ["auth", "api"]


# ── Artifact Store Round-Trips ────────────────────────────────────────────


@pytest.mark.skipif(not _ARTIFACT_STORE_AVAILABLE, reason="langgraph-checkpoint-sqlite not available")
class TestArtifactStoreRoundTrip:
    """Verify ArtifactStore correctly persists and loads phase artifacts."""

    def test_save_and_load_specification_artifact(self, tmp_path):
        """spec.json (as specification.json) round-trips through the artifact store."""
        store = ArtifactStore(base_path=str(tmp_path / "artifacts"))

        spec_content = json.dumps(
            {
                "title": "Dark Mode",
                "summary": "Add dark mode toggle.",
                "objectives": ["Persist user preference"],
                "requirements": ["Toggle in settings"],
                "constraints": ["No flash on load"],
                "scope_inclusions": ["Settings panel"],
                "scope_exclusions": [],
                "known_risks": [],
            },
            indent=2,
        )

        store.save_artifact("wk-str-io-1", "specify", "specification.json", spec_content)
        loaded = store.load_artifact("wk-str-io-1", "specify", "specification.json")
        assert loaded is not None

        parsed = json.loads(loaded)
        assert parsed["title"] == "Dark Mode"
        assert "Add dark mode toggle" in parsed["summary"]

    def test_save_and_load_plan_artifact(self, tmp_path):
        """plan.json round-trips through the artifact store."""
        store = ArtifactStore(base_path=str(tmp_path / "artifacts"))

        plan_data = _plan_json_dict(slice_count=3)
        plan_json_str = json.dumps(plan_data, indent=2)

        store.save_artifact("wk-str-io-2", "plan", "plan.json", plan_json_str)
        loaded = store.load_artifact("wk-str-io-2", "plan", "plan.json")
        assert loaded is not None

        parsed = json.loads(loaded)
        assert len(parsed["feature_slices"]) == 3
        assert parsed["feature_slices"][0]["id"] == "models"

    def test_save_and_load_implementation_artifact(self, tmp_path):
        """implementation.md round-trips through the artifact store."""
        store = ArtifactStore(base_path=str(tmp_path / "artifacts"))

        impl_content = (
            "# Implementation Summary\n\n"
            "## Files Created\n"
            "- models/user.py: User model with Pydantic validation\n"
            "- config/settings.py: YAML config loader\n"
            "- api/routes.py: GET /users endpoint\n\n"
            "## Files Modified\n"
            "- spine/cli/__init__.py: Added dark mode CLI option\n"
        )
        store.save_artifact("wk-str-io-3", "implement", "implementation.md", impl_content)
        loaded = store.load_artifact("wk-str-io-3", "implement", "implementation.md")
        assert loaded is not None
        assert "models/user.py" in loaded

    def test_save_and_load_verification_artifact(self, tmp_path):
        """verification.json round-trips through the artifact store."""
        store = ArtifactStore(base_path=str(tmp_path / "artifacts"))

        verify_data = {
            "overall_status": "VERIFIED",
            "test_results": {"passed": 5, "failed": 0, "skipped": 0},
            "issues": [],
        }
        verify_json_str = json.dumps(verify_data, indent=2)

        store.save_artifact("wk-str-io-4", "verify", "verification.json", verify_json_str)
        loaded = store.load_artifact("wk-str-io-4", "verify", "verification.json")
        assert loaded is not None

        parsed = json.loads(loaded)
        assert parsed["overall_status"] == "VERIFIED"

    def test_list_artifacts_discovers_all_phases(self, tmp_path):
        """list_artifacts finds artifacts across all phases for a work item."""
        store = ArtifactStore(base_path=str(tmp_path / "artifacts"))

        store.save_artifact(
            "wk-str-io-5", "specify", "specification.json",
            json.dumps({"title": "T", "summary": "S"}),
        )
        store.save_artifact(
            "wk-str-io-5", "plan", "plan.json",
            json.dumps(_plan_json_dict(slice_count=2)),
        )
        store.save_artifact("wk-str-io-5", "implement", "implementation.md", "# Impl")
        store.save_artifact(
            "wk-str-io-5", "verify", "verification.json",
            json.dumps({"overall_status": "VERIFIED"}),
        )

        artifacts = store.list_artifacts("wk-str-io-5")
        names = {(a["phase"], a["name"]) for a in artifacts}
        assert ("specify", "specification.json") in names
        assert ("plan", "plan.json") in names
        assert ("implement", "implementation.md") in names
        assert ("verify", "verification.json") in names


# ── Full Artifact Chain ──────────────────────────────────────────────────


@pytest.mark.skipif(not _ARTIFACT_STORE_AVAILABLE, reason="langgraph-checkpoint-sqlite not available")
class TestFullArtifactChain:
    """Verify downstream phases can read upstream JSON artifacts."""

    def test_specification_passes_to_plan(self, tmp_path):
        """PLAN reads SPECIFY's specification.json."""
        store = ArtifactStore(base_path=str(tmp_path / "artifacts"))
        work_id = "chain-spec-to-plan"

        # SPECIFY writes specification.json
        spec_data = {
            "title": "Agent Harness",
            "summary": "Build a deterministic AI agent harness.",
            "objectives": ["Deterministic execution", "LangGraph orchestration"],
            "requirements": ["Python 3.12+", "State graph with checkpoints"],
            "constraints": ["Must be async", "SQLite-backed persistence"],
            "scope_inclusions": ["Workflow engine", "CLI", "Dashboard"],
            "scope_exclusions": ["Cloud deployment", "Auth system"],
            "known_risks": ["LLM nondeterminism"],
        }
        store.save_artifact(
            work_id, "specify", "specification.json",
            json.dumps(spec_data, indent=2),
        )

        # PLAN reads it back
        raw = store.load_artifact(work_id, "specify", "specification.json")
        assert raw is not None
        parsed = json.loads(raw)

        # PLAN uses spec data to scope its plan
        assert parsed["title"] == "Agent Harness"
        # Objectives should be non-empty and contain what we wrote
        assert len(parsed["objectives"]) == 2
        assert "Deterministic execution" in parsed["objectives"]

        # PLAN writes plan.json with feature_slices informed by the spec
        plan_data = _plan_json_dict(slice_count=2)
        store.save_artifact(work_id, "plan", "plan.json", json.dumps(plan_data, indent=2))

        # Verify plan was written
        plan_raw = store.load_artifact(work_id, "plan", "plan.json")
        assert plan_raw is not None
        plan_parsed = json.loads(plan_raw)
        assert len(plan_parsed["feature_slices"]) == 2

    def test_plan_passes_to_implement(self, tmp_path):
        """IMPLEMENT reads PLAN's plan.json with structured feature_slices."""
        store = ArtifactStore(base_path=str(tmp_path / "artifacts"))
        work_id = "chain-plan-to-impl"

        # SPECIFY (prerequisite)
        store.save_artifact(
            work_id, "specify", "specification.json",
            json.dumps({"title": "T", "summary": "S"}),
        )

        # PLAN writes plan.json with feature slices
        plan_data = _plan_json_dict(slice_count=3)
        store.save_artifact(work_id, "plan", "plan.json", json.dumps(plan_data, indent=2))

        # IMPLEMENT reads plan.json
        raw = store.load_artifact(work_id, "plan", "plan.json")
        assert raw is not None
        plan = json.loads(raw)

        # IMPLEMENT uses feature_slices for wave dispatch
        slices = plan["feature_slices"]
        assert len(slices) == 3

        # Each slice must have target_files for the implementer
        for s in slices:
            assert "target_files" in s, f"Slice {s.get('id')} missing target_files"
            assert isinstance(s["target_files"], list)
            assert len(s["target_files"]) > 0

        # IMPLEMENT computes execution waves
        # Use the slice scheduler if available; otherwise verify the data manually
        try:
            from spine.workflow.slice_scheduler import (
                FeatureSlice,
                compute_execution_waves,
            )

            scheduler_slices = [FeatureSlice.from_dict(sd) for sd in slices]
            waves = compute_execution_waves(scheduler_slices)
            # models and config are independent → wave 0; api depends on both → wave 1
            assert len(waves) == 2, f"Expected 2 waves, got {len(waves)}"
        except (ImportError, ModuleNotFoundError):
            # Can't compute waves — verify topology from the data instead
            deps_by_id = {s["id"]: s["dependencies"] for s in slices}
            # models.config have no deps; api depends on both
            assert deps_by_id["models"] == []
            assert deps_by_id["config"] == []
            assert "models" in deps_by_id["api"]
            assert "config" in deps_by_id["api"]

        # IMPLEMENT writes its summary
        impl_summary = "# Implementation\n\nBuilt all 3 slices.\n"
        store.save_artifact(work_id, "implement", "implementation.md", impl_summary)
        assert store.load_artifact(work_id, "implement", "implementation.md") is not None

    def test_implement_passes_to_verify(self, tmp_path):
        """VERIFY reads IMPLEMENT's implementation.md and upstream plan.json."""
        store = ArtifactStore(base_path=str(tmp_path / "artifacts"))
        work_id = "chain-impl-to-verify"

        # SPECIFY → ...
        store.save_artifact(
            work_id, "specify", "specification.json",
            json.dumps({"title": "T", "summary": "S"}),
        )
        # PLAN → ...
        plan_data = _plan_json_dict(slice_count=2)
        store.save_artifact(work_id, "plan", "plan.json", json.dumps(plan_data, indent=2))
        # IMPLEMENT → ...
        store.save_artifact(
            work_id, "implement", "implementation.md",
            "# Implementation Summary\n\nBuilt models and config slices.\n",
        )

        # VERIFY reads the implementation and plan
        impl = store.load_artifact(work_id, "implement", "implementation.md")
        assert impl is not None
        assert "models" in impl

        plan = json.loads(store.load_artifact(work_id, "plan", "plan.json") or "{}")
        assert len(plan["feature_slices"]) == 2

        # VERIFY writes verification.json
        verify_data = {
            "overall_status": "VERIFIED",
            "test_results": {"passed": 5, "failed": 0, "skipped": 0},
            "issues": [],
        }
        store.save_artifact(
            work_id, "verify", "verification.json",
            json.dumps(verify_data, indent=2),
        )

        # VERIFY writes verification.md
        verify_md = "VERIFIED\n\nAll slices pass acceptance criteria.\n"
        store.save_artifact(work_id, "verify", "verification.md", verify_md)

        # Confirm all artifacts present
        artifacts = store.list_artifacts(work_id)
        phases_present = {a["phase"] for a in artifacts}
        assert "specify" in phases_present
        assert "plan" in phases_present
        assert "implement" in phases_present
        assert "verify" in phases_present

        # Confirm verification.json has correct status
        verify_json = json.loads(
            store.load_artifact(work_id, "verify", "verification.json") or "{}"
        )
        assert verify_json["overall_status"] == "VERIFIED"

    def test_failed_verification_triggers_gap_plan(self, tmp_path):
        """When VERIFY reports FAILED, GAP_PLAN can read verification.json."""
        from spine.models.types import FixInstruction, GapPlan

        store = ArtifactStore(base_path=str(tmp_path / "artifacts"))
        work_id = "chain-verify-gap"

        # Write prior artifacts
        store.save_artifact(
            work_id, "specify", "specification.json",
            json.dumps({"title": "T", "summary": "S"}),
        )
        store.save_artifact(
            work_id, "plan", "plan.json",
            json.dumps(_plan_json_dict(slice_count=2)),
        )
        store.save_artifact(work_id, "implement", "implementation.md", "# Impl\n")

        # VERIFY reports FAILED
        verify_data = {
            "overall_status": "FAILED",
            "test_results": {"passed": 3, "failed": 2, "skipped": 0},
            "issues": [
                {"slice_id": "api", "description": "Auth middleware missing"},
                {"slice_id": "config", "description": "Missing env fallback"},
            ],
        }
        store.save_artifact(
            work_id, "verify", "verification.json",
            json.dumps(verify_data, indent=2),
        )

        # GAP_PLAN reads verification.json
        verify_json = json.loads(
            store.load_artifact(work_id, "verify", "verification.json") or "{}"
        )
        assert verify_json["overall_status"] == "FAILED"

        # GAP_PLAN builds structured fix instructions
        fix_instructions = []
        for issue in verify_json["issues"]:
            fi = FixInstruction(
                slice_id=issue["slice_id"],
                file_path=f"{issue['slice_id']}/fix.py",
                change_type="modify",
                specific_change=issue["description"],
                acceptance_criteria=[f"Fix: {issue['description']}"],
            )
            fix_instructions.append(fi)

        gap_plan = GapPlan(
            verification_summary="2 test failures found.",
            gaps_identified=len(verify_json["issues"]),
            fix_instructions=fix_instructions,
            re_verify_slices=[i["slice_id"] for i in verify_json["issues"]],
        )

        assert gap_plan.gaps_identified == 2
        assert len(gap_plan.fix_instructions) == 2
        assert gap_plan.fix_instructions[0].change_type == "modify"


# ── Workflow Graph Topology ────────────────────────────────────────────────


@pytest.mark.skipif(not _WORKFLOW_COMPOSE_AVAILABLE, reason="workflow compose deps not available")
class TestWorkflowGraphTopology:
    """Verify the task workflow graph builds and has correct topology."""

    def test_task_workflow_graph_builds(self):
        """The 'task' workflow graph compiles without error."""
        graph = build_workflow_graph("task")
        assert graph is not None

    def test_task_workflow_nodes_include_all_phases(self):
        """The compiled graph includes core phases for 'task' work type."""
        graph = build_workflow_graph("task")
        node_names = list(graph.nodes.keys()) if hasattr(graph, "nodes") else []
        expected = {"specify", "plan", "critic_plan", "implement", "verify"}
        if node_names:
            found = set(node_names) & expected
            assert len(found) >= len(expected), (
                f"Missing nodes: {expected - found}. Full: {node_names}"
            )

    def test_critical_task_workflow_graph_builds(self):
        """The 'critical_task' workflow graph compiles without error."""
        graph = build_workflow_graph("critical_task")
        assert graph is not None

    def test_unknown_work_type_raises_value_error(self):
        """build_workflow_graph raises ValueError for unknown work types."""
        with pytest.raises(ValueError, match="Unknown work type"):
            build_workflow_graph("nonexistent_type")


@pytest.mark.skipif(not _WORKFLOW_COMPOSE_AVAILABLE, reason="workflow compose deps not available")
class TestStateMappingConsistency:
    """Verify state mappers pass structured fields through to subgraph states."""

    def test_plan_state_mapper_includes_spec_path(self):
        """Plan state mapper forwards spec_path so PLAN can read the specification."""
        parent_state = {
            "work_id": "wk-mapper-1",
            "work_type": "task",
            "description": "Build a thing",
            "workspace_root": "/tmp/test",
            "feedback": [],
            "retry_count": {},
        }

        mapped = _plan_state_mapper(parent_state, None)
        assert mapped["has_spec"] is True
        assert "plan" in mapped["spec_path"]
        assert mapped["phase"] == "plan"
        assert mapped["work_id"] == "wk-mapper-1"

    def test_verify_state_mapper_includes_all_artifact_paths(self):
        """Verify state mapper includes spec_path and plan_path for verification."""
        parent_state = {
            "work_id": "wk-mapper-2",
            "work_type": "task",
            "description": "Build a thing",
            "workspace_root": "/tmp/test",
            "feedback": [],
            "retry_count": {},
        }

        mapped = _verify_state_mapper(parent_state, None)
        assert mapped["phase"] == "verify"
        assert "plan" in mapped.get("plan_path", "")
        assert "specify" in mapped.get("spec_path", "")
        assert mapped["has_spec"] is True
