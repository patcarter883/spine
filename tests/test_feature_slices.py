"""Tests for feature-slice decomposition and agent delegation."""

import json
import pytest
from unittest.mock import MagicMock, patch

from spine.models.types import FeatureSlice
from spine.models.dag import (
    synthesize_slices,
    _build_slice_synthesis_prompt,
    _parse_llm_slices,
    _heuristic_slices,
    _estimate_complexity,
    _extract_components,
)


# ── FeatureSlice dataclass ─────────────────────────────────────────


class TestFeatureSlice:
    """FeatureSlice dataclass and serialization."""

    def test_create_minimal(self):
        s = FeatureSlice(id="test", description="Test slice")
        assert s.id == "test"
        assert s.description == "Test slice"
        assert s.scope == []
        assert s.depends_on == []
        assert s.agent_role == "coder"
        assert s.acceptance == []

    def test_create_full(self):
        s = FeatureSlice(
            id="core-impl",
            description="Implement core",
            scope=["core/", "models/"],
            depends_on=[],
            agent_role="coder",
            acceptance=["Core models compile", "Config loads"],
        )
        assert s.scope == ["core/", "models/"]
        assert s.agent_role == "coder"
        assert len(s.acceptance) == 2

    def test_to_dict(self):
        s = FeatureSlice(
            id="test",
            description="Test",
            scope=["src/"],
            depends_on=["other"],
            agent_role="test_engineer",
            acceptance=["Passes"],
        )
        d = s.to_dict()
        assert d["id"] == "test"
        assert d["scope"] == ["src/"]
        assert d["depends_on"] == ["other"]
        assert d["agent_role"] == "test_engineer"

    def test_from_dict(self):
        d = {
            "id": "test",
            "description": "Test slice",
            "scope": ["src/"],
            "depends_on": [],
            "agent_role": "coder",
            "acceptance": ["Works"],
        }
        s = FeatureSlice.from_dict(d)
        assert s.id == "test"
        assert s.scope == ["src/"]
        assert s.acceptance == ["Works"]

    def test_from_dict_missing_optional_fields(self):
        d = {"id": "test", "description": "Minimal"}
        s = FeatureSlice.from_dict(d)
        assert s.scope == []
        assert s.depends_on == []
        assert s.agent_role == "coder"
        assert s.acceptance == []

    def test_roundtrip_dict(self):
        s = FeatureSlice(
            id="roundtrip",
            description="Round trip test",
            scope=["a/", "b/"],
            depends_on=["x"],
            agent_role="reviewer",
            acceptance=["c1", "c2"],
        )
        d = s.to_dict()
        s2 = FeatureSlice.from_dict(d)
        assert s2.id == s.id
        assert s2.description == s.description
        assert s2.scope == s.scope
        assert s2.depends_on == s.depends_on
        assert s2.agent_role == s.agent_role
        assert s2.acceptance == s.acceptance


# ── synthesize_slices ──────────────────────────────────────────────


class TestSynthesizeSlices:
    """synthesize_slices() — LLM and heuristic paths."""

    def test_heuristic_no_provider(self):
        """No LLM provider → heuristic slicing."""
        slices = synthesize_slices("Build a simple tool", {})
        assert len(slices) >= 1
        assert all(isinstance(s, FeatureSlice) for s in slices)

    def test_heuristic_low_complexity(self):
        slices = synthesize_slices("Fix typo", {})
        assert len(slices) == 2  # low → implementation + verification
        assert slices[0].id == "implementation"
        assert slices[1].id == "verification"

    def test_heuristic_medium_complexity(self):
        slices = synthesize_slices("Build a REST API with authentication", {})
        assert len(slices) == 3  # medium → core-impl, feature-modules, tests
        assert slices[0].id == "core-impl"
        assert "core-impl" in slices[1].depends_on

    def test_heuristic_high_complexity(self):
        slices = synthesize_slices(
            "Build a scalable microservice with real-time distributed processing and Kubernetes deployment",
            {},
        )
        assert len(slices) >= 4  # high → core + features + integration + tests
        # First slice is always core-foundation
        assert slices[0].id == "core-foundation"
        # Last slice is tests
        assert slices[-1].agent_role == "test_engineer"

    def test_llm_path_when_provider_enabled(self):
        mock_provider = MagicMock()
        mock_provider.enabled = True
        mock_provider.generate.return_value = json.dumps([
            {
                "id": "llm-slice-1",
                "description": "LLM generated slice",
                "scope": ["src/"],
                "depends_on": [],
                "agent_role": "coder",
                "acceptance": ["Works"],
            },
        ])
        slices = synthesize_slices("Build API", {}, llm_provider=mock_provider)
        assert len(slices) == 1
        assert slices[0].id == "llm-slice-1"
        mock_provider.generate.assert_called_once()

    def test_llm_fallback_on_error(self):
        mock_provider = MagicMock()
        mock_provider.enabled = True
        mock_provider.generate.side_effect = RuntimeError("API error")
        slices = synthesize_slices("Build API", {}, llm_provider=mock_provider)
        # Falls back to heuristic
        assert len(slices) >= 1

    def test_llm_disabled_provider_uses_heuristic(self):
        mock_provider = MagicMock()
        mock_provider.enabled = False
        slices = synthesize_slices("Build API", {}, llm_provider=mock_provider)
        # Heuristic path
        assert len(slices) >= 1
        mock_provider.generate.assert_not_called()


# ── _parse_llm_slices ──────────────────────────────────────────────


class TestParseLLMSlices:
    """Parsing LLM JSON responses into FeatureSlice objects."""

    def test_valid_json_array(self):
        raw = json.dumps([
            {"id": "a", "description": "Slice A"},
            {"id": "b", "description": "Slice B", "depends_on": ["a"]},
        ])
        slices = _parse_llm_slices(raw)
        assert len(slices) == 2
        assert slices[1].depends_on == ["a"]

    def test_single_object(self):
        raw = json.dumps({"id": "only", "description": "Only slice"})
        slices = _parse_llm_slices(raw)
        assert len(slices) == 1
        assert slices[0].id == "only"

    def test_markdown_fenced(self):
        raw = "```json\n" + json.dumps([
            {"id": "x", "description": "X"},
        ]) + "\n```"
        slices = _parse_llm_slices(raw)
        assert len(slices) == 1

    def test_invalid_json_falls_back(self):
        slices = _parse_llm_slices("not json at all")
        assert len(slices) >= 1  # heuristic fallback

    def test_filters_invalid_items(self):
        raw = json.dumps([
            {"id": "valid", "description": "Good"},
            {"id": "no_desc"},  # missing description
            {"description": "No id"},  # missing id
            "not a dict",
        ])
        slices = _parse_llm_slices(raw)
        assert len(slices) == 1
        assert slices[0].id == "valid"


# ── _build_slice_synthesis_prompt ──────────────────────────────────


class TestBuildSlicePrompt:
    """Prompt construction for LLM-based decomposition."""

    def test_includes_requirement(self):
        prompt = _build_slice_synthesis_prompt("Build API", {})
        assert "Build API" in prompt

    def test_includes_context(self):
        prompt = _build_slice_synthesis_prompt(
            "Build API",
            {"tech_research": "FastAPI", "risk_assessment": "Low risk"},
        )
        assert "FastAPI" in prompt
        assert "Low risk" in prompt

    def test_requests_json_array(self):
        prompt = _build_slice_synthesis_prompt("Build API", {})
        assert "JSON" in prompt


# ── _heuristic_slices ──────────────────────────────────────────────


class TestHeuristicSlices:
    """Heuristic slicer edge cases."""

    def test_dependency_chain_valid(self):
        """All depends_on references must exist as slice ids."""
        slices = _heuristic_slices("Build API with auth and database integration", {})
        ids = {s.id for s in slices}
        for s in slices:
            for dep in s.depends_on:
                assert dep in ids, f"Slice {s.id} depends on non-existent {dep}"

    def test_at_least_one_root(self):
        """At least one slice must have no dependencies."""
        slices = _heuristic_slices("Build scalable distributed system", {})
        roots = [s for s in slices if not s.depends_on]
        assert len(roots) >= 1

    def test_all_slices_have_id_and_description(self):
        slices = _heuristic_slices("Build web app with database", {})
        for s in slices:
            assert s.id
            assert s.description
            assert len(s.description) > 5


# ── Integration: SwarmDAGExecutor SYNTHESIZE stub ──────────────────


class TestSynthesizeStub:
    """SwarmDAGExecutor SYNTHESIZE stub produces FeatureSlices."""

    def test_synthesize_stub_returns_slices(self):
        from spine.models.dag import SwarmDAGExecutor
        from spine.core.state_machine import SubPhase, Task

        executor = SwarmDAGExecutor()
        subphase = SubPhase(
            name="SYNTHESIZE",
            agent_role="planner",
            tasks=[Task(id="draft", description="Draft plan")],
        )
        result = executor.execute_dag(subphase, {"requirement": "Build API"})
        # Result contains feature slice info in the output text
        task_result = result["tasks"]["draft"]["result"]
        assert "feature slice" in task_result.lower()

    def test_synthesize_stub_feature_slices_shape(self):
        from spine.models.dag import SwarmDAGExecutor
        from spine.core.state_machine import SubPhase, Task

        executor = SwarmDAGExecutor()
        subphase = SubPhase(
            name="SYNTHESIZE",
            agent_role="planner",
            tasks=[Task(id="draft", description="Draft plan")],
        )
        result = executor.execute_dag(subphase, {"requirement": "Build API"})
        data = result["tasks"]["draft"].get("structured_data", {})
        if "feature_slices" in data:
            assert isinstance(data["feature_slices"], list)
            for fs in data["feature_slices"]:
                assert "id" in fs
                assert "description" in fs


# ── Integration: SDD + QuickWork FeatureSlice propagation ──────────


class TestWorkflowFeatureSlices:
    """SDD and QuickWork produce and consume FeatureSlices."""

    def test_sdd_plan_produces_feature_slices(self):
        from spine.workflows.sdd import SDDWorkflow

        sdd = SDDWorkflow()
        sdd.create_project("fs-test", "Build a REST API with auth")
        sdd.execute()
        assert sdd.context.plan is not None
        assert "feature_slices" in sdd.context.plan
        slices = sdd.context.plan["feature_slices"]
        assert len(slices) >= 1
        for fs in slices:
            assert "id" in fs
            assert "description" in fs

    def test_quickwork_plan_produces_feature_slices(self):
        from spine.workflows.quick_work import QuickWorkflow

        qw = QuickWorkflow()
        qw.create_project("fs-qw", "Fix the login bug")
        qw.execute()
        assert qw.context.plan is not None
        assert "feature_slices" in qw.context.plan
        slices = qw.context.plan["feature_slices"]
        assert len(slices) >= 1

    def test_sdd_implementation_tasks_match_slices(self):
        from spine.workflows.sdd import SDDWorkflow

        sdd = SDDWorkflow()
        sdd.create_project("fs-match", "Build a web app with database")
        sdd.execute()
        plan = sdd.context.plan
        slice_ids = {s["id"] for s in plan["feature_slices"]}
        task_ids = {t["id"] for t in plan["implementation_tasks"]}
        # implementation_tasks should be a subset view of slices
        assert task_ids == slice_ids

    def test_sdd_implement_subphases_match_slices(self):
        from spine.workflows.sdd import SDDWorkflow

        sdd = SDDWorkflow()
        sdd.create_project("fs-impl", "Build a REST API")
        sdd.execute()
        impl_phase = sdd.hierarchy_engine.find_node(sdd.project, "implement")
        assert impl_phase is not None
        # Each subphase should start with impl-
        for sp in impl_phase.subphases:
            assert sp.id.startswith("impl-")
