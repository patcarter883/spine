"""Tests for the slice scheduler's topological sorting and validation.

Covers all edge cases required by the write-scheduler-tests slice:
- Linear dependencies (A→B→C)
- Parallel slices (no deps)
- Diamond pattern (A→B,C→D)
- Cycle detection (A→B→A)
- Missing dependency references
- Duplicate slice IDs
- Empty slice list
- Self-dependency
- Complex graphs (10+ slices)
- Validation of empty/required fields
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.models.types import FeatureSlice


def _import_slice_scheduler():
    """Import spine.workflow.slice_scheduler directly, avoiding __init__ side-effects.

    The project convention (see test_artifact_gate_quality.py) is to bypass
    spine/workflow/__init__.py which pulls in heavy dependencies (LangGraph,
    deep agents, etc.) by loading the target module file directly.
    """
    module_path = (
        Path(__file__).resolve().parent.parent.parent / "spine" / "workflow" / "slice_scheduler.py"
    )
    if not module_path.exists():
        pytest.skip(
            f"slice_scheduler.py not found at {module_path} "
            "(dependency slice 'create-slice-scheduler' not yet implemented)"
        )
    spec = importlib.util.spec_from_file_location(
        "spine.workflow.slice_scheduler",
        module_path,
    )
    assert spec is not None, f"Could not locate slice_scheduler at {module_path}"
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("spine.workflow.slice_scheduler", mod)
    spec.loader.exec_module(mod)
    return mod


# Load once at module level so all tests share the same module object.
_mod = _import_slice_scheduler()
compute_execution_waves = _mod.compute_execution_waves
validate_feature_slices = _mod.validate_feature_slices
slices_to_state_dict = _mod.slices_to_state_dict


# ── Helpers ──────────────────────────────────────────────────────────────


def _slice(
    slice_id: str,
    *,
    title: str = "",
    deps: list[str] | None = None,
    complexity: str = "small",
) -> FeatureSlice:
    """Shorthand factory for building test slices."""
    return FeatureSlice(
        id=slice_id,
        title=title or f"Slice {slice_id}",
        dependencies=list(deps) if deps else [],
        complexity=complexity,
    )


def _wave_ids(waves: list[list[FeatureSlice]]) -> list[list[str]]:
    """Extract sorted ID lists from waves for deterministic comparison."""
    return sorted(sorted(s.id for s in wave) for wave in waves)


# ── Topological Sorting Tests ────────────────────────────────────────────


class TestComputeExecutionWaves:
    """Tests for compute_execution_waves() — topological grouping."""

    def test_linear_dependencies(self) -> None:
        """A→B→C produces 3 sequential waves: [[A], [B], [C]]."""
        slices = [
            _slice("A"),
            _slice("B", deps=["A"]),
            _slice("C", deps=["B"]),
        ]
        validate_feature_slices(slices)
        waves = compute_execution_waves(slices)

        assert len(waves) == 3
        ids = _wave_ids(waves)
        assert ids == [["A"], ["B"], ["C"]]

    def test_parallel_slices(self) -> None:
        """A, B, C with no dependencies produces 1 wave containing all 3."""
        slices = [
            _slice("A"),
            _slice("B"),
            _slice("C"),
        ]
        validate_feature_slices(slices)
        waves = compute_execution_waves(slices)

        assert len(waves) == 1
        ids = _wave_ids(waves)
        assert ids == [["A", "B", "C"]]

    def test_diamond_pattern(self) -> None:
        """A→B,C→D produces [[A], [B, C], [D]].

        Diamond: A has no deps, B and C depend on A, D depends on B and C.
        """
        slices = [
            _slice("A"),
            _slice("B", deps=["A"]),
            _slice("C", deps=["A"]),
            _slice("D", deps=["B", "C"]),
        ]
        validate_feature_slices(slices)
        waves = compute_execution_waves(slices)

        assert len(waves) == 3
        ids = _wave_ids(waves)
        assert ids == [["A"], ["B", "C"], ["D"]]

    def test_cycle_detection(self) -> None:
        """A→B→A (mutual dependency) raises ValueError."""
        slices = [
            _slice("A", deps=["B"]),
            _slice("B", deps=["A"]),
        ]
        with pytest.raises(ValueError, match="(?i)cycle"):
            validate_feature_slices(slices)

    def test_missing_dependency(self) -> None:
        """A depends on 'X' which doesn't exist → ValueError listing 'X'."""
        slices = [
            _slice("A", deps=["X"]),
        ]
        with pytest.raises(ValueError, match="X"):
            validate_feature_slices(slices)

    def test_duplicate_ids(self) -> None:
        """Two slices with the same ID raise ValueError."""
        slices = [
            _slice("A", title="First A"),
            _slice("A", title="Second A"),
        ]
        with pytest.raises(ValueError, match="(?i)duplicate"):
            validate_feature_slices(slices)

    def test_empty_slices(self) -> None:
        """An empty slice list raises ValueError."""
        with pytest.raises(ValueError):
            validate_feature_slices([])

    def test_self_dependency(self) -> None:
        """A depends on itself → ValueError (cycle of length 1)."""
        slices = [
            _slice("A", deps=["A"]),
        ]
        with pytest.raises(ValueError):
            validate_feature_slices(slices)

    def test_complex_graph(self) -> None:
        """10+ slices with mixed dependencies produce correct wave ordering.

        Graph structure:
            Wave 1 (no deps):  S0, S1, S2
            Wave 2:            S3 (←S0), S4 (←S1), S5 (←S2)
            Wave 3:            S6 (←S3,S4), S7 (←S5)
            Wave 4:            S8 (←S6,S7)
            Wave 5:            S9 (←S8)
        """
        slices = [
            _slice("S0"),
            _slice("S1"),
            _slice("S2"),
            _slice("S3", deps=["S0"]),
            _slice("S4", deps=["S1"]),
            _slice("S5", deps=["S2"]),
            _slice("S6", deps=["S3", "S4"]),
            _slice("S7", deps=["S5"]),
            _slice("S8", deps=["S6", "S7"]),
            _slice("S9", deps=["S8"]),
            _slice("S10", deps=["S0"]),  # Extra parallel branch off S0
        ]
        validate_feature_slices(slices)
        waves = compute_execution_waves(slices)

        ids = _wave_ids(waves)

        # Wave 1: S0, S1, S2 (no deps)
        assert sorted(ids[0]) == ["S0", "S1", "S2"]

        # Wave 2: S3 (←S0), S4 (←S1), S5 (←S2), S10 (←S0)
        assert sorted(ids[1]) == ["S10", "S3", "S4", "S5"]

        # Wave 3: S6 (←S3,S4), S7 (←S5)
        assert sorted(ids[2]) == ["S6", "S7"]

        # Wave 4: S8 (←S6,S7)
        assert sorted(ids[3]) == ["S8"]

        # Wave 5: S9 (←S8)
        assert sorted(ids[4]) == ["S9"]

        assert len(waves) == 5

    def test_single_slice(self) -> None:
        """A single slice with no deps produces exactly 1 wave with 1 element."""
        slices = [_slice("only")]
        validate_feature_slices(slices)
        waves = compute_execution_waves(slices)

        assert len(waves) == 1
        assert len(waves[0]) == 1
        assert waves[0][0].id == "only"

    def test_fan_out_pattern(self) -> None:
        """One root fans out to many dependents: A→B, A→C, A→D → 2 waves."""
        slices = [
            _slice("A"),
            _slice("B", deps=["A"]),
            _slice("C", deps=["A"]),
            _slice("D", deps=["A"]),
        ]
        validate_feature_slices(slices)
        waves = compute_execution_waves(slices)

        assert len(waves) == 2
        ids = _wave_ids(waves)
        assert ids == [["A"], ["B", "C", "D"]]

    def test_fan_in_pattern(self) -> None:
        """Many roots converge on one slice: A→D, B→D, C→D → 2 waves."""
        slices = [
            _slice("A"),
            _slice("B"),
            _slice("C"),
            _slice("D", deps=["A", "B", "C"]),
        ]
        validate_feature_slices(slices)
        waves = compute_execution_waves(slices)

        assert len(waves) == 2
        ids = _wave_ids(waves)
        assert ids == [["A", "B", "C"], ["D"]]

    def test_input_order_does_not_affect_waves(self) -> None:
        """Slice ordering in the input list should not affect wave grouping."""
        # Reversed order from natural
        slices_rev = [
            _slice("D", deps=["B", "C"]),
            _slice("C", deps=["A"]),
            _slice("B", deps=["A"]),
            _slice("A"),
        ]
        slices_fwd = [
            _slice("A"),
            _slice("B", deps=["A"]),
            _slice("C", deps=["A"]),
            _slice("D", deps=["B", "C"]),
        ]

        # Both must validate and produce identical wave structures
        validate_feature_slices(slices_rev)
        validate_feature_slices(slices_fwd)

        waves_rev = compute_execution_waves(slices_rev)
        waves_fwd = compute_execution_waves(slices_fwd)

        assert _wave_ids(waves_rev) == _wave_ids(waves_fwd)


# ── Validation Tests ─────────────────────────────────────────────────────


class TestValidateFeatureSlices:
    """Tests for validate_feature_slices() — input validation."""

    def test_validation_empty_fields(self) -> None:
        """A slice with an empty title raises ValueError."""
        slices = [
            FeatureSlice(id="x", title="", dependencies=[]),
        ]
        with pytest.raises(ValueError):
            validate_feature_slices(slices)

    def test_validation_empty_id(self) -> None:
        """A slice with an empty ID raises ValueError."""
        slices = [
            FeatureSlice(id="", title="Bad Slice", dependencies=[]),
        ]
        with pytest.raises(ValueError):
            validate_feature_slices(slices)

    def test_valid_slices_pass(self) -> None:
        """Well-formed slices should not raise."""
        slices = [
            _slice("A"),
            _slice("B", deps=["A"]),
        ]
        # Should not raise
        validate_feature_slices(slices)

    def test_long_chain(self) -> None:
        """A long linear chain validates and produces N waves."""
        n = 20
        slices = [_slice(f"S{i}", deps=[f"S{i - 1}"] if i > 0 else []) for i in range(n)]
        validate_feature_slices(slices)
        waves = compute_execution_waves(slices)
        assert len(waves) == n


# ── Serialization Tests ──────────────────────────────────────────────────


class TestSlicesToStateDict:
    """Tests for slices_to_state_dict() — JSON-serializable output."""

    def test_roundtrip_basic(self) -> None:
        """Waves convert to list-of-lists of dicts with correct keys."""
        slices = [
            _slice("A"),
            _slice("B", deps=["A"]),
        ]
        validate_feature_slices(slices)
        waves = compute_execution_waves(slices)
        result = slices_to_state_dict(waves)

        # Must be a list of lists (waves → slices)
        assert isinstance(result, list)
        assert len(result) == 2

        # Each inner element must be a list of dicts
        for wave in result:
            assert isinstance(wave, list)
            for entry in wave:
                assert isinstance(entry, dict)
                assert "id" in entry
                assert "title" in entry

    def test_output_is_json_serializable(self) -> None:
        """Output must be serializable with json.dumps (no exotic types)."""
        import json

        slices = [
            _slice("A"),
            _slice("B", deps=["A"]),
            _slice("C", deps=["A"]),
            _slice("D", deps=["B", "C"]),
        ]
        validate_feature_slices(slices)
        waves = compute_execution_waves(slices)
        result = slices_to_state_dict(waves)

        # Must not raise
        serialized = json.dumps(result)
        assert isinstance(serialized, str)
        # Roundtrip back through json.loads
        deserialized = json.loads(serialized)
        assert deserialized == result

    def test_preserves_all_fields(self) -> None:
        """Each dict in the output contains all FeatureSlice fields."""
        slices = [
            FeatureSlice(
                id="my-slice",
                title="My Slice",
                target_files=["spine/foo.py"],
                execution_requirements=["requires X"],
                dependencies=[],
                acceptance_criteria=["test passes"],
                complexity="medium",
            ),
        ]
        validate_feature_slices(slices)
        waves = compute_execution_waves(slices)
        result = slices_to_state_dict(waves)

        entry = result[0][0]
        assert entry["id"] == "my-slice"
        assert entry["title"] == "My Slice"
        assert entry["target_files"] == ["spine/foo.py"]
        assert entry["execution_requirements"] == ["requires X"]
        assert entry["dependencies"] == []
        assert entry["acceptance_criteria"] == ["test passes"]
        assert entry["complexity"] == "medium"


# ── Same-file serialization must not inject a cycle (regression: trace 55ae1919) ──


def test_same_file_chain_respects_existing_dep_direction() -> None:
    """Two slices on one file where the alphabetically-EARLIER id depends on the
    later one must NOT be chained alphabetically — doing so reverses the existing
    edge and creates a 2-cycle that compute_execution_waves then rejects.
    """
    slices = [
        FeatureSlice(
            id="api_phase_timeout",  # alphabetically first…
            title="phase+timeout",
            dependencies=["api_specialized"],  # …but depends on the later one
            target_files=["spine/ui_api/api.py"],
        ),
        FeatureSlice(
            id="api_specialized",
            title="specialized",
            dependencies=[],
            target_files=["spine/ui_api/api.py"],
        ),
    ]
    # Must not raise CycleError, and must order the dependency root first.
    waves = compute_execution_waves(slices, same_file_strategy="chain")
    ids = [sorted(s.id for s in wave) for wave in waves]
    assert ids == [["api_specialized"], ["api_phase_timeout"]]


# ── Same-file strategy: merge for the synthesis editor (run 019f214f) ──


def _samefile_slices() -> list[FeatureSlice]:
    return [
        FeatureSlice(
            id="ui-provider-structure",
            title="Provider structure",
            target_files=["spine/ui/_pages/config_view.py"],
            execution_requirements=["add three expanders"],
            acceptance_criteria=["three expander groups render"],
            provides=["ConfigProviderSection"],
            reference_symbols=["UIApi.get_providers"],
        ),
        FeatureSlice(
            id="ui-phase-config",
            title="Phase config",
            target_files=["spine/ui/_pages/config_view.py"],
            execution_requirements=["add phase forms"],
            dependencies=["ui-provider-structure"],
            acceptance_criteria=["phase forms render"],
            provides=["ConfigViewPhaseEditor"],
            reference_symbols=["SpineConfig.load"],
            complexity="medium",
        ),
        FeatureSlice(
            id="other-file",
            title="Other file",
            target_files=["spine/config.py"],
            execution_requirements=["tweak config"],
            acceptance_criteria=["config tweak lands"],
        ),
    ]


def test_merge_strategy_unions_samefile_slices_into_one() -> None:
    """Chained same-file slices each wholesale-replace the shared symbol to
    satisfy only their own criteria — last writer wins and verify fails every
    slice (run 019f214f). Merge hands ONE editor pass the full criteria set."""
    waves = compute_execution_waves(_samefile_slices(), same_file_strategy="merge")
    all_slices = [s for wave in waves for s in wave]
    assert len(all_slices) == 2  # 2 same-file slices merged + 1 independent
    merged = next(s for s in all_slices if s.target_files == ["spine/ui/_pages/config_view.py"])
    assert set(merged.acceptance_criteria) == {
        "three expander groups render",
        "phase forms render",
    }
    assert set(merged.provides) == {"ConfigProviderSection", "ConfigViewPhaseEditor"}
    assert set(merged.reference_symbols) == {"UIApi.get_providers", "SpineConfig.load"}
    # Both members' requirements survive, labeled by title.
    assert "add three expanders" in str(merged.execution_requirements)
    assert "add phase forms" in str(merged.execution_requirements)
    # No dangling dependency references to merged-away ids.
    valid_ids = {s.id for s in all_slices}
    for s in all_slices:
        for d in s.dependencies or []:
            assert d in valid_ids
    assert merged.complexity == "medium"  # max of small/medium


def test_chain_strategy_keeps_samefile_slices_separate() -> None:
    waves = compute_execution_waves(_samefile_slices(), same_file_strategy="chain")
    all_ids = {s.id for wave in waves for s in wave}
    assert all_ids == {"ui-provider-structure", "ui-phase-config", "other-file"}


def test_default_strategy_follows_synthesis_flag(monkeypatch) -> None:
    class _Cfg:
        implement_synthesis_placement = True

    monkeypatch.setattr(
        "spine.config.SpineConfig.load", classmethod(lambda cls, *a, **k: _Cfg())
    )
    assert len([s for w in compute_execution_waves(_samefile_slices()) for s in w]) == 2

    _Cfg.implement_synthesis_placement = False
    assert len([s for w in compute_execution_waves(_samefile_slices()) for s in w]) == 3
