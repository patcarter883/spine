"""Unit tests for the onboarding RepoManifest dataclass contract."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spine.work.onboarding.manifest import (
    DependencyEdge,
    ModuleBoundary,
    PatternFinding,
    RepoManifest,
    SymbolRef,
)


def _sample_manifest() -> RepoManifest:
    ref = SymbolRef(
        file_path="spine/work/dispatcher.py",
        symbol_name="submit_work",
        symbol_type="function",
        lang="python",
        summary="Submits work to the queue.",
    )
    boundary = ModuleBoundary(
        name="spine.work",
        path="spine/work",
        role="spine.work: dispatch + queue",
        key_symbols=[ref],
    )
    edge = DependencyEdge(src="spine.work", dst="spine.persistence", kind="depends_on")
    pattern = PatternFinding(
        category="logging",
        description="module-level logging.getLogger(__name__)",
        evidence=[ref],
    )
    return RepoManifest(
        workspace_root="/abs/path",
        mode="brownfield",
        tech_stack=["python", "langgraph"],
        core_domains=["spine.work"],
        module_boundaries=[boundary],
        dependency_chains=[edge],
        patterns=[pattern],
        symbol_count=1,
        file_count=1,
        generated_at="2026-05-29T00:00:00+00:00",
        notes="test",
    )


class TestRepoManifestRoundTrip:
    def test_to_dict_is_json_serialisable(self):
        import json

        manifest = _sample_manifest()
        data = manifest.to_dict()
        # Must serialise cleanly to JSON (no dataclass instances leaking).
        text = json.dumps(data)
        assert isinstance(text, str)
        assert "submit_work" in text

    def test_round_trip_preserves_all_fields(self):
        manifest = _sample_manifest()
        restored = RepoManifest.from_dict(manifest.to_dict())
        assert restored == manifest

    def test_round_trip_rebuilds_nested_dataclasses(self):
        restored = RepoManifest.from_dict(_sample_manifest().to_dict())
        assert isinstance(restored.module_boundaries[0], ModuleBoundary)
        assert isinstance(restored.module_boundaries[0].key_symbols[0], SymbolRef)
        assert isinstance(restored.dependency_chains[0], DependencyEdge)
        assert isinstance(restored.patterns[0], PatternFinding)
        assert isinstance(restored.patterns[0].evidence[0], SymbolRef)

    def test_from_dict_tolerates_missing_optional_lists(self):
        data = {
            "workspace_root": "/x",
            "mode": "greenfield",
            "tech_stack": ["python"],
            "core_domains": [],
            "module_boundaries": [],
            "dependency_chains": [],
            "patterns": [],
            "symbol_count": 0,
            "file_count": 0,
            "generated_at": "2026-05-29T00:00:00+00:00",
        }
        restored = RepoManifest.from_dict(data)
        assert restored.mode == "greenfield"
        assert restored.notes == ""
        assert restored.tech_stack == ["python"]

    def test_dataclasses_are_frozen(self):
        import dataclasses

        ref = SymbolRef("a", "b", "function", "python")
        try:
            ref.symbol_name = "c"  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            return
        raise AssertionError("SymbolRef should be frozen")

    def test_symbol_ref_summary_defaults_empty(self):
        ref = SymbolRef("a", "b", "class", "typescript")
        assert ref.summary == ""
