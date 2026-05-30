"""Unit tests for the onboarding RepoAnalyzer / build_repo_manifest.

These tests run the analyzer against the real spine repo using the os.walk
discovery fallback (no MCP server / no vector index required) so they are
deterministic and hermetic in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spine.config import SpineConfig
from spine.work.onboarding.analyzer import RepoAnalyzer, build_repo_manifest
from spine.work.onboarding.manifest import RepoManifest

# Repo root = two levels up from tests/unit/.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)


def _hermetic_config() -> SpineConfig:
    """Config with no MCP servers (forces os.walk) and a bogus vector db.

    With ``mcp_servers={}`` the index discovery short-circuits to the
    filesystem walk, and a nonexistent ``checkpoint_path`` makes summary
    enrichment return an empty map — both exercise the no-dependency path.
    """
    return SpineConfig(
        mcp_servers={},
        checkpoint_path="/nonexistent/spine-onboarding-test.db",
        workspace_root=_REPO_ROOT,
    )


@pytest.mark.asyncio
async def test_brownfield_covers_core_modules():
    manifest = await build_repo_manifest(
        _REPO_ROOT, mode="brownfield", config=_hermetic_config()
    )
    assert isinstance(manifest, RepoManifest)
    assert manifest.mode == "brownfield"
    assert manifest.symbol_count > 0
    assert manifest.file_count > 0

    module_names = {b.name for b in manifest.module_boundaries}
    for expected in ("spine.work", "spine.ui", "spine.agents"):
        assert expected in module_names, f"missing boundary {expected}: {sorted(module_names)}"


@pytest.mark.asyncio
async def test_brownfield_extracts_logging_pattern():
    manifest = await build_repo_manifest(
        _REPO_ROOT, mode="brownfield", config=_hermetic_config()
    )
    categories = {p.category for p in manifest.patterns}
    assert "logging" in categories
    logging_finding = next(p for p in manifest.patterns if p.category == "logging")
    assert "getLogger" in logging_finding.description
    assert logging_finding.evidence  # representative SymbolRefs, not raw text


@pytest.mark.asyncio
async def test_brownfield_manifest_round_trips():
    manifest = await build_repo_manifest(
        _REPO_ROOT, mode="brownfield", config=_hermetic_config()
    )
    restored = RepoManifest.from_dict(manifest.to_dict())
    assert restored == manifest


@pytest.mark.asyncio
async def test_brownfield_infers_python_tech_stack():
    manifest = await build_repo_manifest(
        _REPO_ROOT, mode="brownfield", config=_hermetic_config()
    )
    assert "python" in manifest.tech_stack


@pytest.mark.asyncio
async def test_evidence_carries_no_raw_source():
    """PatternFinding evidence must be SymbolRef metadata, never raw code."""
    manifest = await build_repo_manifest(
        _REPO_ROOT, mode="brownfield", config=_hermetic_config()
    )
    for pattern in manifest.patterns:
        for ref in pattern.evidence:
            assert not hasattr(ref, "raw_code")
            assert ref.file_path
            assert ref.symbol_name


@pytest.mark.asyncio
async def test_dependency_edges_are_cross_module():
    manifest = await build_repo_manifest(
        _REPO_ROOT, mode="brownfield", config=_hermetic_config()
    )
    for edge in manifest.dependency_chains:
        assert edge.src != edge.dst
        assert edge.kind == "depends_on"


@pytest.mark.asyncio
async def test_greenfield_returns_seed_manifest():
    manifest = await build_repo_manifest(
        "/some/new/project",
        mode="greenfield",
        tech_stack=["python"],
        config=_hermetic_config(),
    )
    assert manifest.mode == "greenfield"
    assert manifest.tech_stack == ["python"]
    assert manifest.module_boundaries == []
    assert manifest.dependency_chains == []
    assert manifest.patterns == []
    assert manifest.symbol_count == 0
    assert manifest.file_count == 0


@pytest.mark.asyncio
async def test_analyzer_class_matches_module_helper():
    analyzer = RepoAnalyzer(config=_hermetic_config())
    via_class = await analyzer.analyze(_REPO_ROOT, mode="brownfield")
    assert via_class.symbol_count > 0
    assert any(b.name == "spine.work" for b in via_class.module_boundaries)
