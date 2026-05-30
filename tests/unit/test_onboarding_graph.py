"""Unit tests for the distributed onboarding synthesis hierarchy graph.

Covers Phase B of :mod:`spine.work.onboarding.synthesis_nodes` (design
Revision 2, §2.2-§2.3) with the bare manager/worker LLM calls mocked:

- the documentation manager refines the deterministic skeleton when the LLM
  returns a coherent plan, and falls back to the skeleton when the LLM raises;
- the per-section fan-out (one ``Send`` per section) writes all four documents;
- a missing document raises ``RuntimeError`` (all-or-nothing preserved);
- a section reporting ``status="error"`` raises ``RuntimeError``;
- the greenfield path produces the fixed minimal plan and all four docs.

No live model and no whole-manifest prompt are involved — the test asserts the
graph drives the bounded manager + section-worker hierarchy correctly.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spine.work.onboarding.manifest import (
    DependencyEdge,
    ModuleBoundary,
    PatternFinding,
    RepoManifest,
    SymbolRef,
)
from spine.work.onboarding.onboarding_graph import build_onboarding_graph
from spine.work.onboarding.synthesis_nodes import (
    _doc_manager_node,
    build_synthesis_graph,
)
from spine.work.onboarding.synthesis_plan import (
    SectionPlanSet,
    SectionResult,
    deterministic_section_plan,
)
from spine.work.onboarding.synthesis_tools import ONBOARDING_DOC_NAMES, ONBOARDING_PHASE


# ── Fixtures ────────────────────────────────────────────────────────────────


def _brownfield_manifest(workspace_root: str) -> RepoManifest:
    sym = SymbolRef(
        file_path="spine/work/dispatcher.py",
        symbol_name="submit_work",
        symbol_type="function",
        lang="python",
        summary="Entry point for work dispatch.",
    )
    return RepoManifest(
        workspace_root=workspace_root,
        mode="brownfield",
        tech_stack=["python", "langgraph"],
        core_domains=["spine.work"],
        module_boundaries=[
            ModuleBoundary(
                name="spine.work",
                path="spine/work",
                role="Work dispatch and queue management.",
                key_symbols=[sym],
            ),
            ModuleBoundary(
                name="spine.ui",
                path="spine/ui",
                role="Streamlit UI.",
                key_symbols=[],
            ),
        ],
        dependency_chains=[
            DependencyEdge(src="spine.ui", dst="spine.work", kind="depends_on")
        ],
        patterns=[
            PatternFinding(
                category="logging",
                description="module-level logging.getLogger(__name__)",
                evidence=[sym],
            )
        ],
        symbol_count=1,
        file_count=2,
        generated_at="2026-05-29T00:00:00",
        notes="",
    )


def _greenfield_manifest(workspace_root: str) -> RepoManifest:
    return RepoManifest(
        workspace_root=workspace_root,
        mode="greenfield",
        tech_stack=["python", "fastapi"],
        core_domains=["payments"],
        module_boundaries=[],
        dependency_chains=[],
        patterns=[],
        symbol_count=0,
        file_count=0,
        generated_at="2026-05-29T00:00:00",
        notes="greenfield seed",
    )


def _initial_state(manifest: RepoManifest, workspace_root: str, work_id: str) -> dict[str, Any]:
    return {
        "work_id": work_id,
        "workspace_root": workspace_root,
        "mode": manifest.mode,
        "tech_stack": list(manifest.tech_stack),
        "manifest": manifest.to_dict(),
    }


# ── Stub bare-LLM models ─────────────────────────────────────────────────────


class _StubStructured:
    def __init__(self, schema: Any, factory) -> None:
        self._schema = schema
        self._factory = factory

    async def ainvoke(self, messages: list[Any], **_: Any) -> Any:
        return self._factory(self._schema)


class _StubModel:
    """Structured calls return a canned plan (manager) or section (worker)."""

    def with_structured_output(self, schema: Any) -> _StubStructured:
        return _StubStructured(schema, self._make)

    def _make(self, schema: Any) -> Any:
        if schema is SectionPlanSet:
            return SectionPlanSet(
                sections=[
                    {
                        "doc_id": doc,
                        "order": 0,
                        "title": f"{doc} (refined)",
                        "fragment_keys": {"doc_id": doc},
                        "instruction": f"Write {doc}.",
                    }
                    for doc in ONBOARDING_DOC_NAMES
                ]
            )
        return SectionResult(
            doc_id="", order=0, markdown="Section body.", status="ok"
        )


class _RaisingModel:
    """A model whose structured call always raises (manager failure path)."""

    def with_structured_output(self, schema: Any) -> "_RaisingModel":
        return self

    async def ainvoke(self, messages: list[Any], **_: Any) -> Any:
        raise RuntimeError("local model unavailable")


def _patch_resolve(monkeypatch, model: Any) -> None:
    def fake_resolve_model(config: Any, session_id: Any = None, phase: Any = None) -> Any:
        return model

    monkeypatch.setattr(
        "spine.work.onboarding.synthesis_nodes.resolve_model", fake_resolve_model
    )


# ── Manager (Tier A) ─────────────────────────────────────────────────────────


class TestDocManager:
    def test_refines_plan_when_llm_coherent(self, tmp_path: Path, monkeypatch) -> None:
        manifest = _brownfield_manifest(str(tmp_path))
        _patch_resolve(monkeypatch, _StubModel())
        out = asyncio.run(
            _doc_manager_node(_initial_state(manifest, str(tmp_path), "wk"), None)
        )
        sections = out["sections"]
        # All four docs covered, and the refined titles came from the LLM stub.
        assert {s["doc_id"] for s in sections} == set(ONBOARDING_DOC_NAMES)
        assert all("refined" in s["title"] for s in sections)
        assert "manifest_index" in out

    def test_falls_back_to_skeleton_when_llm_raises(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        manifest = _brownfield_manifest(str(tmp_path))
        _patch_resolve(monkeypatch, _RaisingModel())
        out = asyncio.run(
            _doc_manager_node(_initial_state(manifest, str(tmp_path), "wk"), None)
        )
        index = out["manifest_index"]
        expected = deterministic_section_plan(index, "brownfield")
        # Skeleton used verbatim — no "refined" titles, same section count.
        assert out["sections"] == expected
        assert all("refined" not in s["title"] for s in out["sections"])


# ── Full synthesis graph ─────────────────────────────────────────────────────


class TestSynthesisGraph:
    def _run(self, manifest: RepoManifest, tmp_path: Path, work_id: str) -> dict[str, Any]:
        graph = build_synthesis_graph().compile()
        return asyncio.run(
            graph.ainvoke(_initial_state(manifest, str(tmp_path), work_id))
        )

    def test_fan_out_writes_all_four_docs(self, tmp_path: Path, monkeypatch) -> None:
        manifest = _brownfield_manifest(str(tmp_path))
        _patch_resolve(monkeypatch, _StubModel())
        final = self._run(manifest, tmp_path, "wk-graph")

        doc_dir = tmp_path / ".spine/artifacts" / "wk-graph" / ONBOARDING_PHASE
        for doc in ONBOARDING_DOC_NAMES:
            assert (doc_dir / f"{doc}.md").exists()
        assert set(final["written"]) == set(ONBOARDING_DOC_NAMES)
        # The architecture map fanned out one section per module (2 modules) +
        # the other docs' sections — i.e. real per-section fan-out happened.
        assert len(final["section_results"]) >= len(ONBOARDING_DOC_NAMES)

    def test_greenfield_minimal_plan(self, tmp_path: Path, monkeypatch) -> None:
        manifest = _greenfield_manifest(str(tmp_path))
        # Greenfield skips the LLM refine, but resolve_model is still patched
        # defensively (workers run for the four minimal sections).
        _patch_resolve(monkeypatch, _StubModel())
        final = self._run(manifest, tmp_path, "wk-green")

        doc_dir = tmp_path / ".spine/artifacts" / "wk-green" / ONBOARDING_PHASE
        for doc in ONBOARDING_DOC_NAMES:
            assert (doc_dir / f"{doc}.md").exists()
        # Greenfield = exactly one section per document.
        assert len(final["section_results"]) == len(ONBOARDING_DOC_NAMES)

    def test_missing_document_raises(self, tmp_path: Path, monkeypatch) -> None:
        """If the assembler can't produce a doc file, aggregate must raise.

        We simulate a missing document by patching the assembler to skip one
        document entirely (no write), so aggregate_synthesis finds it absent.
        """
        manifest = _brownfield_manifest(str(tmp_path))
        _patch_resolve(monkeypatch, _StubModel())

        import spine.work.onboarding.synthesis_nodes as nodes

        real_assemble = nodes._assemble_docs_node

        def partial_assemble(state: dict[str, Any]) -> dict[str, Any]:
            out = real_assemble(state)
            # Delete one written doc file to simulate an assembly gap.
            missing_path = Path(out["written"]["CODING_GUIDELINES"])
            if missing_path.exists():
                missing_path.unlink()
            return out

        monkeypatch.setattr(nodes, "_assemble_docs_node", partial_assemble)
        graph = nodes.build_synthesis_graph().compile()

        with pytest.raises(RuntimeError, match="synthesis incomplete"):
            asyncio.run(
                graph.ainvoke(_initial_state(manifest, str(tmp_path), "wk-miss"))
            )

    def test_section_error_raises(self, tmp_path: Path, monkeypatch) -> None:
        """A section reporting status=error must fail the run loudly."""
        manifest = _brownfield_manifest(str(tmp_path))

        class _ErrorWorkerModel(_StubModel):
            def _make(self, schema: Any) -> Any:
                if schema is SectionPlanSet:
                    return super()._make(schema)
                # Worker returns empty markdown → classified as an error.
                return SectionResult(doc_id="", order=0, markdown="", status="error")

        _patch_resolve(monkeypatch, _ErrorWorkerModel())

        graph = build_synthesis_graph().compile()
        with pytest.raises(RuntimeError, match="synthesis incomplete"):
            asyncio.run(
                graph.ainvoke(_initial_state(manifest, str(tmp_path), "wk-err"))
            )


# ── Composed graph: analysis (Phase A) + synthesis (Phase B) ─────────────────


def _write_sample_repo(root: Path) -> None:
    """Lay down a tiny but real Python repo so deterministic analysis runs."""
    pkg = root / "sample_pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "core.py").write_text(
        "import logging\n\n"
        "logger = logging.getLogger(__name__)\n\n\n"
        "def do_work(x: int) -> int:\n"
        '    """Double the input."""\n'
        "    logger.info('working')\n"
        "    return x * 2\n",
        encoding="utf-8",
    )
    (pkg / "service.py").write_text(
        "from sample_pkg.core import do_work\n\n\n"
        "class Service:\n"
        '    """A small service."""\n\n'
        "    def run(self, n: int) -> int:\n"
        "        return do_work(n)\n",
        encoding="utf-8",
    )


class TestComposedOnboardingGraph:
    """End-to-end: Phase A (real deterministic analysis) → Phase B (mocked LLM)."""

    def _patch_both(self, monkeypatch, model: Any) -> None:
        def fake_resolve_model(config: Any, session_id: Any = None, phase: Any = None) -> Any:
            return model

        monkeypatch.setattr(
            "spine.work.onboarding.synthesis_nodes.resolve_model", fake_resolve_model
        )

    def test_brownfield_analyze_then_synthesize_writes_four_docs(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _write_sample_repo(tmp_path)
        self._patch_both(monkeypatch, _StubModel())

        graph = build_onboarding_graph().compile()
        final = asyncio.run(
            graph.ainvoke(
                {
                    "work_id": "wk-e2e",
                    "workspace_root": str(tmp_path),
                    "mode": "brownfield",
                    "tech_stack": ["python"],
                }
            )
        )

        # Phase A produced a manifest persisted to disk, with real modules.
        assert final.get("manifest")
        assert final.get("manifest_path")
        assert Path(final["manifest_path"]).exists()
        manifest = RepoManifest.from_dict(final["manifest"])
        assert manifest.mode == "brownfield"
        assert manifest.module_boundaries  # discovered sample_pkg.* modules

        # Phase B wrote all four documents.
        doc_dir = tmp_path / ".spine/artifacts" / "wk-e2e" / ONBOARDING_PHASE
        for doc in ONBOARDING_DOC_NAMES:
            assert (doc_dir / f"{doc}.md").exists()
        assert set(final["written"]) == set(ONBOARDING_DOC_NAMES)

    def test_greenfield_seed_then_synthesize_writes_four_docs(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        self._patch_both(monkeypatch, _StubModel())

        graph = build_onboarding_graph().compile()
        final = asyncio.run(
            graph.ainvoke(
                {
                    "work_id": "wk-green-e2e",
                    "workspace_root": str(tmp_path),
                    "mode": "greenfield",
                    "tech_stack": ["python", "fastapi"],
                }
            )
        )

        manifest = RepoManifest.from_dict(final["manifest"])
        assert manifest.mode == "greenfield"

        doc_dir = tmp_path / ".spine/artifacts" / "wk-green-e2e" / ONBOARDING_PHASE
        for doc in ONBOARDING_DOC_NAMES:
            assert (doc_dir / f"{doc}.md").exists()
        # Greenfield = exactly one section per document.
        assert len(final["section_results"]) == len(ONBOARDING_DOC_NAMES)
