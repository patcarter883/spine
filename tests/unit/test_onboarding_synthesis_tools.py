"""Unit tests for onboarding synthesis tools and the synthesis driver.

Exercises the tools directly without a live LLM:
- ``ReadRepoManifestTool`` returns the persisted manifest JSON (and a clean
  error when the manifest is absent).
- ``WriteOnboardingDocTool`` writes ``<NAME>.md`` idempotently and rejects
  unknown document names.
- ``synthesize_artifacts`` (now a thin shim over the synthesis hierarchy graph)
  writes all four documents and is idempotent across re-runs, with the bare
  manager/worker LLM calls replaced by stubs returning a ``SectionPlanSet`` /
  ``SectionResult``.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spine.agents.artifacts import artifact_path
from spine.work.onboarding.manifest import (
    DependencyEdge,
    ModuleBoundary,
    PatternFinding,
    RepoManifest,
    SymbolRef,
)
from spine.work.onboarding.synthesis import synthesize_artifacts
from spine.work.onboarding.synthesis_tools import (
    ONBOARDING_DOC_NAMES,
    ONBOARDING_PHASE,
    ReadRepoManifestTool,
    WriteOnboardingDocTool,
    build_synthesis_tools,
)


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
        tech_stack=["python", "langgraph", "streamlit"],
        core_domains=["work dispatch", "agent orchestration"],
        module_boundaries=[
            ModuleBoundary(
                name="spine.work",
                path="spine/work",
                role="Work dispatch and queue management.",
                key_symbols=[sym],
            )
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
        file_count=1,
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


def _write_manifest_to_disk(tmp_path: Path, manifest: RepoManifest, work_id: str) -> Path:
    """Persist repo_manifest.json the way the analysis stage does."""
    rel = artifact_path(work_id, ONBOARDING_PHASE)
    mdir = tmp_path / rel
    mdir.mkdir(parents=True, exist_ok=True)
    mpath = mdir / "repo_manifest.json"
    mpath.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return mpath


# ── ReadRepoManifestTool ──────────────────────────────────────────────────


class TestReadRepoManifestTool:
    def test_returns_manifest_json(self, tmp_path: Path) -> None:
        work_id = "wk-onb"
        manifest = _brownfield_manifest(str(tmp_path))
        _write_manifest_to_disk(tmp_path, manifest, work_id)

        tool = ReadRepoManifestTool(
            workspace_root=str(tmp_path),
            manifest_dir=artifact_path(work_id, ONBOARDING_PHASE),
        )
        out = tool._run()
        data = json.loads(out)
        assert data["mode"] == "brownfield"
        assert data["tech_stack"] == ["python", "langgraph", "streamlit"]
        # round-trips back into the dataclass
        assert RepoManifest.from_dict(data).symbol_count == 1

    def test_missing_manifest_returns_clean_error(self, tmp_path: Path) -> None:
        tool = ReadRepoManifestTool(
            workspace_root=str(tmp_path),
            manifest_dir=artifact_path("nope", ONBOARDING_PHASE),
        )
        data = json.loads(tool._run())
        assert data["error"] == "manifest_not_found"


# ── WriteOnboardingDocTool ────────────────────────────────────────────────


class TestWriteOnboardingDocTool:
    def _tool(self, tmp_path: Path, work_id: str = "wk-onb") -> WriteOnboardingDocTool:
        return WriteOnboardingDocTool(
            workspace_root=str(tmp_path),
            work_id=work_id,
            out_dir=str(tmp_path / ".spine/artifacts"),
        )

    def _doc_dir(self, tmp_path: Path, work_id: str = "wk-onb") -> Path:
        return tmp_path / ".spine/artifacts" / work_id / ONBOARDING_PHASE

    def test_writes_named_doc(self, tmp_path: Path) -> None:
        tool = self._tool(tmp_path)
        result = tool._run(doc="PROJECT_DEFINITION", content="# Project\nbody")
        path = self._doc_dir(tmp_path) / "PROJECT_DEFINITION.md"
        assert path.exists()
        assert path.read_text(encoding="utf-8") == "# Project\nbody"
        assert "PROJECT_DEFINITION.md" in result

    def test_rejects_unknown_doc_name(self, tmp_path: Path) -> None:
        tool = self._tool(tmp_path)
        result = tool._run(doc="RANDOM_DOC", content="x")
        assert result.startswith("VALIDATION_ERROR")
        # nothing should be written
        assert not (self._doc_dir(tmp_path) / "RANDOM_DOC.md").exists()

    def test_rejects_empty_content(self, tmp_path: Path) -> None:
        tool = self._tool(tmp_path)
        result = tool._run(doc="CODING_GUIDELINES", content="   ")
        assert result.startswith("VALIDATION_ERROR")

    def test_idempotent_overwrite_shorter(self, tmp_path: Path) -> None:
        tool = self._tool(tmp_path)
        path = self._doc_dir(tmp_path) / "ARCHITECTURE_MAP.md"
        tool._run(doc="ARCHITECTURE_MAP", content="# Long\n" + ("x" * 500))
        assert path.exists()
        # Second run with shorter content overwrites cleanly (no FileExistsError).
        tool._run(doc="ARCHITECTURE_MAP", content="# Short")
        assert path.read_text(encoding="utf-8") == "# Short"


# ── build_synthesis_tools ──────────────────────────────────────────────────


class TestBuildSynthesisTools:
    def test_returns_two_named_tools(self, tmp_path: Path) -> None:
        tools = build_synthesis_tools(
            workspace_root=str(tmp_path),
            work_id="wk-onb",
            manifest_dir=artifact_path("wk-onb", ONBOARDING_PHASE),
            out_dir=str(tmp_path / ".spine/artifacts"),
        )
        names = {t.name for t in tools}
        assert names == {"read_repo_manifest", "write_onboarding_doc"}


# ── synthesize_artifacts (bare manager/worker LLM calls stubbed) ────────────


class _StubStructured:
    """A ``with_structured_output(...)`` result that returns a canned object."""

    def __init__(self, schema: Any, factory) -> None:
        self._schema = schema
        self._factory = factory

    async def ainvoke(self, messages: list[Any], **_: Any) -> Any:
        return self._factory(self._schema)


class _StubModel:
    """A bare chat model whose structured calls return canned plan/results.

    The manager call binds ``SectionPlanSet`` and the worker call binds
    ``SectionResult``; ``_factory`` inspects the schema to return the right
    canned object. ``worker_status`` lets a test force a worker to report
    ``status="error"`` so the partial-generation RuntimeError path is exercised.
    """

    def __init__(self, worker_status: str = "ok") -> None:
        self._worker_status = worker_status

    def with_structured_output(self, schema: Any) -> _StubStructured:
        return _StubStructured(schema, self._make)

    def _make(self, schema: Any) -> Any:
        from spine.work.onboarding.synthesis_plan import (
            SectionPlanSet,
            SectionResult,
        )

        if schema is SectionPlanSet:
            # Refine = identity on the deterministic skeleton: one section per
            # doc, covering all four documents so _plan_is_coherent() passes.
            return SectionPlanSet(
                sections=[
                    {
                        "doc_id": doc,
                        "order": 0,
                        "title": doc.replace("_", " ").title(),
                        "fragment_keys": {"doc_id": doc},
                        "instruction": f"Write {doc}.",
                    }
                    for doc in ONBOARDING_DOC_NAMES
                ]
            )
        # SectionResult — one section's markdown. The error case returns empty
        # markdown so the worker classifies it as a failed section.
        if self._worker_status == "error":
            return SectionResult(doc_id="", order=0, markdown="", status="error")
        return SectionResult(
            doc_id="",
            order=0,
            markdown="Generated for tests.",
            status="ok",
        )


def _patch_models(monkeypatch, worker_status: str = "ok") -> None:
    """Replace the bare-LLM ``resolve_model`` in synthesis_nodes with a stub."""

    def fake_resolve_model(config: Any, session_id: Any = None, phase: Any = None) -> Any:
        return _StubModel(worker_status=worker_status)

    monkeypatch.setattr(
        "spine.work.onboarding.synthesis_nodes.resolve_model", fake_resolve_model
    )


class TestSynthesizeArtifacts:
    def test_writes_all_four_documents(self, tmp_path: Path, monkeypatch) -> None:
        work_id = "wk-onb"
        manifest = _brownfield_manifest(str(tmp_path))
        _write_manifest_to_disk(tmp_path, manifest, work_id)
        _patch_models(monkeypatch)

        result = asyncio.run(
            synthesize_artifacts(
                manifest=manifest,
                workspace_root=str(tmp_path),
                work_id=work_id,
                config=None,  # type: ignore[arg-type]
            )
        )

        doc_dir = tmp_path / ".spine/artifacts" / work_id / ONBOARDING_PHASE
        for doc in ONBOARDING_DOC_NAMES:
            assert (doc_dir / f"{doc}.md").exists()
            assert result[doc] == str(doc_dir / f"{doc}.md")
        assert set(result) == set(ONBOARDING_DOC_NAMES)

    def test_rerun_is_idempotent(self, tmp_path: Path, monkeypatch) -> None:
        work_id = "wk-onb"
        manifest = _greenfield_manifest(str(tmp_path))
        _write_manifest_to_disk(tmp_path, manifest, work_id)
        _patch_models(monkeypatch)

        first = asyncio.run(
            synthesize_artifacts(
                manifest=manifest,
                workspace_root=str(tmp_path),
                work_id=work_id,
                config=None,  # type: ignore[arg-type]
            )
        )
        # Second run must not raise and must yield the same filenames.
        second = asyncio.run(
            synthesize_artifacts(
                manifest=manifest,
                workspace_root=str(tmp_path),
                work_id=work_id,
                config=None,  # type: ignore[arg-type]
            )
        )
        assert set(first) == set(second) == set(ONBOARDING_DOC_NAMES)
        doc_dir = tmp_path / ".spine/artifacts" / work_id / ONBOARDING_PHASE
        md_files = sorted(p.name for p in doc_dir.glob("*.md"))
        assert md_files == sorted(f"{d}.md" for d in ONBOARDING_DOC_NAMES)

    def test_partial_generation_raises(self, tmp_path: Path, monkeypatch) -> None:
        """A section that reports status=error must fail the run loudly."""
        work_id = "wk-onb"
        manifest = _brownfield_manifest(str(tmp_path))
        _write_manifest_to_disk(tmp_path, manifest, work_id)
        # Every worker reports status="error" → aggregate must raise.
        _patch_models(monkeypatch, worker_status="error")

        with pytest.raises(RuntimeError, match="synthesis incomplete"):
            asyncio.run(
                synthesize_artifacts(
                    manifest=manifest,
                    workspace_root=str(tmp_path),
                    work_id=work_id,
                    config=None,  # type: ignore[arg-type]
                )
            )
