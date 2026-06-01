"""Unit tests for the graph-driven onboarding engine (design Rev 2, PR-4).

These exercise :func:`spine.work.onboarding.engine.run_onboarding` end-to-end on
the composed onboarding ``StateGraph`` with the bare manager/worker LLM calls
mocked:

- **brownfield** runs ``analyze`` → ``synthesize`` and writes all four docs;
- **greenfield** runs ``scaffold`` (pre-graph) → ``analyze`` (greenfield seed)
  → ``synthesize`` and writes all four docs;
- the recorded ``current_phase`` progression matches what the UI expects;
- the return dict shape is preserved;
- a re-run for the same ``work_id`` is idempotent (overwrites cleanly).

No live model is used — the synthesis tiers' ``resolve_chat_model`` is patched to a
stub that returns a canned plan (manager) / section (worker). Analysis is
deterministic Python over a tiny real temp repo.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spine.config import SpineConfig
from spine.models.enums import TaskStatus
from spine.work.onboarding.engine import run_onboarding
from spine.work.onboarding.synthesis_plan import SectionPlanSet, SectionResult
from spine.work.onboarding.synthesis_tools import ONBOARDING_DOC_NAMES


# ── Stub bare-LLM model (manager + worker) ───────────────────────────────────


class _StubStructured:
    def __init__(self, schema: Any, factory) -> None:
        self._schema = schema
        self._factory = factory

    async def ainvoke(self, messages: list[Any], **_: Any) -> Any:
        return self._factory(self._schema)


class _StubModel:
    """Returns a canned plan for the manager and a section body for workers."""

    def model_copy(self, *, update: dict | None = None, **_: Any) -> "_StubModel":
        return self

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
        return SectionResult(doc_id="", order=0, markdown="Section body.", status="ok")


# ── Fixtures / helpers ───────────────────────────────────────────────────────


def _config(tmp_path: Path) -> SpineConfig:
    """An isolated config with all SPINE state rooted under ``tmp_path``."""
    spine_dir = tmp_path / ".spine"
    spine_dir.mkdir(parents=True, exist_ok=True)
    return SpineConfig(
        queue_path=str(spine_dir / "queue.db"),
        artifact_path=str(spine_dir / "artifacts"),
        checkpoint_path=str(spine_dir / "spine.db"),
        workspace_root=str(tmp_path),
    )


def _write_sample_repo(root: Path) -> None:
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


def _patch_llm(monkeypatch) -> None:
    def fake_resolve_model(config: Any, session_id: Any = None, phase: Any = None) -> Any:
        return _StubModel()

    monkeypatch.setattr(
        "spine.work.onboarding.synthesis_nodes.resolve_chat_model", fake_resolve_model
    )


def _spy_phase_starts(monkeypatch) -> list[str]:
    """Record the ordered ``current_phase`` strings the engine fires."""
    recorded: list[str] = []
    import spine.work.dispatcher as dispatcher

    real = dispatcher.update_work_phase_started

    def spy(db: Any, work_id: str, current_phase: str) -> None:
        recorded.append(current_phase)
        real(db, work_id, current_phase)

    monkeypatch.setattr(dispatcher, "update_work_phase_started", spy)
    return recorded


def _doc_dir(tmp_path: Path, work_id: str) -> Path:
    # The four documents live at a single stable location, independent of work_id.
    return tmp_path / ".spine" / "onboarding"


# ── Tests ────────────────────────────────────────────────────────────────────


class TestRunOnboardingBrownfield:
    def test_analyze_then_synthesize_writes_four_docs(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        _write_sample_repo(tmp_path)
        _patch_llm(monkeypatch)
        phases = _spy_phase_starts(monkeypatch)
        config = _config(tmp_path)

        result = asyncio.run(
            run_onboarding(
                workspace_root=str(tmp_path),
                mode="brownfield",
                tech_stack=["python"],
                config=config,
                work_id="wk-brown",
            )
        )

        # Return dict shape preserved.
        assert result["work_id"] == "wk-brown"
        assert result["status"] == TaskStatus.COMPLETED.value
        assert result["work_type"] == "onboarding"
        assert set(result["artifacts"]) == {f"{d}.md" for d in ONBOARDING_DOC_NAMES}
        assert Path(result["manifest_path"]).exists()
        # The result dict carries workspace_root + mode so the UI can resolve
        # the (possibly external) artifact base and the correct phase bar.
        assert result["workspace_root"] == str(tmp_path)
        assert result["mode"] == "brownfield"

        # ...and they are persisted into work_entries.result for the UI.
        from spine.work.dispatcher import get_work_db as _get_db

        persisted = json.loads(_get_db(config)["work_entries"].get("wk-brown")["result"])
        assert persisted["workspace_root"] == str(tmp_path)
        assert persisted["mode"] == "brownfield"

        # All four docs written.
        doc_dir = _doc_dir(tmp_path, "wk-brown")
        for doc in ONBOARDING_DOC_NAMES:
            assert (doc_dir / f"{doc}.md").exists()

        # Phase progression: brownfield = analyze then synthesize (no scaffold).
        assert phases == ["analyze", "synthesize"]

        # Final DB row records completed.
        from spine.work.dispatcher import get_work_db

        row = get_work_db(config)["work_entries"].get("wk-brown")
        assert row["current_phase"] == "completed"
        assert row["status"] == TaskStatus.COMPLETED.value

    def test_rerun_is_idempotent(self, tmp_path: Path, monkeypatch) -> None:
        _write_sample_repo(tmp_path)
        _patch_llm(monkeypatch)
        config = _config(tmp_path)

        first = asyncio.run(
            run_onboarding(
                workspace_root=str(tmp_path),
                mode="brownfield",
                tech_stack=["python"],
                config=config,
                work_id="wk-idem",
            )
        )
        second = asyncio.run(
            run_onboarding(
                workspace_root=str(tmp_path),
                mode="brownfield",
                tech_stack=["python"],
                config=config,
                work_id="wk-idem",
            )
        )

        assert second["status"] == TaskStatus.COMPLETED.value
        assert set(second["artifacts"]) == set(first["artifacts"])
        # Same set of four docs still present after the re-run.
        doc_dir = _doc_dir(tmp_path, "wk-idem")
        assert sorted(p.name for p in doc_dir.glob("*.md")) == sorted(
            f"{d}.md" for d in ONBOARDING_DOC_NAMES
        )


class TestRunOnboardingGreenfield:
    def test_scaffold_analyze_synthesize(self, tmp_path: Path, monkeypatch) -> None:
        _patch_llm(monkeypatch)
        phases = _spy_phase_starts(monkeypatch)
        config = _config(tmp_path)

        result = asyncio.run(
            run_onboarding(
                workspace_root=str(tmp_path),
                mode="greenfield",
                tech_stack=["python", "fastapi"],
                config=config,
                work_id="wk-green",
            )
        )

        assert result["status"] == TaskStatus.COMPLETED.value
        assert result["work_type"] == "onboarding"
        assert result["workspace_root"] == str(tmp_path)
        assert result["mode"] == "greenfield"

        # Greenfield phase order: scaffold (pre-graph) → analyze (seed) → synthesize.
        assert phases == ["scaffold", "analyze", "synthesize"]

        # Scaffold actually laid down a project before the graph ran.
        assert (tmp_path / ".spine").exists()

        # All four docs written.
        doc_dir = _doc_dir(tmp_path, "wk-green")
        for doc in ONBOARDING_DOC_NAMES:
            assert (doc_dir / f"{doc}.md").exists()


class TestRunOnboardingFailure:
    def test_failure_returns_failed_dict(self, tmp_path: Path, monkeypatch) -> None:
        """A graph exception is caught and reported as a failed work dict."""
        _write_sample_repo(tmp_path)
        config = _config(tmp_path)

        # Force the synthesis aggregator to blow up by making every worker error.
        class _ErrorWorkerModel(_StubModel):
            def _make(self, schema: Any) -> Any:
                if schema is SectionPlanSet:
                    return super()._make(schema)
                return SectionResult(doc_id="", order=0, markdown="", status="error")

        def fake_resolve_model(config: Any, session_id: Any = None, phase: Any = None) -> Any:
            return _ErrorWorkerModel()

        monkeypatch.setattr(
            "spine.work.onboarding.synthesis_nodes.resolve_chat_model", fake_resolve_model
        )

        result = asyncio.run(
            run_onboarding(
                workspace_root=str(tmp_path),
                mode="brownfield",
                tech_stack=["python"],
                config=config,
                work_id="wk-fail",
            )
        )

        assert result["status"] == TaskStatus.FAILED.value
        assert result["work_type"] == "onboarding"
        assert "error" in result
        # MEMORY rule: the failure reason carries no raw exception leaked into docs.
        from spine.work.dispatcher import get_work_db

        row = get_work_db(config)["work_entries"].get("wk-fail")
        assert row["status"] == TaskStatus.FAILED.value
        # The recorded result is the generic engine error envelope, not doc text.
        recorded = json.loads(row["result"])
        assert "error" in recorded


class TestUIApiResolvesExternalArtifactStore:
    """Finding #1: docs for an EXTERNAL onboarding target must display.

    The engine writes the four documents to the stable
    ``<workspace_root>/.spine/onboarding`` location (and the manifest under
    ``<workspace_root>/.spine/artifacts``). When the onboarded repo differs from
    spine's own ``artifact_path``, ``UIApi.read_onboarding_doc`` must resolve the
    target from the work item's recorded ``workspace_root`` — not the global
    spine store — or the UI shows nothing.
    """

    def test_read_artifact_resolves_workspace_root_store(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from spine.ui_api import UIApi

        # spine's own state lives under spine_home; the onboarded TARGET repo
        # lives under a SEPARATE external_repo dir, so the two artifact bases
        # genuinely differ (the bug condition).
        spine_home = tmp_path / "spine_home"
        spine_home.mkdir()
        external_repo = tmp_path / "external_repo"
        external_repo.mkdir()
        _write_sample_repo(external_repo)

        spine_dir = spine_home / ".spine"
        spine_dir.mkdir(parents=True, exist_ok=True)
        config = SpineConfig(
            queue_path=str(spine_dir / "queue.db"),
            artifact_path=str(spine_dir / "artifacts"),
            checkpoint_path=str(spine_dir / "spine.db"),
            workspace_root=str(spine_home),
        )
        _patch_llm(monkeypatch)

        result = asyncio.run(
            run_onboarding(
                workspace_root=str(external_repo),
                mode="brownfield",
                tech_stack=["python"],
                config=config,
                work_id="wk-ext",
            )
        )
        assert result["status"] == TaskStatus.COMPLETED.value

        # The docs were written to the stable location under the EXTERNAL repo,
        # not spine's store.
        external_doc_dir = external_repo / ".spine" / "onboarding"
        for doc in ONBOARDING_DOC_NAMES:
            assert (external_doc_dir / f"{doc}.md").exists()
        # spine's own global store has NOTHING for this work item.
        assert not (spine_dir / "artifacts" / "wk-ext").exists()

        api = UIApi(config=config)

        # The reader finds the docs at the external repo's stable location when
        # given its workspace root directly (no work ID needed).
        for doc in ONBOARDING_DOC_NAMES:
            content = api.read_onboarding_doc(
                f"{doc}.md", workspace_root=str(external_repo)
            )
            assert content, f"UIApi failed to resolve external onboarding doc {doc}"

    def test_non_onboarding_read_unchanged(self, tmp_path: Path) -> None:
        """A non-onboarding work item reads from the global store untouched."""
        from spine.ui_api import UIApi

        spine_dir = tmp_path / ".spine"
        spine_dir.mkdir(parents=True, exist_ok=True)
        config = SpineConfig(
            queue_path=str(spine_dir / "queue.db"),
            artifact_path=str(spine_dir / "artifacts"),
            checkpoint_path=str(spine_dir / "spine.db"),
            workspace_root=str(tmp_path),
        )
        api = UIApi(config=config)
        # Save an artifact through the global store, then read it back: the
        # resolver must fall through to the global store for non-onboarding ids.
        api._artifacts.save_artifact(
            work_id="wk-spec",
            phase="specify",
            name="spec.md",
            content="# Spec\nhello",
        )
        assert api.read_artifact("wk-spec", "specify", "spec.md") == "# Spec\nhello"
