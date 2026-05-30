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

No live model is used — the synthesis tiers' ``resolve_model`` is patched to a
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
from spine.work.onboarding.synthesis_tools import ONBOARDING_DOC_NAMES, ONBOARDING_PHASE


# ── Stub bare-LLM model (manager + worker) ───────────────────────────────────


class _StubStructured:
    def __init__(self, schema: Any, factory) -> None:
        self._schema = schema
        self._factory = factory

    async def ainvoke(self, messages: list[Any], **_: Any) -> Any:
        return self._factory(self._schema)


class _StubModel:
    """Returns a canned plan for the manager and a section body for workers."""

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
        "spine.work.onboarding.synthesis_nodes.resolve_model", fake_resolve_model
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
    return tmp_path / ".spine" / "artifacts" / work_id / ONBOARDING_PHASE


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
            "spine.work.onboarding.synthesis_nodes.resolve_model", fake_resolve_model
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
