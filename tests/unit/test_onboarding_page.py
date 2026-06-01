"""Unit tests for the Streamlit Onboarding page (spine.ui._pages.onboarding).

These tests inject a lightweight fake ``streamlit`` module into ``sys.modules``
before importing the page so ``render(api)`` can run without a live Streamlit
session.  The fake records widget calls and lets tests script return values
(e.g. which radio option is selected, what text inputs contain, whether the
Execute button was clicked).

The page must:
  * call ``api.enqueue_onboarding(workspace_root, mode, tech_stack)`` with the
    toggled mode + entered path when Execute is clicked,
  * default the project-path input to ``api._config.workspace_root``,
  * read review documents exclusively via ``api.read_onboarding_doc(...)``,
  * never import ``spine.work.dispatcher`` or ``spine.work.onboarding`` (the
    onboarding engine) directly — asserted via source inspection.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ── Fake streamlit ──


class _FakeColumn:
    """A column / container context manager that forwards widget calls."""

    def __init__(self, st: "_FakeStreamlit") -> None:
        self._st = st

    def __enter__(self) -> "_FakeColumn":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def __getattr__(self, name: str):  # noqa: ANN001 - dynamic widget forwarding
        # Forward any widget method (caption, markdown, metric, write, ...)
        return getattr(self._st, name)


class _FakeStreamlit(types.ModuleType):
    """A minimal fake of the streamlit module for headless page rendering."""

    def __init__(self) -> None:
        super().__init__("streamlit")
        # Scripted inputs keyed by widget key.
        self.radio_values: dict[str, str] = {}
        self.text_input_values: dict[str, str] = {}
        self.button_returns: dict[str, bool] = {}
        # Recorded calls.
        self.errors: list[str] = []
        self.successes: list[str] = []
        self.markdowns: list[str] = []
        self.captions: list[str] = []

    # -- layout / context managers --
    def columns(self, spec: Any) -> list[_FakeColumn]:
        n = spec if isinstance(spec, int) else len(spec)
        return [_FakeColumn(self) for _ in range(n)]

    def expander(self, *_a: Any, **_k: Any) -> _FakeColumn:
        return _FakeColumn(self)

    def container(self, *_a: Any, **_k: Any) -> _FakeColumn:
        return _FakeColumn(self)

    def spinner(self, *_a: Any, **_k: Any) -> _FakeColumn:
        return _FakeColumn(self)

    # -- fragment decorator (no-op passthrough) --
    def fragment(self, *_a: Any, **_k: Any):  # noqa: ANN001
        def _decorator(fn):  # noqa: ANN001
            return fn

        return _decorator

    # -- widgets --
    def radio(self, label: str, options: list[str], **kwargs: Any) -> str:
        key = kwargs.get("key", label)
        return self.radio_values.get(key, options[0])

    def text_input(self, label: str, value: str = "", **kwargs: Any) -> str:
        key = kwargs.get("key", label)
        return self.text_input_values.get(key, value)

    def text_area(self, label: str, value: str = "", **kwargs: Any) -> str:
        key = kwargs.get("key", label)
        return self.text_input_values.get(key, value)

    def button(self, label: str, **kwargs: Any) -> bool:
        key = kwargs.get("key", label)
        return self.button_returns.get(key, False)

    # -- output sinks --
    def title(self, *a: Any, **k: Any) -> None:
        return None

    def subheader(self, *a: Any, **k: Any) -> None:
        return None

    def divider(self, *a: Any, **k: Any) -> None:
        return None

    def metric(self, *a: Any, **k: Any) -> None:
        return None

    def write(self, *a: Any, **k: Any) -> None:
        return None

    def code(self, *a: Any, **k: Any) -> None:
        return None

    def info(self, *a: Any, **k: Any) -> None:
        return None

    def warning(self, *a: Any, **k: Any) -> None:
        return None

    def caption(self, text: str = "", *a: Any, **k: Any) -> None:
        self.captions.append(text)

    def markdown(self, text: str = "", *a: Any, **k: Any) -> None:
        self.markdowns.append(text)

    def error(self, text: str = "", *a: Any, **k: Any) -> None:
        self.errors.append(text)

    def success(self, text: str = "", *a: Any, **k: Any) -> None:
        self.successes.append(text)

    def rerun(self, *a: Any, **k: Any) -> None:
        return None


def _load_onboarding():
    """Import / reload the page module so it binds to the active streamlit."""
    sys.modules.pop("spine.ui._pages.onboarding", None)
    return importlib.import_module("spine.ui._pages.onboarding")


@pytest.fixture
def fake_st(monkeypatch: pytest.MonkeyPatch) -> _FakeStreamlit:
    """Install a fake streamlit module so the page renders headlessly."""
    st = _FakeStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", st)
    return st


def _make_api(workspace_root: str = "/tmp/proj") -> MagicMock:
    """Build a stub UIApi with the methods the page calls."""
    api = MagicMock()
    api._config = types.SimpleNamespace(workspace_root=workspace_root)
    api.enqueue_onboarding.return_value = {
        "queue_id": 7,
        "status": "pending",
        "work_type": "onboarding",
    }
    api.get_queue_overview.return_value = {
        "pending": [],
        "active": None,
        "recent": [],
        "status_summary": {},
    }
    api.get_artifacts.return_value = []
    api.read_artifact.return_value = None
    api.read_onboarding_doc.return_value = None
    return api


# ── Tests ──


def test_render_does_not_raise(fake_st: _FakeStreamlit) -> None:
    onboarding = _load_onboarding()

    api = _make_api()
    onboarding.render(api)  # should not raise


def test_path_defaults_to_workspace_root(fake_st: _FakeStreamlit) -> None:
    onboarding = _load_onboarding()

    api = _make_api(workspace_root="/srv/myrepo")
    captured: dict[str, str] = {}

    real_text_input = fake_st.text_input

    def _capture(label: str, value: str = "", **kwargs: Any) -> str:
        if kwargs.get("key") == "onboarding_path":
            captured["default"] = value
        return real_text_input(label, value, **kwargs)

    fake_st.text_input = _capture  # type: ignore[assignment]
    onboarding.render(api)
    assert captured["default"] == "/srv/myrepo"


def test_execute_brownfield_calls_enqueue_with_mode_and_path(
    fake_st: _FakeStreamlit,
) -> None:
    onboarding = _load_onboarding()

    api = _make_api()
    fake_st.radio_values["onboarding_mode"] = "Brownfield"
    fake_st.text_input_values["onboarding_path"] = "/work/legacy-app"
    fake_st.button_returns["onboarding_execute"] = True

    onboarding.render(api)

    api.enqueue_onboarding.assert_called_once()
    args, _ = api.enqueue_onboarding.call_args
    assert args[0] == "/work/legacy-app"
    assert args[1] == "brownfield"
    # brownfield: no tech stack -> None
    assert args[2] is None


def test_execute_greenfield_passes_tech_stack(fake_st: _FakeStreamlit) -> None:
    onboarding = _load_onboarding()

    api = _make_api()
    fake_st.radio_values["onboarding_mode"] = "Greenfield"
    fake_st.text_input_values["onboarding_path"] = "/work/new-app"
    fake_st.text_input_values["onboarding_tech_stack"] = "python, langgraph , streamlit"
    fake_st.button_returns["onboarding_execute"] = True

    onboarding.render(api)

    args, _ = api.enqueue_onboarding.call_args
    assert args[0] == "/work/new-app"
    assert args[1] == "greenfield"
    assert args[2] == ["python", "langgraph", "streamlit"]


def test_execute_blank_path_does_not_enqueue(fake_st: _FakeStreamlit) -> None:
    onboarding = _load_onboarding()

    api = _make_api(workspace_root="")
    fake_st.text_input_values["onboarding_path"] = "   "
    fake_st.button_returns["onboarding_execute"] = True

    onboarding.render(api)

    api.enqueue_onboarding.assert_not_called()
    assert fake_st.errors  # an error was surfaced


def test_no_execute_means_no_enqueue(fake_st: _FakeStreamlit) -> None:
    onboarding = _load_onboarding()

    api = _make_api()
    # Button defaults to False.
    onboarding.render(api)
    api.enqueue_onboarding.assert_not_called()


def test_artifact_review_reads_via_api_only(fake_st: _FakeStreamlit) -> None:
    onboarding = _load_onboarding()

    api = _make_api()

    def _read(work_id: str, name: str) -> str:
        return f"# {name}\ncontent for {work_id}"

    api.read_onboarding_doc.side_effect = _read
    fake_st.text_input_values["onboarding_review_id"] = "abc123"

    onboarding.render(api)

    # All four docs read through api.read_onboarding_doc.
    read_calls = api.read_onboarding_doc.call_args_list
    read_names = {c.args[1] for c in read_calls}
    for doc in (
        "PROJECT_DEFINITION.md",
        "CODING_GUIDELINES.md",
        "ARCHITECTURE_MAP.md",
        "SPINE_ASSISTANCE_REQUIREMENTS.md",
    ):
        assert doc in read_names
    # Rendered content surfaced via st.markdown.
    assert any("content for abc123" in m for m in fake_st.markdowns)


def test_progress_renders_active_onboarding(fake_st: _FakeStreamlit) -> None:
    onboarding = _load_onboarding()

    api = _make_api()
    api.get_queue_overview.return_value = {
        "pending": [],
        "active": {
            "id": 3,
            "work_id": "wid42",
            "work_type": "onboarding",
            "mode": "brownfield",
            "status": "running",
            "current_phase": "analyze",
            "created_at": "2026-05-29T10:00:00",
            "updated_at": "2026-05-29T10:01:00",
            "description": "Onboard legacy app",
        },
        "recent": [],
        "status_summary": {},
    }
    onboarding.render(api)  # should not raise


def test_phase_bar_helper_handles_completed(fake_st: _FakeStreamlit) -> None:
    onboarding = _load_onboarding()

    # Should not raise for any of: known mid phase, completed, unknown.
    onboarding._render_phase_bar(["analyze", "synthesize"], "analyze")
    onboarding._render_phase_bar(["analyze", "synthesize"], "completed")
    onboarding._render_phase_bar(["analyze", "synthesize"], "???")
    onboarding._render_phase_bar([], "analyze")


def test_phases_for_active_brownfield(fake_st: _FakeStreamlit) -> None:
    """A brownfield active job selects the analyze→synthesize sequence."""
    onboarding = _load_onboarding()

    phases = onboarding._phases_for_active({"mode": "brownfield"})
    assert phases == ["analyze", "synthesize"]


def test_phases_for_active_greenfield_is_scaffold_first(
    fake_st: _FakeStreamlit,
) -> None:
    """Greenfield order is scaffold-first so the progress bar stays monotonic
    (the engine fires "scaffold" pre-graph before "analyze")."""
    onboarding = _load_onboarding()

    phases = onboarding._phases_for_active({"mode": "greenfield"})
    assert phases == ["scaffold", "analyze", "synthesize"]
    # scaffold precedes analyze.
    assert phases.index("scaffold") < phases.index("analyze")


def test_phases_for_active_unknown_mode_defaults_to_greenfield_superset(
    fake_st: _FakeStreamlit,
) -> None:
    """Missing/unknown mode falls back to the greenfield (scaffold-first) list."""
    onboarding = _load_onboarding()

    assert onboarding._phases_for_active({}) == ["scaffold", "analyze", "synthesize"]
    assert onboarding._phases_for_active({"mode": "???"}) == [
        "scaffold",
        "analyze",
        "synthesize",
    ]


def test_page_does_not_import_dispatcher_or_engine() -> None:
    """Zero-duplication: the page must not import the dispatcher, engine, or graph.

    The page may import the dependency-free ``spine.work.onboarding.phases``
    constants module (the shared phase-bar vocabulary, single source of truth
    with the engine) — but NOT the dispatcher, the onboarding engine, the
    workflow graph, or any other backend module that does real work, so all
    backend access still flows through ``UIApi``.
    """
    source = (
        Path(__file__).resolve().parent.parent.parent
        / "spine"
        / "ui"
        / "_pages"
        / "onboarding.py"
    ).read_text()
    assert "spine.work.dispatcher" not in source
    assert "spine.work.onboarding.engine" not in source
    assert "spine.work.onboarding.onboarding_graph" not in source
    assert "spine.work.onboarding.synthesis" not in source
    assert "spine.work.onboarding.analysis" not in source
    assert "build_workflow_graph" not in source
    # The page may ONLY reach into the dependency-free phase-constant module.
    for line in source.splitlines():
        if "spine.work.onboarding" in line:
            assert "spine.work.onboarding.phases" in line
