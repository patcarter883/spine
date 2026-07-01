"""Tests for the synthesis + placement editor (no-tool IMPLEMENT path).

Synthesis itself is an LLM call and is not exercised here; these cover the
deterministic placement half — the part that makes the editor unable to spiral
and that scores best-of-N candidates with the linter as the oracle.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spine.agents.synthesis_implementer import (
    PlacementResult,
    SynthesizedEdit,
    SynthesizedSlice,
    _count_ruff,
    _is_stub_body,
    _stage_files,
    apply_synthesized,
    place_best_candidate,
)

_ORIG = "def greet(name):\n    return 'hi ' + name\n"


def _slice(code: str, *, file: str = "mod.py", symbol: str = "greet") -> SynthesizedSlice:
    return SynthesizedSlice(
        edits=[SynthesizedEdit(file=file, symbol=symbol, action="replace", code=code)]
    )


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "mod.py").write_text(_ORIG, encoding="utf-8")
    return tmp_path


def test_clean_edit_applies_and_writes(workspace: Path) -> None:
    cand = _slice("def greet(name):\n    return f'hello {name}!'\n")
    res = apply_synthesized(cand, workspace_root=str(workspace), target_files=["mod.py"])
    assert res.n_applied == 1 and res.n_failures == 0
    assert res.clean
    assert "hello" in (workspace / "mod.py").read_text()


def test_syntax_error_reverts_and_fails(workspace: Path) -> None:
    cand = _slice("def greet(name)  return 1\n")  # missing colon → syntax error
    res = apply_synthesized(cand, workspace_root=str(workspace), target_files=["mod.py"])
    assert res.n_applied == 0 and res.n_failures == 1
    assert res.failures[0]["status"] == "syntax_error"
    # The linter is the oracle: a failed lint leaves the file byte-identical.
    assert (workspace / "mod.py").read_text() == _ORIG


def test_out_of_scope_file_rejected_locally(workspace: Path) -> None:
    cand = SynthesizedSlice(
        edits=[SynthesizedEdit(file="elsewhere.py", symbol="x", action="replace", code="x = 1\n")]
    )
    res = apply_synthesized(cand, workspace_root=str(workspace), target_files=["mod.py"])
    assert res.n_failures == 1
    assert res.failures[0]["status"] == "out_of_scope"
    assert not (workspace / "elsewhere.py").exists()


def test_best_of_n_keeps_cleanest_candidate(workspace: Path) -> None:
    broken = SynthesizedSlice(summary="A", edits=[
        SynthesizedEdit(file="mod.py", symbol="greet", action="replace", code="def greet(  oops\n")
    ])
    clean = SynthesizedSlice(summary="B", edits=[
        SynthesizedEdit(file="mod.py", symbol="greet", action="replace",
                        code="def greet(name):\n    return name.upper()\n")
    ])
    winner, res = place_best_candidate(
        [broken, clean], workspace_root=str(workspace), target_files=["mod.py"]
    )
    assert winner is not None and winner.summary == "B"
    assert res.n_applied == 1 and res.n_failures == 0
    assert "upper()" in (workspace / "mod.py").read_text()


def test_best_of_n_single_candidate_skips_snapshot(workspace: Path) -> None:
    cand = _slice("def greet(name):\n    return name[::-1]\n")
    winner, res = place_best_candidate(
        [cand], workspace_root=str(workspace), target_files=["mod.py"]
    )
    assert winner is cand and res.clean


def test_best_of_n_empty_returns_none(workspace: Path) -> None:
    winner, res = place_best_candidate(
        [], workspace_root=str(workspace), target_files=["mod.py"]
    )
    assert winner is None and res.n_applied == 0


def test_placement_score_ordering() -> None:
    a = PlacementResult(applied=[{}, {}], failures=[])
    b = PlacementResult(applied=[{}], failures=[{}])
    c = PlacementResult(applied=[{}, {}], failures=[], ruff_issues=3)
    assert a.score() > b.score()          # more applied, no failures wins
    assert a.score() > c.score()          # fewer ruff issues breaks the tie


def test_stage_files_copies_existing_only(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    staging = _stage_files(str(tmp_path), ["a.py", "missing.py"])
    sp = Path(staging)
    assert (sp / "a.py").read_text() == "x = 1\n"
    assert not (sp / "missing.py").exists()  # absent in live tree → not staged


def test_best_of_n_scoring_does_not_touch_live_tree(workspace: Path) -> None:
    """Losing candidates must never mutate the live file — only the winner does.

    Regression for the 019efd92 snapshot/restore stomp: scoring happens in temp
    copies, so the live file changes exactly once (to the winner's content).
    """
    losing = SynthesizedSlice(summary="A", edits=[
        SynthesizedEdit(file="mod.py", symbol="greet", action="replace",
                        code="def greet(name):\n    return 'A'\n")
    ])
    winning = SynthesizedSlice(summary="B", edits=[
        SynthesizedEdit(file="mod.py", symbol="greet", action="replace",
                        code="def greet(name):\n    return 'B' + name\n")
    ])
    before = (workspace / "mod.py").read_text()
    winner, res = place_best_candidate(
        [losing, winning], workspace_root=str(workspace), target_files=["mod.py"]
    )
    text = (workspace / "mod.py").read_text()
    # Both candidates apply+lint clean, so KeepBest is a tie on the score key and
    # the first (losing) wins deterministically — the point of THIS test is only
    # that the live file holds exactly ONE candidate's edit, never a half-reverted
    # mix, and was untouched during scoring.
    assert winner is not None and res.clean
    assert text != before
    assert ("return 'A'" in text) ^ ("return 'B' + name" in text)


@pytest.mark.parametrize(
    "ruff,expected",
    [(None, 0), ([], 0), (["a", "b"], 2), ("", 0), ("one\ntwo", 2), ({"count": 4}, 4)],
)
def test_count_ruff(ruff: object, expected: int) -> None:
    assert _count_ruff(ruff) == expected


# ── stub-body detection (019f1bed: verify caught after the fact; catch it here)


@pytest.mark.parametrize(
    "code",
    [
        "def remove_provider(self, name):\n    pass\n",
        "def remove_provider(self, name):\n    ...\n",
        "def remove_provider(self, name):\n    raise NotImplementedError\n",
        "def remove_provider(self, name):\n    raise NotImplementedError()\n",
        'def remove_provider(self, name):\n    """Docstring only."""\n    pass\n',
    ],
)
def test_is_stub_body_detects_bare_stubs(code: str) -> None:
    assert _is_stub_body(code)


@pytest.mark.parametrize(
    "code",
    [
        "def remove_provider(self, name):\n    self._providers.pop(name, None)\n",
        # A `pass` inside a real branch, not the whole body — not a stub.
        "def f(x):\n    if x:\n        pass\n    return x\n",
        # Non-function edits (e.g. a class or constant) are not stub-checked.
        "class C:\n    pass\n",
        "X = 1\n",
    ],
)
def test_is_stub_body_ignores_real_code(code: str) -> None:
    assert not _is_stub_body(code)


def test_stub_body_places_but_counts_against_score(workspace: Path) -> None:
    # The edit is syntactically valid and lints clean, so it still gets
    # written — but a stub body must not silently score as a clean success.
    cand = _slice("def greet(name):\n    pass\n")
    res = apply_synthesized(cand, workspace_root=str(workspace), target_files=["mod.py"])
    assert res.n_applied == 1  # the write did land
    assert res.n_failures == 1
    assert res.failures[0]["status"] == "stub_body"
    assert not res.clean
    assert "pass" in (workspace / "mod.py").read_text()
