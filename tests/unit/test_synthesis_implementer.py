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
    _restore_files,
    _snapshot_files,
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


def test_snapshot_restore_roundtrip(tmp_path: Path) -> None:
    f = tmp_path / "a.py"
    f.write_text("x = 1\n", encoding="utf-8")
    snap = _snapshot_files(str(tmp_path), ["a.py", "new.py"])  # new.py absent
    f.write_text("x = 999\n", encoding="utf-8")
    (tmp_path / "new.py").write_text("created\n", encoding="utf-8")
    _restore_files(str(tmp_path), snap)
    assert f.read_text() == "x = 1\n"
    assert not (tmp_path / "new.py").exists()  # absent in snapshot → deleted


@pytest.mark.parametrize(
    "ruff,expected",
    [(None, 0), ([], 0), (["a", "b"], 2), ("", 0), ("one\ntwo", 2), ({"count": 4}, 4)],
)
def test_count_ruff(ruff: object, expected: int) -> None:
    assert _count_ruff(ruff) == expected
