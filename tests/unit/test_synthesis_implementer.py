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


def test_creation_anchor_failure_preserves_target_and_next_action(tmp_path: Path) -> None:
    # trace 019f1c10: the synthesizer anchored a NEW method on itself
    # (action='replace' on a symbol that doesn't exist yet, so the code
    # defining it is a CREATE, not an edit). The read_edit_lint guard for
    # this returns a concrete `target` (existing sibling to anchor on
    # instead) and `next_action` (the exact corrective call) — but
    # apply_synthesized used to collapse those into `detail`-or-nothing,
    # so the retry synthesis call never saw the fix-it instruction and
    # re-emitted the same broken anchor for two full cycles.
    (tmp_path / "svc.py").write_text(
        "class UIApi:\n"
        "    def add_llm_provider(self, name, cfg):\n"
        "        return True\n"
    )
    cand = SynthesizedSlice(edits=[SynthesizedEdit(
        file="svc.py",
        symbol="UIApi.add_embedding_provider",
        action="replace",
        code="def add_embedding_provider(self, name):\n    return True\n",
    )])
    res = apply_synthesized(cand, workspace_root=str(tmp_path), target_files=["svc.py"])
    assert res.n_failures == 1
    failure = res.failures[0]
    assert failure["status"] == "no_match"
    assert failure.get("target") == "UIApi.add_llm_provider"
    assert "insert_after" in failure.get("next_action", "")
    assert "UIApi.add_llm_provider" in failure["next_action"]


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


def test_new_target_file_is_created_via_full_replace(tmp_path: Path) -> None:
    """A synthesized edit whose target file doesn't exist yet must CREATE it.

    Regression (run 019f40ac): a new test-file slice synthesized correct
    content three cycles in a row, but placement always routed through
    ast_edit, which bounces not_found on a nonexistent file — the slice
    failed identically every cycle and the run parked with an empty diff.
    """
    cand = SynthesizedSlice(edits=[SynthesizedEdit(
        file="tests/test_new.py",
        symbol="<module>",
        action="replace",
        code="def test_truth():\n    assert True\n",
    )])
    res = apply_synthesized(
        cand, workspace_root=str(tmp_path), target_files=["tests/test_new.py"]
    )
    assert res.n_applied == 1 and res.n_failures == 0
    created = tmp_path / "tests" / "test_new.py"
    assert created.is_file()
    assert "def test_truth" in created.read_text(encoding="utf-8")


def test_new_target_file_with_syntax_error_is_not_created(tmp_path: Path) -> None:
    """Creation still goes through the linter: broken content is refused."""
    cand = SynthesizedSlice(edits=[SynthesizedEdit(
        file="tests/test_new.py",
        symbol="<module>",
        action="replace",
        code="def test_truth(  oops\n",
    )])
    res = apply_synthesized(
        cand, workspace_root=str(tmp_path), target_files=["tests/test_new.py"]
    )
    assert res.n_applied == 0 and res.n_failures == 1
    assert not (tmp_path / "tests" / "test_new.py").exists()


class TestWholeFileReplaceBlocked:
    """Wholesale replacement of an EXISTING file is a policy failure, not a
    coincidence of anchor resolution (run 019f81f1: a rework candidate
    emitted action=replace symbol='routes/api.php' against the 328-line
    shared route file — every route not reproduced would have been
    silently deleted)."""

    def test_file_path_symbol_on_existing_file_is_blocked(self, workspace: Path) -> None:
        cand = _slice("print('whole new file')\n", symbol="mod.py")
        res = apply_synthesized(
            cand, workspace_root=str(workspace), target_files=["mod.py"]
        )
        assert res.n_applied == 0 and res.n_failures == 1
        assert res.failures[0]["status"] == "whole_file_replace_blocked"
        assert "insert_after" in res.failures[0]["detail"]
        assert (workspace / "mod.py").read_text() == _ORIG  # untouched

    def test_slash_path_symbol_is_blocked(self, tmp_path: Path) -> None:
        (tmp_path / "routes").mkdir()
        (tmp_path / "routes" / "api.php").write_text("<?php\n", encoding="utf-8")
        cand = _slice(
            "<?php // regenerated\n", file="routes/api.php", symbol="routes/api.php"
        )
        res = apply_synthesized(
            cand, workspace_root=str(tmp_path), target_files=["routes/api.php"]
        )
        assert res.failures[0]["status"] == "whole_file_replace_blocked"
        assert (tmp_path / "routes" / "api.php").read_text() == "<?php\n"

    def test_new_file_creation_path_unaffected(self, tmp_path: Path) -> None:
        # Path-shaped symbol on a file that does NOT exist yet stays on the
        # full_replace creation path — that fix (run 019f40ac) must survive.
        content = "def newfn():\n    return 1\n"
        cand = _slice(content, file="newmod.py", symbol="newmod.py")
        res = apply_synthesized(
            cand, workspace_root=str(tmp_path), target_files=["newmod.py"]
        )
        assert res.n_applied == 1 and res.n_failures == 0
        assert (tmp_path / "newmod.py").read_text() == content

    def test_real_symbols_pass_through(self, workspace: Path) -> None:
        cand = _slice("def greet(name):\n    return name\n", symbol="greet")
        res = apply_synthesized(
            cand, workspace_root=str(workspace), target_files=["mod.py"]
        )
        assert res.n_applied == 1
        # PHP-style double-colon symbols are real symbols, not paths.
        from spine.agents.synthesis_implementer import _is_whole_file_symbol

        assert _is_whole_file_symbol("RouteServiceProvider::boot", "app/P.php") is False
        assert _is_whole_file_symbol("SpineConfig.load", "spine/config.py") is False
        assert _is_whole_file_symbol("api.php", "routes/api.php") is True
        assert _is_whole_file_symbol("", "routes/api.php") is True


class TestPlaceholderCreationBlocked:
    """Creating a file whose content is an elision must FAIL placement
    (run 019f81c1: RainfallCrudTest.php landed as a literal 3-byte '...'
    in two consecutive cycles — PHP syntax check passes bare text, so the
    lint oracle was blind and each cycle burned a verify pass)."""

    def test_ellipsis_creation_blocked(self, tmp_path: Path) -> None:
        cand = _slice("...", file="RainfallCrudTest.php", symbol="RainfallCrudTest.php")
        res = apply_synthesized(
            cand, workspace_root=str(tmp_path), target_files=["RainfallCrudTest.php"]
        )
        assert res.n_applied == 0 and res.n_failures == 1
        assert res.failures[0]["status"] == "placeholder_content_blocked"
        assert not (tmp_path / "RainfallCrudTest.php").exists()

    def test_empty_and_tiny_creations_blocked(self, tmp_path: Path) -> None:
        from spine.agents.synthesis_implementer import _is_placeholder_content

        assert _is_placeholder_content("") is True
        assert _is_placeholder_content("pass") is True
        assert _is_placeholder_content("// TODO") is True
        assert _is_placeholder_content("<?php\n") is True  # 5 chars stripped... blocked
        assert _is_placeholder_content("x = 1\ny = 2\n") is False

    def test_real_creation_still_lands(self, tmp_path: Path) -> None:
        cand = _slice(
            "def real():\n    return 42\n", file="real.py", symbol="real.py"
        )
        res = apply_synthesized(
            cand, workspace_root=str(tmp_path), target_files=["real.py"]
        )
        assert res.n_applied == 1
        assert (tmp_path / "real.py").exists()
