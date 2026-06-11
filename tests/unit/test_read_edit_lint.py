"""Unit tests for :mod:`spine.agents.tools.read_edit_lint`."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from spine.agents.tools.read_edit_lint import ReadEditLintTool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    return tmp_path


def _tool(workspace: Path) -> ReadEditLintTool:
    return ReadEditLintTool(workspace_root=str(workspace))


def _decode(out: str) -> dict:
    return json.loads(out)


def test_full_replace_creates_file_with_valid_python(workspace: Path) -> None:
    out = _decode(
        _tool(workspace)._run(
            file_path="src/hello.py",
            full_replace="def hello():\n    return 'hi'\n",
        )
    )
    assert out["status"] == "ok"
    assert out["created"] is True
    assert (workspace / "src" / "hello.py").read_text() == "def hello():\n    return 'hi'\n"


def test_full_replace_rejects_invalid_python_without_writing(workspace: Path) -> None:
    target = workspace / "src" / "broken.py"
    out = _decode(
        _tool(workspace)._run(
            file_path="src/broken.py",
            full_replace="def hello(:\n    return 'hi'\n",  # syntax error
        )
    )
    assert out["status"] == "syntax_error"
    assert "SyntaxError" in out["detail"]
    assert out["wrote"] is False
    assert not target.exists()


def test_edit_with_indent_break_does_not_change_mtime(workspace: Path) -> None:
    target = workspace / "src" / "indent.py"
    target.write_text("def hello():\n    return 'hi'\n")
    original_mtime = target.stat().st_mtime

    # Make a syntactically-broken edit (un-indented body).
    out = _decode(
        _tool(workspace)._run(
            file_path="src/indent.py",
            old_str="    return 'hi'",
            new_str="return 'hi'",  # missing indent
        )
    )
    assert out["status"] == "syntax_error"
    assert target.read_text() == "def hello():\n    return 'hi'\n"
    # mtime preserved (atomic write would have changed it)
    assert target.stat().st_mtime == original_mtime


def test_no_match_returns_clear_error(workspace: Path) -> None:
    target = workspace / "src" / "miss.py"
    target.write_text("x = 1\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/miss.py",
            old_str="not in file",
            new_str="replacement",
        )
    )
    assert out["status"] == "no_match"
    assert "not found" in out["detail"]


def test_ambiguous_match_returns_clear_error(workspace: Path) -> None:
    target = workspace / "src" / "dupe.py"
    target.write_text("x = 1\nx = 1\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/dupe.py",
            old_str="x = 1\n",
            new_str="x = 2\n",
        )
    )
    assert out["status"] == "ambiguous_match"
    assert "2 locations" in out["detail"]


def test_exclusive_args_validated(workspace: Path) -> None:
    out = _decode(
        _tool(workspace)._run(
            file_path="src/x.py",
            old_str="a",
            new_str="b",
            full_replace="full",
        )
    )
    assert out["status"] == "input_error"


def test_unknown_extension_skips_lint(workspace: Path) -> None:
    out = _decode(
        _tool(workspace)._run(
            file_path="src/notes.md",
            full_replace="this is `not python` and that is fine",
        )
    )
    assert out["status"] == "ok"


def test_typescript_syntax_error_is_caught(workspace: Path) -> None:
    target = workspace / "src" / "broken.ts"
    out = _decode(
        _tool(workspace)._run(
            file_path="src/broken.ts",
            full_replace="function broken(: string) { return ; }\n",
        )
    )
    assert out["status"] == "syntax_error"
    assert not target.exists()


def test_leading_slash_resolves_workspace_relative(workspace: Path) -> None:
    out = _decode(
        _tool(workspace)._run(
            file_path="/src/leading_slash.py",
            full_replace="def f(): pass\n",
        )
    )
    assert out["status"] == "ok"
    assert (workspace / "src" / "leading_slash.py").exists()


# ── Batch edits (all-or-nothing) ────────────────────────────────────


def test_batch_edits_apply_in_order(workspace: Path) -> None:
    target = workspace / "src" / "batch.py"
    target.write_text("a = 1\nb = 2\nc = 3\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/batch.py",
            edits=[
                {"old_str": "a = 1", "new_str": "a = 10"},
                {"old_str": "c = 3", "new_str": "c = 30"},
            ],
        )
    )
    assert out["status"] == "ok"
    assert target.read_text() == "a = 10\nb = 2\nc = 30\n"


def test_batch_edits_are_all_or_nothing(workspace: Path) -> None:
    target = workspace / "src" / "atomic.py"
    target.write_text("a = 1\nb = 2\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/atomic.py",
            edits=[
                {"old_str": "a = 1", "new_str": "a = 10"},
                {"old_str": "not present", "new_str": "x"},
            ],
        )
    )
    assert out["status"] == "no_match"
    assert out["edit_index"] == 1
    # First edit must NOT have landed — the whole batch is rolled back.
    assert target.read_text() == "a = 1\nb = 2\n"


def test_batch_edit_sees_earlier_results(workspace: Path) -> None:
    # The second edit targets text produced by the first.
    target = workspace / "src" / "chain.py"
    target.write_text("x = 1\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/chain.py",
            edits=[
                {"old_str": "x = 1", "new_str": "x = 2"},
                {"old_str": "x = 2", "new_str": "x = 3"},
            ],
        )
    )
    assert out["status"] == "ok"
    assert target.read_text() == "x = 3\n"


def test_batch_rejected_when_result_has_syntax_error(workspace: Path) -> None:
    target = workspace / "src" / "syn.py"
    target.write_text("def f():\n    return 1\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/syn.py",
            edits=[{"old_str": "    return 1", "new_str": "return 1"}],  # bad indent
        )
    )
    assert out["status"] == "syntax_error"
    assert target.read_text() == "def f():\n    return 1\n"


# ── Line-range mode ─────────────────────────────────────────────────


def test_line_range_replaces_lines(workspace: Path) -> None:
    target = workspace / "src" / "range.py"
    target.write_text("a = 1\nb = 2\nc = 3\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/range.py",
            start_line=2,
            end_line=2,
            replacement="b = 20",
        )
    )
    assert out["status"] == "ok"
    assert target.read_text() == "a = 1\nb = 20\nc = 3\n"


def test_line_range_deletes_when_replacement_empty(workspace: Path) -> None:
    target = workspace / "src" / "del.py"
    target.write_text("a = 1\nb = 2\nc = 3\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/del.py",
            start_line=2,
            end_line=2,
            replacement="",
        )
    )
    assert out["status"] == "ok"
    assert target.read_text() == "a = 1\nc = 3\n"


def test_line_range_out_of_bounds(workspace: Path) -> None:
    target = workspace / "src" / "oob.py"
    target.write_text("a = 1\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/oob.py",
            start_line=1,
            end_line=5,
            replacement="x = 1",
        )
    )
    assert out["status"] == "range_error"
    assert target.read_text() == "a = 1\n"


def test_line_range_expected_guard_rejects_stale(workspace: Path) -> None:
    target = workspace / "src" / "stale.py"
    target.write_text("a = 1\nb = 2\nc = 3\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/stale.py",
            start_line=2,
            end_line=2,
            replacement="b = 20",
            expected="something else",  # does not match current line 2
        )
    )
    assert out["status"] == "stale"
    assert target.read_text() == "a = 1\nb = 2\nc = 3\n"


def test_line_range_expected_guard_allows_match(workspace: Path) -> None:
    target = workspace / "src" / "fresh.py"
    target.write_text("a = 1\nb = 2\nc = 3\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/fresh.py",
            start_line=2,
            end_line=2,
            replacement="b = 20",
            expected="b = 2",
        )
    )
    assert out["status"] == "ok"
    assert target.read_text() == "a = 1\nb = 20\nc = 3\n"


def test_multiple_modes_rejected(workspace: Path) -> None:
    target = workspace / "src" / "multi.py"
    target.write_text("a = 1\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/multi.py",
            old_str="a = 1",
            new_str="a = 2",
            start_line=1,
            end_line=1,
            replacement="a = 3",
        )
    )
    assert out["status"] == "input_error"


# ── Read mode (trace 019eb502: read_file removed from the implementer) ──


def test_read_whole_file_with_line_numbers(workspace: Path) -> None:
    target = workspace / "src" / "r.py"
    target.write_text("a = 1\nb = 2\nc = 3\n")
    out = _tool(workspace)._run(file_path="src/r.py")
    assert out.startswith("[read: src/r.py lines 1-3 of 3]")
    assert "1| a = 1" in out
    assert "3| c = 3" in out


def test_read_line_range(workspace: Path) -> None:
    target = workspace / "src" / "r2.py"
    target.write_text("".join(f"x{i} = {i}\n" for i in range(1, 11)))
    out = _tool(workspace)._run(file_path="src/r2.py", start_line=4, end_line=6)
    assert "[read: src/r2.py lines 4-6 of 10]" in out
    assert "4| x4 = 4" in out
    assert "6| x6 = 6" in out
    assert "x7" not in out


def test_read_missing_file_reports_not_found(workspace: Path) -> None:
    out = _decode(_tool(workspace)._run(file_path="src/nope.py"))
    assert out["status"] == "not_found"


def test_read_truncates_huge_files_with_notice(workspace: Path) -> None:
    from spine.agents.tools.read_edit_lint import _READ_MAX_LINES

    target = workspace / "src" / "big.py"
    target.write_text("".join(f"v{i} = {i}\n" for i in range(_READ_MAX_LINES + 50)))
    out = _tool(workspace)._run(file_path="src/big.py")
    assert "truncated" in out
    assert f"v{_READ_MAX_LINES + 10}" not in out


def test_line_bounds_without_replacement_is_a_read_not_an_error(workspace: Path) -> None:
    """start_line/end_line with no replacement used to input_error; it is
    now a ranged read — the closest useful interpretation."""
    target = workspace / "src" / "rr.py"
    target.write_text("one = 1\ntwo = 2\n")
    out = _tool(workspace)._run(file_path="src/rr.py", start_line=2, end_line=2)
    assert "2| two = 2" in out


# ── already_applied (trace 019eb502: re-sent edits read as failures) ──


def test_already_applied_when_new_str_present(workspace: Path) -> None:
    target = workspace / "src" / "aa.py"
    target.write_text("value = 2\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/aa.py", old_str="value = 1", new_str="value = 2"
        )
    )
    assert out["status"] == "already_applied"
    assert "do NOT retry" in out["detail"]
    assert target.read_text() == "value = 2\n"


def test_plain_no_match_still_reported_when_new_str_absent(workspace: Path) -> None:
    target = workspace / "src" / "nm.py"
    target.write_text("value = 3\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/nm.py", old_str="value = 1", new_str="value = 2"
        )
    )
    assert out["status"] == "no_match"


def test_batch_already_applied_names_the_edit(workspace: Path) -> None:
    target = workspace / "src" / "ba.py"
    target.write_text("a = 10\nb = 2\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/ba.py",
            edits=[
                {"old_str": "a = 1\n", "new_str": "a = 10\n"},  # already applied
                {"old_str": "b = 2", "new_str": "b = 20"},
            ],
        )
    )
    assert out["status"] == "already_applied"
    assert out["edit_index"] == 0
    assert "WITHOUT edits[0]" in out["detail"]
    # All-or-nothing: the second edit must NOT have been applied.
    assert target.read_text() == "a = 10\nb = 2\n"


# ── ruff report on successful Python writes ──


def test_ok_result_carries_ruff_field(workspace: Path) -> None:
    import shutil

    if shutil.which("ruff") is None:
        pytest.skip("ruff not on PATH")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/lint_me.py",
            full_replace="import os\n\n\nvalue = 1\n",  # F401: unused import
        )
    )
    assert out["status"] == "ok"
    assert "ruff" in out
    assert "F401" in out["ruff"]


def test_clean_write_reports_ruff_clean(workspace: Path) -> None:
    import shutil

    if shutil.which("ruff") is None:
        pytest.skip("ruff not on PATH")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/clean.py",
            full_replace="value = 1\n",
        )
    )
    assert out["status"] == "ok"
    assert out.get("ruff") == "clean"


def test_ruff_unavailable_fails_open(workspace: Path, monkeypatch) -> None:
    import spine.agents.tools.read_edit_lint as rel

    def _raise(*a, **kw):
        raise FileNotFoundError("no ruff")

    monkeypatch.setattr(rel.subprocess, "run", _raise)
    out = _decode(
        _tool(workspace)._run(file_path="src/no_ruff.py", full_replace="v = 1\n")
    )
    assert out["status"] == "ok"
    assert "ruff" not in out
