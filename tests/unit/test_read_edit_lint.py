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
