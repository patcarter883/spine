"""Write-side circuit breaker (trace beaa8507).

A full_replace/edit that fails its lint/match check leaves the file unchanged.
A model that re-sends the same broken content must be stopped so it doesn't burn
the token budget re-paying the whole-file prompt every turn.
"""

from __future__ import annotations

import json

from spine.agents.tools.read_edit_lint import (
    _WRITE_FAIL_CAP,
    _WRITE_PRESSURE_WALL,
    ReadEditLintTool,
)

# Unterminated triple-quoted string — fails the python lint check, never writes.
BROKEN = 'def f():\n    x = """unterminated\n'


def _status(out: str) -> str:
    return json.loads(out)["status"]


def test_consecutive_failures_append_circuit_breaker_hint(tmp_path):
    tool = ReadEditLintTool(workspace_root=str(tmp_path), target_files=["m.py"])

    first = tool._run("m.py", full_replace=BROKEN)
    assert _status(first) == "syntax_error"
    assert "circuit_breaker" not in json.loads(first)

    last = first
    for _ in range(_WRITE_FAIL_CAP):
        last = tool._run("m.py", full_replace=BROKEN)
    # Still a real syntax error, now carrying the steering hint.
    assert _status(last) == "syntax_error"
    assert "circuit_breaker" in json.loads(last)


def test_hard_wall_pauses_writes_after_pressure(tmp_path):
    tool = ReadEditLintTool(workspace_root=str(tmp_path), target_files=["m.py"])
    for _ in range(_WRITE_PRESSURE_WALL):
        tool._run("m.py", full_replace=BROKEN)

    # Writes are now paused regardless of content — even a valid full_replace.
    out = tool._run("m.py", full_replace="x = 1\n")
    assert _status(out) == "write_capped"


def test_successful_edit_resets_the_breaker(tmp_path):
    (tmp_path / "m.py").write_text("x = 1\n")
    tool = ReadEditLintTool(workspace_root=str(tmp_path), target_files=["m.py"])

    for _ in range(_WRITE_FAIL_CAP):
        tool._run("m.py", full_replace=BROKEN)

    ok = tool._run("m.py", full_replace="y = 2\n")
    assert _status(ok) == "ok"

    # Counter cleared: the next failure starts fresh (no breaker hint yet).
    after = tool._run("m.py", full_replace=BROKEN)
    assert _status(after) == "syntax_error"
    assert "circuit_breaker" not in json.loads(after)


def test_mode_conflict_write_counts_toward_breaker(tmp_path):
    """A malformed WRITE call (two edit modes) returns input_error but must also
    count toward the breaker — otherwise it loops unbounded past it (trace
    4aa24c6b: find_replace+patch on config_view.py).
    """
    (tmp_path / "m.py").write_text("x = 1\n")
    tool = ReadEditLintTool(workspace_root=str(tmp_path), target_files=["m.py"])

    first = tool._run(
        "m.py", old_str="x = 1", new_str="x = 2",
        patch=[{"search": "x = 1", "replace": "x = 2"}],
    )
    assert _status(first) == "input_error"

    out = first
    for _ in range(_WRITE_PRESSURE_WALL):
        out = tool._run(
            "m.py", old_str="x = 1", new_str="x = 2",
            patch=[{"search": "x = 1", "replace": "x = 2"}],
        )
    assert _status(out) == "write_capped"


def test_read_mode_conflict_also_counts_toward_breaker(tmp_path):
    """A pure-read mode clash (read_symbol+read_around) is malformed too and must
    trip the breaker — a weak model spun exactly this 1M tokens (trace 019efc1a,
    Mellum2) because the breaker used to skip non-write conflicts.
    """
    (tmp_path / "m.py").write_text("x = 1\n")
    tool = ReadEditLintTool(workspace_root=str(tmp_path), target_files=["m.py"])

    first = tool._run("m.py", read_symbol="Foo", read_around="x = 1")
    assert _status(first) == "input_error"

    out = first
    for _ in range(_WRITE_PRESSURE_WALL):
        out = tool._run("m.py", read_symbol="Foo", read_around="x = 1")
    assert _status(out) == "write_capped"
