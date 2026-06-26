"""Tests for the slice-verifier's bounded tool surface.

Covers the two purpose-built tools that replaced the verifier's raw
filesystem+shell surface (trace 019f0212):

- ``VerifyReadFileTool`` — ranged, line-numbered, output-capped, read-only,
  with re-read de-duplication.
- ``RunChecksTool`` — constrained runner that rejects pure-exploration shell
  commands and delegates legitimate test/lint commands to the execute backend.
"""

from __future__ import annotations

from typing import Any

import pytest

from spine.agents.verify_subagent_tools import (
    RunChecksTool,
    VerifyReadFileTool,
    _leading_executable,
    build_verify_subagent_tools,
)


# ── read_file ──────────────────────────────────────────────────────────────


def _write(tmp_path, name: str, n_lines: int) -> str:
    p = tmp_path / name
    p.write_text("\n".join(f"line {i}" for i in range(n_lines)), encoding="utf-8")
    return str(p)


def test_read_returns_line_numbered_window(tmp_path):
    path = _write(tmp_path, "f.py", 50)
    tool = VerifyReadFileTool(workspace_root=str(tmp_path))
    out = tool._run(path, offset=0, limit=5)
    assert "line 0" in out and "line 4" in out
    assert "line 5" not in out  # respected the limit
    assert "1\tline 0" in out  # line-numbered
    assert "more lines" in out  # footer points to the rest


def test_read_limit_is_capped(tmp_path):
    path = _write(tmp_path, "big.py", 1000)
    tool = VerifyReadFileTool(workspace_root=str(tmp_path))
    out = tool._run(path, offset=0, limit=99999)
    # Only up to _MAX_READ_LINES (400) lines, never the whole 1000-line file.
    assert "line 399" in out
    assert "line 400" not in out


def test_read_relative_path_resolves_against_workspace(tmp_path):
    _write(tmp_path, "rel.py", 3)
    tool = VerifyReadFileTool(workspace_root=str(tmp_path))
    out = tool._run("rel.py", offset=0, limit=10)
    assert "line 0" in out


def test_read_dedup_returns_already_read(tmp_path):
    path = _write(tmp_path, "f.py", 10)
    tool = VerifyReadFileTool(workspace_root=str(tmp_path))
    first = tool._run(path, offset=0, limit=5)
    assert "line 0" in first
    second = tool._run(path, offset=0, limit=5)
    assert "already_read" in second
    assert "line 0" not in second  # body withheld on the repeat


def test_read_missing_and_directory(tmp_path):
    tool = VerifyReadFileTool(workspace_root=str(tmp_path))
    assert "not_found" in tool._run(str(tmp_path / "nope.py"))
    assert "is_directory" in tool._run(str(tmp_path))


# ── run_checks policy ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "command,lead",
    [
        ("pytest tests/unit/test_x.py -q", "pytest"),
        ("ruff check spine/", "ruff"),
        ("cd /repo && pytest -q", "pytest"),
        ("PYTHONPATH=. python -m pytest", "python"),
        ("grep -rn foo spine/", "grep"),
        ("cd /repo && grep -rn foo .", "grep"),
        ("find . -name '*.py'", "find"),
        ("cat spine/config.py", "cat"),
        ("pytest -q | grep FAIL", "pytest"),  # grep as a pipe filter is fine
    ],
)
def test_leading_executable(command, lead):
    assert _leading_executable(command) == lead


class _ExecResp:
    """Duck-typed stand-in for the backend's ExecuteResponse."""

    def __init__(self, output: str, exit_code: int = 0, truncated: bool = False) -> None:
        self.output = output
        self.exit_code = exit_code
        self.truncated = truncated


class _FakeBackend:
    """Stand-in for the sandbox backend (execute/aexecute -> ExecuteResponse)."""

    def __init__(self, output: str = "ok") -> None:
        self.calls: list[tuple[str, Any]] = []
        self._output = output

    def execute(self, command: str, *, timeout: int | None = None) -> _ExecResp:
        self.calls.append((command, timeout))
        return _ExecResp(self._output)

    async def aexecute(self, command: str, *, timeout: int | None = None) -> _ExecResp:
        self.calls.append((command, timeout))
        return _ExecResp(self._output)


def test_run_checks_allows_test_command():
    be = _FakeBackend(output="3 passed")
    tool = RunChecksTool(backend=be)
    out = tool._run("pytest -q")
    assert "3 passed" in out
    assert "[Command succeeded with exit code 0]" in out
    assert be.calls == [("pytest -q", None)]


def test_run_checks_forwards_timeout():
    be = _FakeBackend()
    tool = RunChecksTool(backend=be)
    tool._run("ruff check spine/", timeout=30)
    assert be.calls == [("ruff check spine/", 30)]


def test_run_checks_reports_nonzero_exit():
    class _FailBackend(_FakeBackend):
        def execute(self, command, *, timeout=None):  # noqa: ANN001
            return _ExecResp("boom", exit_code=1)

    out = RunChecksTool(backend=_FailBackend())._run("pytest -q")
    assert "[Command failed with exit code 1]" in out


def test_run_checks_async_path():
    import asyncio

    be = _FakeBackend(output="async ok")
    out = asyncio.run(RunChecksTool(backend=be)._arun("pytest -q"))
    assert "async ok" in out
    assert be.calls == [("pytest -q", None)]


@pytest.mark.parametrize(
    "command",
    ["grep -rn foo spine/", "find . -name '*.py'", "cat spine/config.py", "ls -la"],
)
def test_run_checks_rejects_exploration(command):
    be = _FakeBackend()
    tool = RunChecksTool(backend=be)
    out = tool._run(command)
    assert "rejected" in out
    assert "read_file" in out  # points the agent at the right tool
    assert be.calls == []  # never delegated


def test_run_checks_caps_output():
    tool = RunChecksTool(backend=_FakeBackend(output="x" * 100_000))
    out = tool._run("pytest -q")
    assert "capped" in out
    assert len(out) < 100_000


def test_run_checks_without_backend_reports_error():
    tool = RunChecksTool(backend=None)
    assert "no execute backend" in tool._run("pytest -q")


# ── integration: real backend signature (guards the 'runtime' bug) ───────────


def test_run_checks_against_real_backend(tmp_path):
    """run_checks must work against the ACTUAL spine backend.

    A fake backend can't catch a signature mismatch with the real
    SandboxBackendProtocol — that's exactly how the FS-execute-tool `runtime`
    bug slipped past the unit tests (trace 019f02b4). This drives the real
    backend end to end.
    """
    from spine.agents.backend import build_backend

    backend = build_backend(str(tmp_path))
    (tmp_path / "hello.txt").write_text("hi", encoding="utf-8")
    out = RunChecksTool(backend=backend)._run("echo run_checks_ok && cat hello.txt")
    assert "run_checks_ok" in out
    assert "hi" in out
    assert "exit code 0" in out
    # exploration is still policed even with a real backend
    assert "rejected" in RunChecksTool(backend=backend)._run("grep -rn x .")


# ── factory ──────────────────────────────────────────────────────────────────


def test_factory_builds_exactly_two_tools():
    tools = build_verify_subagent_tools(workspace_root="/tmp", backend=_FakeBackend())
    assert [t.name for t in tools] == ["read_file", "run_checks"]
