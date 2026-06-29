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


def test_read_per_file_cap_withholds_body(tmp_path):
    """Re-reading one file at shifting offsets is capped (trace 019f10bf)."""
    path = _write(tmp_path, "f.py", 500)
    tool = VerifyReadFileTool(workspace_root=str(tmp_path), file_read_cap=3)
    # Distinct offsets slip past the verbatim de-dup, so they hit the per-file cap.
    for off in range(3):
        out = tool._run(path, offset=off * 10, limit=5)
        assert "read_capped" not in out
    capped = tool._run(path, offset=999, limit=5)
    assert "read_capped" in capped
    assert "line" not in capped.split("read_capped")[0]  # no body before the status


def test_read_global_wall_refuses_further_reads(tmp_path):
    """Once the global wall is hit, even a new file is refused."""
    tool = VerifyReadFileTool(workspace_root=str(tmp_path), read_wall=3, file_read_cap=99)
    for i in range(3):
        p = _write(tmp_path, f"f{i}.py", 10)
        assert "read_wall" not in tool._run(p, offset=0, limit=5)
    extra = _write(tmp_path, "extra.py", 10)
    assert "read_wall" in tool._run(extra, offset=0, limit=5)


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
    # The original command is preserved verbatim at the tail; only an env-repair
    # prefix may be prepended (PATH/PYTHONPATH exports).
    assert len(be.calls) == 1
    sent, sent_timeout = be.calls[0]
    assert sent.endswith("pytest -q")
    assert sent_timeout is None


def test_run_checks_forwards_timeout():
    be = _FakeBackend()
    tool = RunChecksTool(backend=be)
    tool._run("ruff check spine/", timeout=30)
    assert len(be.calls) == 1
    sent, sent_timeout = be.calls[0]
    assert sent.endswith("ruff check spine/")
    assert sent_timeout == 30


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
    assert len(be.calls) == 1
    assert be.calls[0][0].endswith("pytest -q")
    assert be.calls[0][1] is None


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


def test_factory_passes_workspace_root_to_run_checks():
    tools = build_verify_subagent_tools(workspace_root="/ws", backend=_FakeBackend())
    run_checks = next(t for t in tools if t.name == "run_checks")
    assert run_checks.workspace_root == "/ws"


# ── run_checks: environment repair + static fallback (trace 019f10bf) ────────


def test_run_checks_injects_pythonpath(tmp_path):
    be = _FakeBackend(output="ok")
    tool = RunChecksTool(backend=be, workspace_root=str(tmp_path))
    tool._run("pytest -q")
    sent = be.calls[0][0]
    assert f'PYTHONPATH="{tmp_path}' in sent  # project package made importable
    assert sent.endswith("pytest -q")


class _EnvFailBackend(_FakeBackend):
    """Backend whose commands cannot run (missing runner / package)."""

    def __init__(self, message: str, exit_code: int = 127) -> None:
        super().__init__()
        self._message = message
        self._exit = exit_code

    def execute(self, command, *, timeout=None):  # noqa: ANN001
        self.calls.append((command, timeout))
        return _ExecResp(self._message, exit_code=self._exit)


def test_run_checks_env_unavailable_after_repeated_failures():
    be = _EnvFailBackend("pytest: command not found", exit_code=127)
    tool = RunChecksTool(backend=be, workspace_root="/ws")
    first = tool._run("pytest -q")
    assert "env_unavailable" not in first  # first failure is reported normally
    second = tool._run("pytest tests/ -q")
    assert "env_unavailable" in second
    assert "statically" in second.lower()
    # Further calls short-circuit WITHOUT another backend round-trip.
    n_before = len(be.calls)
    third = tool._run("pytest -q")
    assert "env_unavailable" in third
    assert len(be.calls) == n_before  # no new execution


def test_run_checks_missing_runner_is_env_failure():
    """The test framework itself missing → env failure (static fallback)."""
    be = _EnvFailBackend("No module named pytest", exit_code=1)
    tool = RunChecksTool(backend=be, workspace_root="/ws")
    tool._run("python -m pytest -q")
    out = tool._run("python -m pytest tests/ -q")
    assert "env_unavailable" in out


def test_run_checks_project_import_error_is_reported_not_masked():
    """A project ModuleNotFoundError (likely a real defect) is NOT masked.

    After PYTHONPATH repair, a missing project import is a finding to report,
    not an environment problem to skip — so it must never trip the fallback.
    """
    be = _EnvFailBackend(
        "ModuleNotFoundError: No module named 'spine.ui_api.missing'", exit_code=1
    )
    tool = RunChecksTool(backend=be, workspace_root="/ws")
    for _ in range(3):
        out = tool._run("python -c 'import spine.ui_api.missing'")
        assert "env_unavailable" not in out
        assert "No module named" in out


def test_run_checks_real_failure_does_not_trip_env_fallback():
    """An ordinary test failure (exit 1, real output) must NOT short-circuit."""

    class _AssertFail(_FakeBackend):
        def execute(self, command, *, timeout=None):  # noqa: ANN001
            self.calls.append((command, timeout))
            return _ExecResp("1 failed, 2 passed\nassert 1 == 2", exit_code=1)

    be = _AssertFail()
    tool = RunChecksTool(backend=be, workspace_root="/ws")
    for _ in range(4):
        out = tool._run("pytest -q")
        assert "env_unavailable" not in out
        assert "1 failed" in out
