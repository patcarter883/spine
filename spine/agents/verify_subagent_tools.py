"""Bounded tool surface for the slice-verifier subagent.

The slice-verifier was the last leaf agent still carrying the raw
filesystem+shell surface (``ls``/``read_file``/``glob``/``grep``/``execute``/
``search_codebase``). Trace ``019f0212`` showed it doing exactly the survey
spiral the slice-implementer lockdown (``read_edit_lint``) was built to kill:
~290 ``execute`` calls in a single verify pass, >90% shelling ``grep``/
``find``/``ls``/``cat``, many returning no output.

This module gives the verifier two purpose-built tools — the same shape the
implementer got — so it can inspect and test without a generic shell:

- ``read_file``  — ranged, line-numbered, output-capped, READ-ONLY. Re-reading
  an unchanged range returns an ``already_read`` nudge instead of the body; a
  per-file cap and a global wall stop the same-file-at-shifting-offsets and
  breadth re-read spirals the verbatim de-dup misses (trace 019f10bf).
- ``run_checks`` — a single constrained runner for tests / linters / builds. It
  delegates to the real execute backend but REJECTS pure-exploration commands
  (``grep``/``find``/``ls``/``cat``/``sed``/…): file inspection belongs to
  ``read_file``, not the shell. It also REPAIRS the sandbox environment (PATH +
  PYTHONPATH) so the runner and the edited project package are actually
  reachable, and after repeated *environment* failures it short-circuits with a
  directive to verify statically rather than looping on a runner that cannot run
  (trace 019f10bf: every ``pytest`` returned 127, every ``import spine`` raised
  ModuleNotFoundError, and the verifier spiralled to millions of tokens).

Both are LangChain ``BaseTool`` subclasses that slot directly into a subagent's
``tools=[...]`` list.
"""

from __future__ import annotations

import logging
import os
import shutil
import site
import sys
import sysconfig
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field, PrivateAttr

logger = logging.getLogger(__name__)

# Output bounds — keep a single tool turn from dumping a whole file or an
# unbounded test log into the next prompt cycle (the monotonic-growth failure
# mode behind trace 019e87dd).
_MAX_READ_LINES = 400
_DEFAULT_READ_LINES = 200
_MAX_READ_CHARS = 20_000
_MAX_RUN_CHARS = 12_000

# Per-file read budget — distinct-anchor reads of the SAME file before the body
# is withheld. The exact-(path,offset,limit,mtime) de-dup only catches VERBATIM
# repeats; trace 019f10bf showed one file (api.py) read 8× at *varying* offsets
# (54, 78, 102, 127, 154 …), every one slipping past the de-dup. This is the
# verifier counterpart of read_edit_lint's ``_FILE_READ_CAP``. The verifier has
# no "edit" to reset on, so the cap is cumulative across the whole verify pass.
_FILE_READ_CAP = 5

# Global read wall — total body-returning reads before every further read is
# refused, whatever the file. Stops the breadth-spiral the per-file cap can't
# (reading many files a few times each). The pre-loaded target source + worktree
# diff already hand the verifier most of what it needs, so a healthy pass reads
# only a handful of files; this wall is the backstop, not the budget.
_READ_WALL = 16

# Global run_checks wall — total backend-reaching check runs before every
# further run is refused. The read-side has ``_READ_WALL``; run_checks had no
# equivalent, so a verifier in a healthy sandbox (checks DO run, so the
# env/sticky short-circuits never fire) could loop py_compile/pytest/ruff probes
# indefinitely — the exact spiral behind trace 019f16cf (2.75M tok / 0 verdicts).
# A healthy pass runs only a handful of checks (a py_compile + a lint + maybe a
# targeted pytest); this wall is the backstop, not the budget. Rejected
# exploration commands and env/sticky short-circuits do NOT count — they never
# reach the backend and already return a firm directive.
_RUN_CHECKS_WALL = 12

# Execution-environment failure handling. The slice-verifier runs in a sandbox
# worktree whose shell PATH lacks the test runner and whose interpreter cannot
# ``import`` the project package — trace 019f10bf: every ``pytest`` returned
# 127 (command not found) and every ``python -c "import spine…"`` raised
# ModuleNotFoundError, so the verifier could never obtain a PASS and looped on
# new commands + re-reads, burning millions of tokens. Two defences:
#   1. ``_env_prefix`` repairs the environment (PATH + PYTHONPATH) so the
#      runners and the project package are actually reachable — resolving the
#      runners' bin dirs from the interpreter itself, not just ``shutil.which``,
#      which misses ``~/.local/bin`` under a stripped launch PATH (trace
#      019f111e: ``ruff`` 127'd despite the env-repair shipping).
#   2. A runner that still can't run is recorded as unavailable *per tool*
#      (sticky), so a repeat call short-circuits with a STATIC-verify directive
#      instead of paying another failing round-trip. Sticky-per-tool — not a
#      consecutive streak — because a working ``python -c import`` check
#      interleaved between two dead ``ruff`` checks must not reset the linter
#      back to "available" (trace 019f111e). If the interpreter itself can't
#      run, nothing can, and the verifier is told to stop calling run_checks.
# Python entrypoints whose containing bin dir we surface onto the sandbox PATH.
_RUNNER_TOOLS = ("pytest", "ruff", "mypy", "python", "python3", "pip", "uv", "tox")
# Test/lint frameworks that report unavailability as ``No module named '<tool>'``
# (e.g. when invoked as ``python -m pytest``) rather than a shell 127. A bare
# ``ModuleNotFoundError`` for the *project* package is deliberately NOT treated
# as an environment failure: once ``_env_prefix`` has repaired PYTHONPATH, a
# remaining import error is far more likely a REAL defect the verifier should
# report than an environment problem — matching only framework modules keeps the
# static-fallback from masking real bugs.
_ENV_FAIL_MODULES = ("pytest", "ruff", "mypy", "tox")
# Interpreters whose absence means NO check can run (vs. a single missing tool).
_INTERPRETERS = frozenset({"python", "python3"})

# Pure-exploration shell utilities. The verifier has no reason to run these:
# reading files is ``read_file``'s job, and listing/searching is what drove the
# grep/find spiral. Anything whose *leading* executable is one of these is
# rejected with a pointer back to ``read_file``. They remain usable as
# downstream pipe filters (e.g. ``pytest | grep FAIL``) because only the first
# command is policed.
_EXPLORATION_COMMANDS = frozenset(
    {
        "grep", "egrep", "fgrep", "rg", "ag", "ack",
        "find", "fd", "locate",
        "ls", "ll", "tree", "dir",
        "cat", "bat", "tac", "head", "tail", "less", "more", "nl",
        "sed", "awk", "cut", "sort", "uniq", "wc",
        "stat", "file", "du", "df",
    }
)


# ── read_file (bounded, read-only) ─────────────────────────────────────────


class _VerifyReadInput(BaseModel):
    file_path: str = Field(
        description="Path to the file to read (absolute, or relative to the "
        "project root). Read the implemented target_files named in your slice."
    )
    offset: int = Field(
        default=0,
        description="0-indexed line to start from. Use for paging large files.",
    )
    limit: int = Field(
        default=_DEFAULT_READ_LINES,
        description=f"Max lines to return (capped at {_MAX_READ_LINES}).",
    )


class VerifyReadFileTool(BaseTool):
    """Ranged, line-numbered, output-capped, READ-ONLY file read.

    The verifier's only file-inspection surface. There is no write mode — the
    slice-verifier is report-only by contract.
    """

    name: str = "read_file"
    description: str = (
        "Read a file to inspect the implementation — your ONLY file-inspection "
        "tool (there is no shell `cat`/`grep`/`ls`). Returns line-numbered, "
        "output-capped text. Pass file_path (absolute or project-relative) and "
        "optionally offset/limit to page through large files "
        f"(limit capped at {_MAX_READ_LINES} lines). Read-only: you cannot edit. "
        "status='already_read' means you already read this exact range and the "
        "file is unchanged — the lines are above; do not re-read, move on."
    )
    args_schema: Optional[ArgsSchema] = _VerifyReadInput

    workspace_root: str = ""
    # Caps default to the verifier's tight budget; callers that legitimately
    # survey breadth (project-level review/verify) raise ``read_wall``.
    file_read_cap: int = _FILE_READ_CAP
    read_wall: int = _READ_WALL

    # Per-invocation read de-dup: (resolved_path, offset, limit, mtime) -> seen.
    _seen_reads: set = PrivateAttr(default_factory=set)
    # Body-returning reads of each file this session (drives the per-file cap).
    # Distinct from ``_seen_reads``, which keys on the exact anchor.
    _file_read_count: dict = PrivateAttr(default_factory=dict)
    # Total body-returning reads this session (drives the global wall).
    _total_reads: int = PrivateAttr(default=0)

    def _resolve(self, file_path: str) -> Path:
        p = Path(file_path)
        root = Path(self.workspace_root)
        if p.is_absolute():
            # Strip an accidental doubled workspace prefix, mirroring the
            # backend's normalisation for virtual_mode paths.
            try:
                if str(p).startswith(str(root)):
                    return p
            except Exception:  # noqa: BLE001
                pass
            return p
        return (root / p).resolve()

    def _run(self, file_path: str, offset: int = 0, limit: int = _DEFAULT_READ_LINES) -> str:
        path = self._resolve(file_path)
        if not path.exists():
            return f"status=not_found: {file_path} does not exist (resolved {path})."
        if path.is_dir():
            return (
                f"status=is_directory: {file_path} is a directory. "
                "read_file reads files; name a specific target_file from your slice."
            )

        offset = max(0, int(offset))
        limit = max(1, min(int(limit), _MAX_READ_LINES))

        try:
            mtime = path.stat().st_mtime_ns
        except OSError as exc:
            return f"status=error: cannot stat {file_path}: {exc}"

        key = (str(path), offset, limit, mtime)
        if key in self._seen_reads:
            return (
                f"status=already_read: you already read {file_path} "
                f"lines {offset}–{offset + limit} and it is unchanged. "
                "The content is above — do not re-read; make your verdict."
            )

        # Global wall: too many distinct reads this pass → stop surveying.
        if self._total_reads >= self.read_wall:
            return (
                f"status=read_wall: you have already read {self._total_reads} "
                "file ranges this verification pass — that is enough surveying. "
                "The source you read is above (plus the worktree diff and any "
                "pre-loaded target source). Judge each acceptance criterion from "
                "what you have and emit your verdict; do not read more."
            )

        # Per-file cap: re-reading the SAME file at shifting offsets is the
        # spiral the verbatim de-dup misses (trace 019f10bf). The read that
        # lands ON the cap still returns a fresh body (earlier copies may have
        # been evicted); reads beyond it withhold the body.
        prior_file_reads = self._file_read_count.get(str(path), 0)
        if prior_file_reads >= self.file_read_cap:
            return (
                f"status=read_capped: you have already read {file_path} "
                f"{prior_file_reads} times this pass at different offsets. Its "
                "content is above — do not re-read it. Verify the acceptance "
                "criteria from what you have, or run a check with run_checks."
            )

        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except OSError as exc:
            return f"status=error: cannot read {file_path}: {exc}"

        total = len(lines)
        window = lines[offset : offset + limit]
        if not window:
            return (
                f"status=empty_range: {file_path} has {total} lines; "
                f"offset {offset} is past the end."
            )

        numbered = []
        char_budget = _MAX_READ_CHARS
        truncated_chars = False
        last_lineno = offset
        for i, line in enumerate(window, start=offset + 1):
            rendered = f"{i:6d}\t{line.rstrip(chr(10))}\n"
            if char_budget - len(rendered) < 0:
                truncated_chars = True
                break
            numbered.append(rendered)
            char_budget -= len(rendered)
            last_lineno = i

        self._seen_reads.add(key)
        self._file_read_count[str(path)] = prior_file_reads + 1
        self._total_reads += 1

        body = "".join(numbered)
        header = f"# {file_path} (lines {offset + 1}–{last_lineno} of {total})\n"
        footer = ""
        if truncated_chars:
            footer = (
                f"\n[output capped at {_MAX_READ_CHARS} chars — "
                f"re-read from offset {last_lineno} for more]"
            )
        elif offset + limit < total:
            footer = (
                f"\n[{total - (offset + limit)} more lines — "
                f"re-read from offset {offset + limit} to continue]"
            )
        return header + body + footer

    async def _arun(
        self, file_path: str, offset: int = 0, limit: int = _DEFAULT_READ_LINES
    ) -> str:
        return self._run(file_path, offset=offset, limit=limit)


# ── run_checks (constrained execute) ───────────────────────────────────────


class _RunChecksInput(BaseModel):
    command: str = Field(
        description="Shell command that RUNS a test, linter, type-checker, or "
        "build (e.g. 'pytest tests/unit/test_foo.py -q', 'ruff check spine/', "
        "'mypy spine'). Not for exploring files — use read_file for that."
    )
    timeout: Optional[int] = Field(
        default=None,
        description="Optional timeout in seconds. Use 0 for no timeout on "
        "backends that support it.",
    )


def _leading_executable(command: str) -> str:
    """Return the basename of the first real executable in ``command``.

    Strips leading ``cd <path> &&`` / ``cd <path> ;`` wrappers and common
    inline env-var assignments so a smuggled ``cd x && grep`` is still policed
    on ``grep``, not ``cd``.
    """
    cmd = command.strip()
    # Peel leading `cd ... &&` / `cd ... ;` segments.
    changed = True
    while changed:
        changed = False
        stripped = cmd.lstrip()
        for sep in ("&&", ";"):
            if stripped.startswith("cd "):
                idx = stripped.find(sep)
                if idx != -1:
                    cmd = stripped[idx + len(sep) :]
                    changed = True
                    break
    tokens = cmd.strip().split()
    # Skip leading VAR=value assignments.
    for tok in tokens:
        if "=" in tok and not tok.startswith("-") and "/" not in tok.split("=")[0]:
            continue
        return Path(tok).name
    return ""


def _interpreter_script_dirs() -> list[str]:
    """Console-script bin dirs for the running interpreter, PATH-independent.

    A ``shutil.which`` probe fails when the process is launched with a stripped
    PATH that omits ``~/.local/bin`` (pip ``--user`` installs) or the active
    venv's bin dir, so ``pytest``/``ruff`` return 127 even though they are
    installed (trace 019f111e). ``sysconfig``/``site`` resolve those dirs from
    the interpreter itself, independent of the launching PATH. Best-effort: any
    probe that raises is skipped.
    """
    dirs: list[str] = []
    try:
        dirs.append(sysconfig.get_path("scripts"))
    except Exception:  # noqa: BLE001
        pass
    try:
        # Per-user scheme (``posix_user`` / ``nt_user``) → ``~/.local/bin``.
        dirs.append(sysconfig.get_path("scripts", f"{os.name}_user"))
    except Exception:  # noqa: BLE001
        pass
    try:
        dirs.append(os.path.join(site.getuserbase(), "bin"))
    except Exception:  # noqa: BLE001
        pass
    return [d for d in dirs if d]


class RunChecksTool(BaseTool):
    """Constrained test/lint/build runner.

    Runs the command against the sandbox backend directly (``backend.execute`` /
    ``backend.aexecute``) but rejects pure-exploration commands so the verifier
    inspects files through ``read_file`` instead of spiralling on
    ``grep``/``find``/``cat`` (trace 019f0212).

    It deliberately calls the backend rather than wrapping the
    FilesystemMiddleware ``execute`` BaseTool: that tool requires a LangGraph
    ``runtime`` injected by the agent's ToolNode, which a plain ``.invoke`` from
    inside another tool cannot supply (it raises ``missing 'runtime'``). The
    backend exposes the same execution primitive without that coupling.
    """

    name: str = "run_checks"
    description: str = (
        "Run a test, linter, type-checker, or build command and return its "
        "(output-capped) result — e.g. 'pytest tests/unit/test_x.py -q', "
        "'ruff check spine/', 'mypy spine'. This is for RUNNING checks, not "
        "browsing the codebase: commands that just explore the filesystem "
        "(grep, find, ls, cat, sed, head, tail, …) are rejected — use read_file "
        "to inspect files. Report what the checks show in your verdict; do not "
        "fix anything."
    )
    args_schema: Optional[ArgsSchema] = _RunChecksInput

    # Workspace root of the sandbox worktree. Used to repair PYTHONPATH so the
    # edited project package is importable (``import spine`` failed in the raw
    # sandbox — trace 019f10bf).
    workspace_root: str = ""

    # Cumulative backend-reaching check runs before the wall refuses further
    # runs. Defaults to the verifier's tight backstop; a caller that legitimately
    # needs a wider check budget can raise it.
    run_checks_wall: int = _RUN_CHECKS_WALL

    # The sandbox backend (SandboxBackendProtocol / CompositeBackend), injected
    # at build time. ``run_checks`` is a policy wrapper around backend.execute.
    _backend: Any = PrivateAttr(default=None)
    # Runners proven unable to RUN here (missing binary / framework), tracked
    # sticky per tool so a working check interleaved between two failing ones
    # can't reset a dead runner back to "available" (trace 019f111e). Drives the
    # per-tool and whole-environment static-fallback directives.
    _unavailable_runners: set = PrivateAttr(default_factory=set)
    # Backend-reaching check runs this session (drives the global run_checks wall).
    # Only real backend round-trips increment it — rejected/short-circuited calls
    # never reach the backend, so they are not counted.
    _run_count: int = PrivateAttr(default=0)
    # Cached ``export PATH=…; export PYTHONPATH=…;`` prefix (computed once).
    _prefix_cache: Optional[str] = PrivateAttr(default=None)

    def __init__(
        self, backend: Any = None, workspace_root: str = "", **kwargs: Any
    ) -> None:
        super().__init__(workspace_root=workspace_root, **kwargs)
        self._backend = backend

    def _env_prefix(self) -> str:
        """Build a shell prefix that makes runners + the project importable.

        The sandbox worktree's ``/bin/sh`` PATH lacks the test runner (``pytest``
        lives in a user/venv bin dir, not ``/usr/bin``) and its interpreter
        cannot ``import`` the edited project. We surface the runners' real bin
        dirs onto PATH and add ``PYTHONPATH=<workspace_root>`` so a bare
        ``pytest``/``ruff``/``python -c 'import spine'`` actually runs. Without
        this the verifier can never get a PASS signal and spirals (trace
        019f10bf).

        Discovery does NOT rely on ``shutil.which`` alone: a bench launched with
        a stripped PATH (systemd/cron) that omits ``~/.local/bin`` makes
        ``which('ruff')`` return ``None``, so the prefix never repaired the
        linter and every ``ruff``/``pytest`` returned 127 (trace 019f111e). We
        therefore also resolve the interpreter's console-script dirs directly
        via ``sysconfig``/``site``, which are independent of the launching PATH.
        Best-effort and idempotent; returns ``""`` if nothing useful was found.
        """
        if self._prefix_cache is not None:
            return self._prefix_cache
        bin_dirs: list[str] = []
        seen: set[str] = set()

        def _add(d: Optional[str]) -> None:
            if d and d not in seen and os.path.isdir(d):
                seen.add(d)
                bin_dirs.append(d)

        # The interpreter's own bin dir, then its console-script dirs (default +
        # per-user install schemes) — these resolve ~/.local/bin and venv bin
        # dirs even when the launching PATH is stripped of them.
        _add(os.path.dirname(sys.executable))
        for d in _interpreter_script_dirs():
            _add(d)
        # Finally whatever the live PATH can resolve (rich interactive launches).
        for tool in _RUNNER_TOOLS:
            found = shutil.which(tool)
            if found:
                _add(os.path.dirname(found))

        parts: list[str] = []
        if bin_dirs:
            parts.append(f"export PATH={os.pathsep.join(bin_dirs)}:$PATH;")
        if self.workspace_root:
            parts.append(f'export PYTHONPATH="{self.workspace_root}:$PYTHONPATH";')
        self._prefix_cache = (" ".join(parts) + " ") if parts else ""
        return self._prefix_cache

    def _normalize(self, command: str) -> str:
        """Prepend the environment-repair prefix to ``command``.

        Only adds ``export`` statements — it never rewrites the command's own
        tokens, so ``cd x && pytest`` and ``pytest | grep FAIL`` are preserved.
        """
        prefix = self._env_prefix()
        return prefix + command if prefix else command

    @staticmethod
    def _env_failure_tool(result: Any, formatted: str, lead: str) -> Optional[str]:
        """Name of the runner that could not RUN, or ``None`` if it ran.

        Distinguishes a *missing runner* (127 / command-not-found → the leading
        binary) from a *missing test framework module* (``No module named
        'pytest'`` when invoked as ``python -m pytest`` → the framework, NOT the
        interpreter, which ran fine). Attributing the latter to the framework
        keeps a ``python -m <tool>`` miss from falsely flagging the interpreter
        as dead and stopping every check.
        """
        low = formatted.lower()
        for tool in _ENV_FAIL_MODULES:
            if f"no module named '{tool}'" in low or f"no module named {tool}" in low:
                return tool
        if getattr(result, "exit_code", None) == 127 or "command not found" in low:
            return lead or "the runner"
        return None

    def _interpreter_dead(self) -> bool:
        """True once ``python`` itself proved unrunnable — nothing else can run."""
        return bool(self._unavailable_runners & _INTERPRETERS)

    def _env_unavailable(self, last: str = "") -> str:
        """Terminal directive: stop running checks, verify statically instead."""
        tail = f"\n\nLast error:\n{self._cap(last)}" if last else ""
        return (
            "status=env_unavailable: the sandbox cannot run executable checks — "
            "the interpreter itself is not runnable here (not a test failure). "
            "STOP calling run_checks; retrying will keep failing. Verify "
            "STATICALLY: read the changed code with read_file, compare it against "
            "the worktree diff and the pre-loaded target source, and judge each "
            "acceptance criterion from the source itself. Note in your report "
            "that checks could not be executed." + tail
        )

    def _wall_reached(self) -> str:
        """Terminal directive: the check-run budget is spent; finalize now."""
        return (
            f"status=run_checks_wall: you have already run {self._run_count} "
            "checks this verification (budget "
            f"{self.run_checks_wall}) — that is the backstop, not a per-slice "
            "quota, and further runs will be refused. STOP calling run_checks. "
            "You have enough evidence: judge each acceptance criterion from the "
            "check output you already have plus the worktree diff and pre-loaded "
            "target source, and return your verdict now. Re-running the same or "
            "near-identical checks will not change the result."
        )

    def _runner_unavailable(self, tool: str, last: str = "") -> str:
        """Per-tool directive: this runner can't run; don't retry it."""
        tail = f"\n\nLast error:\n{self._cap(last)}" if last else ""
        return (
            f"status=runner_unavailable: '{tool}' cannot run in this sandbox "
            "(command not found / exit 127 — the tool isn't installed here, not a "
            f"check failure). Do NOT call '{tool}' again; repeat calls will keep "
            "failing. Verify what it would have checked STATICALLY with read_file "
            "(for a linter, eyeball the changed lines), and rely on the runners "
            'that DO work here — a bare `python -c "import <module>"` smoke-import '
            f"succeeds. Note in your report that '{tool}' could not be executed."
            + tail
        )

    def _finish(self, lead: str, result: Any) -> str:
        """Format a backend result, tracking *runner-missing* failures.

        A runner that cannot run is recorded as unavailable (sticky, per tool)
        so subsequent calls short-circuit instead of paying another failing
        round-trip. An interleaved working check does NOT clear the record — a
        permanently-missing ``ruff`` must stay flagged even after a passing
        ``python -c import`` (trace 019f111e).
        """
        formatted = self._format(result)
        tool = self._env_failure_tool(result, formatted, lead)
        if tool:
            self._unavailable_runners.add(tool)
            if self._interpreter_dead():
                return self._env_unavailable(formatted)
            return self._runner_unavailable(tool, formatted)
        return self._cap(formatted)

    def _reject(self, lead: str) -> str:
        return (
            f"status=rejected: '{lead}' is a filesystem-exploration command. "
            "run_checks only runs tests/linters/builds. To inspect a file, use "
            "read_file; the files to verify are named in your slice definition."
        )

    @staticmethod
    def _cap(out: str) -> str:
        if len(out) <= _MAX_RUN_CHARS:
            return out
        half = _MAX_RUN_CHARS // 2
        return (
            out[:half]
            + f"\n…[output capped — {len(out) - _MAX_RUN_CHARS} chars elided]…\n"
            + out[-half:]
        )

    @staticmethod
    def _format(result: Any) -> str:
        """Render a backend ExecuteResponse the way the model expects.

        Mirrors FilesystemMiddleware's execute formatting: output, then an exit
        status line, then a truncation note.
        """
        parts = [str(getattr(result, "output", result) or "")]
        exit_code = getattr(result, "exit_code", None)
        if exit_code is not None:
            status = "succeeded" if exit_code == 0 else "failed"
            parts.append(f"\n[Command {status} with exit code {exit_code}]")
        if getattr(result, "truncated", False):
            parts.append("\n[Output was truncated due to size limits]")
        return "".join(parts)

    def _short_circuit(self, lead: str) -> Optional[str]:
        """Directive to return without a backend round-trip, or ``None`` to run.

        Rejects pure-exploration commands, and skips re-running a runner already
        proven unable to run here (the interpreter being dead skips everything).
        """
        if lead in _EXPLORATION_COMMANDS:
            return self._reject(lead)
        if self._interpreter_dead():
            return self._env_unavailable()
        if lead in self._unavailable_runners:
            return self._runner_unavailable(lead)
        return None

    def _run(self, command: str, timeout: Optional[int] = None) -> str:
        lead = _leading_executable(command)
        short = self._short_circuit(lead)
        if short is not None:
            return short
        if self._run_count >= self.run_checks_wall:
            return self._wall_reached()
        if self._backend is None:
            return "status=error: no execute backend available for run_checks."
        cmd = self._normalize(command)
        self._run_count += 1
        result = (
            self._backend.execute(cmd, timeout=timeout)
            if timeout is not None
            else self._backend.execute(cmd)
        )
        return self._finish(lead, result)

    async def _arun(self, command: str, timeout: Optional[int] = None) -> str:
        lead = _leading_executable(command)
        short = self._short_circuit(lead)
        if short is not None:
            return short
        if self._run_count >= self.run_checks_wall:
            return self._wall_reached()
        if self._backend is None:
            return "status=error: no execute backend available for run_checks."
        cmd = self._normalize(command)
        self._run_count += 1
        result = (
            await self._backend.aexecute(cmd, timeout=timeout)
            if timeout is not None
            else await self._backend.aexecute(cmd)
        )
        return self._finish(lead, result)


# ── Factory ────────────────────────────────────────────────────────────────


def build_verify_subagent_tools(
    workspace_root: str,
    backend: Any = None,
) -> list[BaseTool]:
    """Build the bounded tool surface for the slice-verifier subagent.

    Returns exactly two tools — a ranged read-only ``read_file`` and a
    constrained ``run_checks`` runner — replacing the raw
    ls/read_file/glob/grep/execute/search_codebase surface the verifier
    carried before (trace 019f0212).

    Args:
        workspace_root: Absolute path to the project workspace root.
        backend: The sandbox backend (SandboxBackendProtocol / CompositeBackend)
            that ``run_checks`` executes against. When ``None`` (no
            execution-capable backend), ``run_checks`` returns a structured
            error instead of running.

    Returns:
        ``[VerifyReadFileTool, RunChecksTool]``.
    """
    return [
        VerifyReadFileTool(workspace_root=workspace_root),
        RunChecksTool(backend=backend, workspace_root=workspace_root),
    ]
