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
  an unchanged range returns an ``already_read`` nudge instead of the body, so
  the agent stops re-fetching the same lines.
- ``run_checks`` — a single constrained runner for tests / linters / builds. It
  delegates to the real execute backend but REJECTS pure-exploration commands
  (``grep``/``find``/``ls``/``cat``/``sed``/…): file inspection belongs to
  ``read_file``, not the shell.

Both are LangChain ``BaseTool`` subclasses that slot directly into a subagent's
``tools=[...]`` list.
"""

from __future__ import annotations

import logging
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

    # Per-invocation read de-dup: (resolved_path, offset, limit, mtime) -> seen.
    _seen_reads: set = PrivateAttr(default_factory=set)

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


class RunChecksTool(BaseTool):
    """Constrained test/lint/build runner.

    Delegates to the real ``execute`` backend but rejects pure-exploration
    commands so the verifier inspects files through ``read_file`` instead of
    spiralling on ``grep``/``find``/``cat`` (trace 019f0212).
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

    # The underlying FilesystemMiddleware ``execute`` BaseTool, injected at
    # build time. ``run_checks`` is a policy wrapper around it.
    _inner: Any = PrivateAttr(default=None)

    def __init__(self, execute_tool: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._inner = execute_tool

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

    def _inner_args(self, command: str, timeout: Optional[int]) -> dict[str, Any]:
        args: dict[str, Any] = {"command": command}
        if timeout is not None:
            args["timeout"] = timeout
        return args

    def _run(self, command: str, timeout: Optional[int] = None) -> str:
        lead = _leading_executable(command)
        if lead in _EXPLORATION_COMMANDS:
            return self._reject(lead)
        if self._inner is None:
            return "status=error: no execute backend available for run_checks."
        out = self._inner.invoke(self._inner_args(command, timeout))
        return self._cap(str(out))

    async def _arun(self, command: str, timeout: Optional[int] = None) -> str:
        lead = _leading_executable(command)
        if lead in _EXPLORATION_COMMANDS:
            return self._reject(lead)
        if self._inner is None:
            return "status=error: no execute backend available for run_checks."
        out = await self._inner.ainvoke(self._inner_args(command, timeout))
        return self._cap(str(out))


# ── Factory ────────────────────────────────────────────────────────────────


def build_verify_subagent_tools(
    workspace_root: str,
    execute_tool: Any = None,
) -> list[BaseTool]:
    """Build the bounded tool surface for the slice-verifier subagent.

    Returns exactly two tools — a ranged read-only ``read_file`` and a
    constrained ``run_checks`` runner — replacing the raw
    ls/read_file/glob/grep/execute/search_codebase surface the verifier
    carried before (trace 019f0212).

    Args:
        workspace_root: Absolute path to the project workspace root.
        execute_tool: The FilesystemMiddleware ``execute`` BaseTool that
            ``run_checks`` wraps. When ``None`` (no execution-capable backend),
            ``run_checks`` returns a structured error instead of running.

    Returns:
        ``[VerifyReadFileTool, RunChecksTool]``.
    """
    return [
        VerifyReadFileTool(workspace_root=workspace_root),
        RunChecksTool(execute_tool=execute_tool),
    ]
