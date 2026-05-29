"""Read-edit-lint compound tool for the slice-implementer subagent.

Replaces ``write_file`` + ``edit_file`` with a single tool that resolves an
edit in memory, runs a language-specific syntax check on the proposed new
content, and only writes (atomically) when the check passes. On failure it
returns a structured status WITHOUT touching disk so the model can correct
and retry in-loop.

Four mutually-exclusive edit modes:

1. ``old_str`` → ``new_str`` — exact, single-occurrence find-and-replace.
2. ``full_replace`` — whole-file rewrite (also creates new files).
3. ``edits`` — a batch of find-and-replace ops applied in order to one file,
   **all-or-nothing**: the new content is built entirely in memory and only
   written if every edit matches and the result passes the syntax check.
4. ``start_line`` + ``end_line`` + ``replacement`` — line-range replacement,
   anchored on current 1-indexed line numbers (token-efficient, and
   disambiguates snippets that repeat in the file). An optional ``expected``
   field guards against stale line numbers by verifying the current text of
   the range before applying — borrowed from hashline's snapshot-verify idea.

The exact-match contract on ``old_str`` mirrors Anthropic's ``str_replace``:
no regex, no fuzzy matching, fail loudly if the snippet is missing or appears
more than once.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field

from spine.agents.tools._fs import _atomic_write

logger = logging.getLogger(__name__)


_LINT_LANG_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".php": "php",
    ".ts": "typescript",
    ".tsx": "typescript",
}


class FindReplaceEdit(BaseModel):
    """A single exact find-and-replace operation within a batch."""

    old_str: str = Field(
        description="Exact string to replace. Must match exactly once in the working buffer."
    )
    new_str: str = Field(
        default="",
        description="Replacement string (may be empty to delete old_str).",
    )


class ReadEditLintInput(BaseModel):
    """Input schema for :class:`ReadEditLintTool`."""

    file_path: str = Field(
        description="Workspace-relative path to the file to edit (or create with full_replace)."
    )
    old_str: Optional[str] = Field(
        default=None,
        description=(
            "Exact string to replace. Must appear EXACTLY ONCE in the current file. "
            "Pair with new_str. Mutually exclusive with the other edit modes."
        ),
    )
    new_str: Optional[str] = Field(
        default=None,
        description="Replacement string for old_str (may be empty to delete).",
    )
    full_replace: Optional[str] = Field(
        default=None,
        description=(
            "Full file content to write (creates the file if absent). "
            "Mutually exclusive with the other edit modes."
        ),
    )
    edits: Optional[list[FindReplaceEdit]] = Field(
        default=None,
        description=(
            "Batch of exact find-and-replace edits applied IN ORDER to one file, "
            "all-or-nothing: if any edit fails to match (or the result fails the "
            "syntax check) nothing is written. Each edit must match exactly once "
            "in the buffer at the time it is applied. Mutually exclusive with the "
            "other edit modes."
        ),
    )
    start_line: Optional[int] = Field(
        default=None,
        description="1-indexed first line of the range to replace (line-range mode).",
    )
    end_line: Optional[int] = Field(
        default=None,
        description="1-indexed last line of the range to replace, inclusive (line-range mode).",
    )
    replacement: Optional[str] = Field(
        default=None,
        description=(
            "New text for lines start_line..end_line (line-range mode). "
            "Empty string deletes the range. Mutually exclusive with the other modes."
        ),
    )
    expected: Optional[str] = Field(
        default=None,
        description=(
            "Optional staleness guard for line-range mode: the text you expect to "
            "currently occupy start_line..end_line. If it no longer matches, the "
            "edit is rejected as `stale` (re-read and retry) instead of applied."
        ),
    )


def _result(status: str, **fields: Any) -> str:
    """Encode a result dict as JSON for tool output."""
    payload: dict[str, Any] = {"status": status, **fields}
    return json.dumps(payload, ensure_ascii=False)


def _check_python(source: str) -> Optional[str]:
    """Return a syntax-error description, or None if the source parses."""
    import ast

    try:
        ast.parse(source)
        return None
    except SyntaxError as exc:
        return f"SyntaxError: {exc.msg} (line {exc.lineno}, offset {exc.offset})"


def _check_php(source: str) -> Optional[str]:
    """Try ``php -l``; fall back to tree-sitter when php isn't on PATH."""
    try:
        proc = subprocess.run(
            ["php", "-l"],
            input=source.encode("utf-8"),
            capture_output=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return _check_with_tree_sitter(source, "php")
    if proc.returncode == 0:
        return None
    # php -l prints "PHP Parse error: ... in - on line N"
    msg = proc.stderr.decode("utf-8", errors="replace") or proc.stdout.decode("utf-8", errors="replace")
    return msg.strip().splitlines()[0] if msg.strip() else "php -l reported a syntax error"


def _check_typescript(source: str) -> Optional[str]:
    """Parse via tree-sitter and flag any ERROR node in the tree."""
    return _check_with_tree_sitter(source, "typescript")


def _check_with_tree_sitter(source: str, lang: str) -> Optional[str]:
    """Parse via tree-sitter and report the first ERROR node found."""
    try:
        from spine.agents.tools.ast_extract import _get_parser
    except ImportError:  # pragma: no cover — module always present here
        return None

    try:
        parser = _get_parser(lang)
        tree = parser.parse(source.encode("utf-8"))
    except Exception as exc:  # pragma: no cover — setup error
        logger.debug("tree-sitter %s parse failed: %s", lang, exc)
        return None

    error = _find_error_node(tree.root_node)
    if error is None:
        return None
    return f"Syntax error near line {error.start_point[0] + 1}, column {error.start_point[1] + 1}"


def _find_error_node(node: Any) -> Any | None:
    """Walk the tree depth-first and return the first ``ERROR`` node."""
    if node.type == "ERROR" or getattr(node, "is_missing", False):
        return node
    for child in node.children:
        found = _find_error_node(child)
        if found is not None:
            return found
    return None


def _dispatch_lint(file_path: str, source: str) -> Optional[str]:
    """Pick the right syntax check for the file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    lang = _LINT_LANG_BY_EXT.get(ext)
    if lang == "python":
        return _check_python(source)
    if lang == "php":
        return _check_php(source)
    if lang == "typescript":
        return _check_typescript(source)
    # No linter for this extension — accept the write.
    return None


# ── In-memory edit resolvers ────────────────────────────────────────
# Each returns ``(new_content, None)`` on success or ``(None, error_payload)``
# on failure, where error_payload is a kwargs dict for :func:`_result`.


def _apply_find_replace(
    current: str, old_str: str, new_str: str, file_path: str
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """Resolve a single exact find-and-replace against ``current``."""
    occurrences = current.count(old_str)
    if occurrences == 0:
        return None, {
            "status": "no_match",
            "detail": (
                f"old_str not found in {file_path}. "
                "Include more surrounding context to make the snippet unique."
            ),
        }
    if occurrences > 1:
        return None, {
            "status": "ambiguous_match",
            "detail": (
                f"old_str matches {occurrences} locations in {file_path}. "
                "Include more surrounding context to make the snippet unique."
            ),
        }
    return current.replace(old_str, new_str, 1), None


def _edit_pair(edit: Any) -> tuple[Optional[str], str]:
    """Extract ``(old_str, new_str)`` from a batch item (model or dict)."""
    if isinstance(edit, dict):
        old = edit.get("old_str")
        new = edit.get("new_str")
    else:
        old = getattr(edit, "old_str", None)
        new = getattr(edit, "new_str", None)
    return old, (new or "")


def _apply_batch(
    current: str, edits: list[Any], file_path: str
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """Apply a batch of find-and-replace edits in order, all-or-nothing.

    Each edit matches against the running buffer (so later edits see the
    results of earlier ones). Any miss aborts the whole batch with the
    offending ``edit_index``; nothing is written.
    """
    if not edits:
        return None, {"status": "input_error", "detail": "edits must be a non-empty list."}
    buffer = current
    for index, edit in enumerate(edits):
        old_str, new_str = _edit_pair(edit)
        if old_str is None:
            return None, {
                "status": "input_error",
                "detail": f"edits[{index}] is missing old_str.",
                "edit_index": index,
            }
        occurrences = buffer.count(old_str)
        if occurrences == 0:
            return None, {
                "status": "no_match",
                "detail": (
                    f"edits[{index}] old_str not found in {file_path} "
                    "(after applying earlier edits). Add surrounding context."
                ),
                "edit_index": index,
            }
        if occurrences > 1:
            return None, {
                "status": "ambiguous_match",
                "detail": (
                    f"edits[{index}] old_str matches {occurrences} locations in "
                    f"{file_path}. Add surrounding context to make it unique."
                ),
                "edit_index": index,
            }
        buffer = buffer.replace(old_str, new_str, 1)
    return buffer, None


def _apply_line_range(
    current: str,
    start_line: Optional[int],
    end_line: Optional[int],
    replacement: Optional[str],
    expected: Optional[str],
    file_path: str,
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """Replace 1-indexed lines ``start_line..end_line`` with ``replacement``."""
    if start_line is None or end_line is None or replacement is None:
        return None, {
            "status": "input_error",
            "detail": "start_line, end_line, and replacement are all required for line-range edits.",
        }
    lines = current.splitlines(keepends=True)
    n_lines = len(lines)
    if start_line < 1 or end_line < start_line or end_line > n_lines:
        return None, {
            "status": "range_error",
            "detail": f"Invalid range {start_line}..{end_line} for {file_path} ({n_lines} lines).",
        }

    current_range = "".join(lines[start_line - 1 : end_line])
    if expected is not None and current_range.rstrip("\n") != expected.rstrip("\n"):
        return None, {
            "status": "stale",
            "detail": (
                f"Lines {start_line}..{end_line} of {file_path} no longer match "
                "`expected` — re-read the file and retry."
            ),
            "found": current_range,
        }

    before = "".join(lines[: start_line - 1])
    after = "".join(lines[end_line:])
    # Keep the file well-formed: if there is content after the range, or the
    # last replaced line ended in a newline, ensure the replacement does too.
    range_had_trailing_nl = lines[end_line - 1].endswith("\n")
    if replacement and not replacement.endswith("\n") and (after or range_had_trailing_nl):
        replacement = replacement + "\n"
    return before + replacement + after, None


class ReadEditLintTool(BaseTool):
    """Single write surface for the slice-implementer subagent.

    Exactly one edit mode per call: ``old_str``/``new_str`` find-and-replace,
    ``full_replace`` whole-file rewrite, an ``edits`` batch, or a
    ``start_line``/``end_line``/``replacement`` line-range edit. After
    resolving the edit in memory the tool runs a syntax check and only writes
    to disk on success. On failure it returns a structured status
    (``no_match`` / ``ambiguous_match`` / ``syntax_error`` / ``stale`` / …) and
    leaves the file untouched.
    """

    name: str = "read_edit_lint"
    description: str = (
        "Edit or create a file with a built-in syntax check. Pass exactly ONE "
        "edit mode: old_str+new_str (single-occurrence find-and-replace); "
        "full_replace (whole-file content); edits (a batch of find-and-replace "
        "ops applied atomically — all-or-nothing); or start_line+end_line+"
        "replacement (line-range edit, with optional `expected` staleness "
        "guard). On a syntax error or failed match the write is rejected "
        "without modifying the file — fix and call again. Use this in place of "
        "write_file/edit_file."
    )
    args_schema: Optional[ArgsSchema] = ReadEditLintInput

    workspace_root: str = ""

    def _resolve_path(self, file_path: str) -> Path:
        # Treat absolute paths under the workspace as the user intends them
        # (e.g. ``/spine/foo.py`` resolves to ``<workspace>/spine/foo.py``).
        clean = file_path.lstrip("/")
        return Path(self.workspace_root) / clean

    def _run(
        self,
        file_path: str,
        old_str: Optional[str] = None,
        new_str: Optional[str] = None,
        full_replace: Optional[str] = None,
        edits: Optional[list[Any]] = None,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        replacement: Optional[str] = None,
        expected: Optional[str] = None,
    ) -> str:
        # ── Determine the active mode (exactly one) ─────────────────
        active: list[str] = []
        if full_replace is not None:
            active.append("full_replace")
        if old_str is not None or new_str is not None:
            active.append("find_replace")
        if edits is not None:
            active.append("edits")
        if start_line is not None or end_line is not None or replacement is not None or expected is not None:
            active.append("line_range")
        if not active:
            return _result(
                "input_error",
                detail=(
                    "Pass exactly one edit mode: full_replace; old_str+new_str; "
                    "edits; or start_line+end_line+replacement."
                ),
            )
        if len(active) > 1:
            return _result(
                "input_error",
                detail=f"Pass exactly one edit mode, but several were provided: {active}.",
            )
        mode = active[0]

        path = self._resolve_path(file_path)
        existed = path.exists()

        # ── Build the new content in memory per mode ────────────────
        if mode == "full_replace":
            new_content = full_replace or ""
        else:
            if not existed:
                return _result(
                    "not_found",
                    detail=f"Cannot edit {file_path}: file does not exist.",
                )
            try:
                current = path.read_text(encoding="utf-8")
            except OSError as exc:
                return _result("io_error", detail=f"Could not read {file_path}: {exc}")

            if mode == "find_replace":
                if old_str is None or new_str is None:
                    return _result(
                        "input_error",
                        detail="old_str AND new_str are both required for find-and-replace.",
                    )
                new_content, error = _apply_find_replace(current, old_str, new_str, file_path)
            elif mode == "edits":
                new_content, error = _apply_batch(current, edits or [], file_path)
            else:  # line_range
                new_content, error = _apply_line_range(
                    current, start_line, end_line, replacement, expected, file_path
                )

            if error is not None:
                return _result(**error)
            assert new_content is not None

        # ── Lint the proposed content ───────────────────────────────
        lint_error = _dispatch_lint(file_path, new_content)
        if lint_error is not None:
            return _result(
                "syntax_error",
                detail=lint_error,
                file_path=file_path,
                wrote=False,
            )

        # ── Write atomically ────────────────────────────────────────
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(path, new_content)
        except OSError as exc:
            return _result("io_error", detail=f"Could not write {file_path}: {exc}")

        return _result(
            "ok",
            file_path=file_path,
            bytes_written=len(new_content.encode("utf-8")),
            created=not existed,
        )

    async def _arun(
        self,
        file_path: str,
        old_str: Optional[str] = None,
        new_str: Optional[str] = None,
        full_replace: Optional[str] = None,
        edits: Optional[list[Any]] = None,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        replacement: Optional[str] = None,
        expected: Optional[str] = None,
    ) -> str:
        return self._run(
            file_path,
            old_str=old_str,
            new_str=new_str,
            full_replace=full_replace,
            edits=edits,
            start_line=start_line,
            end_line=end_line,
            replacement=replacement,
            expected=expected,
        )
