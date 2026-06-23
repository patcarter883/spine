"""Read-edit-lint compound tool for the slice-implementer subagent.

The implementer's ONLY filesystem tool: read, edit, and lint in one
surface. Replaces ``write_file`` + ``edit_file`` (and, since trace
019eb502, ``read_file`` + ``execute``) with a single tool that resolves an
edit in memory, runs a language-specific syntax check on the proposed new
content, and only writes (atomically) when the check passes. On failure it
returns a structured status WITHOUT touching disk so the model can correct
and retry in-loop. After a successful Python write it also runs ``ruff``
(when available) and reports a bounded diagnostic summary, so implementers
never need a shell to lint.

Read mode plus four mutually-exclusive edit modes:

0. *(read)* — ``file_path`` alone returns the file with line numbers;
   add ``start_line``/``end_line`` (no ``replacement``) for a range.
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
more than once. One deliberate softening: when ``old_str`` is missing but
``new_str`` is already present in the file, the status is
``already_applied`` rather than ``no_match`` — trace 019eb502 showed models
re-sending an edit that had just succeeded, reading the ``no_match`` as a
failure, and spiralling into re-reads of a file that was already correct.
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
from pydantic import BaseModel, Field, PrivateAttr

from spine.agents.tools._fs import _atomic_write

logger = logging.getLogger(__name__)

# Languages for which ast_edit can resolve a symbol anchor (must match the
# tree-sitter grammars wired into spine.agents.tools.ast_extract).
_AST_EDIT_LANGS = {".py", ".php", ".ts", ".tsx"}

# Edit-pressure thresholds: after this many anchored reads with no intervening
# successful edit, the read result carries a nudge to stop surveying and edit.
# Derived from trace 019ef1e5 where a slice editor made 332 reads / 8 edits and
# spiralled the token budget. Fires once at each threshold (not every call).
_READ_PRESSURE_SOFT = 8
_READ_PRESSURE_HARD = 16

# Directories never worth scanning when suggesting a corrected path.
_PATH_SUGGEST_SKIP = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", ".spine",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "build", "dist",
}


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


class PatchOp(BaseModel):
    """A whitespace-tolerant find-and-replace, for the ``patch`` batch mode."""

    search: str = Field(
        description=(
            "Code to locate. Matched WHITESPACE-TOLERANTLY: per-line leading "
            "indentation and trailing whitespace are ignored, so you do not "
            "need byte-exact indentation — only the trimmed lines must match, "
            "uniquely. Falls back to exact match first."
        )
    )
    replace: str = Field(
        default="",
        description=(
            "Replacement code (empty to delete). Re-indented to the matched "
            "block's indentation when the match was whitespace-tolerant."
        ),
    )


class AstEdit(BaseModel):
    """A symbol-anchored structural edit, for the ``ast_edit`` mode.

    Targets a named definition (function/method/class) by qualified name via
    tree-sitter — drift-proof and indentation-agnostic. No line numbers, no
    exact byte matching: name the symbol and supply the new code.
    """

    symbol: str = Field(
        description=(
            "Qualified name of the target definition, e.g. "
            "'SpineConfig.resolve_model' (method) or 'baseline_config_yaml' "
            "(function) or 'UIApi' (class). Must resolve to exactly one symbol."
        )
    )
    action: str = Field(
        default="replace",
        description=(
            "'replace' the whole definition, or 'insert_before' / "
            "'insert_after' to add a new top-level construct adjacent to it."
        ),
    )
    code: str = Field(
        description=(
            "New source. For 'replace': the complete new definition. For "
            "'insert_before'/'insert_after': a complete construct (def/class/"
            "import) to splice in adjacent to the symbol."
        )
    )


class ReadEditLintInput(BaseModel):
    """Input schema for :class:`ReadEditLintTool`."""

    file_path: str = Field(
        description=(
            "Workspace-relative path to the file to read, edit, or create. To "
            "READ, anchor with read_symbol or read_around (arbitrary whole-file "
            "and line-range reads are disabled — by IMPLEMENT you already know "
            "the symbol or snippet from the plan)."
        )
    )
    read_symbol: Optional[str] = Field(
        default=None,
        description=(
            "READ a single definition's current source by qualified name "
            "(e.g. 'UIApi.update_llm_provider' or 'baseline_config_yaml'). The "
            "anchored way to view code before an ast_edit — no whole-file "
            "survey. Python/PHP/TypeScript."
        ),
    )
    read_around: Optional[str] = Field(
        default=None,
        description=(
            "READ the region around an exact code snippet (whitespace-tolerant, "
            "must match uniquely) with a few lines of surrounding context. Use "
            "for non-symbol targets (imports, module-level code, config)."
        ),
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
    patch: Optional[list[PatchOp]] = Field(
        default=None,
        description=(
            "Batch of WHITESPACE-TOLERANT find-and-replace ops applied in order, "
            "all-or-nothing. Like `edits`, but each op's `search` ignores "
            "per-line indentation and trailing whitespace, and `replace` is "
            "re-indented to the match — so a slightly-off snippet still lands. "
            "Prefer this over `edits` when you are not certain of exact "
            "indentation. Mutually exclusive with the other edit modes."
        ),
    )
    ast_edit: Optional[AstEdit] = Field(
        default=None,
        description=(
            "Symbol-anchored structural edit: name a definition (e.g. "
            "'ClassName.method') and replace it, or insert a construct before/"
            "after it. No line numbers or exact-byte matching — robust to "
            "formatting. Python/PHP/TypeScript only. Mutually exclusive with "
            "the other edit modes."
        ),
    )
    start_line: Optional[int] = Field(
        default=None,
        description=(
            "1-indexed first line of the range to replace (line-range mode) "
            "or to read (read mode, when replacement is omitted)."
        ),
    )
    end_line: Optional[int] = Field(
        default=None,
        description=(
            "1-indexed last line, inclusive — of the range to replace "
            "(line-range mode) or to read (read mode, when replacement is "
            "omitted)."
        ),
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


# Bounded ruff diagnostics appended to a successful .py write. Informational
# only — the write gate stays the syntax check. This exists so implementers
# get lint feedback from their ONE tool instead of shelling out (trace
# 019eb502: `execute("ruff check ...")`, `execute("python -c 'import ast...")`
# after every edit, plus environment spelunking to find the interpreter).
_RUFF_MAX_ISSUES = 10
_RUFF_TIMEOUT_S = 10


def _ruff_report(path: Path, workspace_root: str) -> Optional[str]:
    """Run ``ruff check`` on the written file; return a bounded summary.

    Returns ``"clean"`` when ruff passes, a newline-joined issue list
    (capped at ``_RUFF_MAX_ISSUES``) when it doesn't, and ``None`` when
    ruff is unavailable or errors — fail-open, never blocks the write.
    """
    if path.suffix.lower() != ".py":
        return None
    try:
        proc = subprocess.run(
            ["ruff", "check", "--output-format=concise", str(path)],
            capture_output=True,
            timeout=_RUFF_TIMEOUT_S,
            cwd=workspace_root or None,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode == 0:
        return "clean"
    lines = proc.stdout.decode("utf-8", errors="replace").strip().splitlines()
    issues = [ln for ln in lines if ln.strip()]
    if not issues:
        return None  # non-zero with no parseable output — config error etc.
    shown = issues[:_RUFF_MAX_ISSUES]
    if len(issues) > len(shown):
        shown.append(f"(... {len(issues) - len(shown)} more issue(s))")
    return "\n".join(shown)


# Cap on whole-file read output. Reads beyond this are truncated with a
# notice steering the model to ranged reads — an uncapped read of a 50KB
# file costs ~13K tokens and then rides every subsequent turn's prompt.
_READ_MAX_LINES = 1200


def _render_read(
    file_path: str,
    content: str,
    start_line: Optional[int],
    end_line: Optional[int],
) -> str:
    """Render a line-numbered read of the file (or a 1-indexed range)."""
    lines = content.splitlines()
    n_total = len(lines)
    lo = max(1, start_line or 1)
    hi = min(n_total, end_line or n_total)
    if n_total and (lo > n_total or hi < lo):
        return _result(
            "range_error",
            detail=f"Invalid read range {start_line}..{end_line} for {file_path} ({n_total} lines).",
        )
    window = lines[lo - 1 : hi]
    truncated = len(window) > _READ_MAX_LINES
    if truncated:
        window = window[:_READ_MAX_LINES]
        hi = lo + _READ_MAX_LINES - 1
    body = "\n".join(f"{lo + i}| {line}" for i, line in enumerate(window))
    header = f"[read: {file_path} lines {lo}-{hi} of {n_total}]"
    if truncated:
        header += (
            f" (truncated at {_READ_MAX_LINES} lines — re-call with "
            "start_line/end_line for the rest)"
        )
    return f"{header}\n{body}"


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


def _already_applied_payload(file_path: str) -> dict[str, Any]:
    """Status payload for an edit whose replacement text is already present.

    Distinct from ``no_match`` so the model treats the edit as DONE instead
    of as a failure: trace 019eb502 showed implementers re-sending an edit
    that had just succeeded, reading the resulting ``no_match`` as "the edit
    failed", and spiralling into re-reads of an already-correct file.
    """
    return {
        "status": "already_applied",
        "detail": (
            f"old_str was not found in {file_path}, but new_str is already "
            "present — this edit appears to have been applied previously. "
            "Treat it as done; do NOT retry or re-read."
        ),
    }


def _apply_find_replace(
    current: str, old_str: str, new_str: str, file_path: str
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """Resolve a single exact find-and-replace against ``current``."""
    occurrences = current.count(old_str)
    if occurrences == 0:
        if new_str and new_str in current:
            return None, _already_applied_payload(file_path)
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
            if new_str and new_str in buffer:
                payload = _already_applied_payload(file_path)
                payload["edit_index"] = index
                payload["detail"] = (
                    f"edits[{index}] old_str was not found, but its new_str "
                    f"is already present in {file_path} — that edit was "
                    "applied previously. Nothing was written this call; "
                    f"re-submit the batch WITHOUT edits[{index}]."
                )
                return None, payload
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


# ── Whitespace-tolerant patch resolver (borrowed from opencode #24511) ──
# The exact `old_str`/`edits` contract fails when a model's snippet is
# indentation- or trailing-whitespace-off. `patch` retries an exact match
# first, then a per-line-trimmed match, re-indenting the replacement to the
# matched block. The all-or-nothing in-memory + syntax-gate guarantees still
# hold, so a fuzzy match that produces broken code is rejected, not written.


def _leading_ws(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _reindent(replacement: str, from_indent: str, to_indent: str) -> str:
    """Shift ``replacement`` from ``from_indent`` to ``to_indent`` base indent."""
    if from_indent == to_indent:
        return replacement
    out: list[str] = []
    for line in replacement.split("\n"):
        if not line.strip():
            out.append(line)
            continue
        body = line[len(from_indent) :] if line.startswith(from_indent) else line.lstrip()
        out.append(to_indent + body)
    return "\n".join(out)


def _fuzzy_locate(
    buffer: str, search: str
) -> tuple[Optional[tuple[int, int, str, str]], Optional[dict[str, Any]]]:
    """Locate ``search`` in ``buffer`` exactly, else whitespace-tolerantly.

    Returns ``((start, end, matched_indent, search_indent), None)`` for a
    unique match (char offsets into ``buffer``), or ``(None, error_payload)``.
    """
    exact = buffer.count(search)
    if exact == 1:
        start = buffer.index(search)
        return (start, start + len(search), "", ""), None
    if exact > 1:
        return None, {
            "status": "ambiguous_match",
            "detail": f"search matches {exact} locations. Add surrounding context.",
        }

    # Whitespace-tolerant: match a contiguous run of lines whose trimmed text
    # equals the trimmed search lines.
    buf_lines = buffer.split("\n")
    s_lines = search.split("\n")
    if s_lines and s_lines[-1] == "":  # trailing newline in search → drop empty tail
        s_lines = s_lines[:-1]
    if not s_lines:
        return None, {"status": "input_error", "detail": "search is empty."}
    s_trim = [ln.strip() for ln in s_lines]
    n = len(s_lines)

    hits: list[int] = []
    for i in range(len(buf_lines) - n + 1):
        if [buf_lines[i + j].strip() for j in range(n)] == s_trim:
            hits.append(i)
    if len(hits) == 0:
        return None, {
            "status": "no_match",
            "detail": (
                "search not found (even ignoring indentation). Re-read the "
                "file and copy the target lines, or use ast_edit by symbol."
            ),
        }
    if len(hits) > 1:
        return None, {
            "status": "ambiguous_match",
            "detail": f"search matches {len(hits)} locations (whitespace-insensitive). Add context.",
        }
    i = hits[0]
    # Char span of buffer lines [i, i+n).
    start = sum(len(buf_lines[k]) + 1 for k in range(i))
    end = start + sum(len(buf_lines[i + j]) for j in range(n)) + (n - 1)
    return (start, end, _leading_ws(buf_lines[i]), _leading_ws(s_lines[0])), None


def _apply_patch(
    current: str, patches: list[Any], file_path: str
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """Apply whitespace-tolerant search/replace ops in order, all-or-nothing."""
    if not patches:
        return None, {"status": "input_error", "detail": "patch must be a non-empty list."}
    buffer = current
    for index, op in enumerate(patches):
        search = op.get("search") if isinstance(op, dict) else getattr(op, "search", None)
        replace = (op.get("replace") if isinstance(op, dict) else getattr(op, "replace", "")) or ""
        if not search:
            return None, {
                "status": "input_error",
                "detail": f"patch[{index}] is missing search.",
                "edit_index": index,
            }
        located, error = _fuzzy_locate(buffer, search)
        if error is not None:
            if error.get("status") == "no_match" and replace and replace.strip() in buffer:
                payload = _already_applied_payload(file_path)
                payload["edit_index"] = index
                return None, payload
            error["edit_index"] = index
            return None, error
        start, end, matched_indent, search_indent = located
        buffer = buffer[:start] + _reindent(replace, search_indent, matched_indent) + buffer[end:]
    return buffer, None


# ── Symbol-anchored structural edit (borrowed from opencode #18822) ──
# Reuses spine's existing tree-sitter symbol extractor instead of an external
# ast-grep dependency: locate a definition by qualified name, then replace it
# or splice code adjacent to it — drift-proof and indentation-agnostic.


def _match_symbols(symbols: list, symbol: str) -> list:
    """Match ``symbol`` against extracted symbols, tolerating over-qualification.

    The planner emits module-dotted names ('spine.ui_api.api.UIApi' or
    'pkg.mod.Cls.method'); tree-sitter exposes 'UIApi' / 'Cls.method'. Match the
    qualified_name / bare name, or the trailing segment(s) of what was asked.
    Callers handle the >1-match (ambiguous) case.
    """
    tail1 = symbol.split(".")[-1]
    tail2 = ".".join(symbol.split(".")[-2:])
    return [
        s for s in symbols
        if symbol in (s.qualified_name, s.symbol_name)
        or s.symbol_name == tail1
        or s.qualified_name == tail2
    ]


_READ_AROUND_CONTEXT = 4


def _read_around(file_path: str, content: str, snippet: str) -> str:
    """Render the region around a (whitespace-tolerant) snippet match ± context."""
    located, error = _fuzzy_locate(content, snippet)
    if error is not None:
        return _result(**error)
    start, end, _mi, _si = located
    lo = content.count("\n", 0, start) + 1
    hi = content.count("\n", 0, end) + 1
    return _render_read(
        file_path, content, max(1, lo - _READ_AROUND_CONTEXT), hi + _READ_AROUND_CONTEXT
    )


def _edit_feedback(
    status: str,
    detail: str,
    *,
    target: Optional[str] = None,
    next_action: Optional[str] = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build a reference-shaped edit-failure dict (PR-B feedback contract).

    Every structured ``ast_edit`` failure returns the same shape so a weak model
    can recover deterministically instead of guessing — the three parts are:

    - ``detail``: the specific defect, in prose.
    - ``target``: a *resolvable* reference the model can act on next — a symbol
      qualified name or a ``file:line`` location (omitted when not applicable).
    - ``next_action``: the concrete next call to make.

    Status-specific extras (e.g. ``available_symbols``) pass straight through.
    This is what lets a 30B model self-correct from a failed edit in one step
    rather than spiralling into a blind ``full_replace`` (GLM_QWEN_BENCH_ANALYSIS.md).
    """
    fb: dict[str, Any] = {"status": status, "detail": detail}
    if target is not None:
        fb["target"] = target
    if next_action is not None:
        fb["next_action"] = next_action
    fb.update(extra)
    return fb


def _apply_ast_edit(
    current: str,
    file_path: str,
    symbol: str,
    action: str,
    code: str,
    workspace_path: str,
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """Resolve ``symbol`` via tree-sitter and apply a structural edit."""
    if action not in ("replace", "insert_before", "insert_after"):
        return None, _edit_feedback(
            "input_error",
            f"ast_edit action must be replace|insert_before|insert_after, got {action!r}.",
            next_action="Re-call ast_edit with action set to 'replace', 'insert_before', or 'insert_after'.",
        )
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in _AST_EDIT_LANGS:
        return None, _edit_feedback(
            "input_error",
            f"ast_edit unsupported for {ext or 'this file type'}; use patch/edits instead.",
            next_action="Use the patch or edits mode for this file type instead of ast_edit.",
        )
    try:
        from spine.agents.tools.ast_extract import extract_symbols
    except Exception as exc:  # pragma: no cover — module always present
        return None, _edit_feedback("io_error", f"ast_extract unavailable: {exc}")

    # extract_symbols reads from disk; `current` equals the on-disk content at
    # read time, so its byte offsets index `current` (utf-8) correctly.
    full_path = os.path.join(workspace_path, file_path.lstrip("/"))
    try:
        symbols = extract_symbols(full_path, file_path)
    except Exception as exc:  # noqa: BLE001 — surface as a clean tool status
        return None, _edit_feedback(
            "io_error",
            f"Could not parse {file_path}: {exc}",
            next_action="Fix the syntax error in the file (or use full_replace) before ast_edit.",
        )

    matches = _match_symbols(symbols, symbol)
    if not matches:
        names = sorted({s.qualified_name for s in symbols})[:15]
        # Creation-anchor guard: if the edit body itself defines a symbol whose
        # leaf name matches the requested anchor, the model is trying to CREATE
        # that definition while anchoring to itself — a phantom anchor, and the
        # exact misstep that began the GLM destructive-recovery spiral
        # (scratch/implement_bench/GLM_QWEN_BENCH_ANALYSIS.md). Hand back a
        # concrete, grounded anchor so a weak model recovers in one step instead
        # of falling back to a blind full_replace.
        import re as _re

        leaf = symbol.rsplit(".", 1)[-1]
        defined = set(
            _re.findall(r"(?m)^[ \t]*(?:async\s+)?(?:def|class)\s+(\w+)", code)
        )
        if leaf and leaf in defined and symbols:
            # Prefer the last existing symbol in the target's own scope (e.g. the
            # last method of the same class) so the insert lands in the right
            # place; fall back to the last symbol in the file.
            parent = symbol.rsplit(".", 1)[0] if "." in symbol else ""
            scope = [
                s
                for s in symbols
                if parent and s.qualified_name.startswith(parent + ".")
            ]
            anchor = max(scope or symbols, key=lambda s: s.end_byte).qualified_name
            return None, _edit_feedback(
                "no_match",
                (
                    f"symbol {symbol!r} does not exist in {file_path} yet — your "
                    f"code defines it, so you are CREATING it, not editing it. Do "
                    f"not anchor to a symbol you are creating."
                ),
                target=anchor,
                next_action=(
                    f"Call ast_edit with action='insert_after' anchored to "
                    f"{anchor!r} (the last existing symbol)."
                ),
                available_symbols=names,
            )
        return None, _edit_feedback(
            "no_match",
            f"symbol {symbol!r} not found in {file_path}.",
            next_action=(
                "Anchor to one of available_symbols, or to add a NEW definition "
                "use action='insert_after' anchored to an existing symbol."
            ),
            available_symbols=names,
        )
    if len(matches) > 1:
        enc = current.encode("utf-8")
        locs = [enc[: m.start_byte].count(b"\n") + 1 for m in matches]
        return None, _edit_feedback(
            "ambiguous_match",
            (
                f"symbol {symbol!r} matches {len(matches)} definitions in "
                f"{file_path} at lines {locs}."
            ),
            target=f"{file_path}:{','.join(str(line) for line in locs)}",
            next_action=(
                "Remove the duplicate block(s) with old_str/new_str targeting one "
                "definition, then retry ast_edit."
            ),
        )
    sym = matches[0]
    if action in ("insert_after", "insert_before"):
        import re as _re

        inserted_defs = set(_re.findall(r"(?m)^\s*(?:async\s+)?def\s+(\w+)", code))
        if inserted_defs:
            existing_simple = {s.qualified_name.split(".")[-1] for s in symbols}
            conflicts = sorted(inserted_defs & existing_simple)
            if conflicts:
                return None, _edit_feedback(
                    "conflict_error",
                    (
                        f"Inserted code re-defines {conflicts} which already "
                        f"exist in {file_path}."
                    ),
                    target=f"{file_path}: {', '.join(conflicts)}",
                    next_action=(
                        "Remove the existing definition(s) first, or use "
                        "action='replace' to update one in place."
                    ),
                )
    buf = current.encode("utf-8")
    code_bytes = code.encode("utf-8")
    # A method's start_byte sits AFTER its leading indentation; anchor replace/
    # insert_before at the start of the symbol's LINE so the spliced `code`
    # owns the indentation (it should carry the symbol's natural indent).
    line_start = buf.rfind(b"\n", 0, sym.start_byte) + 1
    if action == "replace":
        new_bytes = buf[:line_start] + code_bytes + buf[sym.end_byte :]
    elif action == "insert_before":
        new_bytes = buf[:line_start] + code_bytes + b"\n\n" + buf[line_start:]
    else:  # insert_after
        tail = buf[sym.end_byte :]
        sep = b"\n\n" if not tail.startswith(b"\n\n") else b""
        new_bytes = buf[: sym.end_byte] + sep + code_bytes + tail
    return new_bytes.decode("utf-8"), None


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
        "Read, edit, or create a file — your single filesystem tool. "
        "READ (anchored only — arbitrary whole-file/line-range reads are "
        "disabled): read_symbol='ClassName.method' returns a definition's "
        "source; read_around='exact snippet' returns the region around a "
        "snippet. "
        "EDIT: pass exactly ONE edit mode: old_str+new_str "
        "(single-occurrence find-and-replace); full_replace (whole-file "
        "content); edits (a batch of EXACT find-and-replace ops applied "
        "atomically — all-or-nothing); patch (a batch of WHITESPACE-TOLERANT "
        "search/replace ops — use when unsure of exact indentation); ast_edit "
        "(symbol-anchored structural edit — name a def/class and replace or "
        "insert before/after it, no line numbers needed); or "
        "start_line+end_line+replacement (line-range edit, with optional "
        "`expected` staleness guard). On a "
        "syntax error or failed match the write is rejected without "
        "modifying the file — fix and call again. status='already_applied' "
        "means the change is ALREADY in the file: move on, do not retry. "
        "status='already_read' means you read this exact symbol/snippet before "
        "and the file is unchanged — its source is already above; do not re-read, "
        "make your edit. status='not_found' includes a `did_you_mean` list of "
        "real paths when the path looks mistyped — use one of those. "
        "Successful Python writes include a `ruff` field with lint "
        "diagnostics — no shell needed to lint."
    )
    args_schema: Optional[ArgsSchema] = ReadEditLintInput

    workspace_root: str = ""
    # The plan's authoritative target files for THIS slice (when known). Used to
    # ground path-correction suggestions so the editor stops inventing variants
    # of files the plan already pinned. Empty for non-slice callers (researcher).
    target_files: list[str] = Field(default_factory=list)

    # ── Per-slice editor-session state ──────────────────────────────────
    # One tool instance is bound per slice-implementer invocation (see
    # spine/agents/subagents.py), so this state spans exactly one editor's
    # read/edit loop. It powers three anti-spiral behaviours observed missing
    # in trace 019ef1e5: read de-duplication, "did you mean" path correction,
    # and edit-pressure nudges.
    _read_cache: dict = PrivateAttr(default_factory=dict)  # key -> (epoch, call#)
    _file_epoch: dict = PrivateAttr(default_factory=dict)  # file_path -> int
    _reads_since_edit: int = PrivateAttr(default=0)
    _read_calls: int = PrivateAttr(default=0)
    _file_index_cache: Optional[dict] = PrivateAttr(default=None)

    def _build_file_index(self) -> dict:
        """Lazily map basename -> [workspace-relative paths] for suggestions."""
        if self._file_index_cache is not None:
            return self._file_index_cache
        idx: dict[str, list[str]] = {}
        root = Path(self.workspace_root)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _PATH_SUGGEST_SKIP]
            for fn in filenames:
                try:
                    rel = (Path(dirpath) / fn).relative_to(root).as_posix()
                except ValueError:
                    continue
                idx.setdefault(fn, []).append(rel)
        self._file_index_cache = idx
        return idx

    def _did_you_mean(self, file_path: str) -> dict:
        """Suggest real workspace paths for a missed path.

        Prefers the slice's ``target_files`` (the plan's authoritative paths),
        then falls back to basename, then stem matches across the tree.
        """
        name = Path(file_path).name
        stem = Path(file_path).stem
        idx = self._build_file_index()
        targets = {t.lstrip("/") for t in self.target_files}
        hits: list[str] = [t for t in targets if Path(t).name == name]
        hits += idx.get(name, [])
        if not hits:
            for n, paths in idx.items():
                if Path(n).stem == stem:
                    hits += paths
        # de-dup preserving order, target_files first
        seen: set[str] = set()
        ordered = [h for h in hits if not (h in seen or seen.add(h))]
        return {"did_you_mean": ordered[:5]} if ordered else {}

    def _pressure(self, out: str) -> str:
        """Append a one-shot edit-pressure nudge at each read threshold."""
        n = self._reads_since_edit
        if n == _READ_PRESSURE_HARD:
            return out + (
                f"\n\n[⚠⚠ {n} anchored reads since your last successful edit and "
                "still no change applied. STOP reading — surveying is done. The "
                "slice's edit_plan / target_files name exactly what to change. "
                "Apply the next edit NOW with read_edit_lint (ast_edit by symbol, "
                "patch, or full_replace), or return a status explaining the "
                "specific blocker.]"
            )
        if n == _READ_PRESSURE_SOFT:
            return out + (
                f"\n\n[⚠ {n} reads since your last edit. The code you need is "
                "already above — switch from surveying to editing: make the next "
                "change with read_edit_lint.]"
            )
        return out

    def _resolve_path(self, file_path: str) -> Path:
        # Treat absolute paths under the workspace as the user intends them
        # (e.g. ``/spine/foo.py`` resolves to ``<workspace>/spine/foo.py``).
        clean = file_path.lstrip("/")
        return Path(self.workspace_root) / clean

    def _read_symbol(self, file_path: str, content: str, symbol: str) -> str:
        """Return the named definition's current source, line-numbered."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in _AST_EDIT_LANGS:
            return _result(
                "input_error",
                detail=f"read_symbol unsupported for {ext or 'this file type'}; "
                "use read_around='snippet' instead.",
            )
        try:
            from spine.agents.tools.ast_extract import extract_symbols

            full_path = os.path.join(self.workspace_root, file_path.lstrip("/"))
            symbols = extract_symbols(full_path, file_path)
        except Exception as exc:  # noqa: BLE001
            return _result("io_error", detail=f"Could not parse {file_path}: {exc}")
        matches = _match_symbols(symbols, symbol)
        if not matches:
            names = sorted({s.qualified_name for s in symbols})[:20]
            return _result(
                "no_match",
                detail=f"symbol {symbol!r} not found in {file_path}.",
                available_symbols=names,
            )
        if len(matches) > 1:
            return _result(
                "ambiguous_match",
                detail=f"symbol {symbol!r} matches {len(matches)} definitions.",
            )
        sym = matches[0]
        lo = content.count("\n", 0, sym.start_byte) + 1
        hi = content.count("\n", 0, sym.end_byte) + 1
        return _render_read(file_path, content, lo, hi)

    def _run(
        self,
        file_path: str,
        old_str: Optional[str] = None,
        new_str: Optional[str] = None,
        full_replace: Optional[str] = None,
        edits: Optional[list[Any]] = None,
        patch: Optional[list[Any]] = None,
        ast_edit: Optional[Any] = None,
        read_symbol: Optional[str] = None,
        read_around: Optional[str] = None,
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
        if patch is not None:
            active.append("patch")
        if ast_edit is not None:
            active.append("ast_edit")
        if read_symbol is not None:
            active.append("read_symbol")
        if read_around is not None:
            active.append("read_around")
        if replacement is not None or expected is not None:
            active.append("line_range")
        elif start_line is not None or end_line is not None:
            # Bare line bounds = an arbitrary ranged read — now disabled.
            active.append("read_disabled")
        if not active:
            active.append("read_disabled")  # file_path alone = whole-file read
        if len(active) > 1:
            return _result(
                "input_error",
                detail=f"Pass exactly one edit mode, but several were provided: {active}.",
            )
        mode = active[0]

        path = self._resolve_path(file_path)
        existed = path.exists()

        # ── Arbitrary reads are disabled ────────────────────────────
        # By IMPLEMENT the plan/decompose stages have already identified the
        # target symbol or snippet, so a whole-file / arbitrary line-range read
        # is the model surveying the codebase a second time (trace: read all of
        # config.py to "understand the structure of SpineConfig"). Steer it to
        # an anchored read instead.
        if mode == "read_disabled":
            return _result(
                "read_disabled",
                detail=(
                    f"Arbitrary reads of {file_path} are disabled. View code by "
                    "ANCHOR: read_symbol='ClassName.method' for a definition, or "
                    "read_around='exact snippet' for a region. Your slice's "
                    "edit_plan names the symbol to target — read_symbol it, then "
                    "apply ast_edit. Do not re-survey the file."
                ),
            )

        if mode in ("read_symbol", "read_around"):
            self._read_calls += 1
            self._reads_since_edit += 1
            anchor = read_symbol if mode == "read_symbol" else f"~{read_around}"
            key = (file_path, anchor)
            if not existed:
                return _result(
                    "not_found",
                    detail=f"Cannot read {file_path}: file does not exist.",
                    **self._did_you_mean(file_path),
                )
            # Per-slice read cache: a repeat anchored read of an UNCHANGED file
            # returns a compact pointer instead of re-injecting the body — the
            # source is already above in the conversation. Invalidated per-file
            # by a successful edit (epoch bump). This is the direct fix for the
            # 56%-redundant re-reads in trace 019ef1e5.
            epoch = self._file_epoch.get(file_path, 0)
            cached = self._read_cache.get(key)
            if cached is not None and cached[0] == epoch:
                return self._pressure(
                    _result(
                        "already_read",
                        detail=(
                            f"You already read {file_path} :: {anchor} (call "
                            f"#{cached[1]}, unchanged since). Its source is above "
                            "in this conversation — do NOT re-read it. Apply your "
                            "edit now, or report what specifically blocks you."
                        ),
                    )
                )
            try:
                content = path.read_text(encoding="utf-8")
            except OSError as exc:
                return _result("io_error", detail=f"Could not read {file_path}: {exc}")
            if mode == "read_symbol":
                out = self._read_symbol(file_path, content, read_symbol or "")
            else:
                out = _read_around(file_path, content, read_around or "")
            # Cache only successful reads (rendered output starts with "[read:");
            # error statuses are JSON and must stay re-tryable.
            if out.startswith("[read:"):
                self._read_cache[key] = (epoch, self._read_calls)
            return self._pressure(out)

        # ── Build the new content in memory per mode ────────────────
        if mode == "full_replace":
            new_content = full_replace or ""
        else:
            if not existed:
                return _result(
                    "not_found",
                    detail=f"Cannot edit {file_path}: file does not exist.",
                    **self._did_you_mean(file_path),
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
            elif mode == "patch":
                new_content, error = _apply_patch(current, patch or [], file_path)
            elif mode == "ast_edit":
                ae = ast_edit if isinstance(ast_edit, dict) else ast_edit.__dict__
                new_content, error = _apply_ast_edit(
                    current,
                    file_path,
                    ae.get("symbol", ""),
                    ae.get("action", "replace"),
                    ae.get("code", ""),
                    self.workspace_root,
                )
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

        ok_fields: dict[str, Any] = {
            "file_path": file_path,
            "bytes_written": len(new_content.encode("utf-8")),
            "created": not existed,
        }
        ruff = _ruff_report(path, self.workspace_root)
        if ruff is not None:
            ok_fields["ruff"] = ruff
        # A successful write changes the file → bump its epoch so cached reads of
        # it are re-fetched fresh, and reset the edit-pressure counter (the agent
        # just made progress).
        self._file_epoch[file_path] = self._file_epoch.get(file_path, 0) + 1
        self._reads_since_edit = 0
        return _result("ok", **ok_fields)

    async def _arun(
        self,
        file_path: str,
        old_str: Optional[str] = None,
        new_str: Optional[str] = None,
        full_replace: Optional[str] = None,
        edits: Optional[list[Any]] = None,
        patch: Optional[list[Any]] = None,
        ast_edit: Optional[Any] = None,
        read_symbol: Optional[str] = None,
        read_around: Optional[str] = None,
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
            patch=patch,
            ast_edit=ast_edit,
            read_symbol=read_symbol,
            read_around=read_around,
            start_line=start_line,
            end_line=end_line,
            replacement=replacement,
            expected=expected,
        )
