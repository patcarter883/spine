"""Read-edit-lint compound tool for the slice-implementer subagent.

Replaces ``write_file`` + ``edit_file`` with a single tool that:

1. Resolves the edit in memory (either an exact ``old_str`` → ``new_str``
   substitution, or a ``full_replace`` rewrite).
2. Runs a language-specific syntax check on the proposed new content.
3. On pass, writes atomically. On fail, returns a syntax-error report
   WITHOUT touching disk so the model can correct and retry.

The exact-match contract on ``old_str`` mirrors Anthropic's ``str_replace``:
no regex, no fuzzy matching, fail loudly if the snippet is missing or
appears more than once.
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


class ReadEditLintInput(BaseModel):
    """Input schema for :class:`ReadEditLintTool`."""

    file_path: str = Field(
        description="Workspace-relative path to the file to edit (or create with full_replace)."
    )
    old_str: Optional[str] = Field(
        default=None,
        description=(
            "Exact string to replace. Must appear EXACTLY ONCE in the current file. "
            "Pair with new_str. Mutually exclusive with full_replace."
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
            "Mutually exclusive with old_str/new_str."
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


class ReadEditLintTool(BaseTool):
    """Single write surface for the slice-implementer subagent.

    Either:
    - Substitute ``old_str`` → ``new_str`` (exact match, single occurrence), or
    - Replace the entire file with ``full_replace``.

    Then run a syntax check; only write to disk on success. On failure,
    return ``{"status": "syntax_error", ...}`` and leave the file untouched.
    """

    name: str = "read_edit_lint"
    description: str = (
        "Edit or create a file with a built-in syntax check. "
        "Pass either old_str+new_str (exact find-and-replace, single occurrence) "
        "OR full_replace (whole-file content). On a syntax error the write is "
        "rejected without modifying the file — fix the snippet and call again. "
        "Use this in place of write_file/edit_file."
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
    ) -> str:
        # ── Validate the input shape ────────────────────────────────
        has_replace = full_replace is not None
        has_edit = old_str is not None or new_str is not None
        if has_replace and has_edit:
            return _result(
                "input_error",
                detail="Pass either full_replace OR old_str+new_str, not both.",
            )
        if not has_replace and not has_edit:
            return _result(
                "input_error",
                detail="Pass either full_replace OR both old_str and new_str.",
            )
        if has_edit and (old_str is None or new_str is None):
            return _result(
                "input_error",
                detail="old_str AND new_str are both required for find-and-replace.",
            )

        path = self._resolve_path(file_path)
        existed = path.exists()

        # ── Build the new content in memory ─────────────────────────
        if has_replace:
            new_content = full_replace or ""
        else:
            if not existed:
                return _result(
                    "not_found",
                    detail=f"Cannot find-and-replace in {file_path}: file does not exist.",
                )
            try:
                current = path.read_text(encoding="utf-8")
            except OSError as exc:
                return _result("io_error", detail=f"Could not read {file_path}: {exc}")
            occurrences = current.count(old_str or "")
            if occurrences == 0:
                return _result(
                    "no_match",
                    detail=(
                        f"old_str not found in {file_path}. "
                        "Include more surrounding context to make the snippet unique."
                    ),
                )
            if occurrences > 1:
                return _result(
                    "ambiguous_match",
                    detail=(
                        f"old_str matches {occurrences} locations in {file_path}. "
                        "Include more surrounding context to make the snippet unique."
                    ),
                )
            new_content = current.replace(old_str or "", new_str or "", 1)

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
    ) -> str:
        return self._run(file_path, old_str=old_str, new_str=new_str, full_replace=full_replace)
