"""Drill-down: fetch a single symbol's source by name.

Bound to the researcher subagent alongside MCP ``get_function_source``.
Looks up the file path for a symbol via the vector store's
``symbol_metadata`` index (always populated by Phase 1's indexing job),
re-runs the tree-sitter extractor on that file, and returns the matching
:class:`spine.agents.tools.ast_extract.Symbol` as JSON.

Falls back to a filesystem walk when the symbol is not in the index —
this covers fresh or uncommitted files that haven't been indexed yet.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field

from spine.agents.tools.ast_extract import _EXT_TO_LANG, extract_symbols

logger = logging.getLogger(__name__)

# Cap the fallback walk so a misnamed symbol doesn't scan the whole tree.
_FALLBACK_MAX_FILES = 200


class AstExtractSymbolInput(BaseModel):
    """Input schema for :class:`AstExtractSymbolTool`."""

    symbol_name: str = Field(description="Exact symbol name (function, class, method, interface).")
    file_hint: Optional[str] = Field(
        default=None,
        description=(
            "Optional workspace-relative file path to narrow the search. "
            "Use when you already know which file the symbol lives in."
        ),
    )


def _open_index(db_path: str) -> Optional[sqlite3.Connection]:
    """Open the vector-store sqlite database, or None if missing."""
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as exc:
        logger.debug("Could not open vector index %s: %s", db_path, exc)
        return None


def _lookup_file_paths(
    conn: sqlite3.Connection, symbol_name: str, file_hint: Optional[str]
) -> list[str]:
    """Return distinct candidate file paths from the vector index."""
    if file_hint:
        cursor = conn.execute(
            "SELECT DISTINCT file_path FROM symbol_metadata "
            "WHERE symbol_name = ? AND file_path = ?",
            (symbol_name, file_hint),
        )
    else:
        cursor = conn.execute(
            "SELECT DISTINCT file_path FROM symbol_metadata WHERE symbol_name = ?",
            (symbol_name,),
        )
    return [row["file_path"] for row in cursor]


def _walk_workspace_for_symbol(
    workspace_root: str, symbol_name: str
) -> list[str]:
    """Filesystem fallback when the index has no row for the symbol."""
    paths: list[str] = []
    root = Path(workspace_root)
    skip_dirs = {".git", ".venv", "node_modules", ".spine", "__pycache__"}
    count = 0
    for dirpath, dirnames, filenames in os.walk(workspace_root):
        # Skip noisy / heavy directories in place.
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext not in _EXT_TO_LANG:
                continue
            count += 1
            if count > _FALLBACK_MAX_FILES:
                return paths
            full = Path(dirpath) / name
            try:
                # Cheap pre-check: only run tree-sitter on files where the
                # name appears as a literal substring.
                if symbol_name.encode("utf-8") not in full.read_bytes():
                    continue
            except OSError:
                continue
            paths.append(str(full.relative_to(root)))
    return paths


class AstExtractSymbolTool(BaseTool):
    """Look up a single symbol's source code by name."""

    name: str = "ast_extract_symbol"
    description: str = (
        "Fetch the full source of a named symbol (function, class, method, interface) "
        "from the vector index, falling back to a filesystem AST walk for files that "
        "haven't been indexed. Use this when you know the symbol name and want its "
        "body without paging through top-k recall hits."
    )
    args_schema: Optional[ArgsSchema] = AstExtractSymbolInput

    workspace_root: str = ""
    db_path: str = ".spine/spine.db"

    def _run(
        self, symbol_name: str, file_hint: Optional[str] = None
    ) -> str:
        candidates: list[str] = []
        source = "index"

        conn = _open_index(self.db_path)
        if conn is not None:
            try:
                candidates = _lookup_file_paths(conn, symbol_name, file_hint)
            finally:
                conn.close()

        if not candidates:
            source = "fallback_walk"
            candidates = _walk_workspace_for_symbol(self.workspace_root, symbol_name)

        if not candidates:
            return json.dumps({
                "status": "not_found",
                "symbol_name": symbol_name,
                "file_hint": file_hint,
                "detail": "No file in the vector index or filesystem fallback contains this symbol.",
            })

        matches: list[dict[str, Any]] = []
        for rel_path in candidates:
            full = os.path.join(self.workspace_root, rel_path)
            for sym in extract_symbols(full, rel_path):
                if sym.symbol_name == symbol_name:
                    matches.append({
                        "file_path": sym.file_path,
                        "symbol_name": sym.symbol_name,
                        "symbol_type": sym.symbol_type,
                        "raw_code": sym.raw_code,
                        "start_byte": sym.start_byte,
                        "end_byte": sym.end_byte,
                        "lang": sym.lang,
                    })

        if not matches:
            return json.dumps({
                "status": "not_found",
                "symbol_name": symbol_name,
                "file_hint": file_hint,
                "searched": candidates,
                "detail": "Found candidate files but the symbol no longer exists in them.",
            })

        return json.dumps({
            "status": "ok",
            "source": source,
            "matches": matches,
        })

    async def _arun(
        self, symbol_name: str, file_hint: Optional[str] = None
    ) -> str:
        return self._run(symbol_name=symbol_name, file_hint=file_hint)
