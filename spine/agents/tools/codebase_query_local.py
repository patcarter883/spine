"""Local-index fallback for ``codebase_query``.

mcp-codebase-index has no PHP support, so for PHP targets the MCP-backed
actions return nothing. These functions serve the same five actions from
the vector store built by Phase 1's indexing job (``symbol_metadata`` +
FTS5 ``symbol_fts`` + ``symbol_edges`` in ``.spine/spine.db``).

Every function returns a JSON string tagged ``"source": "local_index"``,
or ``None`` when the local index has nothing to say — the caller then
surfaces the original MCP result (or error) instead.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from typing import Any

from spine.agents.tools.ast_extract import extract_symbols
from spine.agents.tools.ast_extract_symbol import _open_index

logger = logging.getLogger(__name__)

# Match the documented ~8 KB cap of the MCP search action.
_MAX_OUTPUT_BYTES = 8192
_SNIPPET_CHARS = 300

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

# symbol_metadata stores qualified names ('Class.method'); a bare 'method'
# query must match by suffix too. Used with (name, name) params.
_NAME_MATCH = "(symbol_name = ? OR symbol_name LIKE '%.' || ?)"


def _query(
    db_path: str, sql: str, params: tuple[Any, ...]
) -> list[sqlite3.Row] | None:
    """Run one read-only query; None if the db/table is unavailable."""
    conn = _open_index(db_path)
    if conn is None:
        return None
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        logger.debug("local index query failed (%s): %s", db_path, exc)
        return None
    finally:
        conn.close()


def _ok(payload: dict[str, Any]) -> str:
    return json.dumps({"status": "ok", "source": "local_index", **payload})


def local_list_files(
    workspace_root: str,
    extensions: frozenset[str],
    skip_dirs: frozenset[str] = frozenset(),
) -> list[str]:
    """Walk *workspace_root* for source files matching *extensions*.

    Lists paths only — never reads file contents. Prunes *skip_dirs* and
    every dot-prefixed directory, returning repo-relative paths.
    """
    results: list[str] = []
    for dirpath, dirnames, filenames in os.walk(workspace_root):
        dirnames[:] = [
            d for d in dirnames if d not in skip_dirs and not d.startswith(".")
        ]
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in extensions:
                full = os.path.join(dirpath, fname)
                results.append(os.path.relpath(full, workspace_root))
    return results


def lookup_local_langs(db_path: str, name: str) -> set[str]:
    """Languages of indexed symbols matching *name* (empty set if none)."""
    rows = _query(
        db_path,
        f"SELECT DISTINCT lang FROM symbol_metadata WHERE {_NAME_MATCH}",
        (name, name),
    )
    return {r["lang"] for r in rows} if rows else set()


def local_list_file_symbols(db_path: str, file_path: str) -> list[str]:
    """Return all symbol_name values indexed for *file_path*, alphabetically.

    Used by the decomposer to inject a ground-truth symbol menu into its prompt
    so it never guesses qualified names it can't verify.
    """
    rows = _query(
        db_path,
        "SELECT symbol_name FROM symbol_metadata WHERE file_path = ? ORDER BY symbol_name",
        (file_path,),
    )
    return [r["symbol_name"] for r in rows] if rows else []


def local_find_symbol(db_path: str, name: str) -> str | None:
    """Locate a symbol via symbol_metadata; None when not indexed."""
    rows = _query(
        db_path,
        "SELECT file_path, symbol_name, symbol_type, lang, "
        f"substr(enriched_summary, 1, {_SNIPPET_CHARS}) AS summary "
        f"FROM symbol_metadata WHERE {_NAME_MATCH} LIMIT 20",
        (name, name),
    )
    if not rows:
        return None
    return _ok({"matches": [dict(r) for r in rows]})


def local_get_source(db_path: str, workspace_root: str, name: str) -> str | None:
    """Full source of a symbol — fresh AST slice, else stored raw_code."""
    rows = _query(
        db_path,
        "SELECT file_path, symbol_name, symbol_type, lang, raw_code "
        f"FROM symbol_metadata WHERE {_NAME_MATCH} LIMIT 5",
        (name, name),
    )
    if not rows:
        return None

    matches: list[dict[str, Any]] = []
    for row in rows:
        rel_path = row["file_path"]
        raw_code = row["raw_code"]
        freshness = "indexed"
        # Prefer a fresh byte slice from disk — the index may be stale.
        full = os.path.join(workspace_root, rel_path)
        for sym in extract_symbols(full, rel_path):
            if row["symbol_name"] in (sym.symbol_name, sym.qualified_name):
                raw_code = sym.raw_code
                freshness = "fresh"
                break
        matches.append({
            "file_path": rel_path,
            "symbol_name": row["symbol_name"],
            "symbol_type": row["symbol_type"],
            "lang": row["lang"],
            "freshness": freshness,
            "raw_code": raw_code,
        })
    return _ok({"matches": matches})


def _fts_match_expr(pattern: str) -> str | None:
    """Reduce a regex-shaped pattern to a safe FTS5 AND-of-tokens query."""
    tokens = [t for t in _TOKEN_RE.findall(pattern) if len(t) > 1]
    if not tokens:
        return None
    seen: set[str] = set()
    terms = []
    for t in tokens:
        low = t.lower()
        if low not in seen:
            seen.add(low)
            terms.append(f'"{t}"')
    return " AND ".join(terms)


def local_search(db_path: str, pattern: str, max_results: int) -> str | None:
    """Lexical search over the symbol index.

    The tool's ``pattern`` arg is regex-shaped; FTS5 can't take regexes,
    so the pattern is reduced to AND-of-tokens. When that yields nothing
    (or no hits), fall back to a literal substring LIKE over raw_code.
    """
    rows: list[sqlite3.Row] = []
    match = _fts_match_expr(pattern)
    if match:
        rows = _query(
            db_path,
            "SELECT m.file_path, m.symbol_name, m.symbol_type, m.lang, "
            f"substr(m.raw_code, 1, {_SNIPPET_CHARS}) AS snippet "
            "FROM symbol_fts JOIN symbol_metadata m ON symbol_fts.rowid = m.id "
            "WHERE symbol_fts MATCH ? "
            "ORDER BY bm25(symbol_fts, 10.0, 5.0, 1.0) LIMIT ?",
            (match, max_results),
        ) or []
    if not rows:
        rows = _query(
            db_path,
            "SELECT file_path, symbol_name, symbol_type, lang, "
            f"substr(raw_code, 1, {_SNIPPET_CHARS}) AS snippet "
            "FROM symbol_metadata WHERE raw_code LIKE '%' || ? || '%' LIMIT ?",
            (pattern, max_results),
        ) or []
    if not rows:
        return None

    matches: list[dict[str, Any]] = []
    size = 0
    truncated = False
    for r in rows:
        item = dict(r)
        size += len(json.dumps(item))
        if size > _MAX_OUTPUT_BYTES:
            truncated = True
            break
        matches.append(item)
    payload: dict[str, Any] = {"matches": matches}
    if truncated:
        payload["truncated"] = True
    return _ok(payload)


def local_dependencies(db_path: str, name: str, direction: str) -> str | None:
    """Serve get_dependencies / get_dependents from ``symbol_edges``.

    Only answers for symbols the local index knows as PHP — for other
    languages the MCP server's graph is authoritative, so this returns
    ``None`` and the caller keeps the MCP result.
    """
    langs = lookup_local_langs(db_path, name)
    if langs != {"php"}:
        return None

    if direction == "dependencies":
        rows = _query(
            db_path,
            "SELECT edge_kind, dst_name FROM symbol_edges "
            "WHERE src_symbol = ? OR src_symbol LIKE '%.' || ? "
            "ORDER BY edge_kind, dst_name LIMIT 100",
            (name, name),
        )
        edges = (
            [{"kind": r["edge_kind"], "uses": r["dst_name"]} for r in rows]
            if rows else []
        )
    else:
        rows = _query(
            db_path,
            "SELECT src_file, src_symbol, edge_kind FROM symbol_edges "
            "WHERE dst_name = ? OR dst_name LIKE '%.' || ? "
            "ORDER BY src_file, src_symbol LIMIT 100",
            (name, name),
        )
        edges = (
            [
                {
                    "file": r["src_file"],
                    "symbol": r["src_symbol"] or "<file-level>",
                    "kind": r["edge_kind"],
                }
                for r in rows
            ]
            if rows else []
        )

    if edges:
        return _ok({"symbol": name, direction: edges})

    # Symbol is known PHP but no edges: distinguish a pre-edges index
    # (re-index needed) from a genuinely edge-less symbol.
    any_php_edges = _query(
        db_path, "SELECT 1 FROM symbol_edges WHERE lang = 'php' LIMIT 1", ()
    )
    if not any_php_edges:
        return json.dumps({
            "status": "unavailable",
            "source": "local_index",
            "detail": (
                "PHP dependency edges are not in the index yet — it predates "
                "edge extraction. Re-run indexing (spine index), or use "
                "action='search' with the symbol name to find call sites."
            ),
        })
    return json.dumps({
        "status": "ok",
        "source": "local_index",
        "symbol": name,
        direction: [],
        "detail": f"No {direction} recorded for this PHP symbol.",
    })
