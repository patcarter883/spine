"""codebase-memory-mcp backend for :class:`CodebaseQueryTool` (Phase 1).

Maps the facade's five actions onto the DeusData/codebase-memory-mcp graph
server and normalizes its response shapes. Selected by
``codebase_query_backend: codebase-memory`` in config; the default backend
remains ``codebase-index``. See docs/codebase-memory-mcp-migration-plan.md.

Everything backend-specific lives HERE (the isolation layer the plan's
"schema drift" risk calls for): tool names, argument shapes, the project
naming rule, response adaptation. The facade keeps owning arg validation,
nullish handling, and phase-aware search caps — those are model-side
protections, not backend concerns.

Ground-truth notes (captured live from v0.9.0, 2026-07-17 spike):

- The MCP wire surface is 8 tools; names are prefixed by spine's MCP client
  as ``mcp_codebase-memory_<tool>``.
- A repo indexes under a PROJECT name derived from its absolute path:
  ``/home/pat/projects/spine`` → ``home-pat-projects-spine``.
- ``search_graph`` returns ``{"total": N, "results": [{name,
  qualified_name, label, file_path, in_degree, ...}]}``.
- ``get_code_snippet`` needs the full ``qualified_name`` and returns
  ``{... "start_line": N, "source": "..."}`` — so ``get_source`` is a
  two-step: resolve via ``search_graph``, then fetch.
- ``query_graph`` (openCypher subset; ``[r:CALLS|USAGE|IMPORTS]`` union
  verified working) returns ``{"columns": [...], "rows": [[...], ...]}``.
- ``search_code`` returns ``{"results": [{node, qualified_name, file,
  start_line, match_lines, ...}], "raw_matches": [...], "total_results": N}``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SERVER = "codebase-memory"

# action → backing MCP tool name on the codebase-memory server. get_source
# ALSO uses search_graph internally (qualified-name resolution).
ACTION_TO_CBM: dict[str, str] = {
    "find_symbol":      f"mcp_{_SERVER}_search_graph",
    "get_source":       f"mcp_{_SERVER}_get_code_snippet",
    "get_dependencies": f"mcp_{_SERVER}_query_graph",
    "get_dependents":   f"mcp_{_SERVER}_query_graph",
    "search":           f"mcp_{_SERVER}_search_code",
}

INDEX_TOOL = f"mcp_{_SERVER}_index_repository"
RESOLVE_TOOL = f"mcp_{_SERVER}_search_graph"

# Edges that constitute a "uses" relationship for the deps/dependents
# actions. TESTS is deliberately included for dependents (callers include
# tests — useful to the researcher) but excluded for dependencies.
_DEP_EDGES = "CALLS|USAGE|IMPORTS"


# Patterns that hang or bloat the v0.9.0 indexer, written into the target
# repo's .cbmignore before indexing (append-only, idempotent). Root cause of
# the 2026-07-17 "12-minute index" incident was a SINGLE 6.2MB Python pickle
# at the repo root: the indexer spins on large binary blobs indefinitely
# (bisected file-by-file; the full clone with vendor/ indexes in 0.9s once
# the pickle is excluded — the gitignore/.cbmignore layers themselves work).
# Keep this list to KNOWN-pathological shapes, not a kitchen sink: exclusion
# is the upstream default's job, this is a hang guard.
CBMIGNORE_GUARDS: tuple[str, ...] = (
    "*.pkl",
    "*.db",
    "*.sqlite*",
    ".spine/",
)


def ensure_cbmignore(workspace_root: str) -> bool:
    """Ensure the hang-guard patterns are active for *workspace_root*.

    Returns True when every guard is already present in ``.cbmignore``.
    In a GIT worktree the file is never written: spine's own sandbox
    preflight requires a clean tree, and a sandbox-side write would leak
    into verified patch diffs (both observed live 2026-07-17) — the
    operator commits the guards once instead (the warning prints the
    exact block). Non-git directories get the file written directly.
    """
    root = Path(workspace_root or ".")
    path = root / ".cbmignore"
    try:
        existing = (
            path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        )
        have = {ln.strip() for ln in existing.splitlines()}
        missing = [g for g in CBMIGNORE_GUARDS if g not in have]
        if not missing:
            return True
        if (root / ".git").exists():
            logger.warning(
                "codebase_query: %s is missing hang-guard pattern(s) %s — not "
                "writing into a git worktree (dirty-tree preflights / patch "
                "diffs). Commit them once:\n  printf '%%s\\n' %s >> %s && "
                "git add %s && git commit -m 'chore: codebase-memory hang guards'",
                path, missing, " ".join(repr(m) for m in missing), path, path.name,
            )
            return False
        block = "".join(f"{g}\n" for g in missing)
        header = (
            "" if existing.endswith("\n") or not existing else "\n"
        ) + "# spine: codebase-memory-mcp hang guards (large binary blobs)\n"
        path.write_text(existing + header + block, encoding="utf-8")
        logger.info("codebase_query: added %d .cbmignore guard(s) at %s", len(missing), path)
        return True
    except OSError:
        logger.warning("codebase_query: could not read/write %s", path, exc_info=True)
        return False


def indexing_hazards(workspace_root: str) -> list[str]:
    """Indexer-visible files matching the guard patterns (>512KB).

    "Visible" = what the indexer will see given its gitignore layer:
    tracked + untracked-unignored files (``git ls-files -co
    --exclude-standard``). Non-git dirs return [] — the guard file is
    writable there, so hazards are already handled. Best-effort.
    """
    import fnmatch
    import subprocess

    root = Path(workspace_root or ".")
    try:
        out = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard"],
            cwd=root, capture_output=True, text=True, timeout=30, check=True,
        ).stdout
    except Exception:  # noqa: BLE001 — non-git / git failure ⇒ no hazard scan
        return []
    hazards: list[str] = []
    for rel in out.splitlines():
        base = rel.rsplit("/", 1)[-1]
        if any(fnmatch.fnmatch(base, g.rstrip("/")) for g in CBMIGNORE_GUARDS if not g.endswith("/")):
            try:
                if (root / rel).stat().st_size > 512 * 1024:
                    hazards.append(rel)
            except OSError:
                continue
    return hazards


def project_name_for(workspace_root: str) -> str:
    """The server's project name for a repo path (observed naming rule).

    Slashes become dashes, consecutive dashes collapse, edges strip — and
    NOTHING else changes: dots survive (live 2026-07-17: the sandbox path
    /home/pat/projects/.agripath-spine-sandbox-<id> indexed as
    home-pat-projects-.agripath-spine-sandbox-<id>, and the previous
    flatten-everything rule made every query miss with "project not found").
    This is the FALLBACK only — the authoritative name is captured from the
    index_repository response (see the facade's _cbm_ensure_indexed).
    """
    name = (workspace_root or "").strip().replace("/", "-")
    return re.sub(r"-{2,}", "-", name).strip("-")


def _cypher_str(value: str) -> str:
    """Escape a value for embedding in a single-quoted Cypher literal.

    The facade's validators already reject markup/whitespace-only args;
    this is defense-in-depth for quotes and backslashes only.
    """
    return value.replace("\\", "\\\\").replace("'", "\\'")


def backing_args_for(
    action: str,
    project: str,
    name: str | None,
    pattern: str | None,
    max_results: int,
) -> dict[str, Any]:
    """Build the backing tool's argument dict for a validated facade call."""
    if action == "find_symbol":
        # Qualified inputs ('UIApi.get_providers') match on the qualified-name
        # suffix; bare names match exactly on the symbol name.
        if name and "." in name:
            return {"project": project, "qn_pattern": f"*{name}", "limit": max_results}
        return {"project": project, "name_pattern": name, "limit": max_results}
    if action == "search":
        return {
            "project": project,
            "pattern": pattern,
            "regex": True,
            "limit": max_results,
        }
    if action in ("get_dependencies", "get_dependents"):
        leaf = (name or "").rsplit(".", 1)[-1]
        safe = _cypher_str(leaf)
        if action == "get_dependents":
            q = (
                f"MATCH (a)-[r:{_DEP_EDGES}|TESTS]->(b) WHERE b.name = '{safe}' "
                "RETURN DISTINCT a.qualified_name, a.file_path, type(r) LIMIT 40"
            )
        else:
            q = (
                f"MATCH (a)-[r:{_DEP_EDGES}]->(b) WHERE a.name = '{safe}' "
                "RETURN DISTINCT b.qualified_name, b.file_path, type(r) LIMIT 40"
            )
        # ⚠ Deliberately NO max_rows: v0.9.0 returns ZERO rows whenever the
        # max_rows request parameter is present (verified live: identical
        # query, 29 rows without it, 0 with max_rows=40 — upstream bug #2,
        # see the migration plan's Phase 1 log). The in-query LIMIT bounds
        # the result instead.
        return {"project": project, "query": q}
    if action == "get_source":
        # qualified_name is resolved by the caller via RESOLVE_TOOL first.
        return {"project": project, "qualified_name": name}
    raise ValueError(f"unknown action {action!r}")


# ── Response adaptation ──────────────────────────────────────────────────────
def _payload(result: Any) -> Any:
    """Unwrap the MCP text envelope to a parsed JSON payload (or raw text)."""
    if isinstance(result, list) and result and isinstance(result[0], dict):
        result = result[0].get("text", result)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return result
    return result


def _strip_project(qn: str, project: str) -> str:
    prefix = f"{project}."
    return qn[len(prefix):] if isinstance(qn, str) and qn.startswith(prefix) else qn


def adapt_response(
    action: str, project: str, result: Any
) -> str:
    """Normalize a codebase-memory response to the facade's text contract.

    Consumers are LLM agents reading text — the contract is "compact,
    line-oriented, says 'not found' when empty" (the facade's
    ``_looks_empty`` empty-result detection keys on those phrasings).
    """
    data = _payload(result)
    if not isinstance(data, (dict, list)):
        return str(data)

    if action == "find_symbol":
        rows = (data.get("results") or []) if isinstance(data, dict) else data
        if not rows:
            return "not found"
        # Code symbols first — when this listing is an ambiguity report,
        # the agent retries with the first plausible candidate, and a doc
        # heading at the top sends it down a dead end.
        rows = sorted(rows, key=is_doc_row)
        lines = []
        for r in rows:
            lines.append(
                f"{_strip_project(r.get('qualified_name', r.get('name', '?')), project)}"
                f"  [{r.get('label', '?')}]  {r.get('file_path', '?')}"
                + (f":{r['start_line']}" if r.get("start_line") else "")
            )
        return "\n".join(lines)

    if action == "get_source":
        if isinstance(data, dict) and data.get("source"):
            fp = data.get("file_path", "?")
            start = data.get("start_line", "?")
            return f"# {fp}:{start}\n{data['source']}"
        return "not found"

    if action in ("get_dependencies", "get_dependents"):
        rows = data.get("rows") if isinstance(data, dict) else None
        if not rows:
            return "no results"
        lines = []
        for row in rows:
            qn, fp, rel = (list(row) + ["?", "?", "?"])[:3]
            lines.append(f"{_strip_project(qn, project)}  ({rel})  {fp}")
        return "\n".join(dict.fromkeys(lines))  # dedupe, keep order

    if action == "search":
        rows = (data.get("results") or []) if isinstance(data, dict) else []
        raw = (data.get("raw_matches") or []) if isinstance(data, dict) else []
        if not rows and not raw:
            return "no matches"
        lines = []
        for r in rows:
            ml = r.get("match_lines") or []
            lines.append(
                f"{r.get('file', '?')}:{r.get('start_line', '?')}  "
                f"{_strip_project(r.get('qualified_name', r.get('node', '?')), project)}"
                + (f"  (match lines: {ml})" if ml else "")
            )
        for r in raw:
            if isinstance(r, dict):
                lines.append(f"{r.get('file', '?')}:{r.get('line', '?')}  {r.get('text', '')}".rstrip())
        return "\n".join(lines)

    return json.dumps(data) if isinstance(data, (dict, list)) else str(data)


_DOC_FILE_EXT_RE = re.compile(r"\.(md|mdx|rst|txt|adoc)$", re.IGNORECASE)


def is_doc_row(row: Any) -> bool:
    """True when a search_graph row is a DOCUMENT node, not a code symbol.

    codebase-memory-mcp indexes markdown headings as graph symbols, so a
    name query returns rows like "8.2 Exception Handler" or "1. Farm
    Managers" alongside the real class (runs 019f6e2d/019f81c1: 4/9 then
    6/15 get_source calls died on "did not resolve to exactly one symbol"
    because a doc heading shared the pool with the code definition).
    Classified by file extension (doc formats) or name shape — real code
    symbols never contain whitespace; headings almost always do. A code
    file that happens to live under docs/ (e.g. docs/generate_erd.py)
    stays a code row.
    """
    if not isinstance(row, dict):
        return False
    if _DOC_FILE_EXT_RE.search(str(row.get("file_path") or "")):
        return True
    return bool(re.search(r"\s", str(row.get("name") or "").strip()))


def resolve_qualified_name(project: str, name: str, resolve_result: Any) -> str | None:
    """Pick the qualified_name for ``get_source`` from a search_graph result.

    Code symbols outrank document nodes (markdown headings share the graph
    — see :func:`is_doc_row`); within the ranked pool, exact-name matches
    win; a single result wins; residual ambiguity → None (the caller
    reports the candidates to the agent instead of guessing).
    """
    data = _payload(resolve_result)
    rows = (data.get("results") or []) if isinstance(data, dict) else []
    if not rows:
        return None
    leaf = name.rsplit(".", 1)[-1]
    exact = [r for r in rows if r.get("name") == leaf]
    pool = exact or rows
    # Doc headings only compete when no code row exists at all (an agent
    # may genuinely ask for a doc section).
    code_rows = [r for r in pool if not is_doc_row(r)]
    pool = code_rows or pool
    if len(pool) == 1:
        return pool[0].get("qualified_name")
    # Prefer a qualified-suffix match for dotted inputs — on a DOT
    # boundary, else 'RelationshipFarmScope.apply' string-matches a query
    # for 'FarmScope.apply' and re-ambiguates it.
    if "." in name:
        suffix = [
            r for r in pool
            if str(r.get("qualified_name", "")) == name
            or str(r.get("qualified_name", "")).endswith("." + name)
        ]
        if len(suffix) == 1:
            return suffix[0].get("qualified_name")
    return None
