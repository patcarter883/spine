"""Shared per-work_id cache for deterministic MCP codebase-index lookups.

The exploration subgraph fans out one ``Send("explore", …)`` per topic
with no upper bound — observed runs spin up ~9 concurrent researcher
agents per round, each independently calling
``mcp_codebase-index_get_*`` / ``find_symbol`` against the same hot
symbols. ``ReadCacheMiddleware`` dedupes inside one researcher loop
via the per-branch ``SpineContext.read_cache``, but sibling branches
launched in the same super-step never see each other's writes — the
``_merge_read_cache`` reducer only merges at fan-in, after every
branch has already paid for duplicate MCP round-trips.

This module sits in front of the MCP handler: a process-level dict
keyed by ``(work_id, tool_name, args_fingerprint)`` returning the
**full** deterministic tool result string. Each entry carries an
``asyncio.Lock`` so concurrent sibling fetches for the same key
coalesce to a single in-flight MCP call (single-flight); the rest
await the lock and read the cached result.

Restricted to known-deterministic, read-only codebase-index tools.
Mutating tools never appear here.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


# Allowlist of deterministic, read-only MCP tool prefixes whose results
# can be safely memoised across sibling branches for the same work_id.
# Keyed by server name; values are the tool-name-suffix prefixes (after
# the ``mcp_<server>_`` namespace) that qualify as deterministic lookups.
#
# Servers register themselves via ``register_cacheable_server``; the
# codebase-index entry is preloaded because spine has always treated it
# as deterministic and existing callers rely on the prior behaviour.
_DEFAULT_DETERMINISTIC_SUFFIXES: tuple[str, ...] = (
    "get_",
    "find_",
    "list_",
    "search_",
)

_cacheable_servers: dict[str, tuple[str, ...]] = {
    # Preserves the historical prefix behaviour for codebase-index:
    # any ``mcp_codebase-index_get_*`` and ``mcp_codebase-index_find_symbol``.
    # The default suffixes also cover find_symbol (find_ prefix).
    "codebase-index": _DEFAULT_DETERMINISTIC_SUFFIXES,
}

# Exact tool names (non-``mcp_``-prefixed) opted into cross-branch result
# sharing. The ``codebase_query`` facade (commit b2f60ac) replaced the raw
# ``mcp_codebase-index_*`` surface for all subagents; without this entry
# ``is_cacheable`` returned False for the facade and cross-branch dedupe
# silently stopped applying — trace 019eaecf showed the same
# ``get_source(SpineConfig)`` fetched 19× across sibling scouts.
_cacheable_tools: set[str] = {
    "codebase_query",
}


def register_cacheable_tool(tool_name: str) -> None:
    """Opt an exact (non-MCP) tool name into cross-branch result sharing.

    Only deterministic, read-only tools may be registered — the cache
    lives for the whole work_id, so results from tools that observe
    mutable state would go stale across phases.
    """
    if tool_name:
        _cacheable_tools.add(tool_name)


def register_cacheable_server(
    server_name: str,
    suffix_prefixes: tuple[str, ...] = _DEFAULT_DETERMINISTIC_SUFFIXES,
) -> None:
    """Opt a deterministic MCP server into cross-branch result sharing.

    Call once at MCP-client init for each server whose read-only tools
    are safe to memoise within a single ``work_id``. Re-registering a
    server overwrites the previous suffix list.
    """
    if not server_name:
        return
    _cacheable_servers[server_name] = tuple(suffix_prefixes)


def is_cacheable(tool_name: str) -> bool:
    """Return True when *tool_name* is a deterministic codebase lookup.

    Exact names registered via ``register_cacheable_tool`` qualify (e.g.
    the ``codebase_query`` facade); otherwise matches the
    ``mcp_<server>_<suffix>...`` naming convention against the server
    allowlist registered via ``register_cacheable_server``.
    """
    if not tool_name:
        return False
    if tool_name in _cacheable_tools:
        return True
    if not tool_name.startswith("mcp_"):
        return False
    for server, suffixes in _cacheable_servers.items():
        prefix = f"mcp_{server}_"
        if not tool_name.startswith(prefix):
            continue
        rest = tool_name[len(prefix):]
        return any(rest.startswith(suf) for suf in suffixes)
    return False


# ── Args canonicalization ────────────────────────────────────────────
#
# Fingerprinting raw facade args lets superficially different calls miss
# each other: ``codebase_query {"action":"get_source","name":"render"}``
# and the same call plus a hallucinated ``file_hint`` key (dropped by
# Pydantic ``extra="ignore"``) resolve to the *identical* backing MCP
# call but produced distinct cache keys — trace 019eafac fetched
# ``get_function_source(render)`` 3× in one work_id. A canonicalizer
# maps ``(tool_name, raw args)`` → the resolved backing call before
# fingerprinting, so every variant (and any raw ``mcp_*`` caller)
# coalesces onto one entry.

def _codebase_query_canonicalizer(
    args: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    # Lazy import: symbol_cache must stay importable independent of the
    # tools package; codebase_query does not import symbol_cache (no cycle).
    from spine.agents.tools.codebase_query import canonical_backing_call

    return canonical_backing_call(args)


_args_canonicalizers: dict[
    str, Callable[[dict[str, Any]], tuple[str, dict[str, Any]] | None]
] = {
    "codebase_query": _codebase_query_canonicalizer,
}


def register_args_canonicalizer(
    tool_name: str,
    fn: Callable[[dict[str, Any]], tuple[str, dict[str, Any]] | None],
) -> None:
    """Register a canonical-call mapper for *tool_name*.

    ``fn`` receives the raw tool args and returns ``(canonical_tool_name,
    canonical_args)``, or ``None`` to keep the raw key (e.g. when the args
    don't validate). The canonical name should itself be ``is_cacheable``
    so direct callers of the backing tool share the same entries.
    """
    if tool_name and fn is not None:
        _args_canonicalizers[tool_name] = fn


def _canonical_call(
    tool_name: str, args: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    fn = _args_canonicalizers.get(tool_name)
    if fn is not None:
        try:
            mapped = fn(args)
            if mapped is not None:
                return mapped
        except Exception:
            logger.debug(
                "symbol_cache: canonicalizer for %r failed — using raw key",
                tool_name,
                exc_info=True,
            )
    return tool_name, args


def args_fingerprint(args: dict[str, Any]) -> str:
    """Stable short hash of a tool call's arguments for cache keying."""
    try:
        canonical = json.dumps(args, sort_keys=True, default=str)
    except Exception:
        canonical = repr(sorted(args.items()))
    return hashlib.sha1(canonical.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


class _Entry:
    """Cache slot: a lock plus an optional cached result string.

    The lock is acquired by the *first* fetcher; concurrent siblings
    block on it. Once the first fetcher stores ``result`` and releases
    the lock, the others see the populated value and return it without
    invoking the MCP handler.

    ``hit_count`` tracks how many times this entry has been served
    *after* the initial fetch — used by the dedupe middleware to stamp
    a "tried-already" sentinel so the model recognises the loop.
    """

    __slots__ = ("lock", "result", "hit_count")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.result: str | None = None
        self.hit_count: int = 0


# Process-level cache. Keys are (work_id, tool_name, args_fingerprint).
_cache: dict[tuple[str, str, str], _Entry] = {}
# Guards _cache itself during entry creation. Per-entry critical
# sections use _Entry.lock, not this one.
_index_lock = asyncio.Lock()


def _key(work_id: str, tool_name: str, args: dict[str, Any]) -> tuple[str, str, str]:
    tool_name, args = _canonical_call(tool_name, args)
    return (work_id or "_unknown", tool_name, args_fingerprint(args))


async def get_or_fetch(
    work_id: str,
    tool_name: str,
    args: dict[str, Any],
    fetch: Any,
) -> tuple[str | None, int]:
    """Return the cached result for (work_id, tool_name, args) or fetch it.

    ``fetch`` is an awaitable returning the raw tool result string (or
    None when the underlying handler did not produce a string payload).
    Only the *first* concurrent caller invokes ``fetch``; siblings
    block on the per-key lock and receive the same result.

    Returns ``(content, hit_count)``:
        - ``content`` is the cached string, or ``None`` if the fetcher
          produced a non-string payload (caller falls back to the handler
          return verbatim — we don't memoise non-string results).
        - ``hit_count`` is 0 when *this* caller did the fetch and >=1 when
          the result was served from a previously populated entry. The
          dedupe middleware uses this to stamp a sentinel header.
    """
    cache_key = _key(work_id, tool_name, args)

    # Fast path: entry already populated. Still go through _index_lock
    # below for the increment so hit_count is race-free.
    entry = _cache.get(cache_key)
    if entry is not None and entry.result is not None:
        entry.hit_count += 1
        return entry.result, entry.hit_count

    async with _index_lock:
        entry = _cache.get(cache_key)
        if entry is None:
            entry = _Entry()
            _cache[cache_key] = entry

    async with entry.lock:
        if entry.result is not None:
            entry.hit_count += 1
            return entry.result, entry.hit_count
        result = await fetch()
        if isinstance(result, str):
            entry.result = result
            return result, 0
        # Non-string payload — don't memoise, but signal the caller
        # to use the handler result directly.
        return None, 0


def clear(work_id: str) -> int:
    """Evict every cache entry for *work_id*. Returns the count removed.

    Call on rework / restart when stale entries could mask code changes
    that landed between attempts. Not auto-wired — the orchestrator
    decides when to invalidate.
    """
    if not work_id:
        return 0
    keys = [k for k in _cache if k[0] == work_id]
    for k in keys:
        _cache.pop(k, None)
    if keys:
        logger.info("symbol_cache: cleared %d entries for work_id=%s", len(keys), work_id)
    return len(keys)


def _stats() -> dict[str, int]:
    """Diagnostic snapshot — populated/empty entries per work_id. Test-only."""
    populated = sum(1 for e in _cache.values() if e.result is not None)
    return {"total": len(_cache), "populated": populated}
