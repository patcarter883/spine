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
from typing import Any

logger = logging.getLogger(__name__)


# Allowlist of deterministic, read-only MCP tools whose results can be
# safely memoised across sibling branches for the same work_id. These
# query the codebase index (file/symbol structure), which only changes
# when the user edits files on disk — not during a single work item.
_CACHEABLE_TOOL_PREFIXES: tuple[str, ...] = (
    "mcp_codebase-index_get_",
    "mcp_codebase-index_find_symbol",
)


def is_cacheable(tool_name: str) -> bool:
    """Return True when *tool_name* is a deterministic codebase-index lookup."""
    if not tool_name:
        return False
    return any(tool_name.startswith(prefix) for prefix in _CACHEABLE_TOOL_PREFIXES)


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
    """

    __slots__ = ("lock", "result")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.result: str | None = None


# Process-level cache. Keys are (work_id, tool_name, args_fingerprint).
_cache: dict[tuple[str, str, str], _Entry] = {}
# Guards _cache itself during entry creation. Per-entry critical
# sections use _Entry.lock, not this one.
_index_lock = asyncio.Lock()


def _key(work_id: str, tool_name: str, args: dict[str, Any]) -> tuple[str, str, str]:
    return (work_id or "_unknown", tool_name, args_fingerprint(args))


async def get_or_fetch(
    work_id: str,
    tool_name: str,
    args: dict[str, Any],
    fetch: Any,
) -> str | None:
    """Return the cached result for (work_id, tool_name, args) or fetch it.

    ``fetch`` is an awaitable returning the raw tool result string (or
    None when the underlying handler did not produce a string payload).
    Only the *first* concurrent caller invokes ``fetch``; siblings
    block on the per-key lock and receive the same result.

    Returns ``None`` if the fetcher itself produced a non-string result —
    in that case the middleware should fall back to the handler return
    value verbatim (we don't memoise non-string payloads).
    """
    cache_key = _key(work_id, tool_name, args)

    # Fast path: entry already populated, no lock needed for read of
    # an immutable string. We still go through the index_lock to grab
    # the entry object so we don't race with concurrent creation.
    entry = _cache.get(cache_key)
    if entry is not None and entry.result is not None:
        return entry.result

    async with _index_lock:
        entry = _cache.get(cache_key)
        if entry is None:
            entry = _Entry()
            _cache[cache_key] = entry

    async with entry.lock:
        if entry.result is not None:
            return entry.result
        result = await fetch()
        if isinstance(result, str):
            entry.result = result
            return result
        # Non-string payload — don't memoise, but signal the caller
        # to use the handler result directly.
        return None


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
