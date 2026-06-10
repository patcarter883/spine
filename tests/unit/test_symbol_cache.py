"""Tests for the per-work_id MCP symbol-fetch cache."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.agents import symbol_cache


@pytest.fixture(autouse=True)
def _reset_cache():
    """Wipe the module-level cache between tests."""
    symbol_cache._cache.clear()
    yield
    symbol_cache._cache.clear()


class TestIsCacheable:
    def test_codebase_index_get_tools_are_cacheable(self):
        assert symbol_cache.is_cacheable("mcp_codebase-index_get_function_source")
        assert symbol_cache.is_cacheable("mcp_codebase-index_get_class_source")
        assert symbol_cache.is_cacheable("mcp_codebase-index_get_dependencies")

    def test_find_symbol_is_cacheable(self):
        assert symbol_cache.is_cacheable("mcp_codebase-index_find_symbol")

    def test_search_codebase_is_cacheable(self):
        # search_* is in the default deterministic suffix set — read-only
        # regex against the index is safe to share across sibling branches.
        assert symbol_cache.is_cacheable("mcp_codebase-index_search_codebase")

    def test_codebase_query_facade_is_cacheable(self):
        # The facade replaced the raw mcp_codebase-index_* surface for
        # subagents (commit b2f60ac); it must stay on the allowlist or
        # cross-branch dedupe silently stops applying (trace 019eaecf:
        # get_source(SpineConfig) fetched 19× by sibling scouts).
        assert symbol_cache.is_cacheable("codebase_query")

    def test_non_mcp_and_mutating_tools_are_not_cacheable(self):
        assert not symbol_cache.is_cacheable("read_file")
        assert not symbol_cache.is_cacheable("write_file")
        assert not symbol_cache.is_cacheable("")
        # File-observing research tools are NOT globally cacheable — files
        # change during implement; only the exploration worker path opts
        # them in (where the workspace is read-only).
        assert not symbol_cache.is_cacheable("ast_extract_symbol")
        assert not symbol_cache.is_cacheable("search_codebase")
        # No suffix match — create_/update_/delete_/run_ are excluded.
        assert not symbol_cache.is_cacheable("mcp_codebase-index_create_thing")
        assert not symbol_cache.is_cacheable("mcp_codebase-index_run_query")
        # Unregistered server — even read-shaped names don't qualify.
        assert not symbol_cache.is_cacheable("mcp_unregistered_get_thing")


class TestRegisterCacheableServer:
    def test_registering_a_server_opts_in_its_read_tools(self):
        # Sanity: unregistered server's read-shaped tools don't qualify.
        assert not symbol_cache.is_cacheable("mcp_my-server_get_thing")
        try:
            symbol_cache.register_cacheable_server("my-server")
            assert symbol_cache.is_cacheable("mcp_my-server_get_thing")
            assert symbol_cache.is_cacheable("mcp_my-server_find_widget")
            assert symbol_cache.is_cacheable("mcp_my-server_list_all")
            # Mutating tools still excluded (no matching suffix).
            assert not symbol_cache.is_cacheable("mcp_my-server_update_thing")
        finally:
            symbol_cache._cacheable_servers.pop("my-server", None)

    def test_custom_suffixes_override_defaults(self):
        try:
            symbol_cache.register_cacheable_server("strict", suffix_prefixes=("get_",))
            assert symbol_cache.is_cacheable("mcp_strict_get_thing")
            # search_ not in the custom suffix list — excluded.
            assert not symbol_cache.is_cacheable("mcp_strict_search_thing")
        finally:
            symbol_cache._cacheable_servers.pop("strict", None)


class TestRegisterCacheableTool:
    def test_exact_name_registration(self):
        assert not symbol_cache.is_cacheable("my_lookup_tool")
        try:
            symbol_cache.register_cacheable_tool("my_lookup_tool")
            assert symbol_cache.is_cacheable("my_lookup_tool")
        finally:
            symbol_cache._cacheable_tools.discard("my_lookup_tool")

    def test_empty_name_is_noop(self):
        before = set(symbol_cache._cacheable_tools)
        symbol_cache.register_cacheable_tool("")
        assert symbol_cache._cacheable_tools == before


class TestArgsFingerprint:
    def test_order_independent(self):
        assert symbol_cache.args_fingerprint({"a": 1, "b": 2}) == \
            symbol_cache.args_fingerprint({"b": 2, "a": 1})

    def test_different_args_differ(self):
        assert symbol_cache.args_fingerprint({"name": "Foo"}) != \
            symbol_cache.args_fingerprint({"name": "Bar"})


class TestGetOrFetch:
    @pytest.mark.asyncio
    async def test_first_call_invokes_fetcher(self):
        calls = 0

        async def fetch():
            nonlocal calls
            calls += 1
            return "payload"

        result, hit_count = await symbol_cache.get_or_fetch(
            "work-1", "mcp_codebase-index_find_symbol", {"name": "X"}, fetch,
        )
        assert result == "payload"
        assert hit_count == 0, "first call is a fetch, not a hit"
        assert calls == 1

    @pytest.mark.asyncio
    async def test_second_call_returns_cached(self):
        calls = 0

        async def fetch():
            nonlocal calls
            calls += 1
            return "payload"

        await symbol_cache.get_or_fetch(
            "work-1", "mcp_codebase-index_find_symbol", {"name": "X"}, fetch,
        )
        result, hit_count = await symbol_cache.get_or_fetch(
            "work-1", "mcp_codebase-index_find_symbol", {"name": "X"}, fetch,
        )
        assert result == "payload"
        assert hit_count == 1, "second call is the first hit on a populated entry"
        assert calls == 1, "second call should hit the cache"

    @pytest.mark.asyncio
    async def test_concurrent_callers_coalesce(self):
        """9 sibling branches fetching the same key → 1 underlying call."""
        calls = 0
        gate = asyncio.Event()

        async def fetch():
            nonlocal calls
            calls += 1
            # Block until all callers are queued so we exercise the
            # per-key lock, not just a fast serial path.
            await gate.wait()
            return "payload"

        async def caller():
            return await symbol_cache.get_or_fetch(
                "work-1",
                "mcp_codebase-index_find_symbol",
                {"name": "X"},
                fetch,
            )

        tasks = [asyncio.create_task(caller()) for _ in range(9)]
        # Let all tasks queue up on the lock / index lock
        await asyncio.sleep(0.05)
        gate.set()
        results = await asyncio.gather(*tasks)

        assert calls == 1, f"expected single-flight, got {calls} underlying calls"
        payloads = [r[0] for r in results]
        assert payloads == ["payload"] * 9
        # One fetcher (hit_count=0); the other 8 see a populated entry.
        hit_counts = sorted(r[1] for r in results)
        assert hit_counts[0] == 0
        assert all(h >= 1 for h in hit_counts[1:])

    @pytest.mark.asyncio
    async def test_different_work_ids_do_not_collide(self):
        calls = 0

        async def fetch():
            nonlocal calls
            calls += 1
            return f"payload-{calls}"

        r1, h1 = await symbol_cache.get_or_fetch(
            "work-A", "mcp_codebase-index_find_symbol", {"name": "X"}, fetch,
        )
        r2, h2 = await symbol_cache.get_or_fetch(
            "work-B", "mcp_codebase-index_find_symbol", {"name": "X"}, fetch,
        )
        assert r1 != r2
        assert h1 == 0 and h2 == 0, "different work_ids are independent fetches"
        assert calls == 2

    @pytest.mark.asyncio
    async def test_non_string_payload_not_memoised(self):
        calls = 0

        async def fetch():
            nonlocal calls
            calls += 1
            return None

        r1, _ = await symbol_cache.get_or_fetch(
            "work-1", "mcp_codebase-index_find_symbol", {"name": "X"}, fetch,
        )
        r2, _ = await symbol_cache.get_or_fetch(
            "work-1", "mcp_codebase-index_find_symbol", {"name": "X"}, fetch,
        )
        assert r1 is None and r2 is None
        assert calls == 2, "non-string payloads must not be memoised"


class TestClear:
    @pytest.mark.asyncio
    async def test_clear_evicts_only_target_work_id(self):
        async def fetch_a():
            return "A"

        async def fetch_b():
            return "B"

        await symbol_cache.get_or_fetch(
            "work-A", "mcp_codebase-index_find_symbol", {"n": 1}, fetch_a,
        )
        await symbol_cache.get_or_fetch(
            "work-B", "mcp_codebase-index_find_symbol", {"n": 1}, fetch_b,
        )

        removed = symbol_cache.clear("work-A")
        assert removed == 1
        assert symbol_cache._stats()["total"] == 1

    def test_clear_empty_work_id_is_noop(self):
        assert symbol_cache.clear("") == 0
