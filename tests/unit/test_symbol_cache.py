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

    def test_other_tools_are_not_cacheable(self):
        assert not symbol_cache.is_cacheable("mcp_codebase-index_search_codebase")
        assert not symbol_cache.is_cacheable("read_file")
        assert not symbol_cache.is_cacheable("write_file")
        assert not symbol_cache.is_cacheable("")


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

        result = await symbol_cache.get_or_fetch(
            "work-1", "mcp_codebase-index_find_symbol", {"name": "X"}, fetch,
        )
        assert result == "payload"
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
        result = await symbol_cache.get_or_fetch(
            "work-1", "mcp_codebase-index_find_symbol", {"name": "X"}, fetch,
        )
        assert result == "payload"
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
        assert results == ["payload"] * 9

    @pytest.mark.asyncio
    async def test_different_work_ids_do_not_collide(self):
        calls = 0

        async def fetch():
            nonlocal calls
            calls += 1
            return f"payload-{calls}"

        r1 = await symbol_cache.get_or_fetch(
            "work-A", "mcp_codebase-index_find_symbol", {"name": "X"}, fetch,
        )
        r2 = await symbol_cache.get_or_fetch(
            "work-B", "mcp_codebase-index_find_symbol", {"name": "X"}, fetch,
        )
        assert r1 != r2
        assert calls == 2

    @pytest.mark.asyncio
    async def test_non_string_payload_not_memoised(self):
        calls = 0

        async def fetch():
            nonlocal calls
            calls += 1
            return None

        r1 = await symbol_cache.get_or_fetch(
            "work-1", "mcp_codebase-index_find_symbol", {"name": "X"}, fetch,
        )
        r2 = await symbol_cache.get_or_fetch(
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
