"""Regression coverage for ReadCacheMiddleware dedupe of multimodal lookups
and the TurnBudgetGuard model-turn ceiling.

Trace 019ed413: a single slice-implementer issued ``codebase_query
find_symbol UIApi`` 4× (and other symbols 2×) with ZERO cache hits, grinding
69 model turns ≈ 954K input tokens on a config-UI edit. Root cause: the
``codebase_query`` / ``mcp_codebase-index_*`` tools return *multimodal*
content (a list of ``{"type": "text", ...}`` blocks), but the symbol_cache
``_fetch`` closure only memoised bare ``str`` payloads — so nothing was ever
cached and every lookup re-executed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.agents import symbol_cache
from spine.agents.context_editing import ReadCacheMiddleware, TurnBudgetGuard


@pytest.fixture(autouse=True)
def _reset_cache():
    symbol_cache._cache.clear()
    yield
    symbol_cache._cache.clear()


@dataclass
class FakeRequest:
    tool_call: dict
    runtime: object
    messages: list = field(default_factory=list)
    tools: list = field(default_factory=lambda: [object(), object()])

    def override(self, **kw) -> "FakeRequest":
        new = FakeRequest(
            tool_call=dict(self.tool_call),
            runtime=self.runtime,
            messages=list(self.messages),
            tools=list(self.tools),
        )
        for k, v in kw.items():
            setattr(new, k, v)
        return new


def _ctx(work_id: str = "w1"):
    return SimpleNamespace(read_cache={}, read_cache_turn=0, work_id=work_id)


def _multimodal(text: str) -> list[dict]:
    """A codebase_query-style multimodal ToolMessage content payload."""
    return [{"type": "text", "text": text}]


def _request(ctx, *, name="codebase_query", args=None, call_id="c1"):
    return FakeRequest(
        tool_call={
            "id": call_id,
            "name": name,
            "args": args or {"action": "find_symbol", "name": "UIApi"},
            "type": "tool_call",
        },
        runtime=SimpleNamespace(context=ctx),
    )


class TestMultimodalDedupe:
    @pytest.mark.asyncio
    async def test_multimodal_codebase_query_is_memoised(self):
        """Second identical codebase_query is served from the cache.

        Before the fix the multimodal result was never memoised, so the
        handler ran on every call (trace 019ed413: UIApi fetched 4×).
        """
        mw = ReadCacheMiddleware()
        ctx = _ctx()
        calls = {"n": 0}

        async def handler(req):
            calls["n"] += 1
            return ToolMessage(
                content=_multimodal('{"file": "spine/ui_api/api.py", "line": 40}'),
                tool_call_id=req.tool_call["id"],
                name=req.tool_call["name"],
            )

        # First lookup — handler runs once, result memoised.
        out1 = await mw.awrap_tool_call(_request(ctx, call_id="c1"), handler)
        assert calls["n"] == 1
        # First call returns the original (multimodal) result verbatim.
        assert isinstance(out1, ToolMessage)

        # Second identical lookup — handler must NOT run again.
        out2 = await mw.awrap_tool_call(_request(ctx, call_id="c2"), handler)
        assert calls["n"] == 1, "handler re-ran — multimodal result was not cached"
        assert "ALREADY FETCHED" in str(out2.content)
        assert "spine/ui_api/api.py" in str(out2.content)

    @pytest.mark.asyncio
    async def test_raw_mcp_lookup_is_memoised(self):
        """Raw mcp_codebase-index_* multimodal results memoise too."""
        mw = ReadCacheMiddleware()
        ctx = _ctx()
        calls = {"n": 0}

        async def handler(req):
            calls["n"] += 1
            return ToolMessage(
                content=_multimodal("symbol body here"),
                tool_call_id=req.tool_call["id"],
                name=req.tool_call["name"],
            )

        args = {"name": "UIApi"}
        await mw.awrap_tool_call(
            _request(ctx, name="mcp_codebase-index_find_symbol", args=args, call_id="c1"),
            handler,
        )
        await mw.awrap_tool_call(
            _request(ctx, name="mcp_codebase-index_find_symbol", args=args, call_id="c2"),
            handler,
        )
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_distinct_args_not_deduped(self):
        """Different symbols are independent cache entries."""
        mw = ReadCacheMiddleware()
        ctx = _ctx()
        calls = {"n": 0}

        async def handler(req):
            calls["n"] += 1
            return ToolMessage(
                content=_multimodal(f"body for {req.tool_call['args']['name']}"),
                tool_call_id=req.tool_call["id"],
                name=req.tool_call["name"],
            )

        await mw.awrap_tool_call(
            _request(ctx, args={"action": "find_symbol", "name": "UIApi"}, call_id="c1"),
            handler,
        )
        await mw.awrap_tool_call(
            _request(ctx, args={"action": "find_symbol", "name": "SpineConfig"}, call_id="c2"),
            handler,
        )
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_error_results_not_cached(self):
        """An error ToolMessage must not be memoised — a corrected retry runs."""
        mw = ReadCacheMiddleware()
        ctx = _ctx()
        calls = {"n": 0}

        async def handler(req):
            calls["n"] += 1
            return ToolMessage(
                content=_multimodal("boom"),
                tool_call_id=req.tool_call["id"],
                name=req.tool_call["name"],
                status="error",
            )

        await mw.awrap_tool_call(_request(ctx, call_id="c1"), handler)
        await mw.awrap_tool_call(_request(ctx, call_id="c2"), handler)
        assert calls["n"] == 2, "error result was cached and masked the retry"


class TestTurnBudgetGuard:
    @dataclass
    class _Req:
        messages: list
        tools: list = field(default_factory=lambda: [object()])

        def override(self, **kw):
            new = TestTurnBudgetGuard._Req(messages=list(self.messages), tools=list(self.tools))
            for k, v in kw.items():
                setattr(new, k, v)
            return new

    @staticmethod
    async def _identity(req):
        return req

    @pytest.mark.asyncio
    async def test_below_threshold_passthrough(self):
        mw = TurnBudgetGuard(threshold=5)
        msgs = [HumanMessage(content="go"), AIMessage(content="a"), AIMessage(content="b")]
        out = await mw.awrap_model_call(self._Req(messages=msgs), self._identity)
        assert len(out.messages) == len(msgs)
        assert not any(isinstance(m, SystemMessage) for m in out.messages)

    @pytest.mark.asyncio
    async def test_threshold_triggers_nudge(self):
        mw = TurnBudgetGuard(threshold=3)
        msgs = [HumanMessage(content="go"), *[AIMessage(content=str(i)) for i in range(3)]]
        out = await mw.awrap_model_call(self._Req(messages=msgs), self._identity)
        appended = out.messages[-1]
        assert isinstance(appended, SystemMessage)
        assert "TURN BUDGET GUARD" in appended.content

    @pytest.mark.asyncio
    async def test_escalates_past_threshold(self):
        mw = TurnBudgetGuard(threshold=3)
        msgs = [HumanMessage(content="go"), *[AIMessage(content=str(i)) for i in range(5)]]
        out = await mw.awrap_model_call(self._Req(messages=msgs), self._identity)
        assert "reminder #" in out.messages[-1].content

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            TurnBudgetGuard(threshold=0)
