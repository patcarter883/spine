"""Tests for SearchLoopGuard — break zero-result search spins.

Regression coverage for trace 019ed3b8, where an implement subagent issued
~7 near-duplicate reranker/recall searches that each returned ``[]`` before
accepting the section did not exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from spine.agents.context_editing import (
    SearchLoopGuard,
    _is_empty_search_result,
    _is_search_tool,
)


@dataclass
class FakeRequest:
    messages: list
    tools: list = field(default_factory=lambda: [object(), object()])

    def override(self, **kw) -> "FakeRequest":
        new = FakeRequest(messages=list(self.messages), tools=list(self.tools))
        for k, v in kw.items():
            setattr(new, k, v)
        return new


async def _identity_handler(req: FakeRequest):
    return req


def _search(content, name="codebase_query", idx=0):
    """A matched AIMessage(tool_call) + ToolMessage(result) pair."""
    tc_id = f"q{idx}"
    return [
        AIMessage(
            content="",
            tool_calls=[{"id": tc_id, "name": name, "args": {}, "type": "tool_call"}],
        ),
        ToolMessage(content=content, tool_call_id=tc_id, name=name),
    ]


class TestHelpers:
    def test_is_search_tool(self):
        assert _is_search_tool("codebase_query")
        assert _is_search_tool("grep")
        assert _is_search_tool("mcp_codebase-index_find_symbol")
        assert not _is_search_tool("read_edit_lint")
        assert not _is_search_tool("write_file")

    @pytest.mark.parametrize(
        "content",
        ["[]", "{}", "null", "no results", "", "   ",
         "[codebase_query search 'x' → no results]",
         "[grep: 'x' in . — 0 files, 0 lines]"],
    )
    def test_empty_results_detected(self, content):
        assert _is_empty_search_result(content)

    @pytest.mark.parametrize(
        "content",
        ['[{"file": "a.py", "line": 1}]',
         "[codebase_query find_symbol 'X' → 1 hit(s): a.py:1]",
         "[grep: 'x' in . — 2 files, 5 lines]"],
    )
    def test_non_empty_results_not_flagged(self, content):
        assert not _is_empty_search_result(content)


class TestSearchLoopGuard:
    @pytest.mark.asyncio
    async def test_below_threshold_passthrough(self):
        mw = SearchLoopGuard(threshold=3)
        msgs = [HumanMessage(content="go"), *_search("[]", idx=0), *_search("[]", idx=1)]
        out = await mw.awrap_model_call(FakeRequest(messages=msgs), _identity_handler)
        assert not any(isinstance(m, SystemMessage) for m in out.messages)
        assert len(out.tools) == 2  # tools never dropped

    @pytest.mark.asyncio
    async def test_streak_triggers_nudge(self):
        mw = SearchLoopGuard(threshold=3)
        msgs = [HumanMessage(content="go")]
        for i in range(3):
            msgs += _search("[]", idx=i)
        out = await mw.awrap_model_call(FakeRequest(messages=msgs), _identity_handler)
        sys = [m for m in out.messages if isinstance(m, HumanMessage)
               and 'SEARCH LOOP' in str(m.content)]
        assert len(sys) == 1
        assert "SEARCH LOOP GUARD" in sys[0].content
        assert len(out.tools) == 2  # tools stay bound

    @pytest.mark.asyncio
    async def test_hit_resets_streak(self):
        mw = SearchLoopGuard(threshold=3)
        msgs = [HumanMessage(content="go")]
        msgs += _search("[]", idx=0)
        msgs += _search("[]", idx=1)
        msgs += _search('[{"file": "a.py", "line": 3}]', idx=2)  # a hit → reset
        msgs += _search("[]", idx=3)
        out = await mw.awrap_model_call(FakeRequest(messages=msgs), _identity_handler)
        assert not any(isinstance(m, SystemMessage) for m in out.messages)

    @pytest.mark.asyncio
    async def test_multimodal_empty_result_counts(self):
        """codebase_query empties arrive as list-of-blocks, not str."""
        mw = SearchLoopGuard(threshold=2)
        msgs = [HumanMessage(content="go")]
        for i in range(2):
            msgs += _search([{"type": "text", "text": "[]"}], idx=i)
        out = await mw.awrap_model_call(FakeRequest(messages=msgs), _identity_handler)
        assert any(
            isinstance(m, HumanMessage) and "SEARCH LOOP GUARD" in m.content
            for m in out.messages
        )

    @pytest.mark.asyncio
    async def test_non_search_tools_ignored(self):
        mw = SearchLoopGuard(threshold=2)
        msgs = [HumanMessage(content="go")]
        # read_edit_lint returning short content must not count as a search.
        for i in range(4):
            msgs += _search("ok", name="read_edit_lint", idx=i)
        out = await mw.awrap_model_call(FakeRequest(messages=msgs), _identity_handler)
        assert not any(isinstance(m, SystemMessage) for m in out.messages)

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            SearchLoopGuard(threshold=0)
