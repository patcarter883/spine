"""Tests for TokenBudgetCompactor — token-threshold eviction with
hardened preservation rules.

Verifies that:
  * compaction does not fire below the token threshold,
  * compaction fires above the threshold and replaces older tool messages
    with the structured metadata placeholder,
  * the most-recent ``keep_recent`` tool messages are preserved verbatim,
  * tool messages from ``preserved_tools`` are never evicted, even when
    older than the preservation window,
  * the corresponding ``AIMessage.tool_calls`` args for evicted tools
    are trimmed (write_file content → ``[N chars written to ...]``).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from spine.agents.context_editing import (
    DEFAULT_PRESERVED_TOOLS,
    TokenBudgetCompactor,
)


# ── Test doubles ─────────────────────────────────────────────────────────


@dataclass
class FakeRequest:
    messages: list

    def override(self, **kw) -> "FakeRequest":
        new = FakeRequest(messages=list(self.messages))
        for k, v in kw.items():
            setattr(new, k, v)
        return new


async def _identity_handler(req: FakeRequest):
    return req


# ── Message factories ────────────────────────────────────────────────────


def _ai_with_tool_call(tc_id: str, name: str, args: dict) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"id": tc_id, "name": name, "args": args, "type": "tool_call"}],
    )


def _tool_msg(tc_id: str, name: str, content: str) -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=tc_id, name=name)


def _build_history(n_pairs: int, big_content: str) -> list:
    """Build [Human, (AI(tool_call) → Tool(result)) × n_pairs].

    All tool calls are read_file on a synthetic path so they are
    eviction candidates (not in preserved_tools).
    """
    msgs: list = [HumanMessage(content="kick off")]
    for i in range(n_pairs):
        tc_id = f"call-{i}"
        msgs.append(_ai_with_tool_call(tc_id, "read_file", {"file_path": f"/f{i}.py"}))
        msgs.append(_tool_msg(tc_id, "read_file", big_content))
    return msgs


# ── Tests ────────────────────────────────────────────────────────────────


class TestTokenBudgetCompactor:
    @pytest.mark.asyncio
    async def test_below_threshold_passes_through(self):
        compactor = TokenBudgetCompactor(threshold_tokens=10_000, keep_recent=2)
        msgs = _build_history(n_pairs=3, big_content="tiny")
        req = FakeRequest(messages=msgs)
        out = await compactor.awrap_model_call(req, _identity_handler)
        # Below threshold: messages untouched
        assert out.messages == msgs

    @pytest.mark.asyncio
    async def test_evicts_old_tool_messages_above_threshold(self):
        # ~80K characters → well over a 5000-token threshold.
        big = "x" * 80_000
        msgs = _build_history(n_pairs=5, big_content=big)
        compactor = TokenBudgetCompactor(threshold_tokens=5_000, keep_recent=2)
        req = FakeRequest(messages=msgs)
        out = await compactor.awrap_model_call(req, _identity_handler)
        # All tool messages: 5. keep_recent=2 → 3 oldest must be evicted.
        tool_msgs = [m for m in out.messages if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 5
        evicted = [t for t in tool_msgs if t.content.startswith("[read:")]
        preserved = [t for t in tool_msgs if t.content == big]
        assert len(evicted) == 3
        assert len(preserved) == 2

    @pytest.mark.asyncio
    async def test_preserved_tools_never_evicted(self):
        big = "y" * 80_000
        msgs: list = [HumanMessage(content="kick off")]
        # 5 old write_file pairs (preserved) + 5 old read_file pairs (evictable)
        for i in range(5):
            tc = f"w-{i}"
            msgs.append(
                _ai_with_tool_call(tc, "write_file", {"file_path": f"/w{i}.py", "content": big})
            )
            msgs.append(_tool_msg(tc, "write_file", big))
        for i in range(5):
            tc = f"r-{i}"
            msgs.append(_ai_with_tool_call(tc, "read_file", {"file_path": f"/r{i}.py"}))
            msgs.append(_tool_msg(tc, "read_file", big))

        compactor = TokenBudgetCompactor(
            threshold_tokens=5_000,
            keep_recent=2,
            preserved_tools=DEFAULT_PRESERVED_TOOLS,
        )
        req = FakeRequest(messages=msgs)
        out = await compactor.awrap_model_call(req, _identity_handler)

        write_outputs = [
            m for m in out.messages if isinstance(m, ToolMessage) and m.name == "write_file"
        ]
        # All 5 write_file ToolMessages must survive verbatim.
        assert len(write_outputs) == 5
        for m in write_outputs:
            assert m.content == big

    @pytest.mark.asyncio
    async def test_ai_args_trimmed_for_evicted_writes(self):
        # write_file is in preserved_tools by default — its ToolMessage is
        # kept, BUT this test exercises the AI-arg trim path by removing
        # write_file from the preserved set so the eviction path runs and
        # we can observe the corresponding tool_call args being compacted.
        big = "z" * 80_000
        write_content = "a" * 5_000
        msgs: list = [HumanMessage(content="kick off")]
        # First a write_file pair (will be evicted now), then enough read
        # pairs to push past threshold and to leave write_file outside the
        # keep_recent window.
        msgs.append(
            _ai_with_tool_call(
                "w-0", "write_file", {"file_path": "/out.py", "content": write_content}
            )
        )
        msgs.append(_tool_msg("w-0", "write_file", "ok"))
        for i in range(5):
            tc = f"r-{i}"
            msgs.append(_ai_with_tool_call(tc, "read_file", {"file_path": f"/r{i}.py"}))
            msgs.append(_tool_msg(tc, "read_file", big))

        compactor = TokenBudgetCompactor(
            threshold_tokens=5_000,
            keep_recent=2,
            preserved_tools=frozenset(),  # nothing preserved → write_file evictable
        )
        req = FakeRequest(messages=msgs)
        out = await compactor.awrap_model_call(req, _identity_handler)

        # The AIMessage with the write_file call must have its content arg
        # replaced by a short summary marker.
        write_ai = next(
            m for m in out.messages
            if isinstance(m, AIMessage)
            and m.tool_calls
            and m.tool_calls[0].get("name") == "write_file"
        )
        trimmed_content = write_ai.tool_calls[0]["args"]["content"]
        assert trimmed_content.startswith("[") and "chars written to /out.py" in trimmed_content
        assert len(trimmed_content) < len(write_content)

    @pytest.mark.asyncio
    async def test_zero_threshold_disables(self):
        msgs = _build_history(n_pairs=10, big_content="x" * 80_000)
        compactor = TokenBudgetCompactor(threshold_tokens=0, keep_recent=2)
        req = FakeRequest(messages=msgs)
        out = await compactor.awrap_model_call(req, _identity_handler)
        # Disabled: pass through untouched.
        assert out.messages == msgs
