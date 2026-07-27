"""Tests for ReadBudgetGuard — nudge toward output when reads pile up.

The survey trap that spine has fought since the implement-spiral incidents
is not a search loop (those results are empty) and not a turn overrun (that
fires long after the budget is gone): it is an agent reading file after file,
each read succeeding, writing nothing. Neither existing guard sees it.

The prompt has been asking the model to police this itself — "Exploration
budget: maximum 2 turns of read/lookup before your first write" — which is
turn-counting across a conversation, the thing small models are worst at.
This moves the counting into the harness, in the shape smallcode's
early_stop governor uses (soft nudge at 5 read-only calls, firm at 8, reset
by any write), which was observed working on the same model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from spine.agents.context_editing import ReadBudgetGuard, _is_read_only_call


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


def _call(name: str, args: dict, idx: int, content: str = "ok"):
    tc_id = f"c{idx}"
    return [
        AIMessage(
            content="",
            tool_calls=[{"id": tc_id, "name": name, "args": args, "type": "tool_call"}],
        ),
        ToolMessage(content=content, tool_call_id=tc_id, name=name),
    ]


def _read(idx: int):
    return _call("read_edit_lint", {"file_path": "a.py", "read_symbol": "F"}, idx)


def _write(idx: int):
    return _call(
        "read_edit_lint",
        {"file_path": "a.py", "old_str": "x", "new_str": "y"},
        idx,
    )


class TestCallClassification:
    def test_search_tools_are_reads(self):
        assert _is_read_only_call("codebase_query", {}) is True
        assert _is_read_only_call("search_codebase", {}) is True
        assert _is_read_only_call("ast_extract_symbol", {}) is True

    def test_anchored_reads_are_reads(self):
        assert _is_read_only_call(
            "read_edit_lint", {"file_path": "a.py", "read_symbol": "F"}
        ) is True
        assert _is_read_only_call(
            "read_edit_lint", {"file_path": "a.py", "read_around": "import os"}
        ) is True

    @pytest.mark.parametrize(
        "args",
        [
            {"file_path": "a.py", "old_str": "x", "new_str": "y"},
            {"file_path": "a.py", "full_replace": "..."},
            {"file_path": "a.py", "read_symbol": "F", "new_str": "z"},
        ],
    )
    def test_edit_args_make_it_a_write(self, args):
        assert _is_read_only_call("read_edit_lint", args) is False

    def test_unrelated_tools_are_neutral(self):
        """`execute` is not producing output either — it must not reset."""
        assert _is_read_only_call("execute", {}) is None


class TestStreak:
    def test_counts_reads(self):
        msgs = [m for i in range(4) for m in _read(i)]
        assert ReadBudgetGuard._reads_since_last_write(msgs) == 4

    def test_a_write_resets(self):
        msgs = [m for i in range(4) for m in _read(i)]
        msgs += _write(99)
        msgs += [m for i in range(2) for m in _read(100 + i)]
        assert ReadBudgetGuard._reads_since_last_write(msgs) == 2

    def test_neutral_calls_do_not_reset(self):
        msgs = [m for i in range(3) for m in _read(i)]
        msgs += _call("execute", {"cmd": "ls"}, 50)
        msgs += _read(60)
        assert ReadBudgetGuard._reads_since_last_write(msgs) == 4


class TestIntervention:
    @pytest.mark.asyncio
    async def test_silent_below_soft_threshold(self):
        guard = ReadBudgetGuard(soft=5, hard=8)
        msgs = [m for i in range(4) for m in _read(i)]
        out = await guard.awrap_model_call(FakeRequest(messages=msgs), _identity_handler)
        assert len(out.messages) == len(msgs)

    @pytest.mark.asyncio
    async def test_soft_nudge(self):
        guard = ReadBudgetGuard(soft=5, hard=8)
        msgs = [m for i in range(5) for m in _read(i)]
        out = await guard.awrap_model_call(FakeRequest(messages=msgs), _identity_handler)
        last = out.messages[-1]
        assert isinstance(last, HumanMessage)
        assert "READ BUDGET GUARD" in last.content
        assert "start" in last.content.lower()

    @pytest.mark.asyncio
    async def test_hard_nudge_is_firmer(self):
        guard = ReadBudgetGuard(soft=5, hard=8)
        msgs = [m for i in range(8) for m in _read(i)]
        out = await guard.awrap_model_call(FakeRequest(messages=msgs), _identity_handler)
        assert "STOP reading" in out.messages[-1].content

    @pytest.mark.asyncio
    async def test_read_edit_rhythm_never_trips(self):
        """A healthy editor alternates; it must never see this guard."""
        guard = ReadBudgetGuard(soft=5, hard=8)
        msgs: list = []
        for i in range(10):
            msgs += _read(i)
            msgs += _write(1000 + i)
        out = await guard.awrap_model_call(FakeRequest(messages=msgs), _identity_handler)
        assert len(out.messages) == len(msgs)

    @pytest.mark.asyncio
    async def test_tools_stay_bound(self):
        guard = ReadBudgetGuard(soft=2, hard=3)
        msgs = [m for i in range(5) for m in _read(i)]
        req = FakeRequest(messages=msgs)
        out = await guard.awrap_model_call(req, _identity_handler)
        assert len(out.tools) == len(req.tools)

    def test_rejects_incoherent_thresholds(self):
        with pytest.raises(ValueError):
            ReadBudgetGuard(soft=0, hard=3)
        with pytest.raises(ValueError):
            ReadBudgetGuard(soft=5, hard=2)


class TestWiring:
    def test_disabled_by_default(self):
        """Off until a live run validates it — merging must be inert."""
        from spine.config import SpineConfig

        assert SpineConfig().implement_read_budget == []
