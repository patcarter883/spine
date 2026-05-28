"""Tests for ResearcherConvergenceMiddleware — soft nudge / hard forcing.

Verifies that:
  * below soft threshold: middleware is a no-op (no message added),
  * at/above soft threshold: a CONVERGENCE NUDGE SystemMessage is appended
    and tools remain bound,
  * at/above hard threshold: a CONVERGENCE FORCING SystemMessage is
    appended AND tools are dropped (``request.tools == []``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from spine.agents.context_editing import ResearcherConvergenceMiddleware


@dataclass
class FakeRequest:
    messages: list
    tools: list = field(default_factory=lambda: [object(), object()])  # 2 fake tools

    def override(self, **kw) -> "FakeRequest":
        new = FakeRequest(messages=list(self.messages), tools=list(self.tools))
        for k, v in kw.items():
            setattr(new, k, v)
        return new


async def _identity_handler(req: FakeRequest):
    return req


def _make_history(n_tool_calls: int) -> list:
    """Build a history with `n_tool_calls` AIMessage tool_calls + matching ToolMessages."""
    msgs: list = [HumanMessage(content="research topic X")]
    for i in range(n_tool_calls):
        tc_id = f"call-{i}"
        msgs.append(
            AIMessage(
                content="",
                tool_calls=[
                    {"id": tc_id, "name": "mcp_x_find", "args": {"q": str(i)}, "type": "tool_call"}
                ],
            )
        )
        msgs.append(ToolMessage(content="result", tool_call_id=tc_id, name="mcp_x_find"))
    return msgs


class TestConvergenceMiddleware:
    @pytest.mark.asyncio
    async def test_below_soft_is_passthrough(self):
        mw = ResearcherConvergenceMiddleware(soft_threshold=25, hard_threshold=40)
        msgs = _make_history(n_tool_calls=10)
        req = FakeRequest(messages=msgs)
        out = await mw.awrap_model_call(req, _identity_handler)
        # No SystemMessage appended; tools intact.
        assert not any(isinstance(m, SystemMessage) for m in out.messages)
        assert len(out.tools) == 2

    @pytest.mark.asyncio
    async def test_at_soft_threshold_appends_nudge(self):
        mw = ResearcherConvergenceMiddleware(soft_threshold=25, hard_threshold=40)
        msgs = _make_history(n_tool_calls=25)
        req = FakeRequest(messages=msgs)
        out = await mw.awrap_model_call(req, _identity_handler)
        sys_msgs = [m for m in out.messages if isinstance(m, SystemMessage)]
        assert len(sys_msgs) == 1
        assert "CONVERGENCE NUDGE" in sys_msgs[0].content
        # Tools still bound.
        assert len(out.tools) == 2

    @pytest.mark.asyncio
    async def test_at_hard_threshold_drops_tools_and_forces(self):
        mw = ResearcherConvergenceMiddleware(
            soft_threshold=25, hard_threshold=40, recursion_limit=50
        )
        msgs = _make_history(n_tool_calls=40)
        req = FakeRequest(messages=msgs)
        out = await mw.awrap_model_call(req, _identity_handler)
        sys_msgs = [m for m in out.messages if isinstance(m, SystemMessage)]
        assert len(sys_msgs) == 1
        assert "CONVERGENCE FORCING" in sys_msgs[0].content
        assert "40" in sys_msgs[0].content
        assert "50" in sys_msgs[0].content
        # Tools dropped so model has no choice but to emit final message.
        assert out.tools == []

    @pytest.mark.asyncio
    async def test_invalid_thresholds_raise(self):
        with pytest.raises(ValueError):
            ResearcherConvergenceMiddleware(soft_threshold=40, hard_threshold=25)
