"""Finite-window overflow guards (trace 019ece87).

Trace 019ece87 spun the fallback decomposer 75 times: every slice touching a
1200-line file sent a ~27K prompt + a statically-reserved 12K completion cap
against a 32K window and the provider 400'd with "Context size has been
exceeded". The decomposer then re-ran the same whole-file work, which overflowed
identically, until the depth cap.

This covers the four guards added in response:
  * window_aware_completion_cap — per-turn clamp keeping prompt+gen in-window
  * window_aware_compaction_threshold — eviction trigger clamped below window
  * DynamicCompletionCapMiddleware — applies the clamp each model call
  * _is_context_overflow — routes overflow failures to narrower decomposition
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

from langchain_core.messages import HumanMessage, SystemMessage

from spine.agents.context_editing import DynamicCompletionCapMiddleware
from spine.agents.synthesis_budget import (
    window_aware_compaction_threshold,
    window_aware_completion_cap,
)
from spine.workflow.subgraphs.implement_subgraph import _is_context_overflow


# ── window_aware_completion_cap ──────────────────────────────────────────


def test_completion_cap_legacy_window_is_noop():
    # No declared window (cloud/legacy) → never clamp.
    assert window_aware_completion_cap(
        window=0, prompt_tokens=99999, base_cap=12000, overhead=2000
    ) == 12000


def test_completion_cap_keeps_base_when_room():
    assert window_aware_completion_cap(
        window=32000, prompt_tokens=5000, base_cap=12000, overhead=2000
    ) == 12000


def test_completion_cap_lowers_when_tight():
    # 32000 - 27000 - 2000 = 3000 of room left.
    assert window_aware_completion_cap(
        window=32000, prompt_tokens=27000, base_cap=12000, overhead=2000
    ) == 3000


def test_completion_cap_floors_when_overfull():
    # Prompt already eats the window → never request <= 0; floor instead.
    assert window_aware_completion_cap(
        window=32000, prompt_tokens=31900, base_cap=12000, overhead=2000, floor=512
    ) == 512


# ── window_aware_compaction_threshold ────────────────────────────────────


def test_threshold_legacy_window_unchanged():
    assert window_aware_compaction_threshold(
        window=0, configured_threshold=30000, reserve=12000
    ) == 30000


def test_threshold_large_window_keeps_configured():
    assert window_aware_compaction_threshold(
        window=60000, configured_threshold=30000, reserve=12000
    ) == 30000


def test_threshold_small_window_clamps_below_ceiling():
    # The exact trace condition: a 30K trigger is useless at a 32K window.
    assert window_aware_compaction_threshold(
        window=32000, configured_threshold=30000, reserve=12000
    ) == 20000


def test_threshold_disabled_stays_disabled():
    assert window_aware_compaction_threshold(
        window=32000, configured_threshold=0, reserve=12000
    ) == 0


# ── DynamicCompletionCapMiddleware ───────────────────────────────────────


class _FakeModel:
    def __init__(self, max_tokens=12000):
        self.max_tokens = max_tokens
        self.max_completion_tokens = None

    def model_copy(self, update):
        m = _FakeModel(self.max_tokens)
        for k, v in update.items():
            setattr(m, k, v)
        return m


@dataclass
class _FakeReq:
    model: object
    messages: list
    system_message: object = None

    def override(self, **kw):
        return replace(self, **kw)


async def _echo(req):
    return req


def test_dynamic_cap_keeps_base_on_small_prompt():
    mw = DynamicCompletionCapMiddleware(window=32000, overhead=2000)
    req = _FakeReq(
        model=_FakeModel(12000),
        messages=[HumanMessage("hello")],
        system_message=SystemMessage("sys"),
    )
    out = asyncio.run(mw.awrap_model_call(req, _echo))
    assert out.model.max_tokens == 12000


def test_dynamic_cap_lowers_on_large_prompt():
    mw = DynamicCompletionCapMiddleware(window=32000, overhead=2000, floor=512)
    big = "token " * 28000  # well over the window once tokenized
    req = _FakeReq(
        model=_FakeModel(12000),
        messages=[HumanMessage(big)],
        system_message=SystemMessage("sys"),
    )
    out = asyncio.run(mw.awrap_model_call(req, _echo))
    assert 512 <= out.model.max_tokens < 12000


def test_dynamic_cap_legacy_window_is_noop():
    mw = DynamicCompletionCapMiddleware(window=0)
    big = "token " * 28000
    req = _FakeReq(model=_FakeModel(12000), messages=[HumanMessage(big)])
    out = asyncio.run(mw.awrap_model_call(req, _echo))
    assert out.model.max_tokens == 12000


# ── overflow routing ─────────────────────────────────────────────────────


def test_is_context_overflow_detects_provider_error():
    assert _is_context_overflow("APIError('Context size has been exceeded.')")
    assert _is_context_overflow("maximum context length is 32768 tokens")
    assert _is_context_overflow("This model's CONTEXT WINDOW is full")


def test_is_context_overflow_ignores_logic_failures():
    assert not _is_context_overflow("ValueError: bad slice json")
    assert not _is_context_overflow("AssertionError: test failed")
    assert not _is_context_overflow("")
