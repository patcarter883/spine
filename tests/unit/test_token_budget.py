"""Token-budget circuit breaker in spine.agents.retry."""
from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage

from spine.agents.retry import (
    MaxTokenBudgetExceeded,
    _check_and_update_token_budget,
    _cumulative_tokens,
    _get_token_budget,
    _tokens_from_result,
    ainvoke_with_retry,
    reset_token_budget,
)


@pytest.fixture(autouse=True)
def _clear_cumulative():
    _cumulative_tokens.clear()
    yield
    _cumulative_tokens.clear()


def _msg(input_tokens: int, output_tokens: int, msg_id: str | None = None) -> AIMessage:
    msg = AIMessage(content="ok", id=msg_id)
    msg.usage_metadata = {  # type: ignore[attr-defined]
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return msg


def test_tokens_from_result_sums_input_and_output():
    result = {"messages": [_msg(100, 50, msg_id="a")]}
    assert _tokens_from_result(result) == 150


def test_tokens_from_result_deduplicates_by_message_id():
    # Same id appearing twice (e.g. streamed-then-finalised) must not
    # be double-counted.
    msg = _msg(100, 50, msg_id="dup")
    result = {"messages": [msg, msg]}
    assert _tokens_from_result(result) == 150


def test_tokens_from_result_handles_missing_usage_metadata():
    bare = AIMessage(content="no usage")
    assert _tokens_from_result({"messages": [bare]}) == 0


def test_tokens_from_result_zero_when_no_messages():
    assert _tokens_from_result({}) == 0
    assert _tokens_from_result({"messages": []}) == 0
    assert _tokens_from_result(None) == 0


def test_get_token_budget_respects_env_override(monkeypatch):
    monkeypatch.setenv("SPINE_TOKEN_BUDGET", "5000")
    assert _get_token_budget("quick") == 5000
    assert _get_token_budget("critical_reviewed_task") == 5000


def test_get_token_budget_falls_back_to_work_type_defaults(monkeypatch):
    monkeypatch.delenv("SPINE_TOKEN_BUDGET", raising=False)
    assert _get_token_budget("quick") == 200_000
    assert _get_token_budget("critical_reviewed_task") == 1_000_000
    assert _get_token_budget("totally_unknown") == 1_000_000


def test_check_and_update_accumulates_and_returns_total():
    result1 = {"messages": [_msg(100, 50, msg_id="a")]}
    result2 = {"messages": [_msg(200, 25, msg_id="b")]}
    assert _check_and_update_token_budget(
        work_id="w1", work_type="quick", result=result1
    ) == 150
    assert _check_and_update_token_budget(
        work_id="w1", work_type="quick", result=result2
    ) == 375


def test_check_and_update_raises_when_budget_exceeded(monkeypatch):
    monkeypatch.setenv("SPINE_TOKEN_BUDGET", "100")
    result = {"messages": [_msg(80, 50, msg_id="a")]}
    with pytest.raises(MaxTokenBudgetExceeded) as exc:
        _check_and_update_token_budget(
            work_id="w1", work_type="quick", result=result
        )
    assert exc.value.work_id == "w1"
    assert exc.value.cumulative == 130
    assert exc.value.budget == 100


def test_check_and_update_noops_when_work_id_blank():
    result = {"messages": [_msg(1_000_000, 1_000_000, msg_id="a")]}
    assert _check_and_update_token_budget(
        work_id="", work_type="quick", result=result
    ) == 0


def test_reset_clears_counter():
    _cumulative_tokens["w1"] = 999
    reset_token_budget("w1")
    assert "w1" not in _cumulative_tokens
    # Idempotent — no error for missing key.
    reset_token_budget("never_seen")


class _FakeAgent:
    """Minimal agent stub for ainvoke_with_retry coverage."""

    def __init__(self, result: dict):
        self._result = result
        self.calls = 0

    async def ainvoke(self, input_, **kwargs):
        self.calls += 1
        return self._result


def test_ainvoke_with_retry_raises_max_token_budget(monkeypatch):
    monkeypatch.setenv("SPINE_TOKEN_BUDGET", "50")
    agent = _FakeAgent({"messages": [_msg(40, 20, msg_id="a")]})
    with pytest.raises(MaxTokenBudgetExceeded):
        asyncio.run(
            ainvoke_with_retry(
                agent,
                {"messages": []},
                work_id="wid",
                work_type="quick",
            )
        )
    # The wrapper must not retry on a budget breach.
    assert agent.calls == 1


def test_ainvoke_with_retry_passes_through_when_under_budget(monkeypatch):
    monkeypatch.setenv("SPINE_TOKEN_BUDGET", "10000")
    agent = _FakeAgent({"messages": [_msg(40, 20, msg_id="a")], "ok": True})
    out = asyncio.run(
        ainvoke_with_retry(
            agent,
            {"messages": []},
            work_id="wid",
            work_type="quick",
        )
    )
    assert out.get("ok") is True
    assert _cumulative_tokens["wid"] == 60
