"""Tests for the empty-parse retry guard in ``spine.agents.helpers``.

Covers the transient json_schema failure where an OpenAI-style (vLLM-served)
model returns ``finish_reason='stop'`` with empty content and no parsed object,
which LangChain surfaces as a bare ``ValueError``. The guard retries that case
once and re-raises everything else so each caller's own fallback still runs.
"""

from __future__ import annotations

import pytest

from spine.agents.helpers import (
    ainvoke_structured_with_retry,
    is_empty_structured_parse,
)

# The exact message LangChain's _oai_structured_outputs_parser raises.
_EMPTY_PARSE_MSG = (
    "Structured Output response does not have a 'parsed' field nor a "
    "'refusal' field. Received message:\n\ncontent=''"
)


class _FakeStructuredModel:
    """Stub model whose ``ainvoke`` replays a scripted sequence of outcomes.

    Each item is either an exception instance (raised) or a value (returned).
    Records the messages it was invoked with so the nudge can be asserted.
    """

    def __init__(self, outcomes: list):
        self._outcomes = list(outcomes)
        self.calls: list[list] = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_is_empty_structured_parse_matches_marker():
    assert is_empty_structured_parse(ValueError(_EMPTY_PARSE_MSG)) is True


def test_is_empty_structured_parse_rejects_other_errors():
    # A different ValueError, and a non-ValueError, must not be treated as the
    # transient empty-parse case.
    assert is_empty_structured_parse(ValueError("some other problem")) is False
    assert is_empty_structured_parse(RuntimeError(_EMPTY_PARSE_MSG)) is False


@pytest.mark.asyncio
async def test_retries_once_and_succeeds():
    sentinel = object()
    model = _FakeStructuredModel([ValueError(_EMPTY_PARSE_MSG), sentinel])

    result = await ainvoke_structured_with_retry(
        model, [("human", "do the thing")], label="t"
    )

    assert result is sentinel
    assert len(model.calls) == 2


@pytest.mark.asyncio
async def test_retry_appends_nudge_message():
    model = _FakeStructuredModel([ValueError(_EMPTY_PARSE_MSG), "ok"])
    base = [("human", "original")]

    await ainvoke_structured_with_retry(model, base, label="t")

    # First attempt uses the messages verbatim; the retry appends one nudge.
    assert model.calls[0] == base
    assert len(model.calls[1]) == len(base) + 1
    nudge = model.calls[1][-1]
    assert "ONLY the JSON" in nudge.content
    # The caller's list must not be mutated.
    assert base == [("human", "original")]


@pytest.mark.asyncio
async def test_reraises_after_retries_exhausted():
    err = ValueError(_EMPTY_PARSE_MSG)
    model = _FakeStructuredModel([err, ValueError(_EMPTY_PARSE_MSG)])

    with pytest.raises(ValueError, match="does not have a 'parsed' field"):
        await ainvoke_structured_with_retry(model, [], retries=1, label="t")

    assert len(model.calls) == 2  # initial + one retry, then re-raise


@pytest.mark.asyncio
async def test_non_empty_parse_error_propagates_without_retry():
    model = _FakeStructuredModel([RuntimeError("boom"), "unreached"])

    with pytest.raises(RuntimeError, match="boom"):
        await ainvoke_structured_with_retry(model, [], label="t")

    assert len(model.calls) == 1  # no retry on unrelated errors


@pytest.mark.asyncio
async def test_first_attempt_success_makes_one_call():
    model = _FakeStructuredModel(["immediate"])

    result = await ainvoke_structured_with_retry(model, [], label="t")

    assert result == "immediate"
    assert len(model.calls) == 1


# ── transient API/transport retry (mid-stream SSE break, 5xx, dropped socket) ──


def _bare_api_error(msg: str = "An error occurred during streaming"):
    """The exact shape llama.cpp/vLLM raise on a mid-stream SSE break."""
    import httpx
    from openai import APIError

    return APIError(msg, request=httpx.Request("POST", "http://local/v1"), body=None)


def _status_error(status: int):
    import httpx
    from openai import APIStatusError

    req = httpx.Request("POST", "http://local/v1")
    return APIStatusError("x", response=httpx.Response(status, request=req), body=None)


def test_retryable_api_error_classification():
    from openai import APIConnectionError, APITimeoutError

    from spine.agents.helpers import _retryable_api_error

    import httpx

    req = httpx.Request("POST", "http://local/v1")
    # Transient transport failures → retry.
    assert _retryable_api_error(_bare_api_error()) is True
    assert _retryable_api_error(APIConnectionError(request=req)) is True
    assert _retryable_api_error(APITimeoutError(request=req)) is True
    assert _retryable_api_error(_status_error(503)) is True
    assert _retryable_api_error(_status_error(429)) is True
    # Deterministic client errors → do not retry.
    assert _retryable_api_error(_status_error(400)) is False
    assert _retryable_api_error(_status_error(404)) is False
    # Unrelated exceptions → not our concern.
    assert _retryable_api_error(ValueError("x")) is False
    assert _retryable_api_error(RuntimeError("x")) is False


@pytest.mark.asyncio
async def test_retries_transient_api_error_then_succeeds():
    sentinel = object()
    model = _FakeStructuredModel([_bare_api_error(), sentinel])

    result = await ainvoke_structured_with_retry(
        model, [("human", "go")], api_backoff=0, label="t"
    )

    assert result is sentinel
    assert len(model.calls) == 2
    # Transport retries resend the request verbatim — no nudge appended.
    assert model.calls[1] == [("human", "go")]


@pytest.mark.asyncio
async def test_transient_api_error_reraises_after_api_retries_exhausted():
    model = _FakeStructuredModel([_bare_api_error(), _bare_api_error(), _bare_api_error()])

    from openai import APIError

    with pytest.raises(APIError):
        await ainvoke_structured_with_retry(
            model, [], api_retries=2, api_backoff=0, label="t"
        )

    assert len(model.calls) == 3  # initial + 2 retries, then re-raise


@pytest.mark.asyncio
async def test_non_retryable_status_error_propagates_immediately():
    model = _FakeStructuredModel([_status_error(400), "unreached"])

    from openai import APIStatusError

    with pytest.raises(APIStatusError):
        await ainvoke_structured_with_retry(model, [], api_backoff=0, label="t")

    assert len(model.calls) == 1  # 4xx is not retried


@pytest.mark.asyncio
async def test_api_retries_zero_disables_transport_retry():
    model = _FakeStructuredModel([_bare_api_error(), "unreached"])

    from openai import APIError

    with pytest.raises(APIError):
        await ainvoke_structured_with_retry(
            model, [], api_retries=0, api_backoff=0, label="t"
        )

    assert len(model.calls) == 1


# ── disable_streaming ───────────────────────────────────────────────────


class TestDisableStreaming:
    def test_turns_streaming_off(self):
        from spine.agents.helpers import disable_streaming

        model = _local_chat_openai(streaming=True)
        assert disable_streaming(model).streaming is False

    def test_returns_a_copy_leaving_original_untouched(self):
        from spine.agents.helpers import disable_streaming

        model = _local_chat_openai(streaming=True)
        copy = disable_streaming(model)
        assert copy is not model
        assert model.streaming is True

    def test_fails_open_on_non_model(self):
        from spine.agents.helpers import disable_streaming

        sentinel = object()
        assert disable_streaming(sentinel) is sentinel


# ── suppress_reasoning ──────────────────────────────────────────────────


def _local_chat_openai(**overrides):
    from langchain_openai import ChatOpenAI

    kwargs = {
        "model": "Qwen3.6-35B-A3B-MTP-GGUF",
        "base_url": "http://localhost:8000/v1",
        "api_key": "vllm",
    }
    kwargs.update(overrides)
    return ChatOpenAI(**kwargs)


class TestSuppressReasoning:
    def test_local_chat_openai_gets_suppression_levers(self):
        from spine.agents.helpers import suppress_reasoning

        model = suppress_reasoning(_local_chat_openai())
        extra = model.extra_body
        assert extra["reasoning_budget"] == 0
        assert extra["chat_template_kwargs"]["enable_thinking"] is False

    def test_existing_extra_body_keys_survive_merge(self):
        from spine.agents.helpers import suppress_reasoning

        model = suppress_reasoning(
            _local_chat_openai(
                extra_body={
                    "top_k": 40,
                    "chat_template_kwargs": {"custom_flag": True},
                }
            )
        )
        extra = model.extra_body
        assert extra["top_k"] == 40
        assert extra["chat_template_kwargs"]["custom_flag"] is True
        assert extra["chat_template_kwargs"]["enable_thinking"] is False
        assert extra["reasoning_budget"] == 0

    def test_idempotent(self):
        from spine.agents.helpers import suppress_reasoning

        once = suppress_reasoning(_local_chat_openai())
        twice = suppress_reasoning(once)
        assert twice.extra_body == once.extra_body

    def test_non_chat_openai_passthrough(self):
        from spine.agents.helpers import suppress_reasoning

        sentinel = object()
        assert suppress_reasoning(sentinel) is sentinel

    def test_real_openai_endpoint_passthrough(self):
        from spine.agents.helpers import suppress_reasoning

        model = _local_chat_openai(base_url="https://api.openai.com/v1")
        assert suppress_reasoning(model) is model

    def test_no_base_url_passthrough(self):
        from langchain_openai import ChatOpenAI

        from spine.agents.helpers import suppress_reasoning

        model = ChatOpenAI(model="gpt-4o-mini", api_key="sk-test")
        assert suppress_reasoning(model) is model
