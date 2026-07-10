"""resolve_chat_model must forward stream_usage on the init_chat_model path.

Streamed responses only carry token counts when stream_usage=True triggers
``stream_options: {"include_usage": true}``. The string/init_chat_model path
used to drop this, producing 0-token spans in LangSmith and starving the
per-work_id budget tracker (trace 019ec965). Parity with the local/OpenRouter
builders, which already set it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import langchain.chat_models as lc_chat_models
import langchain_openai
import spine.agents.helpers as helpers


def _capture_chat_openai(monkeypatch):
    """Patch ChatOpenAI to capture constructor kwargs without a real client."""
    captured: dict = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    return captured


def test_local_completion_cap_clamped_to_half_window(monkeypatch):
    """A completion reservation larger than half the context_window is clamped
    so the prompt always fits — an over-large reservation OOM-crashes a finite
    local backend (trace 019ed360)."""
    captured = _capture_chat_openai(monkeypatch)
    helpers._build_local_model(
        "openai:Qwen-Local",
        {
            "base_url": "http://localhost:8010/v1",
            "max_completion_tokens": 30000,
            "context_window": 40000,
        },
    )
    # 30000 > 40000 // 2 -> clamped to 20000.
    assert captured.get("max_completion_tokens") == 20000


def test_local_completion_cap_within_window_untouched(monkeypatch):
    """A sane per-phase cap (e.g. implement=8K) under half the window is left
    alone."""
    captured = _capture_chat_openai(monkeypatch)
    helpers._build_local_model(
        "openai:Qwen-Local",
        {
            "base_url": "http://localhost:8010/v1",
            "max_completion_tokens": 8000,
            "context_window": 40000,
        },
    )
    assert captured.get("max_completion_tokens") == 8000


def test_local_sampler_knobs_routed_correctly(monkeypatch):
    """top_p is a native ChatOpenAI kwarg; top_k/repetition_penalty have no
    field and must ride extra_body or the llama.cpp server never sees them."""
    captured = _capture_chat_openai(monkeypatch)
    helpers._build_local_model(
        "openai:Qwen-Local",
        {
            "base_url": "http://localhost:8010/v1",
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.05,
        },
    )
    assert captured.get("temperature") == 0.7
    assert captured.get("top_p") == 0.8
    extra = captured.get("extra_body") or {}
    assert extra.get("top_k") == 20
    assert extra.get("repetition_penalty") == 1.05


def test_local_sampler_knobs_merge_with_reasoning_suppression(monkeypatch):
    """Sampler extra_body and the reasoning:false suppression coexist."""
    captured = _capture_chat_openai(monkeypatch)
    helpers._build_local_model(
        "openai:Qwen-Local",
        {
            "base_url": "http://localhost:8010/v1",
            "repetition_penalty": 1.05,
            "reasoning": False,
        },
    )
    extra = captured.get("extra_body") or {}
    assert extra.get("repetition_penalty") == 1.05
    assert extra.get("reasoning_budget") == 0
    assert extra.get("chat_template_kwargs") == {"enable_thinking": False}


def test_local_no_sampler_config_no_extra_body(monkeypatch):
    """With no sampler/reasoning config, extra_body is not set at all."""
    captured = _capture_chat_openai(monkeypatch)
    helpers._build_local_model(
        "openai:Qwen-Local", {"base_url": "http://localhost:8010/v1"}
    )
    assert "extra_body" not in captured


def test_local_rsa_patch_dict_rides_extra_body_and_disables_streaming(monkeypatch):
    """An active RSA patch dict passes through verbatim on extra_body and forces
    the non-streaming endpoint (RSA is multi-round, emits nothing until final)."""
    captured = _capture_chat_openai(monkeypatch)
    helpers._build_local_model(
        "openai:Zaya-Local",
        {
            "base_url": "http://localhost:8010/v1",
            "rsa": {"n": 32, "k": 4, "t": 3, "selection": "majority"},
        },
    )
    extra = captured.get("extra_body") or {}
    assert extra.get("rsa") == {"n": 32, "k": 4, "t": 3, "selection": "majority"}
    assert captured.get("streaming") is False
    # Non-streaming: no stream_usage / stream_options on the request.
    assert "stream_usage" not in captured
    # Active RSA gets the long default request timeout (multi-round run).
    assert captured.get("request_timeout") == 1800


def test_local_rsa_true_disables_streaming(monkeypatch):
    """rsa: true (server defaults) is still an active run -> non-streaming."""
    captured = _capture_chat_openai(monkeypatch)
    helpers._build_local_model(
        "openai:Zaya-Local",
        {"base_url": "http://localhost:8010/v1", "rsa": True},
    )
    assert (captured.get("extra_body") or {}).get("rsa") is True
    assert captured.get("streaming") is False


def test_local_rsa_false_passthrough_keeps_streaming(monkeypatch):
    """rsa: false is a plain backend passthrough on the shim, so streaming and
    stream_usage stay on and the normal single-call timeout applies."""
    captured = _capture_chat_openai(monkeypatch)
    helpers._build_local_model(
        "openai:Zaya-Local",
        {"base_url": "http://localhost:8010/v1", "rsa": False},
    )
    # The field is still forwarded so the shim explicitly bypasses RSA.
    assert (captured.get("extra_body") or {}).get("rsa") is False
    assert captured.get("streaming") is True
    assert captured.get("stream_usage") is True
    assert captured.get("request_timeout") == 300


def test_local_rsa_dict_enabled_false_keeps_streaming(monkeypatch):
    """A patch dict with enabled:false opts the call out -> treat as inactive."""
    captured = _capture_chat_openai(monkeypatch)
    helpers._build_local_model(
        "openai:Zaya-Local",
        {"base_url": "http://localhost:8010/v1", "rsa": {"enabled": False, "n": 8}},
    )
    assert (captured.get("extra_body") or {}).get("rsa") == {"enabled": False, "n": 8}
    assert captured.get("streaming") is True


def test_local_streaming_false_opts_out(monkeypatch):
    """`streaming: false` in provider config forces the non-streaming endpoint.

    Needed for backends whose tool-call parser only runs on the complete
    response (mini-sglang): streamed replies leak raw tool-call markup into
    content and never emit tool_calls (probe 13, run eb0aacc8).
    """
    captured = _capture_chat_openai(monkeypatch)
    helpers._build_local_model(
        "openai:Qwen-Remote",
        {"base_url": "http://10.50.1.51:1919/v1", "streaming": False},
    )
    assert captured.get("streaming") is False
    # Non-streaming: usage arrives on the response body, stream_options inert.
    assert "stream_usage" not in captured


def test_local_streaming_unset_defaults_on(monkeypatch):
    """Without the key, streaming stays on (stall-detector default)."""
    captured = _capture_chat_openai(monkeypatch)
    helpers._build_local_model(
        "openai:Qwen-Local", {"base_url": "http://localhost:8010/v1"}
    )
    assert captured.get("streaming") is True
    assert captured.get("stream_usage") is True


class TestFlattenTextBlockContent:
    """Local models flatten all-text content-block lists to plain strings.

    Middleware can rewrite message content into the OpenAI array-of-parts
    form; minimal local backends (mini-sglang) 422 on it (probe 17, run
    a3f963b9 — the plan critic died with "Input should be a valid string").
    """

    def test_all_text_blocks_joined(self):
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a reviewer."},
                        {
                            "type": "text",
                            "text": "Static prefix.",
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
                {"role": "user", "content": "plain string untouched"},
            ]
        }
        out = helpers._flatten_text_block_content(payload)
        assert out["messages"][0]["content"] == "You are a reviewer.\nStatic prefix."
        assert out["messages"][1]["content"] == "plain string untouched"

    def test_non_text_parts_left_alone(self):
        blocks = [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ]
        payload = {"messages": [{"role": "user", "content": list(blocks)}]}
        out = helpers._flatten_text_block_content(payload)
        assert out["messages"][0]["content"] == blocks

    def test_built_local_model_flattens_at_the_wire(self):
        """The built model's request payload flattens block content end-to-end."""
        from langchain_core.messages import SystemMessage

        model = helpers._build_local_model(
            "openai:Qwen-Remote",
            {"base_url": "http://10.50.1.51:1919/v1", "api_key": "vllm"},
        )
        payload = model._get_request_payload(
            [SystemMessage(content=[{"type": "text", "text": "hello"}])]
        )
        assert payload["messages"][0]["content"] == "hello"


def _capture_init_chat_model(monkeypatch):
    captured: dict = {}

    def fake_init(model, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return object()  # stand-in BaseChatModel

    monkeypatch.setattr(lc_chat_models, "init_chat_model", fake_init)
    return captured


def test_stream_usage_forwarded_for_local_provider(monkeypatch):
    captured = _capture_init_chat_model(monkeypatch)
    monkeypatch.setattr(helpers, "resolve_model", lambda *a, **k: "openai:Qwen-Local")
    monkeypatch.setattr(
        helpers, "_active_provider_config", lambda phase=None, escalation_level=0: {"model": "openai:Qwen-Local"}
    )

    helpers.resolve_chat_model(None, phase="specify")

    assert captured["kwargs"].get("stream_usage") is True
    assert captured["kwargs"].get("streaming") is True


def test_stream_usage_opt_out_respected(monkeypatch):
    captured = _capture_init_chat_model(monkeypatch)
    monkeypatch.setattr(helpers, "resolve_model", lambda *a, **k: "openai:Qwen-Local")
    monkeypatch.setattr(
        helpers,
        "_active_provider_config",
        lambda phase=None, escalation_level=0: {"model": "openai:Qwen-Local", "stream_usage": False},
    )

    helpers.resolve_chat_model(None, phase="specify")

    # Explicit opt-out: stream_usage not forced on; streaming still enabled.
    assert "stream_usage" not in captured["kwargs"]
    assert captured["kwargs"].get("streaming") is True

def test_base_url_and_api_key_forwarded_for_local_provider(monkeypatch):
    """The string/init_chat_model path must pass the provider's base_url + api_key,
    else the bare openai: client falls back to OPENAI_* env vars and fails with
    "Missing credentials" on a local provider (trace 019efca3: fallback enrich).
    """
    captured = _capture_init_chat_model(monkeypatch)
    monkeypatch.setattr(helpers, "resolve_model", lambda *a, **k: "openai:Mellum-Local")
    monkeypatch.setattr(
        helpers,
        "_active_provider_config",
        lambda phase=None, escalation_level=0: {
            "model": "openai:Mellum-Local",
            "base_url": "http://localhost:8010/v1",
            "api_key": "vllm",
        },
    )

    helpers.resolve_chat_model(None, phase="implement/decomposer")

    assert captured["kwargs"].get("base_url") == "http://localhost:8010/v1"
    assert captured["kwargs"].get("api_key") == "vllm"


class TestReasoningLock:
    """`reasoning: true` locks reasoning on: suppress_reasoning is a no-op.

    Batch 1 (run b15cee51): slice workers suppressed reasoning on a serve
    that separates channels correctly — the model thought INSIDE the JSON
    string fields instead, rambling ~30K chars past the completion cap and
    degrading to stub-fallback detail three times in one run."""

    def test_reasoning_true_stamps_lock_tag(self, monkeypatch):
        captured = _capture_chat_openai(monkeypatch)
        helpers._build_local_model(
            "openai:Qwen-Remote",
            {"base_url": "http://10.50.1.51:1919/v1", "reasoning": True},
        )
        assert helpers._REASONING_LOCK_TAG in captured.get("tags", [])
        # No suppression levers on the wire.
        assert "reasoning_budget" not in (captured.get("extra_body") or {})

    def test_suppress_reasoning_noops_on_locked_model(self):
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model="m", api_key="x", base_url="http://local:1919/v1",
            tags=[helpers._REASONING_LOCK_TAG],
        )
        out = helpers.suppress_reasoning(model)
        assert (getattr(out, "extra_body", None) or {}).get("reasoning_budget") is None

    def test_suppress_reasoning_still_works_unlocked(self):
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="m", api_key="x", base_url="http://local:1919/v1")
        out = helpers.suppress_reasoning(model)
        extra = getattr(out, "extra_body", None) or {}
        assert extra.get("reasoning_budget") == 0
        assert extra.get("chat_template_kwargs", {}).get("enable_thinking") is False

    def test_reasoning_false_still_injects_levers(self, monkeypatch):
        captured = _capture_chat_openai(monkeypatch)
        helpers._build_local_model(
            "openai:Qwen-Local",
            {"base_url": "http://localhost:8010/v1", "reasoning": False},
        )
        extra = captured.get("extra_body") or {}
        assert extra.get("reasoning_budget") == 0
        assert helpers._REASONING_LOCK_TAG not in (captured.get("tags") or [])

    def test_cap_gets_reasoning_allowance_on_locked_model(self):
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model="m", api_key="x", base_url="http://local:1919/v1",
            max_tokens=100, tags=[helpers._REASONING_LOCK_TAG],
        )
        out = helpers.cap_completion_tokens(model, 4096)
        assert out.max_tokens == 4096 + helpers._REASONING_CAP_ALLOWANCE

    def test_cap_unchanged_on_unlocked_model(self):
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model="m", api_key="x", base_url="http://local:1919/v1",
            max_tokens=100,
        )
        out = helpers.cap_completion_tokens(model, 4096)
        assert out.max_tokens == 4096
