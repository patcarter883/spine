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
