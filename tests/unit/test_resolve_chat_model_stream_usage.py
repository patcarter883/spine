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
import spine.agents.helpers as helpers


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
        helpers, "_active_provider_config", lambda phase=None: {"model": "openai:Qwen-Local"}
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
        lambda phase=None: {"model": "openai:Qwen-Local", "stream_usage": False},
    )

    helpers.resolve_chat_model(None, phase="specify")

    # Explicit opt-out: stream_usage not forced on; streaming still enabled.
    assert "stream_usage" not in captured["kwargs"]
    assert captured["kwargs"].get("streaming") is True
