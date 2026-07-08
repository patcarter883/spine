"""CAM provider-config passthrough on the local ChatOpenAI builder.

With a `cam:` block on the provider, every model built for that provider must
(1) scope requests to the project namespace via the X-CAM-Namespace default
header and (2) pin the server's ambient auto-write gate with an explicit
top-level `cam_write` body field — True only under `write: ambient`, because
agent prompts are not conversational turns and an explicit False must override
a server booted with MINISGL_CAM_AUTO_WRITE=1. Without a `cam:` block the
request is byte-unchanged (no header, no field).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import langchain_openai

import spine.agents.helpers as helpers
import spine.config as spine_config
from spine.config import SpineConfig


def _capture_chat_openai(monkeypatch):
    captured: dict = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    return captured


def test_no_cam_config_leaves_request_untouched(monkeypatch):
    captured = _capture_chat_openai(monkeypatch)
    helpers._build_local_model(
        "openai:Qwen-Local", {"base_url": "http://localhost:1919/v1"}
    )
    assert "default_headers" not in captured
    assert "cam_write" not in (captured.get("extra_body") or {})


def test_cam_explicit_namespace_sets_header_and_write_false(monkeypatch):
    captured = _capture_chat_openai(monkeypatch)
    helpers._build_local_model(
        "openai:Qwen-Local",
        {
            "base_url": "http://localhost:1919/v1",
            "cam": {"namespace": "myproj"},
        },
    )
    assert captured.get("default_headers") == {"X-CAM-Namespace": "myproj"}
    # Default write mode is `distill`: ambient auto-write explicitly pinned off.
    assert (captured.get("extra_body") or {}).get("cam_write") is False
    # CAM must not disturb the normal streaming path.
    assert captured.get("streaming") is True


def test_cam_ambient_write_mode_sends_true(monkeypatch):
    captured = _capture_chat_openai(monkeypatch)
    helpers._build_local_model(
        "openai:Qwen-Local",
        {
            "base_url": "http://localhost:1919/v1",
            "cam": {"namespace": "myproj", "write": "ambient"},
        },
    )
    assert (captured.get("extra_body") or {}).get("cam_write") is True


def test_cam_auto_namespace_slugs_workspace_root(monkeypatch):
    captured = _capture_chat_openai(monkeypatch)
    monkeypatch.setattr(
        SpineConfig,
        "load",
        staticmethod(
            lambda: SimpleNamespace(
                workspace_root="/home/x/My Repo", max_completion_tokens=0
            )
        ),
    )
    helpers._build_local_model(
        "openai:Qwen-Local",
        {"base_url": "http://localhost:1919/v1", "cam": True},
    )
    assert captured.get("default_headers") == {"X-CAM-Namespace": "my-repo"}


def test_cam_enabled_false_dict_is_off(monkeypatch):
    captured = _capture_chat_openai(monkeypatch)
    helpers._build_local_model(
        "openai:Qwen-Local",
        {
            "base_url": "http://localhost:1919/v1",
            "cam": {"enabled": False, "namespace": "myproj"},
        },
    )
    assert "default_headers" not in captured
    assert "cam_write" not in (captured.get("extra_body") or {})


def test_cam_coexists_with_sampler_extra_body(monkeypatch):
    captured = _capture_chat_openai(monkeypatch)
    helpers._build_local_model(
        "openai:Qwen-Local",
        {
            "base_url": "http://localhost:1919/v1",
            "top_k": 20,
            "cam": {"namespace": "myproj"},
        },
    )
    extra = captured.get("extra_body") or {}
    assert extra.get("top_k") == 20
    assert extra.get("cam_write") is False


def test_cam_is_a_phase_overridable_provider_key():
    """Per-phase routing entries may carry a `cam` override (e.g. a different
    namespace, or read: off for a mechanical phase)."""
    assert "cam" in spine_config.SpineConfig._PROVIDER_KEYS
