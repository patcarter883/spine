"""CAM memory-organ client: settings resolution + fail-open transport.

The /cam/* edit plane is best-effort infrastructure — every client method must
degrade to None (CAM not loaded, unreachable host, auth rejection) rather than
raise into the workflow. Settings resolution mirrors the rsa-field conventions
(true = defaults, dict enabled unless enabled: false).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import pytest

from spine.services.cam_client import (
    CAMClient,
    CamSettings,
    resolve_cam_settings,
)


# ── settings resolution ──────────────────────────────────────────────────────
def test_absent_or_disabled_cam_resolves_none():
    assert resolve_cam_settings({"base_url": "http://h:1919/v1"}) is None
    assert resolve_cam_settings({"base_url": "http://h:1919/v1", "cam": False}) is None
    assert (
        resolve_cam_settings(
            {"base_url": "http://h:1919/v1", "cam": {"enabled": False}}
        )
        is None
    )


def test_cam_true_enables_with_defaults():
    s = resolve_cam_settings({"base_url": "http://h:1919/v1", "cam": True})
    assert s is not None
    assert s.server_root == "http://h:1919"  # /v1 stripped
    assert s.write == "distill"
    # facts_block is the principled default (frozen-base scorecard, plan §6.2)
    assert s.read == "facts_block"
    assert s.mode is None  # pre-hybrid: no delivery-mode field sent
    assert s.namespace is None  # auto with no workspace root -> server default


def test_namespace_auto_slugs_workspace_root():
    s = resolve_cam_settings(
        {"base_url": "http://h:1919/v1", "cam": {}},
        workspace_root="/home/pat/Projects/My Repo",
    )
    assert s is not None
    assert s.namespace == "my-repo"


def test_explicit_namespace_and_base_url_override():
    s = resolve_cam_settings(
        {
            "base_url": "http://h:1919/v1",
            "cam": {"namespace": "teamstore", "base_url": "http://cam-host:2020"},
        },
        workspace_root="/x/spine",
    )
    assert s is not None
    assert s.namespace == "teamstore"
    assert s.server_root == "http://cam-host:2020"


def test_api_token_env_indirection_fail_open(monkeypatch):
    monkeypatch.delenv("CAM_TOK", raising=False)
    monkeypatch.delenv("MINISGL_CAM_API_TOKEN", raising=False)
    s = resolve_cam_settings(
        {"base_url": "http://h:1919/v1", "cam": {"api_token": "env:CAM_TOK"}}
    )
    assert s is not None and s.api_token is None  # unset env never raises

    monkeypatch.setenv("CAM_TOK", "sekrit")
    s = resolve_cam_settings(
        {"base_url": "http://h:1919/v1", "cam": {"api_token": "env:CAM_TOK"}}
    )
    assert s is not None and s.api_token == "sekrit"


def test_api_token_falls_back_to_minisgl_env(monkeypatch):
    monkeypatch.setenv("MINISGL_CAM_API_TOKEN", "from-env")
    s = resolve_cam_settings({"base_url": "http://h:1919/v1", "cam": {}})
    assert s is not None and s.api_token == "from-env"


def test_unknown_write_read_modes_fall_back():
    s = resolve_cam_settings(
        {"base_url": "http://h:1919/v1", "cam": {"write": "yolo", "read": "psychic"}}
    )
    assert s is not None
    assert s.write == "distill"
    assert s.read == "facts_block"


def test_delivery_mode_resolution():
    s = resolve_cam_settings(
        {"base_url": "http://h:1919/v1", "cam": {"mode": "pointer"}}
    )
    assert s is not None and s.mode == "pointer"
    # unknown mode -> omit the field entirely (pre-hybrid behavior), not error
    s = resolve_cam_settings(
        {"base_url": "http://h:1919/v1", "cam": {"mode": "telepathy"}}
    )
    assert s is not None and s.mode is None


# ── transport (httpx.MockTransport) ──────────────────────────────────────────
def _client_with(handler) -> CAMClient:
    settings = CamSettings(
        server_root="http://h:1919", api_token="tok", namespace="proj"
    )
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return CAMClient(settings, client=http)


@pytest.mark.asyncio
async def test_remember_sends_auth_and_namespace_headers():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["ns"] = request.headers.get("x-cam-namespace")
        return httpx.Response(200, json={"stored": True, "base_p": 0.12})

    out = await _client_with(handler).remember(
        "spine default branch", "The default branch of spine is", "main"
    )
    assert out == {"stored": True, "base_p": 0.12}
    assert seen["url"] == "http://h:1919/cam/remember"
    assert seen["auth"] == "Bearer tok"
    assert seen["ns"] == "proj"


@pytest.mark.asyncio
async def test_503_cam_not_loaded_is_none():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "CAM not loaded"})

    assert await _client_with(handler).stats() is None


@pytest.mark.asyncio
async def test_connection_error_is_none_after_retry():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ConnectError("refused")

    assert await _client_with(handler).facts() is None
    assert calls["n"] == 2  # one retry on transport errors


@pytest.mark.asyncio
async def test_auth_rejection_is_none_not_raise():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "invalid or missing CAM API token"})

    assert await _client_with(handler).save() is None


@pytest.mark.asyncio
async def test_ask_extracts_text():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": " Tokyo."})

    assert await _client_with(handler).ask("Capital of France is", "France") == " Tokyo."


# ── hybrid delivery mode (pointer pivot, plan §6.3) ──────────────────────────
def _pointer_client(handler) -> CAMClient:
    settings = CamSettings(server_root="http://h:1919", mode="pointer")
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return CAMClient(settings, client=http)


@pytest.mark.asyncio
async def test_remember_omits_mode_field_pre_hybrid():
    # A pre-hybrid server must see the exact payload it always did.
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"stored": True, "base_p": 0.1})

    await _client_with(handler).remember("s", "The s is", "o")
    assert "mode" not in seen["body"]


@pytest.mark.asyncio
async def test_remember_and_ask_send_configured_mode():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen[request.url.path] = json.loads(request.content)
        return httpx.Response(
            200, json={"stored": True, "text": " ok", "mode_served": "pointer"}
        )

    client = _pointer_client(handler)
    await client.remember("s", "The s is", "o")
    assert seen["/cam/remember"]["mode"] == "pointer"
    assert await client.ask("The s is", "s") == " ok"
    assert seen["/cam/ask"]["mode"] == "pointer"


@pytest.mark.asyncio
async def test_per_call_mode_overrides_settings_and_ask_full_surfaces_mode_served():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen[request.url.path] = json.loads(request.content)
        return httpx.Response(200, json={"text": " ok", "mode_served": "tap"})

    client = _pointer_client(handler)
    out = await client.ask_full("The s is", "s", mode="tap")
    assert seen["/cam/ask"]["mode"] == "tap"
    assert out == {"text": " ok", "mode_served": "tap"}


@pytest.mark.asyncio
async def test_lookup_namespaces_and_delete_namespace():
    def handler(request: httpx.Request) -> httpx.Response:
        p = request.url.path
        if p == "/cam/lookup":
            if request.url.params.get("subject"):
                return httpx.Response(
                    200, json={"delivered": True, "subject": "s", "object": "o"}
                )
            return httpx.Response(
                200, json={"matches": [{"subject": "s", "object": "o"}]}
            )
        if p == "/cam/namespaces" and request.method == "GET":
            return httpx.Response(
                200, json=[{"namespace": "p", "facts": 3, "frozen": False}]
            )
        if p.startswith("/cam/namespaces/"):
            assert request.method == "DELETE"
            return httpx.Response(200, json={"dropped": True})
        raise AssertionError(f"unexpected path {p}")

    c = _client_with(handler)
    assert (await c.lookup(subject="s"))["delivered"] is True
    assert (await c.lookup(text="what is s"))["matches"][0]["object"] == "o"
    assert (await c.namespaces())[0]["namespace"] == "p"
    assert await c.delete_namespace("p") is True


@pytest.mark.asyncio
async def test_freeze_uses_query_param():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"frozen": True})

    await _client_with(handler).freeze(True)
    assert seen["url"] == "http://h:1919/cam/freeze?frozen=true"


@pytest.mark.asyncio
async def test_delete_fact_parses_deleted():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "DELETE"
        assert str(request.url).endswith("/cam/facts/France")
        return httpx.Response(200, json={"deleted": True})

    assert await _client_with(handler).delete_fact("France") is True
