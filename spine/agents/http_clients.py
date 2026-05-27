"""Shared httpx clients with per-provider connection caps.

A provider's ``max_concurrent_calls`` config field is enforced by handing
every chat-model instance for that provider the **same** ``httpx`` client
configured with ``httpx.Limits(max_connections=N)``.  Because streaming
is enabled in :mod:`spine.agents.helpers`, one in-flight LLM call holds
one connection for the duration of the stream — so the connection cap
acts as a concurrent-call cap.

The clients are cached by provider name so every agent build for the
same provider reuses the same pool, making the cap global across the
process rather than per-agent.
"""

from __future__ import annotations

import threading

import httpx


_async_clients: dict[str, httpx.AsyncClient] = {}
_sync_clients: dict[str, httpx.Client] = {}
_lock = threading.Lock()


def _limits(max_concurrent: int) -> httpx.Limits:
    return httpx.Limits(
        max_connections=max_concurrent,
        max_keepalive_connections=max_concurrent,
    )


def get_async_http_client(provider_name: str, max_concurrent: int) -> httpx.AsyncClient:
    """Return a cached async client capped at ``max_concurrent`` connections."""
    with _lock:
        client = _async_clients.get(provider_name)
        if client is None:
            client = httpx.AsyncClient(limits=_limits(max_concurrent))
            _async_clients[provider_name] = client
        return client


def get_sync_http_client(provider_name: str, max_concurrent: int) -> httpx.Client:
    """Return a cached sync client capped at ``max_concurrent`` connections."""
    with _lock:
        client = _sync_clients.get(provider_name)
        if client is None:
            client = httpx.Client(limits=_limits(max_concurrent))
            _sync_clients[provider_name] = client
        return client
