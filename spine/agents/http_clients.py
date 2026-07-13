"""Shared httpx clients with per-provider connection caps.

A provider's ``max_concurrent_calls`` config field is enforced by handing
every chat-model instance for that provider the **same** ``httpx`` client
configured with ``httpx.Limits(max_connections=N)``.  Because streaming
is enabled in :mod:`spine.agents.helpers`, one in-flight LLM call holds
one connection for the duration of the stream — so the connection cap
acts as a concurrent-call cap.

``max_concurrent_streams`` adds a second, *weighted* budget for RSA
backends: one RSA request fans out into ``rsa.n`` concurrent rollout
streams server-side, so a connection count is the wrong unit — 4
in-flight n=8 requests already occupy 32 of the server's decode streams.
When the field is set, the provider's async client routes through
:class:`_WeightedStreamTransport`, which reads ``rsa.n`` out of each
request body and holds that many permits from a FIFO
:class:`_AsyncWeightedSemaphore` until the response body is closed.
Non-RSA requests weigh 1. The budget is enforced *before* the connection
pool, so queued requests don't pin connections.

Only the async client is weighted — every dispatcher LLM call is async.
Sync calls (bench scripts) still respect the plain connection cap.

The clients are cached by provider name so every agent build for the
same provider reuses the same pool, making the cap global across the
process rather than per-agent.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections import deque
from collections.abc import AsyncIterator
from typing import Callable

import httpx


_async_clients: dict[str, httpx.AsyncClient] = {}
_sync_clients: dict[str, httpx.Client] = {}
_lock = threading.Lock()

# Weight assumed for ``"rsa": true`` (server-default RSA): the shim's
# default population is N=16 (docs/RSA_KNOBS.md). Spine's configs always
# send explicit dicts, so this is a conservative backstop.
_DEFAULT_RSA_N = 16


def _limits(max_concurrent: int) -> httpx.Limits:
    return httpx.Limits(
        max_connections=max_concurrent,
        max_keepalive_connections=max_concurrent,
    )


class _AsyncWeightedSemaphore:
    """FIFO semaphore where each acquire takes a caller-chosen weight.

    Strict FIFO on purpose: with wake-any semantics a stream of light
    (n=2) requests would starve heavy (n=8) ones indefinitely. The head
    waiter blocks everything behind it until its full weight fits.

    Single-event-loop only (matching the cached ``httpx.AsyncClient``,
    which is likewise bound to the loop that first uses it).
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._capacity = capacity
        self._free = capacity
        self._waiters: deque[tuple[int, asyncio.Future]] = deque()

    def _clamp(self, weight: int) -> int:
        # A weight above capacity must still be admittable — clamp rather
        # than deadlock. It simply gets the whole budget to itself.
        return max(1, min(int(weight), self._capacity))

    async def acquire(self, weight: int) -> int:
        """Take ``weight`` permits (clamped to capacity); returns the held count."""
        held = self._clamp(weight)
        if not self._waiters and self._free >= held:
            self._free -= held
            return held
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._waiters.append((held, fut))
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                # Granted between the grant and the cancellation landing —
                # hand the permits straight back.
                self.release(held)
            else:
                try:
                    self._waiters.remove((held, fut))
                except ValueError:
                    pass
            raise
        return held

    def release(self, weight: int) -> None:
        self._free = min(self._capacity, self._free + weight)
        while self._waiters:
            head_weight, fut = self._waiters[0]
            if fut.cancelled():
                self._waiters.popleft()
                continue
            if self._free < head_weight:
                break
            self._waiters.popleft()
            self._free -= head_weight
            fut.set_result(None)

    @property
    def in_use(self) -> int:
        return self._capacity - self._free


def _request_stream_weight(request: httpx.Request) -> int:
    """How many server decode streams this request will occupy (= ``rsa.n``)."""
    try:
        body = json.loads(request.content)
    except Exception:
        # Non-JSON or streaming-upload body — not a chat completion we
        # know how to weigh.
        return 1
    if not isinstance(body, dict):
        return 1
    rsa = body.get("rsa")
    if isinstance(rsa, dict):
        if rsa.get("enabled") is False:
            return 1
        try:
            return max(1, int(rsa.get("n", _DEFAULT_RSA_N)))
        except (TypeError, ValueError):
            return _DEFAULT_RSA_N
    if rsa is True:
        return _DEFAULT_RSA_N
    return 1


class _ReleasingStream(httpx.AsyncByteStream):
    """Wraps a response stream; runs ``on_close`` exactly once at close."""

    def __init__(self, inner: httpx.AsyncByteStream, on_close: Callable[[], None]) -> None:
        self._inner = inner
        self._on_close = on_close
        self._released = False

    async def __aiter__(self) -> AsyncIterator[bytes]:  # type: ignore[override]
        async for chunk in self._inner:
            yield chunk

    async def aclose(self) -> None:
        try:
            await self._inner.aclose()
        finally:
            if not self._released:
                self._released = True
                self._on_close()


class _WeightedStreamTransport(httpx.AsyncBaseTransport):
    """Holds ``rsa.n`` permits per request for the response's lifetime.

    Permits are released when the response body is closed — httpx closes
    it after a full read (non-streaming calls) or when the caller
    finishes/abandons a stream, including on errors.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport, budget: _AsyncWeightedSemaphore) -> None:
        self._inner = inner
        self._budget = budget

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        held = await self._budget.acquire(_request_stream_weight(request))
        try:
            response = await self._inner.handle_async_request(request)
        except BaseException:
            self._budget.release(held)
            raise
        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            stream=_ReleasingStream(response.stream, lambda: self._budget.release(held)),
            extensions=response.extensions,
            request=request,
        )

    async def aclose(self) -> None:
        await self._inner.aclose()


def get_async_http_client(
    provider_name: str,
    max_concurrent: int,
    max_streams: int | None = None,
) -> httpx.AsyncClient:
    """Return a cached async client capped at ``max_concurrent`` connections.

    With ``max_streams``, requests additionally pass through a weighted
    stream budget (see module docstring). The first build for a provider
    wins — later calls return the cached client unchanged.
    """
    with _lock:
        client = _async_clients.get(provider_name)
        if client is None:
            if max_streams and int(max_streams) > 0:
                client = httpx.AsyncClient(
                    transport=_WeightedStreamTransport(
                        httpx.AsyncHTTPTransport(limits=_limits(max_concurrent)),
                        _AsyncWeightedSemaphore(int(max_streams)),
                    )
                )
            else:
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
