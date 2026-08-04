"""Weighted stream concurrency: rsa.n permits per request against a server budget."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from spine.agents.http_clients import (
    _AsyncWeightedSemaphore,
    _DEFAULT_RSA_N,
    _request_stream_weight,
    _WeightedStreamTransport,
)


def _chat_request(rsa=None, **body_extra) -> httpx.Request:
    body = {"model": "zaya", "messages": [{"role": "user", "content": "hi"}], **body_extra}
    if rsa is not None:
        body["rsa"] = rsa
    return httpx.Request(
        "POST",
        "http://server/v1/chat/completions",
        content=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )


class TestRequestStreamWeight:
    def test_rsa_dict_uses_n(self):
        assert _request_stream_weight(_chat_request(rsa={"n": 8, "k": 2, "t": 1})) == 8

    def test_no_rsa_weighs_one(self):
        assert _request_stream_weight(_chat_request()) == 1

    def test_rsa_false_weighs_one(self):
        assert _request_stream_weight(_chat_request(rsa=False)) == 1

    def test_rsa_disabled_dict_weighs_one(self):
        assert _request_stream_weight(_chat_request(rsa={"enabled": False, "n": 8})) == 1

    def test_rsa_true_uses_server_default(self):
        assert _request_stream_weight(_chat_request(rsa=True)) == _DEFAULT_RSA_N

    def test_rsa_dict_without_n_uses_server_default(self):
        assert _request_stream_weight(_chat_request(rsa={"t": 2})) == _DEFAULT_RSA_N

    def test_non_json_body_weighs_one(self):
        req = httpx.Request("POST", "http://server/x", content=b"\x00not json")
        assert _request_stream_weight(req) == 1


class TestWeightedSemaphore:
    @pytest.mark.asyncio
    async def test_acquire_release_roundtrip(self):
        sem = _AsyncWeightedSemaphore(10)
        held = await sem.acquire(4)
        assert held == 4
        assert sem.in_use == 4
        sem.release(held)
        assert sem.in_use == 0

    @pytest.mark.asyncio
    async def test_overweight_request_clamps_to_capacity(self):
        sem = _AsyncWeightedSemaphore(4)
        held = await sem.acquire(16)  # rsa n=16 through a budget of 4 must not deadlock
        assert held == 4
        sem.release(held)

    @pytest.mark.asyncio
    async def test_fifo_prevents_heavy_starvation(self):
        sem = _AsyncWeightedSemaphore(8)
        first = await sem.acquire(6)
        order: list[str] = []

        async def take(label, weight):
            held = await sem.acquire(weight)
            order.append(label)
            return held

        heavy = asyncio.create_task(take("heavy-8", 8))
        light = asyncio.create_task(take("light-2", 2))
        await asyncio.sleep(0)  # both queued; 2 free — light would fit, heavy is head
        assert not heavy.done() and not light.done()

        sem.release(first)  # 8 free: heavy (head) admitted, light waits behind it
        held_heavy = await heavy
        assert order == ["heavy-8"]
        assert not light.done()

        sem.release(held_heavy)
        held_light = await light
        assert order == ["heavy-8", "light-2"]
        sem.release(held_light)
        assert sem.in_use == 0

    @pytest.mark.asyncio
    async def test_cancelled_waiter_is_skipped(self):
        sem = _AsyncWeightedSemaphore(4)
        first = await sem.acquire(4)
        waiter = asyncio.create_task(sem.acquire(2))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        sem.release(first)
        assert sem.in_use == 0
        # Budget intact after the cancellation.
        held = await sem.acquire(4)
        sem.release(held)


class TestWeightedTransport:
    @pytest.mark.asyncio
    async def test_concurrent_streams_never_exceed_budget(self):
        budget = 8
        in_flight = {"weighted": 0, "peak": 0}
        release_gate = asyncio.Event()

        class SlowInner(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                w = _request_stream_weight(request)
                in_flight["weighted"] += w
                in_flight["peak"] = max(in_flight["peak"], in_flight["weighted"])
                await release_gate.wait()
                in_flight["weighted"] -= w
                return httpx.Response(200, json={"ok": True}, request=request)

        transport = _WeightedStreamTransport(SlowInner(), _AsyncWeightedSemaphore(budget))
        client = httpx.AsyncClient(transport=transport)
        try:
            tasks = [
                asyncio.create_task(client.post("http://s/v1/chat/completions",
                                                json={"rsa": {"n": n}, "messages": []}))
                for n in (4, 4, 4, 2, 2, 8, 2)  # 26 weighted total through a budget of 8
            ]
            await asyncio.sleep(0.05)
            release_gate.set()
            responses = await asyncio.gather(*tasks)
            assert all(r.status_code == 200 for r in responses)
            assert in_flight["peak"] <= budget
            assert in_flight["weighted"] == 0
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_permits_released_on_transport_error(self):
        sem = _AsyncWeightedSemaphore(4)

        class FailingInner(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                raise httpx.ConnectError("server crashed", request=request)

        transport = _WeightedStreamTransport(FailingInner(), sem)
        client = httpx.AsyncClient(transport=transport)
        try:
            for _ in range(3):  # repeated failures must not leak permits
                with pytest.raises(httpx.ConnectError):
                    await client.post("http://s/v1/chat/completions", json={"rsa": {"n": 4}})
            assert sem.in_use == 0
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_permits_released_after_body_read(self):
        sem = _AsyncWeightedSemaphore(8)

        class OkInner(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                return httpx.Response(200, json={"ok": True}, request=request)

        client = httpx.AsyncClient(transport=_WeightedStreamTransport(OkInner(), sem))
        try:
            r = await client.post("http://s/v1/chat/completions", json={"rsa": {"n": 8}})
            assert r.json() == {"ok": True}
            assert sem.in_use == 0
        finally:
            await client.aclose()
