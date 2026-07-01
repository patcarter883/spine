"""Unit tests for ephemeral GPU pod lifecycle (spine.infra.ephemeral_pod).

All tests run WITHOUT the optional ``runpod`` SDK and without touching a real
cloud: the boot step is monkeypatched, so these exercise config parsing, the
reference-counted singleton, the failure policies, the decorator, and the
``env:`` provider-value expansion.
"""

from __future__ import annotations

import os

import pytest

import spine.infra.ephemeral_pod as ep
from spine.agents.helpers import _expand_env_ref
from spine.infra.ephemeral_pod import (
    POD_BASE_URL_ENV,
    EphemeralPodConfig,
    PodStartupError,
    _create_kwargs,
    _docker_args,
    acquire,
    release,
    with_ephemeral_pod,
)


class FakeConfig:
    """Minimal stand-in for SpineConfig (acquire reads only these two attrs)."""

    def __init__(self, ephemeral_pod: dict | None = None, providers: dict | None = None):
        self.ephemeral_pod = ephemeral_pod or {}
        self.providers = providers or {}


class FakeRunpod:
    def __init__(self) -> None:
        self.terminated: list[str] = []

    def terminate_pod(self, pod_id: str) -> None:
        self.terminated.append(pod_id)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the module-level pod singleton + env var around every test."""
    ep._POD = None
    ep._REFCOUNT = 0
    os.environ.pop(POD_BASE_URL_ENV, None)
    yield
    ep._POD = None
    ep._REFCOUNT = 0
    os.environ.pop(POD_BASE_URL_ENV, None)


def _patch_boot(monkeypatch, runpod: FakeRunpod, pod_id: str = "pod123", events=None):
    async def fake_boot(pod_cfg, work_id):  # noqa: ANN001
        url = f"https://{pod_id}-8000.proxy.runpod.net/v1"
        os.environ[POD_BASE_URL_ENV] = url
        if events is not None:
            events.append("boot")
        return ep._RunPodLease(runpod, pod_id, url)

    monkeypatch.setattr(ep, "_boot_runpod", fake_boot)


# ── Config parsing ──────────────────────────────────────────────────────────


def test_disabled_by_default():
    assert EphemeralPodConfig.from_config(FakeConfig()).enabled is False


def test_model_derived_from_bound_provider():
    cfg = FakeConfig(
        ephemeral_pod={"enabled": True, "binds_provider": "pod"},
        providers={"llm": [{"name": "pod", "model": "openai:Qwen/Qwen3.6-35B-A3B-FP8"}]},
    )
    pod = EphemeralPodConfig.from_config(cfg)
    assert pod.enabled
    assert pod.model == "Qwen/Qwen3.6-35B-A3B-FP8"  # openai: prefix stripped


def test_explicit_model_wins_over_provider():
    cfg = FakeConfig(
        ephemeral_pod={"enabled": True, "binds_provider": "pod", "model": "explicit/M"},
        providers={"llm": [{"name": "pod", "model": "openai:other"}]},
    )
    assert EphemeralPodConfig.from_config(cfg).model == "explicit/M"


def test_unknown_keys_ignored():
    pod = EphemeralPodConfig.from_config(
        FakeConfig(ephemeral_pod={"enabled": True, "bogus_key": 1, "gpu_count": 4})
    )
    assert pod.gpu_count == 4


def test_docker_args_tp_matches_gpu_count():
    pod = EphemeralPodConfig(model="M", gpu_count=2, port=8000, vllm_args="--max-model-len 60000")
    da = _docker_args(pod)
    assert "--model M" in da
    assert "--tensor-parallel-size 2" in da
    assert "--host 0.0.0.0" in da
    assert "--max-model-len 60000" in da


def test_create_kwargs_shape():
    pod = EphemeralPodConfig(model="M", gpu_count=2, gpu_type_id="NVIDIA A40", network_volume_id="v1")
    ck = _create_kwargs(pod, "spine-pod-w1")
    assert ck["ports"] == "8000/http"
    assert ck["gpu_count"] == 2
    assert ck["gpu_type_id"] == "NVIDIA A40"
    assert ck["network_volume_id"] == "v1"
    assert ck["volume_mount_path"] == "/root/.cache/huggingface"


# ── Reference-counted singleton ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_acquire_noop_when_disabled():
    lease = await acquire(FakeConfig(), "w1")
    assert lease.active is False
    await release(lease)  # no-op
    assert POD_BASE_URL_ENV not in os.environ


@pytest.mark.asyncio
async def test_refcount_single_pod_and_teardown(monkeypatch):
    rp = FakeRunpod()
    _patch_boot(monkeypatch, rp)
    cfg = FakeConfig(ephemeral_pod={"enabled": True, "model": "M"})

    l1 = await acquire(cfg, "w1")
    assert os.environ[POD_BASE_URL_ENV].endswith("/v1")
    assert ep._REFCOUNT == 1

    l2 = await acquire(cfg, "w1")  # nested entry point → same pod
    assert ep._REFCOUNT == 2
    assert ep._POD is not None and ep._POD.pod_id == "pod123"

    await release(l2)
    assert ep._REFCOUNT == 1
    assert rp.terminated == []  # outer lease still holds it

    await release(l1)
    assert ep._REFCOUNT == 0
    assert rp.terminated == ["pod123"]  # terminated exactly once
    assert POD_BASE_URL_ENV not in os.environ


@pytest.mark.asyncio
async def test_boot_failure_aborts(monkeypatch):
    async def boom(pod_cfg, work_id):  # noqa: ANN001
        raise PodStartupError("no capacity")

    monkeypatch.setattr(ep, "_boot_runpod", boom)
    cfg = FakeConfig(ephemeral_pod={"enabled": True, "model": "M", "on_failure": "abort"})

    with pytest.raises(PodStartupError):
        await acquire(cfg, "w1")
    assert ep._REFCOUNT == 0
    assert POD_BASE_URL_ENV not in os.environ


@pytest.mark.asyncio
async def test_boot_failure_local_fallback(monkeypatch):
    async def boom(pod_cfg, work_id):  # noqa: ANN001
        raise PodStartupError("no capacity")

    monkeypatch.setattr(ep, "_boot_runpod", boom)
    cfg = FakeConfig(
        ephemeral_pod={
            "enabled": True,
            "model": "M",
            "on_failure": "local_fallback",
            "fallback_base_url": "http://localhost:8010/v1",
        }
    )
    lease = await acquire(cfg, "w1")
    assert lease.active
    assert os.environ[POD_BASE_URL_ENV] == "http://localhost:8010/v1"
    await release(lease)
    assert POD_BASE_URL_ENV not in os.environ


# ── Decorator ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_decorator_brings_up_and_tears_down(monkeypatch):
    rp = FakeRunpod()
    events: list[str] = []
    _patch_boot(monkeypatch, rp, pod_id="podX", events=events)
    cfg = FakeConfig(ephemeral_pod={"enabled": True, "model": "M"})

    @with_ephemeral_pod()
    async def run(work_id, config=None):  # noqa: ANN001
        assert os.environ[POD_BASE_URL_ENV]  # pod live during the body
        events.append("body")
        return "ok"

    assert await run("w1", config=cfg) == "ok"
    assert events == ["boot", "body"]
    assert rp.terminated == ["podX"]
    assert POD_BASE_URL_ENV not in os.environ


@pytest.mark.asyncio
async def test_decorator_tears_down_on_exception(monkeypatch):
    rp = FakeRunpod()
    _patch_boot(monkeypatch, rp, pod_id="podE")
    cfg = FakeConfig(ephemeral_pod={"enabled": True, "model": "M"})

    @with_ephemeral_pod()
    async def run(work_id, config=None):  # noqa: ANN001
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await run("w1", config=cfg)
    assert rp.terminated == ["podE"]  # finally fired
    assert POD_BASE_URL_ENV not in os.environ


@pytest.mark.asyncio
async def test_decorator_skip_does_not_boot(monkeypatch):
    async def boom(pod_cfg, work_id):  # noqa: ANN001
        raise AssertionError("must not boot a pod for a skipped call")

    monkeypatch.setattr(ep, "_boot_runpod", boom)
    cfg = FakeConfig(ephemeral_pod={"enabled": True, "model": "M"})

    @with_ephemeral_pod(skip=lambda a: a.get("start") is False)
    async def run(work_id, start=True, config=None):  # noqa: ANN001
        return "skipped-ok"

    assert await run("w1", start=False, config=cfg) == "skipped-ok"
    assert POD_BASE_URL_ENV not in os.environ


# ── env: provider-value expansion (helpers) ─────────────────────────────────


def test_expand_env_ref_passthrough():
    assert _expand_env_ref("http://x/v1") == "http://x/v1"
    assert _expand_env_ref(None) is None


def test_expand_env_ref_resolves(monkeypatch):
    monkeypatch.setenv("SPINE_POD_BASE_URL", "https://pod/v1")
    assert _expand_env_ref("env:SPINE_POD_BASE_URL") == "https://pod/v1"


def test_expand_env_ref_unset_raises(monkeypatch):
    monkeypatch.delenv("SPINE_POD_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="unset"):
        _expand_env_ref("env:SPINE_POD_BASE_URL")
