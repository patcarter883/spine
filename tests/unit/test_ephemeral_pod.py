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
    """Minimal stand-in for SpineConfig (parse_pods reads only these attrs)."""

    def __init__(
        self,
        ephemeral_pod: dict | None = None,
        providers: dict | None = None,
        ephemeral_pods: list | None = None,
    ):
        self.ephemeral_pod = ephemeral_pod or {}
        self.ephemeral_pods = ephemeral_pods or []
        self.providers = providers or {}


class FakeRunpod:
    def __init__(self) -> None:
        self.terminated: list[str] = []

    def terminate_pod(self, pod_id: str) -> None:
        self.terminated.append(pod_id)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the module-level pod registry + env vars around every test."""

    def _clear():
        ep._PODS.clear()
        ep._FALLBACKS.clear()
        ep._REFCOUNT = 0
        for k in [k for k in os.environ if k.startswith("SPINE_POD_URL__")]:
            os.environ.pop(k, None)
        os.environ.pop(POD_BASE_URL_ENV, None)

    _clear()
    yield
    _clear()


def _patch_boot(monkeypatch, runpod: FakeRunpod, pod_id: str = "pod123", events=None):
    """Patch the real boot with one that publishes the pod's own url_env and
    returns a lease keyed to that env var, per-pod-name aware."""

    async def fake_boot(pod_cfg, work_id):  # noqa: ANN001
        # Give each named pod a distinct pod_id so terminations are attributable.
        pid = f"{pod_id}-{pod_cfg.name}" if pod_cfg.name else pod_id
        url = f"https://{pid}-8000.proxy.runpod.net/v1"
        os.environ[pod_cfg.url_env] = url
        if events is not None:
            events.append(f"boot:{pod_cfg.name or 'default'}")
        return ep._RunPodLease(runpod, pid, url, url_env=pod_cfg.url_env)

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


def test_resolved_image_by_engine():
    assert EphemeralPodConfig(engine="vllm").resolved_image == "vllm/vllm-openai:latest"
    assert EphemeralPodConfig(engine="sglang").resolved_image == "lmsysorg/sglang:latest"
    # explicit image always wins
    assert (
        EphemeralPodConfig(engine="sglang", image="vllm/vllm-openai:v0.20.0").resolved_image
        == "vllm/vllm-openai:v0.20.0"
    )


def test_docker_args_sglang_with_dflash():
    pod = EphemeralPodConfig(
        engine="sglang",
        model="poolside/Laguna-XS-2.1",
        gpu_count=1,
        port=8000,
        speculative_algorithm="DFLASH",
        draft_model="poolside/Laguna-XS-2.1-DFlash",
        sglang_args="--kv-cache-dtype fp8_e5m2 --mem-fraction-static 0.9",
    )
    da = _docker_args(pod)
    assert da.startswith("python3 -m sglang.launch_server")
    assert "--model-path poolside/Laguna-XS-2.1" in da
    assert "--tp 1" in da
    assert "--host 0.0.0.0" in da and "--port 8000" in da
    assert "--speculative-algorithm DFLASH" in da
    assert "--speculative-draft-model-path poolside/Laguna-XS-2.1-DFlash" in da
    assert "--mem-fraction-static 0.9" in da


def test_docker_args_vllm_unchanged_by_engine_default():
    # engine defaults to vllm → no sglang prefix, vLLM flag form
    da = _docker_args(EphemeralPodConfig(model="M", gpu_count=4, vllm_args="--kv-cache-dtype fp8"))
    assert "sglang" not in da
    assert da.startswith("--model M")
    assert "--tensor-parallel-size 4" in da


def test_create_kwargs_sglang_image():
    pod = EphemeralPodConfig(engine="sglang", model="poolside/Laguna-XS-2.1", gpu_count=1)
    ck = _create_kwargs(pod, "spine-pod-coder")
    assert ck["image_name"] == "lmsysorg/sglang:latest"
    assert ck["docker_args"].startswith("python3 -m sglang.launch_server")


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
    # Legacy single pod keeps an empty name (so its url_env stays
    # SPINE_POD_BASE_URL); it is registered under that empty key.
    assert "" in ep._PODS and ep._PODS[""].pod_id == "pod123"

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
    assert events == ["boot:default", "body"]
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


# ── Multi-pod: parsing, selection, per-lane boot ────────────────────────────

from spine.infra.ephemeral_pod import (  # noqa: E402
    EXECUTION_PHASES,
    PLANNING_PHASES,
    executed_phases_for_run,
    lane_phase_set,
    parse_pods,
    select_pods,
)

_TWO_POD_CFG = FakeConfig(
    ephemeral_pods=[
        {
            "name": "reasoner",
            "enabled": True,
            "model": "openai:GLM-Air",
            "phases": ["specify", "plan", "critic", "gap_plan"],
        },
        {
            "name": "coder",
            "enabled": True,
            "model": "openai:Qwen3.6-35B-A3B-FP8",
            "phases": ["implement", "verify"],
        },
    ]
)


def test_lane_phase_set():
    assert lane_phase_set("plan") == set(PLANNING_PHASES)
    assert lane_phase_set("critic/plan") == set(PLANNING_PHASES)  # sub-phase path
    assert lane_phase_set("implement") == set(EXECUTION_PHASES)
    assert lane_phase_set("verify") == set(EXECUTION_PHASES)
    assert lane_phase_set("") is None
    assert lane_phase_set("onboarding") is None


def test_executed_phases_for_run():
    # reviewed task ENDs at the approval gate → planning lane only
    assert executed_phases_for_run(work_type="reviewed_task") == set(PLANNING_PHASES)
    # autonomous task spans both lanes
    both = executed_phases_for_run(work_type="task")
    assert both == set(PLANNING_PHASES) | set(EXECUTION_PHASES)
    # seed phase pins the lane regardless of work_type (approve→implement)
    assert executed_phases_for_run(work_type="task", seed_phase="implement") == set(
        EXECUTION_PHASES
    )


def test_parse_pods_multi():
    pods = parse_pods(_TWO_POD_CFG)
    assert [p.name for p in pods] == ["reasoner", "coder"]
    assert pods[0].url_env == "SPINE_POD_URL__REASONER"
    assert pods[1].url_env == "SPINE_POD_URL__CODER"
    assert pods[1].model == "Qwen3.6-35B-A3B-FP8"  # openai: stripped


def test_parse_pods_skips_disabled():
    cfg = FakeConfig(
        ephemeral_pods=[
            {"name": "a", "enabled": True, "model": "m", "phases": ["plan"]},
            {"name": "b", "enabled": False, "model": "m", "phases": ["verify"]},
        ]
    )
    assert [p.name for p in parse_pods(cfg)] == ["a"]


def test_select_pods_by_lane():
    plan_sel = select_pods(_TWO_POD_CFG, set(PLANNING_PHASES))
    assert [p.name for p in plan_sel] == ["reasoner"]
    exec_sel = select_pods(_TWO_POD_CFG, set(EXECUTION_PHASES))
    assert [p.name for p in exec_sel] == ["coder"]
    # None (undeterminable) → conservative: every pod
    assert {p.name for p in select_pods(_TWO_POD_CFG, None)} == {"reasoner", "coder"}


def test_list_form_wins_over_singular(caplog):
    cfg = FakeConfig(
        ephemeral_pod={"enabled": True, "model": "legacy"},
        ephemeral_pods=[{"name": "r", "enabled": True, "model": "m", "phases": ["plan"]}],
    )
    pods = parse_pods(cfg)
    assert [p.name for p in pods] == ["r"]  # singular ignored


@pytest.mark.asyncio
async def test_planning_run_boots_only_reasoner(monkeypatch):
    rp = FakeRunpod()
    _patch_boot(monkeypatch, rp)

    lease = await acquire(_TWO_POD_CFG, "w1", set(PLANNING_PHASES))
    assert os.environ["SPINE_POD_URL__REASONER"].endswith("/v1")
    assert "SPINE_POD_URL__CODER" not in os.environ  # coder NOT booted
    assert set(ep._PODS) == {"reasoner"}

    await release(lease)
    assert rp.terminated == ["pod123-reasoner"]
    assert "SPINE_POD_URL__REASONER" not in os.environ


@pytest.mark.asyncio
async def test_execution_run_boots_only_coder(monkeypatch):
    rp = FakeRunpod()
    _patch_boot(monkeypatch, rp)

    lease = await acquire(_TWO_POD_CFG, "w1", set(EXECUTION_PHASES))
    assert os.environ["SPINE_POD_URL__CODER"].endswith("/v1")
    assert "SPINE_POD_URL__REASONER" not in os.environ
    assert set(ep._PODS) == {"coder"}

    await release(lease)
    assert rp.terminated == ["pod123-coder"]


@pytest.mark.asyncio
async def test_conservative_boots_both_when_phases_none(monkeypatch):
    rp = FakeRunpod()
    _patch_boot(monkeypatch, rp)

    lease = await acquire(_TWO_POD_CFG, "w1", None)  # undeterminable → both
    assert set(ep._PODS) == {"reasoner", "coder"}

    await release(lease)
    assert sorted(rp.terminated) == ["pod123-coder", "pod123-reasoner"]
    assert "SPINE_POD_URL__REASONER" not in os.environ
    assert "SPINE_POD_URL__CODER" not in os.environ


# ── Hardening: pod skipped once providers.phases stops routing to it ───────


def test_select_pods_skips_when_provider_unreferenced():
    """A pod stays 'enabled' with a matching 'phases' gate, but
    providers.phases has been rewired away from its binds_provider — the
    real-world case where reasoner-pod/coder-pod were dropped in favor of a
    single 'pat' provider but the ephemeral_pods block was left enabled."""
    cfg = FakeConfig(
        ephemeral_pods=[
            {
                "name": "reasoner",
                "enabled": True,
                "model": "openai:GLM-Air",
                "binds_provider": "reasoner-pod",
                "phases": ["specify", "plan"],
            },
        ],
        providers={"phases": {"specify": {"provider": "pat"}, "plan": {"provider": "pat"}}},
    )
    assert select_pods(cfg, {"specify"}) == []
    assert select_pods(cfg, None) == []  # conservative superset still filters


def test_select_pods_keeps_pod_when_provider_referenced_directly():
    cfg = FakeConfig(
        ephemeral_pods=[
            {
                "name": "reasoner",
                "enabled": True,
                "model": "openai:GLM-Air",
                "binds_provider": "reasoner-pod",
                "phases": ["specify", "plan"],
            },
        ],
        providers={"phases": {"specify": {"provider": "reasoner-pod"}}},
    )
    assert [p.name for p in select_pods(cfg, {"specify"})] == ["reasoner"]


def test_select_pods_keeps_pod_when_provider_referenced_via_escalation():
    cfg = FakeConfig(
        ephemeral_pods=[
            {
                "name": "coder",
                "enabled": True,
                "model": "openai:Qwen",
                "binds_provider": "coder-pod",
                "phases": ["implement"],
            },
        ],
        providers={
            "phases": {
                "implement": {
                    "provider": "pat",
                    "escalation": [{"provider": "coder-pod"}],
                }
            }
        },
    )
    assert [p.name for p in select_pods(cfg, {"implement"})] == ["coder"]


def test_select_pods_no_binds_provider_is_unaffected():
    """Pods without a declared binds_provider (e.g. legacy configs that set
    model: directly) can't be checked, so they're never dropped by this."""
    assert [p.name for p in select_pods(_TWO_POD_CFG, {"plan"})] == ["reasoner"]


@pytest.mark.asyncio
async def test_abort_rolls_back_pods_booted_this_call(monkeypatch):
    """If the second pod fails to boot under on_failure=abort, the first
    (booted in the same acquire) is rolled back and the error propagates."""
    rp = FakeRunpod()
    boots: list[str] = []

    async def flaky_boot(pod_cfg, work_id):  # noqa: ANN001
        if pod_cfg.name == "coder":
            raise PodStartupError("no capacity")
        boots.append(pod_cfg.name)
        url = f"https://x-{pod_cfg.name}/v1"
        os.environ[pod_cfg.url_env] = url
        return ep._RunPodLease(rp, f"pid-{pod_cfg.name}", url, url_env=pod_cfg.url_env)

    monkeypatch.setattr(ep, "_boot_runpod", flaky_boot)

    with pytest.raises(PodStartupError):
        await acquire(_TWO_POD_CFG, "w1", None)  # both selected; coder fails

    assert boots == ["reasoner"]  # reasoner booted first
    assert rp.terminated == ["pid-reasoner"]  # …then rolled back
    assert ep._PODS == {}
    assert ep._REFCOUNT == 0
    assert "SPINE_POD_URL__REASONER" not in os.environ
