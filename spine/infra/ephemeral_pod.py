"""Ephemeral GPU pod lifecycle scoped to a single ``spine`` run.

Brings a remote vLLM pod up at the start of a run and tears it down at the
end — on success, on exception, **and** on ``KeyboardInterrupt`` — so a run
only pays for GPU while it is actually executing. The model the pod serves
and the phases that use it are configured declaratively:

* ``ephemeral_pod:`` in ``.spine/config.yaml`` says HOW to bring the pod up
  (backend, GPU, image, vLLM args, lifecycle policy).
* The existing ``providers.phases`` routing says WHICH phases use it — point
  a phase at a ``providers.llm[]`` entry whose ``base_url`` is
  ``env:SPINE_POD_BASE_URL`` (resolved at runtime once the pod is live, see
  :func:`spine.agents.helpers._expand_env_ref`).

The pod's URL is published into the process environment (``SPINE_POD_BASE_URL``)
rather than threaded through the config object, because provider resolution
reloads ``SpineConfig`` from disk on every phase — see the design note on
``_active_provider_config`` in ``helpers.py``.

Coverage is by a small reference-counted singleton: every graph-running
dispatcher entry point acquires a lease and releases it in a ``finally``;
nested entry points (e.g. ``resume_interrupted_work`` →
``restart_from_phase`` → ``_run_workflow_graph``) share one pod and only the
outermost release tears it down.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from spine.config import SpineConfig

logger = logging.getLogger(__name__)

#: Environment variable the bound provider's ``base_url: env:SPINE_POD_BASE_URL``
#: resolves against. Set when the pod is live, unset on teardown.
POD_BASE_URL_ENV = "SPINE_POD_BASE_URL"


class PodStartupError(RuntimeError):
    """The ephemeral pod could not be brought up and policy is ``abort``."""


# ── Config ────────────────────────────────────────────────────────────────


@dataclass
class EphemeralPodConfig:
    """Parsed ``ephemeral_pod:`` section of ``.spine/config.yaml``."""

    enabled: bool = False
    backend: str = "runpod"
    # Name of the providers.llm[] entry this pod backs. Its ``model`` (minus the
    # ``openai:`` prefix) is used as the vLLM ``--model`` unless ``model`` below
    # is set explicitly, and its ``base_url`` must be ``env:SPINE_POD_BASE_URL``.
    binds_provider: str = ""
    api_key_env: str = "RUNPOD_API_KEY"
    image: str = "vllm/vllm-openai:latest"
    gpu_type_id: str = ""
    gpu_count: int = 1
    cloud_type: str = "SECURE"
    container_disk_gb: int = 20
    volume_gb: int = 0
    network_volume_id: str = ""
    volume_mount_path: str = "/root/.cache/huggingface"
    port: int = 8000
    model: str = ""
    vllm_args: str = ""
    hf_token_env: str = "HF_TOKEN"
    env: dict = field(default_factory=dict)
    startup_timeout_s: int = 900
    poll_interval_s: int = 10
    # Hard watchdog: force-terminate the pod this many seconds after boot even
    # if the run hangs or the event loop wedges. 0 disables it.
    max_lifetime_s: int = 0
    # "abort": fail the run if the pod can't come up. "local_fallback": point
    # SPINE_POD_BASE_URL at ``fallback_base_url`` and continue (requires it set).
    on_failure: str = "abort"
    fallback_base_url: str = ""
    name_prefix: str = "spine"
    # Tried in order on a capacity error: each is a dict that may override
    # gpu_type_id / gpu_count / cloud_type.
    gpu_fallback: list = field(default_factory=list)

    @classmethod
    def from_config(cls, config: "SpineConfig") -> "EphemeralPodConfig":
        """Build from a :class:`SpineConfig`, deriving ``model`` from the
        bound provider when not set explicitly."""
        raw = getattr(config, "ephemeral_pod", {}) or {}
        if not isinstance(raw, dict):
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in raw.items() if k in known}
        pod = cls(**kwargs)
        if not pod.model and pod.binds_provider:
            for prov in (config.providers or {}).get("llm", []):
                if prov.get("name") == pod.binds_provider:
                    pod.model = str(prov.get("model", "")).removeprefix("openai:")
                    break
        return pod


# ── RunPod backend (lazy SDK import; only needed when enabled) ──────────────


def _import_runpod() -> Any:
    try:
        import runpod  # type: ignore[import-untyped,import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise PodStartupError(
            "ephemeral_pod is enabled but the 'runpod' package is not "
            "installed. Install it with: pip install runpod"
        ) from exc
    return runpod


def _docker_args(pod: EphemeralPodConfig) -> str:
    """vLLM OpenAI-server flags. TP size must equal the GPU count."""
    parts = [
        f"--model {pod.model}",
        f"--tensor-parallel-size {pod.gpu_count}",
        "--host 0.0.0.0",
        f"--port {pod.port}",
    ]
    if pod.vllm_args:
        parts.append(pod.vllm_args)
    return " ".join(parts)


def _create_kwargs(pod: EphemeralPodConfig, name: str) -> dict[str, Any]:
    env: dict[str, str] = dict(pod.env)
    hf_token = os.environ.get(pod.hf_token_env)
    if hf_token:
        env.setdefault("HF_TOKEN", hf_token)
    kwargs: dict[str, Any] = {
        "name": name,
        "image_name": pod.image,
        "gpu_type_id": pod.gpu_type_id,
        "gpu_count": pod.gpu_count,
        "cloud_type": pod.cloud_type,
        "container_disk_in_gb": pod.container_disk_gb,
        "ports": f"{pod.port}/http",
        "docker_args": _docker_args(pod),
        "volume_mount_path": pod.volume_mount_path,
        "env": env,
    }
    if pod.volume_gb:
        kwargs["volume_in_gb"] = pod.volume_gb
    if pod.network_volume_id:
        kwargs["network_volume_id"] = pod.network_volume_id
    return kwargs


def _http_ok(url: str, timeout: float = 5.0) -> bool:
    """True iff GET ``url`` returns HTTP 200 (used to detect a live server)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


class _RunPodLease:
    """A booted RunPod pod plus its teardown. ``terminate`` is idempotent."""

    def __init__(self, runpod: Any, pod_id: str, base_url: str) -> None:
        self._runpod = runpod
        self.pod_id = pod_id
        self.base_url = base_url
        self._watchdog: asyncio.Task | None = None
        self._terminated = False

    def start_watchdog(self, max_lifetime_s: int) -> None:
        if max_lifetime_s > 0:
            self._watchdog = asyncio.create_task(self._watch(max_lifetime_s))

    async def _watch(self, max_lifetime_s: int) -> None:
        try:
            await asyncio.sleep(max_lifetime_s)
        except asyncio.CancelledError:  # normal teardown
            return
        logger.error(
            "ephemeral_pod %s exceeded max_lifetime_s=%s — force-terminating",
            self.pod_id,
            max_lifetime_s,
        )
        await self.terminate()

    async def terminate(self) -> None:
        if self._terminated:
            return
        self._terminated = True
        if self._watchdog is not None:
            self._watchdog.cancel()
        # terminate_pod (not stop_pod): ends GPU *and* disk billing; a network
        # volume persists. Best-effort — never raise during teardown.
        try:
            await asyncio.to_thread(self._runpod.terminate_pod, self.pod_id)
            logger.info("ephemeral_pod %s terminated", self.pod_id)
        except Exception as exc:  # noqa: BLE001 - teardown must not mask errors
            logger.warning("ephemeral_pod %s terminate failed: %s", self.pod_id, exc)
        if os.environ.get(POD_BASE_URL_ENV) == self.base_url:
            os.environ.pop(POD_BASE_URL_ENV, None)


def _gpu_attempts(pod: EphemeralPodConfig) -> list[EphemeralPodConfig]:
    """Primary spec followed by the configured fallback ladder."""
    attempts = [pod]
    for alt in pod.gpu_fallback:
        if not isinstance(alt, dict):
            continue
        attempts.append(
            EphemeralPodConfig(
                **{
                    **pod.__dict__,
                    "gpu_type_id": alt.get("gpu_type_id", pod.gpu_type_id),
                    "gpu_count": int(alt.get("gpu_count", pod.gpu_count)),
                    "cloud_type": alt.get("cloud_type", pod.cloud_type),
                    "gpu_fallback": [],
                }
            )
        )
    return attempts


async def _sweep_orphans(runpod: Any, name_prefix: str) -> None:
    """Terminate any pod left over from a previously-stranded run."""
    try:
        pods = await asyncio.to_thread(runpod.get_pods)
    except Exception as exc:  # noqa: BLE001 - sweep is best-effort
        logger.debug("ephemeral_pod orphan sweep skipped: %s", exc)
        return
    tag = f"{name_prefix}-pod-"
    for p in pods or []:
        name = (p or {}).get("name", "")
        pid = (p or {}).get("id")
        if pid and isinstance(name, str) and name.startswith(tag):
            logger.warning("ephemeral_pod sweeping orphan %s (%s)", pid, name)
            try:
                await asyncio.to_thread(runpod.terminate_pod, pid)
            except Exception:  # noqa: BLE001
                pass


async def _boot_runpod(pod: EphemeralPodConfig, work_id: str | None) -> _RunPodLease:
    """Create a RunPod pod, wait until the vLLM endpoint answers, publish URL."""
    runpod = _import_runpod()
    api_key = os.environ.get(pod.api_key_env)
    if not api_key:
        raise PodStartupError(
            f"ephemeral_pod: RunPod API key env '{pod.api_key_env}' is unset."
        )
    runpod.api_key = api_key
    if not pod.model:
        raise PodStartupError(
            "ephemeral_pod: no model resolved. Set ephemeral_pod.model or "
            "ephemeral_pod.binds_provider to a providers.llm[] entry with a model."
        )

    await _sweep_orphans(runpod, pod.name_prefix)

    name = f"{pod.name_prefix}-pod-{work_id or 'run'}"
    created: dict[str, Any] | None = None
    last_exc: Exception | None = None
    for attempt in _gpu_attempts(pod):
        try:
            created = await asyncio.to_thread(
                functools.partial(runpod.create_pod, **_create_kwargs(attempt, name))
            )
            pod = attempt
            break
        except Exception as exc:  # noqa: BLE001 - capacity errors are QueryError
            last_exc = exc
            logger.warning(
                "ephemeral_pod create failed on %s x%s: %s",
                attempt.gpu_type_id,
                attempt.gpu_count,
                exc,
            )
    if created is None:
        raise PodStartupError(
            f"ephemeral_pod: could not allocate a pod across "
            f"{len(_gpu_attempts(pod))} GPU option(s): {last_exc}"
        )

    pod_id = created.get("id")
    if not pod_id:
        raise PodStartupError(f"ephemeral_pod: create returned no id: {created!r}")
    base_url = f"https://{pod_id}-{pod.port}.proxy.runpod.net/v1"
    lease = _RunPodLease(runpod, pod_id, base_url)

    # Wait for the vLLM OpenAI server to actually answer (the pod reaching
    # RUNNING only means the container started; weights still have to load).
    deadline = pod.startup_timeout_s
    waited = 0
    models_url = f"{base_url}/models"
    try:
        while waited < deadline:
            if await asyncio.to_thread(_http_ok, models_url):
                break
            await asyncio.sleep(pod.poll_interval_s)
            waited += pod.poll_interval_s
        else:
            raise PodStartupError(
                f"ephemeral_pod {pod_id}: vLLM endpoint did not become ready "
                f"within {deadline}s."
            )
    except BaseException:
        await lease.terminate()
        raise

    os.environ[POD_BASE_URL_ENV] = base_url
    lease.start_watchdog(pod.max_lifetime_s)
    logger.info(
        "ephemeral_pod %s ready after ~%ss → %s (model=%s)",
        pod_id,
        waited,
        base_url,
        pod.model,
    )
    return lease


# ── Reference-counted process singleton ─────────────────────────────────────


@dataclass
class _Lease:
    """Returned by :func:`acquire`; passed back to :func:`release`."""

    active: bool


_LOCK = asyncio.Lock()
_POD: _RunPodLease | None = None
_REFCOUNT = 0


async def acquire(config: "SpineConfig", work_id: str | None) -> _Lease:
    """Ensure the pod is up (booting it on the first lease) and return a lease.

    No-op (returns an inactive lease) when ``ephemeral_pod.enabled`` is false.
    On boot failure: raises :class:`PodStartupError` under ``on_failure: abort``;
    under ``local_fallback`` publishes ``fallback_base_url`` and continues.
    """
    pod_cfg = EphemeralPodConfig.from_config(config)
    if not pod_cfg.enabled:
        return _Lease(active=False)

    global _POD, _REFCOUNT
    async with _LOCK:
        if _POD is None and _REFCOUNT == 0:
            try:
                _POD = await _boot_runpod(pod_cfg, work_id)
            except BaseException as exc:
                if pod_cfg.on_failure == "local_fallback" and pod_cfg.fallback_base_url:
                    logger.warning(
                        "ephemeral_pod boot failed (%s); falling back to %s",
                        exc,
                        pod_cfg.fallback_base_url,
                    )
                    os.environ[POD_BASE_URL_ENV] = pod_cfg.fallback_base_url
                    # _POD stays None; refcount still tracks the run so we don't
                    # re-attempt a boot on every nested entry point.
                    _REFCOUNT += 1
                    return _Lease(active=True)
                raise
        _REFCOUNT += 1
        return _Lease(active=True)


async def release(lease: _Lease | None) -> None:
    """Release a lease; tear the pod down when the last lease is released."""
    if lease is None or not lease.active:
        return
    global _POD, _REFCOUNT
    async with _LOCK:
        _REFCOUNT = max(0, _REFCOUNT - 1)
        if _REFCOUNT == 0:
            if _POD is not None:
                await _POD.terminate()
                _POD = None
            os.environ.pop(POD_BASE_URL_ENV, None)


# ── Decorator for dispatcher entry points ───────────────────────────────────


def with_ephemeral_pod(
    *, skip: Callable[[dict[str, Any]], bool] | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap an async dispatcher entry point so a configured pod is live for the
    whole call and torn down in a ``finally`` (covers success, ``Exception``,
    and ``KeyboardInterrupt``).

    ``skip(bound_args)`` lets a caller suppress bring-up for cheap paths that
    never run the graph (e.g. ``submit_work(start=False)``).
    """

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                bound = sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                argmap: dict[str, Any] = dict(bound.arguments)
            except TypeError:
                argmap = {}
            if skip is not None and skip(argmap):
                return await fn(*args, **kwargs)
            config = argmap.get("config")
            if config is None:
                from spine.config import SpineConfig

                config = SpineConfig.load()
            lease = await acquire(config, argmap.get("work_id"))
            try:
                return await fn(*args, **kwargs)
            finally:
                await release(lease)

        return wrapper

    return deco
