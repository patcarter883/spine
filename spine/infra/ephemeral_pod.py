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

#: Env var the *legacy single-pod* provider ``base_url: env:SPINE_POD_BASE_URL``
#: resolves against. A named multi-pod publishes ``SPINE_POD_URL__<NAME>``
#: instead (see :attr:`EphemeralPodConfig.url_env`).
POD_BASE_URL_ENV = "SPINE_POD_BASE_URL"


# ── Lane / phase-set helpers (pure) ─────────────────────────────────────────
# A pod declares the phases it serves (``phases:``). A run boots a pod iff the
# pod's phases intersect the set of phases the run will actually execute. With
# the Specify/Plan and Implement/Verify lanes always run as SEPARATE spine runs
# (human approval gate between them), each run touches exactly one lane, so this
# selection boots exactly one pod per run — phase-scoped economics without any
# mid-run boot/teardown. Names use the canonical PhaseName values plus the
# ``critic``/``critic_plan``/``critic_specify`` aliases the workflow emits.
PLANNING_PHASES: frozenset[str] = frozenset(
    {
        "specify",
        "plan",
        "tasks",
        "critic",
        "critic_plan",
        "critic_specify",
        "adversarial",
        "adversarial_plan",
        "gap_plan",
    }
)
EXECUTION_PHASES: frozenset[str] = frozenset({"implement", "verify"})


def lane_phase_set(phase: str | None) -> set[str] | None:
    """The full phase set of the lane ``phase`` belongs to, or ``None``.

    Used by resume/restart paths: given the phase a run resumes at, boot the
    pod(s) for that whole lane (a resume never crosses the human-review gate,
    so it stays within one lane).
    """
    if not phase:
        return None
    base = phase.split("/")[0]
    if phase in EXECUTION_PHASES or base in EXECUTION_PHASES:
        return set(EXECUTION_PHASES)
    if phase in PLANNING_PHASES or base in PLANNING_PHASES:
        return set(PLANNING_PHASES)
    return None


def executed_phases_for_run(
    work_type: str | None = None, seed_phase: str | None = None
) -> set[str] | None:
    """Phases a run will execute, for pod selection.

    ``seed_phase`` (the phase a run starts/resumes at) wins when it pins a lane.
    Otherwise the phase set is derived from ``work_type``:

    * reviewed_task / critical_reviewed_task → planning lane only (ENDs at the
      human-approval gate).
    * task / critical_task → both lanes (a fully autonomous run spans them).

    Returns ``None`` (→ boot every configured pod, conservative) when neither
    input determines the set.
    """
    if seed_phase:
        lane = lane_phase_set(seed_phase)
        if lane is not None:
            return lane
    if work_type in ("reviewed_task", "critical_reviewed_task"):
        return set(PLANNING_PHASES)
    if work_type in ("task", "critical_task"):
        return set(PLANNING_PHASES) | set(EXECUTION_PHASES)
    return None


class PodStartupError(RuntimeError):
    """The ephemeral pod could not be brought up and policy is ``abort``."""


# ── Config ────────────────────────────────────────────────────────────────


@dataclass
class EphemeralPodConfig:
    """Parsed ``ephemeral_pod:`` section of ``.spine/config.yaml``."""

    enabled: bool = False
    backend: str = "runpod"
    # Distinguishes pods in a multi-pod (``ephemeral_pods:``) config and derives
    # this pod's URL env var (see ``url_env``). Empty for the legacy single-pod
    # (``ephemeral_pod:``) form, which uses SPINE_POD_BASE_URL.
    name: str = ""
    # Phases this pod serves. A run boots the pod iff this intersects the phases
    # the run will execute (see ``executed_phases_for_run``). Empty = always boot
    # when enabled (the legacy single-pod behaviour).
    phases: list = field(default_factory=list)
    # Name of the providers.llm[] entry this pod backs. Its ``model`` (minus the
    # ``openai:`` prefix) is used as the vLLM ``--model`` unless ``model`` below
    # is set explicitly, and its ``base_url`` must be ``env:<url_env>``.
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

    @property
    def url_env(self) -> str:
        """Env var this pod publishes its live OpenAI URL to. A named pod uses
        ``SPINE_POD_URL__<NAME>``; the legacy unnamed pod uses
        ``SPINE_POD_BASE_URL``. The bound provider's ``base_url`` must be
        ``env:<this value>``."""
        if self.name:
            return f"SPINE_POD_URL__{self.name.upper()}"
        return POD_BASE_URL_ENV

    @classmethod
    def _from_raw(cls, raw: dict, providers: dict) -> "EphemeralPodConfig":
        """Build one pod from a raw dict, deriving ``model`` from its bound
        provider when not set explicitly."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        pod = cls(**{k: v for k, v in raw.items() if k in known})
        if not pod.model and pod.binds_provider:
            for prov in (providers or {}).get("llm", []):
                if prov.get("name") == pod.binds_provider:
                    pod.model = str(prov.get("model", ""))
                    break
        # vLLM's --model needs the bare HF repo, never the ``openai:`` spec
        # prefix — strip it whether the model was set explicitly or derived.
        pod.model = pod.model.removeprefix("openai:")
        return pod

    @classmethod
    def from_config(cls, config: "SpineConfig") -> "EphemeralPodConfig":
        """Back-compat single-pod accessor: the first pod of :func:`parse_pods`,
        or a disabled default. Prefer :func:`parse_pods` for multi-pod configs."""
        pods = parse_pods(config)
        return pods[0] if pods else cls()


def parse_pods(config: "SpineConfig") -> list["EphemeralPodConfig"]:
    """All configured pods, from either config shape.

    * ``ephemeral_pods:`` (list) — the multi-pod form; each entry needs a
      ``name`` and a ``phases`` list.
    * ``ephemeral_pod:`` (dict) — the legacy single-pod form (unnamed, always
      boots when enabled).

    Returns only enabled pods. The two forms are mutually exclusive; if both are
    present the list form wins and the singular is ignored (logged).
    """
    providers = getattr(config, "providers", {}) or {}
    multi = getattr(config, "ephemeral_pods", None)
    single = getattr(config, "ephemeral_pod", None)
    pods: list[EphemeralPodConfig] = []
    if isinstance(multi, list) and multi:
        if single:
            logger.warning(
                "Both 'ephemeral_pods' and 'ephemeral_pod' are set; using the "
                "'ephemeral_pods' list and ignoring the singular block."
            )
        for i, raw in enumerate(multi):
            if not isinstance(raw, dict):
                continue
            pod = EphemeralPodConfig._from_raw(raw, providers)
            if not pod.name:
                pod.name = f"pod{i}"
            if pod.enabled:
                pods.append(pod)
    elif isinstance(single, dict) and single:
        pod = EphemeralPodConfig._from_raw(single, providers)
        if pod.enabled:
            pods.append(pod)
    return pods


def select_pods(
    config: "SpineConfig", executed_phases: set[str] | None
) -> list["EphemeralPodConfig"]:
    """Pods a run needs: those whose ``phases`` intersect ``executed_phases``.

    A pod with an empty ``phases`` list always qualifies (legacy always-on). When
    ``executed_phases`` is ``None`` (the run's phase set couldn't be determined)
    every enabled pod is returned — the safe superset.
    """
    pods = parse_pods(config)
    if executed_phases is None:
        return pods
    selected: list[EphemeralPodConfig] = []
    for pod in pods:
        if not pod.phases or (set(pod.phases) & executed_phases):
            selected.append(pod)
    return selected


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

    def __init__(
        self, runpod: Any, pod_id: str, base_url: str, url_env: str = POD_BASE_URL_ENV
    ) -> None:
        self._runpod = runpod
        self.pod_id = pod_id
        self.base_url = base_url
        self.url_env = url_env
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
        if os.environ.get(self.url_env) == self.base_url:
            os.environ.pop(self.url_env, None)


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
    lease = _RunPodLease(runpod, pod_id, base_url, url_env=pod.url_env)

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

    os.environ[pod.url_env] = base_url
    lease.start_watchdog(pod.max_lifetime_s)
    logger.info(
        "ephemeral_pod '%s' %s ready after ~%ss → %s=%s (model=%s)",
        pod.name or "default",
        pod_id,
        waited,
        pod.url_env,
        base_url,
        pod.model,
    )
    return lease


# ── Reference-counted, multi-pod process registry ───────────────────────────


@dataclass
class _Lease:
    """Returned by :func:`acquire`; passed back to :func:`release`."""

    active: bool


_LOCK = asyncio.Lock()
#: name → live pod lease, for every pod currently booted this run.
_PODS: dict[str, _RunPodLease] = {}
#: fallback env vars published this run (name → url_env), cleared on teardown.
_FALLBACKS: dict[str, str] = {}
_REFCOUNT = 0


async def _boot_or_fallback(pod: "EphemeralPodConfig", work_id: str | None) -> None:
    """Boot one pod into the registry, or apply its failure policy (caller holds
    ``_LOCK``). Raises :class:`PodStartupError` under ``on_failure: abort``."""
    try:
        _PODS[pod.name] = await _boot_runpod(pod, work_id)
    except BaseException as exc:
        if pod.on_failure == "local_fallback" and pod.fallback_base_url:
            logger.warning(
                "ephemeral_pod '%s' boot failed (%s); falling back to %s",
                pod.name or "default",
                exc,
                pod.fallback_base_url,
            )
            os.environ[pod.url_env] = pod.fallback_base_url
            _FALLBACKS[pod.name] = pod.url_env
            return
        raise


async def _terminate_all_locked() -> None:
    """Terminate every booted pod and clear published env vars (holds ``_LOCK``)."""
    for lease in list(_PODS.values()):
        await lease.terminate()
    _PODS.clear()
    for url_env in _FALLBACKS.values():
        os.environ.pop(url_env, None)
    _FALLBACKS.clear()


async def acquire(
    config: "SpineConfig",
    work_id: str | None,
    executed_phases: set[str] | None = None,
) -> _Lease:
    """Ensure the pods this run needs are up, and return a lease.

    Selection: boots the pods whose ``phases`` intersect ``executed_phases``
    (``None`` → every configured pod). Returns an inactive lease when no pod is
    configured/selected. Already-booted pods (a nested entry point in the same
    run) are reused, not re-booted. On boot failure: raises under
    ``on_failure: abort`` (rolling back anything booted in this call);
    ``local_fallback`` publishes ``fallback_base_url`` and continues.
    """
    selected = select_pods(config, executed_phases)
    if not selected:
        return _Lease(active=False)

    global _REFCOUNT
    async with _LOCK:
        booted_here: list[str] = []
        try:
            for pod in selected:
                if pod.name in _PODS or pod.name in _FALLBACKS:
                    continue  # already up for this run
                await _boot_or_fallback(pod, work_id)
                booted_here.append(pod.name)
        except BaseException:
            # Roll back only what THIS call booted; leave any pods a parent
            # entry point already holds intact (its finally will release them).
            for name in booted_here:
                lease = _PODS.pop(name, None)
                if lease is not None:
                    await lease.terminate()
                _FALLBACKS.pop(name, None)
            raise
        _REFCOUNT += 1
        return _Lease(active=True)


async def release(lease: _Lease | None) -> None:
    """Release a lease; tear every pod down when the last lease is released."""
    if lease is None or not lease.active:
        return
    global _REFCOUNT
    async with _LOCK:
        _REFCOUNT = max(0, _REFCOUNT - 1)
        if _REFCOUNT == 0:
            await _terminate_all_locked()


# ── Decorator for dispatcher entry points ───────────────────────────────────


def with_ephemeral_pod(
    *,
    skip: Callable[[dict[str, Any]], bool] | None = None,
    phases: Callable[[dict[str, Any]], set[str] | None] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap an async dispatcher entry point so the pod(s) it needs are live for
    the whole call and torn down in a ``finally`` (covers success, ``Exception``,
    and ``KeyboardInterrupt``).

    ``skip(bound_args)`` suppresses bring-up for cheap paths that never run the
    graph (e.g. ``submit_work(start=False)``).

    ``phases(bound_args) -> set[str] | None`` returns the phases this call will
    execute, used to select which pods to boot (see :func:`select_pods`). Return
    ``None`` to boot every configured pod (the conservative default when the
    phase set can't be determined).
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
            executed = phases(argmap) if phases is not None else None
            lease = await acquire(config, argmap.get("work_id"), executed)
            try:
                return await fn(*args, **kwargs)
            finally:
                await release(lease)

        return wrapper

    return deco
