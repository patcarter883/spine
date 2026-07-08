"""Client for the minisgl CAM memory-organ edit plane (``/cam/*``).

The CAM serving path (minisgl-rdna4 ``cam-serving`` milestone) exposes an
editable, subject-keyed fact store next to the served model: writes go through
a base-uncertainty gate (``/cam/remember`` stores a fact only when the base
model can't already recall it), reads are delivered inside the forward pass,
and the store is scoped per tenant via an ``X-CAM-Namespace`` header.

Spine treats this plane as strictly optional infrastructure: **every method
here is fail-open** — a server without CAM (503), an unreachable host, or a
malformed response degrades to ``None`` with a debug log, never an exception.
Nothing in the workflow may block on the memory organ (mirrors the
``capture_run_experience`` never-raise contract).

The client owns a small dedicated ``httpx.AsyncClient`` rather than the shared
per-provider pool in :mod:`spine.agents.http_clients`: that pool's connection
cap is the LLM concurrency limiter, and streaming LLM calls hold connections
for minutes — edit-plane calls must not queue behind them.

See ``docs/memory-organ-integration-plan.md`` for the feature plan this
implements (Phase 0).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 15.0
# One retry on transport errors only — the plane is best-effort; anything more
# belongs to the caller's reconciliation pass, not the request path.
_TRANSPORT_ATTEMPTS = 2

_WRITE_MODES = ("off", "distill", "ambient")
_READ_MODES = ("off", "transparent", "facts_block", "both")


@dataclass(frozen=True)
class CamSettings:
    """Resolved ``cam:`` provider-config block (see config.reference.yaml)."""

    server_root: str
    """Server origin, no trailing ``/v1`` (the ``/cam`` router sits beside it)."""
    api_token: str | None = None
    namespace: str | None = None
    """Resolved namespace; ``None`` means the server's ``default`` store."""
    write: str = "distill"
    read: str = "transparent"
    capacity_alert: int = 100


def _expand_env(value: Any) -> Any:
    """Resolve an ``env:VARNAME`` indirection, fail-open to ``None`` when unset.

    Unlike :func:`spine.agents.helpers._expand_env_ref` this never raises — a
    missing CAM token must not break model building.
    """
    if isinstance(value, str) and value.startswith("env:"):
        return os.environ.get(value[len("env:") :]) or None
    return value


def _slugify(name: str) -> str:
    """Lowercase, non-alphanumerics collapsed to ``-`` — a stable namespace key."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _server_root_from(base_url: str) -> str:
    """Derive the server origin from an OpenAI-style base_url.

    The CAM router is mounted at ``/cam`` on the same server that serves
    ``/v1/...``, so strip a trailing ``/v1`` (and any trailing slash).
    """
    return re.sub(r"/v1/?$", "", base_url.rstrip("/")) or base_url


def resolve_cam_settings(
    provider_cfg: dict[str, Any],
    workspace_root: str | None = None,
) -> CamSettings | None:
    """Parse a provider's ``cam:`` block into :class:`CamSettings`.

    Returns ``None`` when CAM is not configured or explicitly disabled.
    Accepted forms (consistent with the ``rsa`` field conventions):

    - absent / ``null`` / ``false`` → ``None`` (CAM off)
    - ``cam: true`` → enabled with all defaults
    - ``cam: {...}`` → enabled unless ``enabled: false``

    ``namespace: auto`` (the default) resolves to a slug of the project's main
    repo directory name, so every run against the same project shares one
    durable store; pass ``workspace_root`` to resolve it. Without a root the
    namespace stays ``None`` (server default store).
    """
    cam = provider_cfg.get("cam")
    if cam is True:
        cam = {}
    if not isinstance(cam, dict) or cam.get("enabled") is False:
        return None

    base_url = _expand_env(cam.get("base_url") or provider_cfg.get("base_url"))
    if not base_url:
        logger.debug("CAM configured but no base_url available; disabling")
        return None

    token = _expand_env(cam.get("api_token")) or os.environ.get(
        "MINISGL_CAM_API_TOKEN"
    )

    namespace = cam.get("namespace", "auto")
    if namespace == "auto":
        namespace = _slugify(Path(workspace_root).name) if workspace_root else None

    write = cam.get("write", "distill")
    if write not in _WRITE_MODES:
        logger.warning("Unknown cam.write %r; falling back to 'distill'", write)
        write = "distill"
    read = cam.get("read", "transparent")
    if read not in _READ_MODES:
        logger.warning("Unknown cam.read %r; falling back to 'transparent'", read)
        read = "transparent"

    return CamSettings(
        server_root=_server_root_from(str(base_url)),
        api_token=token,
        namespace=namespace or None,
        write=write,
        read=read,
        capacity_alert=int(cam.get("capacity_alert", 100)),
    )


class CAMClient:
    """Async, fail-open client for the ``/cam/*`` edit plane.

    All methods return ``None`` on any failure (CAM not loaded, network error,
    auth rejection, bad payload) after logging at debug level — callers branch
    on ``None`` and carry on.
    """

    def __init__(
        self,
        settings: CamSettings,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        headers: dict[str, str] = {}
        if settings.api_token:
            headers["Authorization"] = f"Bearer {settings.api_token}"
        if settings.namespace:
            headers["X-CAM-Namespace"] = settings.namespace
        # An injected client is the test seam (httpx.MockTransport).
        self._client = client or httpx.AsyncClient(
            base_url=settings.server_root,
            headers=headers,
            timeout=_DEFAULT_TIMEOUT,
        )
        self._owns_client = client is None
        # When a client is injected, base_url/headers may not be set on it —
        # keep them for per-request use.
        self._headers = headers

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ── transport ─────────────────────────────────────────────────────────
    async def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any | None:
        url = f"{self.settings.server_root}{path}"
        for attempt in range(1, _TRANSPORT_ATTEMPTS + 1):
            try:
                resp = await self._client.request(
                    method, url, json=json, params=params, headers=self._headers
                )
                if resp.status_code == 503:
                    logger.debug("CAM not loaded on %s (503)", self.settings.server_root)
                    return None
                resp.raise_for_status()
                if not resp.content:
                    return {}
                return resp.json()
            except httpx.TransportError as e:
                if attempt < _TRANSPORT_ATTEMPTS:
                    continue
                logger.debug("CAM %s %s unreachable: %s", method, path, e)
                return None
            except Exception as e:  # noqa: BLE001 — fail-open by contract
                logger.debug("CAM %s %s failed: %s", method, path, e)
                return None
        return None

    # ── edit plane ────────────────────────────────────────────────────────
    async def remember(
        self, subject: str, prompt: str, object_: str
    ) -> dict[str, Any] | None:
        """Write-gated store: ``{stored: bool, base_p: float}`` or ``None``."""
        return await self._request(
            "POST",
            "/cam/remember",
            json={"subject": subject, "prompt": prompt, "object": object_},
        )

    async def ask(
        self, prompt: str, subject: str, max_tokens: int = 32
    ) -> str | None:
        """Router-gated seed-once generation; the ground-truth readback probe."""
        data = await self._request(
            "POST",
            "/cam/ask",
            json={"prompt": prompt, "subject": subject, "max_tokens": max_tokens},
        )
        return data.get("text") if isinstance(data, dict) else None

    async def facts(self) -> list[dict[str, Any]] | None:
        data = await self._request("GET", "/cam/facts")
        return data if isinstance(data, list) else None

    async def delete_fact(self, subject: str) -> bool | None:
        data = await self._request("DELETE", f"/cam/facts/{subject}")
        return data.get("deleted") if isinstance(data, dict) else None

    async def stats(self) -> dict[str, Any] | None:
        data = await self._request("GET", "/cam/stats")
        return data if isinstance(data, dict) else None

    async def audit(self) -> list[dict[str, Any]] | None:
        data = await self._request("GET", "/cam/audit")
        return data if isinstance(data, list) else None

    # ── lifecycle ─────────────────────────────────────────────────────────
    async def freeze(self, frozen: bool = True) -> dict[str, Any] | None:
        return await self._request("POST", "/cam/freeze", params={"frozen": frozen})

    async def save(self) -> dict[str, Any] | None:
        return await self._request("POST", "/cam/save")

    async def undo(self) -> dict[str, Any] | None:
        return await self._request("POST", "/cam/undo")

    async def rebuild(self) -> dict[str, Any] | None:
        return await self._request("POST", "/cam/rebuild")


def cam_client_for(
    provider_cfg: dict[str, Any],
    workspace_root: str | None = None,
) -> CAMClient | None:
    """Build a :class:`CAMClient` from a provider config, or ``None`` if CAM is off."""
    settings = resolve_cam_settings(provider_cfg, workspace_root=workspace_root)
    return CAMClient(settings) if settings else None
