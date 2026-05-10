"""POST /work/submit endpoint with validation, idempotency, and async dispatch."""

import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request
from starlette.status import (
    HTTP_202_ACCEPTED,
    HTTP_400_BAD_REQUEST,
    HTTP_409_CONFLICT,
    HTTP_429_TOO_MANY_REQUESTS,
)

from ..models.job import Job, JobStatus, JobStore
from ..schemas.work import WorkSubmitRequest, WorkSubmitResponse, sanitize_input

router = APIRouter(prefix="/work", tags=["work"])

job_store = JobStore()

# ── Rate Limiter ──────────────────────────────────────────────────────


class TokenBucket:
    """Simple token bucket rate limiter per client key."""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._buckets: dict[str, dict] = {}
        self._lock = threading.Lock()

    def _get_bucket(self, key: str) -> dict:
        now = time.monotonic()
        if key not in self._buckets:
            self._buckets[key] = {"tokens": self.capacity, "last_refill": now}
            return self._buckets[key]
        bucket = self._buckets[key]
        elapsed = now - bucket["last_refill"]
        bucket["tokens"] = min(self.capacity, bucket["tokens"] + elapsed * self.refill_rate)
        bucket["last_refill"] = now
        return bucket

    def consume(self, key: str, tokens: int = 1) -> bool:
        with self._lock:
            bucket = self._get_bucket(key)
            if bucket["tokens"] >= tokens:
                bucket["tokens"] -= tokens
                return True
            return False


rate_limiter = TokenBucket(capacity=10, refill_rate=1.0)


def _resolve_client_key(request: Request) -> str:
    """Extract a unique client key from the request.

    Uses X-API-Key header if present, otherwise falls back to client IP.
    """
    api_key = request.headers.get("X-API-Key")
    if api_key:
        sanitized = sanitize_input(api_key)
        return f"apikey:{sanitized}"
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"
    return f"ip:{client_ip}"


# ── Async Worker ─────────────────────────────────────────────────────


def _dispatch_work(job: Job) -> None:
    """Run work in a background thread and update job status."""
    try:
        from spine.core.state_machine import SpineStateMachine

        machine = SpineStateMachine(checkpoint_path=".spine/spine.db")

        config = load_config()
        providers_by_type = _load_providers_from_config(config)
        providers_dict = {}
        for category, provider_list in providers_by_type.items():
            if provider_list:
                providers_dict[category] = provider_list[0][1]

        thread_id = str(uuid.uuid4())

        initial_state = {
            "phase": "INIT",
            "previous_phase": None,
            "requirement": job.requirement,
            "plan": None,
            "tasks": {},
            "completed_tasks": [],
            "failed_tasks": [],
            "swarm_state": {"active_subphases": [], "file_reservations": {}, "pending_gates": []},
            "hive_cells": {},
            "swarm_events": [],
            "variables": {
                "thread_id": thread_id,
                "work_item_id": thread_id,
                "checkpoint_path": ".spine/spine.db",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "errors": [],
            "providers": providers_dict,
            "critic_gate_result": None,
            "error_state": None,
            "error_history": [],
        }

        job_store.update_status(job.job_id, JobStatus.RUNNING, thread_id=thread_id)

        result = machine.app.invoke(
            initial_state,
            {"configurable": {"thread_id": thread_id}},
        )

        job_store.update_status(job.job_id, JobStatus.COMPLETED)

    except Exception as exc:
        error_msg = str(exc)
        job_store.update_status(job.job_id, JobStatus.FAILED, error_message=error_msg)


def _dispatch_async(job: Job) -> None:
    """Launch work dispatch in a daemon thread."""
    thread = threading.Thread(target=_dispatch_work, args=(job,), daemon=True)
    thread.start()


# ── Route ────────────────────────────────────────────────────────────


@router.post("/submit", status_code=HTTP_202_ACCEPTED)
async def submit_work(
    request: Request,
    body: WorkSubmitRequest,
    x_idempotency_key: Optional[str] = Header(None, alias="X-Idempotency-Key"),
) -> WorkSubmitResponse:
    """Submit a new work item.

    Validates the payload, checks idempotency, enforces rate limits,
    creates a job record, and dispatches work to the background queue.
    Returns 202 Accepted immediately with the job ID.
    """
    client_key = _resolve_client_key(request)
    if not rate_limiter.consume(client_key):
        raise HTTPException(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait before submitting again.",
        )

    if x_idempotency_key:
        sanitized_key = sanitize_input(x_idempotency_key)
        existing = job_store.get_by_idempotency_key(sanitized_key)
        if existing:
            raise HTTPException(
                status_code=HTTP_409_CONFLICT,
                detail=f"Duplicate submission. Existing job_id: {existing.job_id}",
            )
    else:
        sanitized_key = None

    job = Job(
        requirement=body.requirement,
        method=body.method,
        project_type=body.project_type,
        llm_provider=body.llm_provider,
        parallel_agents=body.parallel_agents,
        idempotency_key=sanitized_key,
    )

    job_store.create(job)
    _dispatch_async(job)

    return WorkSubmitResponse(job_id=job.job_id, status="queued")


def _load_providers_from_config(config: dict) -> dict[str, list]:
    """Load provider configurations from a parsed config dict."""
    providers_by_category: dict[str, list] = {}
    for category, provider_list in config.get("providers", {}).items():
        if not provider_list:
            continue
        for instance in provider_list:
            name = instance.get("name", "unnamed")
            enabled = instance.get("enabled", True)
            if not enabled:
                continue
            providers_by_category.setdefault(category, []).append((name, instance))
    return providers_by_category


def load_config(config_path: str = ".spine/config.yaml") -> dict:
    """Load configuration from YAML file."""
    from pathlib import Path
    import yaml

    path = Path(config_path)
    if not path.exists():
        return {}

    with open(path) as f:
        config = yaml.safe_load(f) or {}
    return config
