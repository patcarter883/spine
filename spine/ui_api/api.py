"""SPINE UI API — sole read/write interface for Streamlit.

UI pages MUST use UIApi for all data access. Never import directly from
workflow/, phases/, or work.dispatcher. This maintains the zero-duplication
principle: CLI and UI share the same backend code paths.
"""

from __future__ import annotations

import logging
from typing import Any

from spine.config import SpineConfig
from spine.models.enums import TaskStatus
from spine.persistence.artifacts import ArtifactStore
from spine.services.audit_service import AuditService
from spine.work.dispatcher import get_work_status, list_work
from spine.work.ralph_worker import get_worker

logger = logging.getLogger(__name__)


class UIApi:
    """The sole read/write interface for Streamlit UI pages.

    Wraps the dispatcher, artifact store, and audit service
    to provide a unified API for all UI operations.

    Usage::

        api = UIApi()
        items = api.list_work(status="running")
        artifacts = api.get_artifacts(work_id="abc123")
    """

    def __init__(self, config: SpineConfig | None = None) -> None:
        self._config = config or SpineConfig.load()
        self._artifacts = ArtifactStore(base_path=self._config.artifact_path)
        self._audit = AuditService(
            db_path=str(__import__("pathlib").Path(self._config.queue_path).parent / "audit.db")
        )

    # ── Work operations ──

    def get_work(self, work_id: str) -> dict[str, Any] | None:
        """Get details for a specific work item.

        Args:
            work_id: The work item ID.

        Returns:
            Work entry dict, or None.
        """
        return get_work_status(work_id, self._config)

    def list_work(self, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """List work items.

        Args:
            status: Optional status filter.
            limit: Maximum items to return.

        Returns:
            List of work entry dicts.
        """
        return list_work(status=status, limit=limit, config=self._config)

    async def submit_work(self, description: str, work_type: str = "spec") -> dict[str, Any]:
        """Submit new work via the dispatcher (blocking — prefer enqueue_work for UI).

        This method blocks until the entire workflow completes. Use
        ``enqueue_work()`` from UI pages to avoid blocking Streamlit.

        Args:
            description: Work description.
            work_type: Workflow type.

        Returns:
            Result dict with work_id and status.
        """
        from spine.work.dispatcher import submit_work

        return await submit_work(description, work_type, self._config)

    def enqueue_work(self, description: str, work_type: str = "spec") -> dict[str, Any]:
        """Enqueue work for async processing via the RalphLoopWorker.

        Non-blocking: adds the item to the persistent SQLite queue and
        starts the background worker if not already running, then returns
        immediately with a queue reference. The work_status page can poll
        for progress using the returned queue_id.

        Args:
            description: Work description.
            work_type: Workflow type.

        Returns:
            Dict with queue_id, status, and work_type.
        """
        from spine.work.ralph_worker import get_worker

        worker = get_worker(self._config)
        queue_id = worker.enqueue(description=description, work_type=work_type)
        worker.start()  # no-op if already running
        logger.info(f"Enqueued work via RalphLoopWorker: queue_id={queue_id}")
        return {
            "queue_id": queue_id,
            "status": "pending",
            "work_type": work_type,
        }

    # ── Artifact operations ──

    def get_artifacts(self, work_id: str) -> list[dict[str, Any]]:
        """List all artifacts for a work item.

        Args:
            work_id: The work item ID.

        Returns:
            List of artifact metadata dicts.
        """
        return self._artifacts.list_artifacts(work_id)

    def read_artifact(self, work_id: str, phase: str, name: str) -> str | None:
        """Read the content of a specific artifact.

        Args:
            work_id: The work item ID.
            phase: The phase name.
            name: The artifact filename.

        Returns:
            The artifact content, or None.
        """
        return self._artifacts.load_artifact(work_id, phase, name)

    # ── Audit operations ──

    def get_audit_log(
        self,
        work_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query audit events.

        Args:
            work_id: Optional work item filter.
            event_type: Optional event type filter.
            limit: Maximum events to return.

        Returns:
            List of audit event dicts.
        """
        return self._audit.query_events(work_id=work_id, event_type=event_type, limit=limit)

    # ── Config ──

    def get_config(self) -> dict[str, Any]:
        """Return the current configuration as a dict."""
        return {
            "checkpoint_path": self._config.checkpoint_path,
            "artifact_path": self._config.artifact_path,
            "max_critic_retries": self._config.max_critic_retries,
            "work_type": self._config.work_type,
            "queue_backend": self._config.queue_backend,
            "workspace_root": self._config.workspace_root,
        }

    # ── Worker ──

    def get_worker_status(self) -> dict[str, Any]:
        """Get the RalphLoopWorker queue status.

        Returns:
            Dict with queue item counts by status.
        """
        worker = get_worker(self._config)
        return {
            "running": worker.running,
            "queue": worker.queue_status(),
        }

    # ── Queue operations ──

    def get_queue_overview(self) -> dict[str, Any]:
        """Get a combined view of the queue: pending items, active job with
        phase and timing, and recent history.

        The active job is sourced from the **queue** table (via
        ``RalphLoopWorker.get_active()``) rather than the ``work_entries``
        table.  The ``work_entries`` status can go stale because
        ``dispatcher.py`` finalises it independently of the worker's queue
        lifecycle.  The queue table is the authoritative source for the
        currently-running item.

        Phase-timing details (``current_phase``, ``created_at``,
        ``updated_at``) are enriched from the corresponding ``work_entries``
        record, so the UI can display a live phase progress bar and timing
        captions.

        Returns:
            Dict with keys: pending, active, recent, status_summary.
        """
        worker = get_worker(self._config)
        pending = worker.list_pending()
        recent = worker.list_recent_completed()

        # Active job from the queue table (authoritative source).
        active = worker.get_active()

        # Enrich with phase-timing details from work_entries.
        if active is not None:
            self._enrich_active_with_work_entry(active)

        return {
            "pending": pending,
            "active": active,
            "recent": recent,
            "status_summary": worker.queue_status(),
        }

    def _enrich_active_with_work_entry(self, active: dict[str, Any]) -> None:
        """Merge ``current_phase``, ``created_at``, ``updated_at`` from the
        corresponding ``work_entries`` record into *active* (mutated in
        place).

        Correlation logic:
        1. If the queue row already has a ``work_id`` (stored by the worker
           after ``submit_work()`` returns), use it directly.
        2. Otherwise, fall back to finding the most recently-created
           ``work_entries`` row with ``status = "running"``.  Since the
           worker processes items sequentially, there should be at most
           one running work entry at any time.
        """
        work_id = active.get("work_id")
        if not work_id:
            # Fallback: find the running work entry by status.
            entries = list_work(status="running", limit=1, config=self._config)
            if entries:
                work_id = entries[0].get("id")

        if not work_id:
            return

        entry = get_work_status(work_id, self._config)
        if entry is None:
            return

        for key in ("current_phase", "created_at", "updated_at"):
            if key in entry and entry[key]:
                active[key] = entry[key]

    # ── Resume operations ──

    def resume_work(
        self,
        work_id: str,
        human_feedback: str,
        action: str = "rework",
    ) -> dict[str, Any]:
        """Resume a work item in ``needs_review`` status.

        Non-blocking: enqueues the resume for async processing via
        RalphLoopWorker and returns immediately. The work_detail page
        can poll for progress.

        Args:
            work_id: The work item ID.
            human_feedback: The human's review input.
            action: ``"rework"`` to rerun from the flagged phase,
                ``"approve"`` to proceed without rework.

        Returns:
            Dict with queue_id, status, work_id.
        """
        from spine.work.dispatcher import resume_work as _async_resume

        # Mark the work entry as running immediately so the UI
        # shows progress right away.
        self._mark_running(work_id)

        # Run resume in the background via RalphLoopWorker thread pool
        import asyncio
        import concurrent.futures

        def _run():
            asyncio.run(_async_resume(work_id, human_feedback, action, self._config))

        # Submit to the worker's executor (or a fresh one)
        executor = getattr(get_worker(self._config), "_executor", None)
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        executor.submit(_run)

        return {
            "work_id": work_id,
            "status": "running",
            "action": action,
        }

    def _mark_running(self, work_id: str) -> None:
        """Transition a needs_review work entry back to running."""
        from spine.work.dispatcher import update_work_status

        update_work_status(work_id, TaskStatus.RUNNING.value, config=self._config)
