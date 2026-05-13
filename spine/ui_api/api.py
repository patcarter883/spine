"""SPINE UI API — sole read/write interface for Streamlit.

UI pages MUST use UIApi for all data access. Never import directly from
workflow/, phases/, or work.dispatcher. This maintains the zero-duplication
principle: CLI and UI share the same backend code paths.
"""

from __future__ import annotations

import logging
from typing import Any

from spine.config import SpineConfig
from spine.persistence.artifacts import ArtifactStore
from spine.services.audit_service import AuditService
from spine.work.dispatcher import get_work_status, list_work

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
        from spine.work.ralph_worker import get_worker

        worker = get_worker(self._config)
        return {
            "running": worker.running,
            "queue": worker.queue_status(),
        }
