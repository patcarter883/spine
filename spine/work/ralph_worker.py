"""SPINE RalphLoopWorker — background queue processor (singleton).

Processes work items from a SQLite-backed queue. The worker runs in a
daemon thread and dequeues items for execution via ``submit_work()``.

Usage:
    worker = get_worker()
    worker.enqueue(description="Fix the login bug", work_type="quick")
    worker.start()  # begins processing in background thread
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import sqlite_utils

from spine.config import SpineConfig

logger = logging.getLogger(__name__)

_WORKER_LOCK = threading.Lock()
_WORKER_INSTANCE: RalphLoopWorker | None = None


class RalphLoopWorker:
    """Background worker that dequeues and processes work items.

    Uses a SQLite queue for persistence. Runs in a daemon thread.
    Access via ``get_worker()`` — do not instantiate directly.

    Attributes:
        config: The SpineConfig instance.
        running: Whether the worker loop is active.
    """

    def __init__(self, config: SpineConfig | None = None) -> None:
        self.config = config or SpineConfig.load()
        self.config.ensure_dirs()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.running = False

    def _get_db(self) -> sqlite_utils.Database:
        """Return a fresh connection — safe across threads."""
        db_path = Path(self.config.queue_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite_utils.Database(str(db_path))

        if "queue" not in db.table_names():
            db["queue"].create(
                {
                    "id": int,
                    "description": str,
                    "work_type": str,
                    "status": str,
                    "enqueued_at": str,
                    "started_at": str,
                    "completed_at": str,
                    "result": str,
                },
                pk="id",
            )

        return db

    def enqueue(self, description: str, work_type: str = "spec") -> int:
        """Add a work item to the queue.

        Args:
            description: The work description.
            work_type: One of the valid work types.

        Returns:
            The queue item ID.
        """
        row = self._get_db()["queue"].insert(
            {
                "description": description,
                "work_type": work_type,
                "status": "pending",
                "enqueued_at": datetime.now().isoformat(),
                "started_at": "",
                "completed_at": "",
                "result": "",
            }
        )
        logger.info(f"Enqueued work item {row.last_pk}: {description[:80]}")
        return row.last_pk

    def dequeue(self) -> dict[str, Any] | None:
        """Get the next pending work item from the queue.

        Returns:
            The queue item dict, or None if queue is empty.
        """
        rows = list(
            self._get_db()["queue"].rows_where(
                "status = ?",
                ["pending"],
                order_by="enqueued_at",
                limit=1,
            )
        )
        return rows[0] if rows else None

    def start(self) -> None:
        """Start the worker loop in a background daemon thread."""
        if self.running:
            logger.warning("Worker already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.running = True
        logger.info("RalphLoopWorker started")

    def stop(self) -> None:
        """Signal the worker loop to stop."""
        self._stop_event.set()
        self.running = False
        logger.info("RalphLoopWorker stopping")

    def _loop(self) -> None:
        """Main worker loop — dequeues and processes items."""
        import asyncio

        while not self._stop_event.is_set():
            item = self.dequeue()
            if item is None:
                self._stop_event.wait(timeout=5.0)
                continue

            item_id = item["id"]
            logger.info(f"Processing queue item {item_id}")

            # Mark as started in the queue
            self._get_db()["queue"].update(
                item_id,
                {
                    "status": "running",
                    "started_at": datetime.now().isoformat(),
                },
            )

            # ── Push event to WebSocket bus ──
            try:
                from spine.ui.ws_bus import get_bus

                get_bus().publish_sync(
                    "queue_started",
                    {"queue_id": item_id, "description": item["description"][:200]},
                )
            except Exception:
                pass

            try:
                from spine.work.dispatcher import submit_work

                result = asyncio.run(
                    submit_work(
                        description=item["description"],
                        work_type=item["work_type"],
                        config=self.config,
                        created_at=item.get("enqueued_at"),
                    )
                )

                # Derive the queue item status from the work result so
                # the queue page can distinguish failed from completed.
                work_status = result.get("status", "completed") if isinstance(result, dict) else "completed"
                queue_status = work_status if work_status in ("failed", "needs_review") else "completed"

                self._get_db()["queue"].update(
                    item_id,
                    {
                        "status": queue_status,
                        "completed_at": datetime.now().isoformat(),
                        "result": json.dumps(result),
                    },
                )

                # ── Push completion event to WebSocket bus ──
                try:
                    from spine.ui.ws_bus import get_bus

                    get_bus().publish_sync(
                        "queue_completed",
                        {"queue_id": item_id, "result": result},
                    )
                except Exception:
                    pass

            except Exception as e:
                logger.error(f"Queue item {item_id} failed: {e}", exc_info=True)
                self._get_db()["queue"].update(
                    item_id,
                    {
                        "status": "failed",
                        "completed_at": datetime.now().isoformat(),
                        "result": json.dumps({"error": str(e)}),
                    },
                )

                # ── Push failure event to WebSocket bus ──
                try:
                    from spine.ui.ws_bus import get_bus

                    get_bus().publish_sync(
                        "queue_failed",
                        {"queue_id": item_id, "error": str(e)},
                    )
                except Exception:
                    pass

    def reset_stuck_items(self) -> int:
        """Reset any queue items stuck in "running" back to "pending".

        When the worker or UI dies mid-execution, items remain in "running"
        status indefinitely.  This method resets them so they can be picked
        up again on the next dequeue.

        Also purges LangGraph checkpoints for those items so the graph
        restarts cleanly from phase 0.

        Returns:
            The number of items that were reset.
        """
        from spine.persistence.checkpoint import CheckpointStore

        db = self._get_db()
        stuck = list(
            db["queue"].rows_where("status = ?", ["running"], order_by="started_at")
        )
        if not stuck:
            return 0

        checkpoint_store = CheckpointStore(db_path=self.config.checkpoint_path)
        count = 0
        for item in stuck:
            item_id = item["id"]
            # Reset queue row to pending
            db["queue"].update(
                item_id,
                {
                    "status": "pending",
                    "started_at": "",
                    "completed_at": "",
                    "result": "",
                },
            )
            # Purge any LangGraph checkpoint so the graph starts fresh
            # (must use async purge via the checkpointer)
            import asyncio

            saver = asyncio.run(checkpoint_store.get_checkpointer())
            asyncio.run(saver.apurge({"configurable": {"thread_id": str(item_id)}}))
            count += 1
            logger.info(f"Reset stuck queue item {item_id}")

        logger.info(f"reset_stuck_items: reset {count}/{len(stuck)} running items")
        return count

    def queue_status(self) -> dict[str, int]:
        """Get counts of queue items by status.

        Returns:
            A dict mapping status to count.
        """
        counts: dict[str, int] = {}
        for row in self._get_db()["queue"].rows_where():
            status = row.get("status", "unknown")
            counts[status] = counts.get(status, 0) + 1
        return counts

    def list_pending(self, limit: int = 50) -> list[dict[str, Any]]:
        """List pending queue items.

        Args:
            limit: Maximum number of items to return.

        Returns:
            List of pending queue item dicts, newest first.
        """
        rows = list(
            self._get_db()["queue"].rows_where(
                "status = ?", ["pending"], order_by="enqueued_at DESC", limit=limit,
            )
        )
        return rows

    def get_active(self) -> dict[str, Any] | None:
        """Get the currently running queue item, if any.

        Returns:
            The running queue item dict, or None.
        """
        rows = list(
            self._get_db()["queue"].rows_where(
                "status = ?", ["running"], order_by="started_at", limit=1,
            )
        )
        return rows[0] if rows else None

    def list_recent_completed(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recently completed/failed queue items.

        Args:
            limit: Maximum number of items to return.

        Returns:
            List of completed/failed queue item dicts, newest first.
        """
        rows = list(
            self._get_db()["queue"].rows_where(
                "status IN (?, ?)", ["completed", "failed"],
                order_by="completed_at DESC", limit=limit,
            )
        )
        return rows


def get_worker(config: SpineConfig | None = None) -> RalphLoopWorker:
    """Get the RalphLoopWorker singleton.

    Thread-safe — uses a lock to prevent double instantiation.

    Args:
        config: Optional SpineConfig (used only on first call).

    Returns:
        The global RalphLoopWorker instance.
    """
    global _WORKER_INSTANCE
    with _WORKER_LOCK:
        if _WORKER_INSTANCE is None:
            _WORKER_INSTANCE = RalphLoopWorker(config)
        return _WORKER_INSTANCE
