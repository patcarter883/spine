"""SPINE RalphLoopWorker — background queue processor (singleton).

Processes work items from a SQLite-backed queue. The worker runs in a
daemon thread and dequeues items for execution via ``submit_work()``.

Usage:
    worker = get_worker()
    worker.enqueue(description="Fix the login bug", work_type="task")
    worker.start()  # begins processing in background thread
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import sqlite_utils

from spine.config import SpineConfig
from spine.models.enums import TaskStatus
from spine.persistence.sqlite_tuning import retry_on_locked, tune_connection

_TERMINAL_STATUSES: frozenset[str] = frozenset(
    s.value for s in TaskStatus
    if s.value not in ("pending", "running")
)

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
        self._executor: Any = None
        self._executor_lock = threading.Lock()
        self.running = False

    def is_alive(self) -> bool:
        """Whether the processing-loop thread is currently alive.

        This is the authoritative signal for "is the worker running" —
        unlike the ``running`` flag it cannot go stale if the daemon
        thread dies (e.g. an unhandled error), so the queue page reports
        the real state rather than a remembered one.
        """
        return self._thread is not None and self._thread.is_alive()

    def get_executor(self) -> Any:
        """Return the shared background executor for out-of-band jobs.

        Restart and resume run outside the queue loop. They previously
        each spawned a throwaway ``ThreadPoolExecutor`` that was never
        shut down; routing them through one persistent, worker-owned
        pool keeps those jobs tracked alongside the worker.
        """
        import concurrent.futures

        # Double-checked locking: concurrent restart/resume calls (Streamlit
        # re-runs across threads) must not each build a separate pool and
        # leak all but one.
        if self._executor is None:
            with self._executor_lock:
                if self._executor is None:
                    self._executor = concurrent.futures.ThreadPoolExecutor(
                        max_workers=4, thread_name_prefix="ralph-oob"
                    )
        return self._executor

    def _get_db(self) -> sqlite_utils.Database:
        """Return a fresh connection — safe across threads."""
        db_path = Path(self.config.queue_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db = tune_connection(sqlite_utils.Database(str(db_path)))

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
                    "work_id": str,
                },
                pk="id",
            )
        else:
            # Migrate: add work_id column if missing on existing DBs.
            queue_table = db["queue"]
            existing_cols = {c.name for c in queue_table.columns}
            if "work_id" not in existing_cols and hasattr(queue_table, "add_column"):
                queue_table.add_column("work_id", str)  # type: ignore[union-attr]

        return db

    def enqueue(self, description: str, work_type: str = "task") -> int:
        """Add a work item to the queue.

        Args:
            description: The work description.
            work_type: One of the valid work types.

        Returns:
            The queue item ID.
        """
        row = retry_on_locked(
            lambda: self._get_db()["queue"].insert(
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
        )
        logger.info(f"Enqueued work item {row.last_pk}: {description[:80]}")
        return row.last_pk

    def dequeue(self) -> dict[str, Any] | None:
        """Atomically claim the next pending work item.

        Uses a single UPDATE … WHERE status='pending' … RETURNING so that
        concurrent callers (e.g. a stale daemon thread racing a freshly
        restarted worker) can never both claim the same row.  SQLite
        serialises writers, so whichever connection executes the UPDATE first
        wins; the other sees zero rows returned and gets None.

        Returns:
            The claimed queue item dict (with status already set to
            ``"running"``), or None if the queue is empty.
        """
        db = self._get_db()
        started_at = datetime.now().isoformat()
        work_id = str(uuid.uuid4())[:8]

        # The whole claim is wrapped so a writer-vs-writer lock-upgrade BUSY
        # (which bypasses busy_timeout) retries the atomic UPDATE rather than
        # surfacing as ``database is locked``. The UPDATE … WHERE status =
        # 'pending' is idempotent under retry: re-running only ever claims a
        # still-pending row, never the one a prior attempt already claimed.
        def _claim() -> dict[str, Any] | None:
            cursor = db.execute(
                """
                UPDATE queue
                   SET status     = 'running',
                       started_at = ?,
                       work_id    = ?
                 WHERE id = (
                       SELECT id FROM queue
                        WHERE status = 'pending'
                        ORDER BY enqueued_at
                        LIMIT 1
                 )
                RETURNING *
                """,
                [started_at, work_id],
            )
            row = cursor.fetchone()
            if row is None:
                return None
            cols = [d[0] for d in cursor.description]
            db.conn.commit()
            return dict(zip(cols, row))

        return retry_on_locked(_claim)

    def _dequeue_work_id(self, item: dict[str, Any]) -> str:
        """Return the work_id that was stamped by :meth:`dequeue`."""
        return item.get("work_id") or str(uuid.uuid4())[:8]

    def start(self) -> None:
        """Start the worker loop in a background daemon thread.

        Idempotent and self-healing: guards on actual thread liveness, so
        a worker whose thread previously died is restarted rather than
        left permanently wedged by a stale ``running`` flag.
        """
        if self.is_alive():
            logger.debug("Worker already running")
            self.running = True
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="ralph-worker"
        )
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
            # work_id was stamped atomically by dequeue(); no separate update needed.
            work_id = self._dequeue_work_id(item)
            logger.info(f"Processing queue item {item_id}")

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
                        work_id=work_id,
                    )
                )

                # Derive the queue item status from the work result so
                # the queue page can distinguish failed from completed.
                work_status = (
                    result.get("status", "completed") if isinstance(result, dict) else "completed"
                )
                queue_status = (
                    work_status if work_status in _TERMINAL_STATUSES else "completed"
                )
                work_id = result.get("work_id") if isinstance(result, dict) else None

                update_payload: dict[str, Any] = {
                    "status": queue_status,
                    "completed_at": datetime.now().isoformat(),
                    "result": json.dumps(result),
                }
                if work_id:
                    update_payload["work_id"] = work_id

                retry_on_locked(
                    lambda: self._get_db()["queue"].update(item_id, update_payload)
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
                # Capture the message in a local: ``e`` is deleted when the
                # except block exits, and closing the retry lambda over it would
                # be both fragile and flagged as an undefined name.
                error_text = str(e)
                retry_on_locked(
                    lambda: self._get_db()["queue"].update(
                        item_id,
                        {
                            "status": "failed",
                            "completed_at": datetime.now().isoformat(),
                            "result": json.dumps({"error": error_text}),
                        },
                    )
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
        """Reset any queue items stuck in "running" or "stalled" back to "pending".

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
        stuck = list(db["queue"].rows_where(
            "status IN (?, ?)", ["running", "stalled"], order_by="started_at"
        ))
        if not stuck:
            return 0

        checkpoint_store = CheckpointStore(db_path=self.config.checkpoint_path)
        count = 0
        for item in stuck:
            item_id = item["id"]
            # Reset queue row to pending
            retry_on_locked(
                lambda item_id=item_id: db["queue"].update(
                    item_id,
                    {
                        "status": "pending",
                        "started_at": "",
                        "completed_at": "",
                        "result": "",
                    },
                )
            )
            # Purge any LangGraph checkpoint so the graph starts fresh.
            # Best-effort: a purge failure (no checkpoint, version skew)
            # must not abort the queue-row reset, which is the point.
            import asyncio

            try:
                saver = asyncio.run(checkpoint_store.get_checkpointer())
                asyncio.run(saver.adelete_thread(str(item_id)))
            except Exception:
                logger.warning(
                    "Checkpoint purge failed for queue item %s (continuing)",
                    item_id,
                    exc_info=True,
                )
            count += 1
            logger.info(f"Reset stuck queue item {item_id}")

        logger.info(f"reset_stuck_items: reset {count}/{len(stuck)} running/stalled items")
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
                "status = ?",
                ["pending"],
                order_by="enqueued_at DESC",
                limit=limit,
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
                "status = ?",
                ["running"],
                order_by="started_at",
                limit=1,
            )
        )
        return rows[0] if rows else None

    def list_running(self) -> list[dict[str, Any]]:
        """List every queue item currently in the ``running`` state.

        Normally there is at most one (the loop is sequential), but the
        caller correlates these against ``work_entries`` to build the
        full active set, so return all of them.

        Returns:
            List of running queue item dicts, oldest first.
        """
        return list(
            self._get_db()["queue"].rows_where(
                "status = ?",
                ["running"],
                order_by="started_at",
            )
        )

    def list_recent_completed(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recently terminated queue items (all non-pending, non-running statuses).

        Args:
            limit: Maximum number of items to return.

        Returns:
            List of completed/failed queue item dicts, newest first.
        """
        rows = list(
            self._get_db()["queue"].rows_where(
                "status NOT IN (?, ?)",
                ["pending", "running"],
                order_by="completed_at DESC",
                limit=limit,
            )
        )
        return rows

    def cancel_item(self, item_id: int) -> bool:
        """Cancel a pending queue item.

        Args:
            item_id: The queue item ID to cancel.

        Returns:
            True if the item was cancelled, False if not found or not pending.
        """
        db = self._get_db()
        item = db["queue"].get(item_id)
        if not item:
            return False

        if item["status"] != "pending":
            return False

        retry_on_locked(
            lambda: db["queue"].update(
                item_id,
                {
                    "status": "cancelled",
                    "completed_at": datetime.now().isoformat(),
                    "result": json.dumps({"cancelled": True}),
                },
            )
        )
        logger.info(f"Cancelled queue item {item_id}")
        return True

    def cancel_running(self, work_id: str, purge_checkpoint: bool = True) -> bool:
        """Cancel a running work item by work_id.

        Marks the queue item as cancelled and (optionally) purges the LangGraph
        checkpoint so the graph can be restarted cleanly if needed.

        Args:
            work_id: The work ID to cancel.
            purge_checkpoint: When True, delete the LangGraph checkpoint thread.
                Pass False for a cooperative stop of a job that is still
                streaming (e.g. onboarding): deleting the checkpoint out from
                under a live run only races with its in-flight superstep write —
                the run halts at its next node boundary on its own, and a later
                reset purges any stale checkpoint.

        Returns:
            True if a running item was cancelled, False if not found.
        """
        db = self._get_db()
        # Find the queue item by work_id. ``rows_where`` returns a generator —
        # materialise it before indexing (a generator is always truthy and not
        # subscriptable, so ``rows[0] if rows`` on the raw generator raised
        # TypeError and silently broke every running-job cancel).
        rows = list(
            db["queue"].rows_where(
                "work_id = ? AND status = ?",
                [work_id, "running"],
                limit=1,
            )
        )
        item = rows[0] if rows else None

        if not item:
            return False

        item_id = item["id"]
        # Mark as cancelled
        retry_on_locked(
            lambda: db["queue"].update(
                item_id,
                {
                    "status": "cancelled",
                    "completed_at": datetime.now().isoformat(),
                    "result": json.dumps({"cancelled": True, "work_id": work_id}),
                },
            )
        )

        if purge_checkpoint:
            self._purge_checkpoint(work_id)

        logger.info(f"Cancelled running work {work_id} (queue item {item_id})")
        return True

    def _purge_checkpoint(self, work_id: str) -> None:
        """Best-effort delete of a work item's LangGraph checkpoint thread.

        Runs the async getter + delete inside ONE event loop. The previous code
        called ``asyncio.run`` twice, so the saver built in the first loop was
        bound to a loop that was already closed before ``adelete_thread`` ran in
        the second — every purge silently failed. Failures here are non-fatal.
        """
        from spine.persistence.checkpoint import CheckpointStore
        import asyncio

        async def _purge() -> None:
            store = CheckpointStore(db_path=self.config.checkpoint_path)
            saver = await store.get_checkpointer()
            await saver.adelete_thread(work_id)

        try:
            asyncio.run(_purge())
        except Exception:
            logger.debug("Checkpoint purge for %s failed (continuing)", work_id, exc_info=True)


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
