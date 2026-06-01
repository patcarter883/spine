"""Tests for RalphLoopWorker ordering behavior."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.config import SpineConfig
from spine.work.ralph_worker import RalphLoopWorker


def _fresh_worker(tmpdir: str) -> RalphLoopWorker:
    """Create an isolated RalphLoopWorker (resets the singleton)."""
    import spine.work.ralph_worker as rw_mod

    rw_mod._WORKER_INSTANCE = None
    config = SpineConfig()
    config.queue_path = str(Path(tmpdir) / "queue.db")
    config.checkpoint_path = str(Path(tmpdir) / "spine.db")
    config.ensure_dirs()
    return RalphLoopWorker(config)


class TestCancelRunning:
    """cancel_running() must actually flip a running queue row to cancelled.

    Regression: ``rows_where(...)`` returns a generator, so the previous
    ``item = item[0] if item else None`` raised ``TypeError`` (a generator is
    truthy but not subscriptable) on every call — silently breaking Stop Work
    for a running job.
    """

    def test_flips_running_row_to_cancelled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = _fresh_worker(tmpdir)
            db = worker._get_db()
            db["queue"].insert(
                {
                    "id": 1,
                    "description": "job",
                    "work_type": "task",
                    "status": "running",
                    "enqueued_at": "2024-01-01T00:00:00",
                    "started_at": "2024-01-01T00:00:01",
                    "completed_at": "",
                    "result": "",
                    "work_id": "wk-1",
                }
            )

            # purge_checkpoint=False avoids touching a non-existent checkpoint DB.
            assert worker.cancel_running("wk-1", purge_checkpoint=False) is True
            assert db["queue"].get(1)["status"] == "cancelled"

    def test_returns_false_when_no_running_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = _fresh_worker(tmpdir)
            worker._get_db()  # ensure the table exists
            assert worker.cancel_running("nope", purge_checkpoint=False) is False


class TestRalphLoopWorkerListPendingOrdering:
    """Tests for list_pending() ordering (enqueued_at DESC, newest first)."""

    def _make_worker(self, tmpdir: str) -> RalphLoopWorker:
        """Create a RalphLoopWorker with an isolated database."""
        # Reset singleton for test isolation
        import spine.work.ralph_worker as rw_mod

        rw_mod._WORKER_INSTANCE = None

        config = SpineConfig()
        config.queue_path = str(Path(tmpdir) / "queue.db")
        config.ensure_dirs()
        return RalphLoopWorker(config)

    def _insert_pending(self, worker: RalphLoopWorker, items: list[dict]) -> None:
        """Insert pending queue items directly into the database."""
        db = worker._get_db()
        for item in items:
            db["queue"].insert(item)

    def test_empty_queue_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = self._make_worker(tmpdir)
            pending = worker.list_pending()
            assert pending == []

    def test_single_item_returns_that_item(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = self._make_worker(tmpdir)
            self._insert_pending(
                worker,
                [
                    {
                        "id": 1,
                        "description": "only item",
                        "work_type": "task",
                        "status": "pending",
                        "enqueued_at": "2024-06-15T12:00:00",
                        "started_at": "",
                        "completed_at": "",
                        "result": "",
                    },
                ],
            )
            pending = worker.list_pending()
            assert len(pending) == 1
            assert pending[0]["id"] == 1

    def test_multiple_items_ordered_newest_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = self._make_worker(tmpdir)
            self._insert_pending(
                worker,
                [
                    {
                        "id": 1,
                        "description": "oldest",
                        "work_type": "task",
                        "status": "pending",
                        "enqueued_at": "2024-01-01T00:00:00",
                        "started_at": "",
                        "completed_at": "",
                        "result": "",
                    },
                    {
                        "id": 2,
                        "description": "middle",
                        "work_type": "task",
                        "status": "pending",
                        "enqueued_at": "2024-06-15T12:00:00",
                        "started_at": "",
                        "completed_at": "",
                        "result": "",
                    },
                    {
                        "id": 3,
                        "description": "newest",
                        "work_type": "task",
                        "status": "pending",
                        "enqueued_at": "2024-12-31T23:59:59",
                        "started_at": "",
                        "completed_at": "",
                        "result": "",
                    },
                ],
            )
            pending = worker.list_pending()
            assert len(pending) == 3
            timestamps = [item["enqueued_at"] for item in pending]
            assert timestamps == sorted(timestamps, reverse=True)
            # Explicit order check
            assert timestamps == [
                "2024-12-31T23:59:59",
                "2024-06-15T12:00:00",
                "2024-01-01T00:00:00",
            ]

    def test_excludes_non_pending_items(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = self._make_worker(tmpdir)
            self._insert_pending(
                worker,
                [
                    {
                        "id": 1,
                        "description": "pending item",
                        "work_type": "task",
                        "status": "pending",
                        "enqueued_at": "2024-06-01T00:00:00",
                        "started_at": "",
                        "completed_at": "",
                        "result": "",
                    },
                    {
                        "id": 2,
                        "description": "completed item",
                        "work_type": "task",
                        "status": "completed",
                        "enqueued_at": "2024-01-01T00:00:00",
                        "started_at": "",
                        "completed_at": "",
                        "result": "",
                    },
                    {
                        "id": 3,
                        "description": "failed item",
                        "work_type": "task",
                        "status": "failed",
                        "enqueued_at": "2024-01-01T00:00:00",
                        "started_at": "",
                        "completed_at": "",
                        "result": "",
                    },
                ],
            )
            pending = worker.list_pending()
            assert len(pending) == 1
            assert pending[0]["id"] == 1

    def test_list_pending_respects_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = self._make_worker(tmpdir)
            self._insert_pending(
                worker,
                [
                    {
                        "id": i,
                        "description": f"item {i}",
                        "work_type": "task",
                        "status": "pending",
                        "enqueued_at": f"2024-06-{i:02d}T12:00:00",
                        "started_at": "",
                        "completed_at": "",
                        "result": "",
                    }
                    for i in range(1, 11)
                ],
            )
            pending = worker.list_pending(limit=3)
            assert len(pending) == 3


class TestRalphLoopWorkerListRecentCompletedOrdering:
    """Tests for list_recent_completed() ordering (completed_at DESC, newest first)."""

    def _make_worker(self, tmpdir: str) -> RalphLoopWorker:
        import spine.work.ralph_worker as rw_mod

        rw_mod._WORKER_INSTANCE = None

        config = SpineConfig()
        config.queue_path = str(Path(tmpdir) / "queue.db")
        config.ensure_dirs()
        return RalphLoopWorker(config)

    def test_empty_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = self._make_worker(tmpdir)
            recent = worker.list_recent_completed()
            assert recent == []

    def test_completed_items_ordered_newest_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = self._make_worker(tmpdir)
            db = worker._get_db()
            db["queue"].insert_all(
                [
                    {
                        "id": 1,
                        "description": "old completed",
                        "work_type": "task",
                        "status": "completed",
                        "enqueued_at": "2024-01-01T00:00:00",
                        "started_at": "",
                        "completed_at": "2024-01-01T01:00:00",
                        "result": "",
                    },
                    {
                        "id": 2,
                        "description": "new completed",
                        "work_type": "task",
                        "status": "completed",
                        "enqueued_at": "2024-06-01T00:00:00",
                        "started_at": "",
                        "completed_at": "2024-06-01T01:00:00",
                        "result": "",
                    },
                ]
            )
            recent = worker.list_recent_completed()
            assert len(recent) == 2
            timestamps = [item["completed_at"] for item in recent]
            assert timestamps == sorted(timestamps, reverse=True)

    def test_includes_failed_items(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            worker = self._make_worker(tmpdir)
            db = worker._get_db()
            db["queue"].insert_all(
                [
                    {
                        "id": 1,
                        "description": "failed item",
                        "work_type": "task",
                        "status": "failed",
                        "enqueued_at": "2024-01-01T00:00:00",
                        "started_at": "",
                        "completed_at": "2024-01-01T02:00:00",
                        "result": '{"error": "bad"}',
                    },
                    {
                        "id": 2,
                        "description": "completed item",
                        "work_type": "task",
                        "status": "completed",
                        "enqueued_at": "2024-01-01T00:00:00",
                        "started_at": "",
                        "completed_at": "2024-01-01T01:00:00",
                        "result": "",
                    },
                ]
            )
            recent = worker.list_recent_completed()
            assert len(recent) == 2
