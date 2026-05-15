"""Tests for UIApi.get_queue_overview() ordering behavior."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.config import SpineConfig
from spine.ui_api.api import UIApi
from spine.work.ralph_worker import RalphLoopWorker


def _reset_worker_singleton() -> None:
    """Reset the RalphLoopWorker singleton for test isolation."""
    import spine.work.ralph_worker as rw_mod
    rw_mod._WORKER_INSTANCE = None


def _init_queue_db(config: SpineConfig) -> RalphLoopWorker:
    """Create a fresh worker with a queue DB and return it."""
    _reset_worker_singleton()
    worker = RalphLoopWorker(config)
    # _get_db() creates the table if it doesn't exist
    worker._get_db()
    return worker


class TestUIApiGetQueueOverviewOrdering:
    """Tests for get_queue_overview() ordering."""

    def test_pending_ordered_newest_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpineConfig()
            config.queue_path = str(Path(tmpdir) / "queue.db")
            config.ensure_dirs()

            worker = _init_queue_db(config)
            db = worker._get_db()
            db["queue"].insert_all([
                {
                    "id": 1,
                    "description": "oldest",
                    "work_type": "quick",
                    "status": "pending",
                    "enqueued_at": "2024-01-01T00:00:00",
                    "started_at": "",
                    "completed_at": "",
                    "result": "",
                },
                {
                    "id": 2,
                    "description": "newest",
                    "work_type": "quick",
                    "status": "pending",
                    "enqueued_at": "2024-12-31T23:59:59",
                    "started_at": "",
                    "completed_at": "",
                    "result": "",
                },
                {
                    "id": 3,
                    "description": "middle",
                    "work_type": "quick",
                    "status": "pending",
                    "enqueued_at": "2024-06-15T12:00:00",
                    "started_at": "",
                    "completed_at": "",
                    "result": "",
                },
            ])

            api = UIApi(config)
            overview = api.get_queue_overview()
            pending = overview["pending"]
            assert len(pending) == 3
            timestamps = [item["enqueued_at"] for item in pending]
            assert timestamps == sorted(timestamps, reverse=True)
            assert timestamps == [
                "2024-12-31T23:59:59",
                "2024-06-15T12:00:00",
                "2024-01-01T00:00:00",
            ]

    def test_recent_completed_ordered_newest_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpineConfig()
            config.queue_path = str(Path(tmpdir) / "queue.db")
            config.ensure_dirs()

            worker = _init_queue_db(config)
            db = worker._get_db()
            db["queue"].insert_all([
                {
                    "id": 1,
                    "description": "old completed",
                    "work_type": "quick",
                    "status": "completed",
                    "enqueued_at": "2024-01-01T00:00:00",
                    "started_at": "",
                    "completed_at": "2024-01-01T01:00:00",
                    "result": "",
                },
                {
                    "id": 2,
                    "description": "new completed",
                    "work_type": "quick",
                    "status": "completed",
                    "enqueued_at": "2024-06-01T00:00:00",
                    "started_at": "",
                    "completed_at": "2024-06-01T01:00:00",
                    "result": "",
                },
            ])

            api = UIApi(config)
            overview = api.get_queue_overview()
            recent = overview["recent"]
            assert len(recent) == 2
            timestamps = [item["completed_at"] for item in recent]
            assert timestamps == sorted(timestamps, reverse=True)
            assert timestamps == [
                "2024-06-01T01:00:00",
                "2024-01-01T01:00:00",
            ]

    def test_empty_queue_returns_empty_lists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpineConfig()
            config.queue_path = str(Path(tmpdir) / "queue.db")
            config.ensure_dirs()

            _init_queue_db(config)
            api = UIApi(config)
            overview = api.get_queue_overview()
            assert overview["pending"] == []
            assert overview["recent"] == []
            assert overview["status_summary"] == {}

    def test_pending_and_recent_are_separate(self):
        """Verify that pending and recent lists don't overlap."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpineConfig()
            config.queue_path = str(Path(tmpdir) / "queue.db")
            config.ensure_dirs()

            worker = _init_queue_db(config)
            db = worker._get_db()
            db["queue"].insert_all([
                {
                    "id": 1,
                    "description": "pending item",
                    "work_type": "quick",
                    "status": "pending",
                    "enqueued_at": "2024-06-15T12:00:00",
                    "started_at": "",
                    "completed_at": "",
                    "result": "",
                },
                {
                    "id": 2,
                    "description": "completed item",
                    "work_type": "quick",
                    "status": "completed",
                    "enqueued_at": "2024-01-01T00:00:00",
                    "started_at": "",
                    "completed_at": "2024-01-01T01:00:00",
                    "result": "",
                },
            ])

            api = UIApi(config)
            overview = api.get_queue_overview()
            assert len(overview["pending"]) == 1
            assert len(overview["recent"]) == 1
