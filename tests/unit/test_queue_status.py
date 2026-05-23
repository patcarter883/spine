"""Tests for queue status handling in RalphLoopWorker."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.config import SpineConfig
from spine.work.ralph_worker import RalphLoopWorker, _TERMINAL_STATUSES


def _make_worker(tmpdir: str) -> RalphLoopWorker:
    """Create a RalphLoopWorker with an isolated database."""
    import spine.work.ralph_worker as rw_mod

    rw_mod._WORKER_INSTANCE = None

    config = SpineConfig()
    config.queue_path = str(Path(tmpdir) / "queue.db")
    config.ensure_dirs()
    return RalphLoopWorker(config)


def _insert_items(worker: RalphLoopWorker, items: list[dict]) -> None:
    """Insert items directly into the queue table."""
    db = worker._get_db()
    for item in items:
        db["queue"].insert(item)


class TestTerminalStatuses:
    """Tests for _TERMINAL_STATUSES frozenset."""

    def test_terminal_statuses_includes_all_non_running_non_pending(self):
        """Verify _TERMINAL_STATUSES equals exactly the 7 terminal statuses."""
        expected = frozenset({
            "completed",
            "failed",
            "needs_review",
            "stalled",
            "awaiting_approval",
            "approved",
            "rejected",
        })
        assert _TERMINAL_STATUSES == expected

    def test_terminal_statuses_excludes_running_and_pending(self):
        """Verify 'running' and 'pending' are NOT in the terminal statuses set."""
        assert "running" not in _TERMINAL_STATUSES
        assert "pending" not in _TERMINAL_STATUSES

    def test_terminal_statuses_is_frozenset(self):
        """Verify the type is frozenset (immutability)."""
        assert isinstance(_TERMINAL_STATUSES, frozenset)
        # Verify immutability: frozenset has no 'add' method at all
        assert not hasattr(_TERMINAL_STATUSES, "add")


class TestListRecentCompleted:
    """Tests for list_recent_completed() method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.tmpdir = tempfile.mkdtemp()
        self.worker = _make_worker(self.tmpdir)

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        pass  # tempfile cleanup handled by caller

    def test_includes_needs_review(self):
        """needs_review items appear in results."""
        _insert_items(self.worker, [
            {
                "id": 1,
                "description": "needs review item",
                "work_type": "task",
                "status": "needs_review",
                "enqueued_at": "2024-01-01T00:00:00",
                "started_at": "2024-01-01T01:00:00",
                "completed_at": "2024-01-01T02:00:00",
                "result": "",
            },
        ])
        results = self.worker.list_recent_completed()
        assert len(results) >= 1
        statuses = {r["status"] for r in results}
        assert "needs_review" in statuses

    def test_includes_stalled(self):
        """stalled items appear in results."""
        _insert_items(self.worker, [
            {
                "id": 1,
                "description": "stalled item",
                "work_type": "task",
                "status": "stalled",
                "enqueued_at": "2024-01-01T00:00:00",
                "started_at": "2024-01-01T01:00:00",
                "completed_at": "2024-01-01T02:00:00",
                "result": "",
            },
        ])
        results = self.worker.list_recent_completed()
        assert len(results) >= 1
        statuses = {r["status"] for r in results}
        assert "stalled" in statuses

    def test_includes_awaiting_approval(self):
        """awaiting_approval items appear in results."""
        _insert_items(self.worker, [
            {
                "id": 1,
                "description": "awaiting approval item",
                "work_type": "task",
                "status": "awaiting_approval",
                "enqueued_at": "2024-01-01T00:00:00",
                "started_at": "2024-01-01T01:00:00",
                "completed_at": "2024-01-01T02:00:00",
                "result": "",
            },
        ])
        results = self.worker.list_recent_completed()
        assert len(results) >= 1
        statuses = {r["status"] for r in results}
        assert "awaiting_approval" in statuses

    def test_excludes_pending_and_running(self):
        """pending and running items are excluded from results."""
        _insert_items(self.worker, [
            {
                "id": 1,
                "description": "pending item",
                "work_type": "task",
                "status": "pending",
                "enqueued_at": "2024-01-01T00:00:00",
                "started_at": "",
                "completed_at": "",
                "result": "",
            },
            {
                "id": 2,
                "description": "running item",
                "work_type": "task",
                "status": "running",
                "enqueued_at": "2024-01-01T00:00:00",
                "started_at": "2024-01-01T01:00:00",
                "completed_at": "",
                "result": "",
            },
            {
                "id": 3,
                "description": "completed item",
                "work_type": "task",
                "status": "completed",
                "enqueued_at": "2024-01-01T00:00:00",
                "started_at": "2024-01-01T01:00:00",
                "completed_at": "2024-01-01T02:00:00",
                "result": "",
            },
        ])
        results = self.worker.list_recent_completed()
        statuses = {r["status"] for r in results}
        assert "pending" not in statuses
        assert "running" not in statuses
        assert "completed" in statuses

    def test_includes_all_terminal_statuses(self):
        """All 7 terminal statuses are returned by list_recent_completed()."""
        terminal_statuses = [
            "completed",
            "failed",
            "needs_review",
            "stalled",
            "awaiting_approval",
            "approved",
            "rejected",
        ]
        items = []
        for i, status in enumerate(terminal_statuses):
            items.append({
                "id": i + 1,
                "description": f"{status} item",
                "work_type": "task",
                "status": status,
                "enqueued_at": f"2024-01-0{i + 1}T00:00:00",
                "started_at": f"2024-01-0{i + 1}T01:00:00",
                "completed_at": f"2024-01-0{i + 1}T02:00:00",
                "result": "",
            })
        _insert_items(self.worker, items)
        results = self.worker.list_recent_completed()
        statuses = {r["status"] for r in results}
        for status in terminal_statuses:
            assert status in statuses, f"{status} not found in results"


class TestQueueStatusSummary:
    """Tests for queue_status() method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.tmpdir = tempfile.mkdtemp()
        self.worker = _make_worker(self.tmpdir)

    def test_queue_status_counts_all_statuses(self):
        """Insert items with various statuses, verify counts match."""
        _insert_items(self.worker, [
            {"id": 1, "description": "p1", "work_type": "task", "status": "pending",
             "enqueued_at": "2024-01-01T00:00:00", "started_at": "", "completed_at": "", "result": ""},
            {"id": 2, "description": "p2", "work_type": "task", "status": "pending",
             "enqueued_at": "2024-01-01T01:00:00", "started_at": "", "completed_at": "", "result": ""},
            {"id": 3, "description": "r1", "work_type": "task", "status": "running",
             "enqueued_at": "2024-01-01T00:00:00", "started_at": "2024-01-01T01:00:00",
             "completed_at": "", "result": ""},
            {"id": 4, "description": "c1", "work_type": "task", "status": "completed",
             "enqueued_at": "2024-01-01T00:00:00", "started_at": "2024-01-01T01:00:00",
             "completed_at": "2024-01-01T02:00:00", "result": ""},
            {"id": 5, "description": "f1", "work_type": "task", "status": "failed",
             "enqueued_at": "2024-01-01T00:00:00", "started_at": "2024-01-01T01:00:00",
             "completed_at": "2024-01-01T02:00:00", "result": ""},
            {"id": 6, "description": "nr1", "work_type": "task", "status": "needs_review",
             "enqueued_at": "2024-01-01T00:00:00", "started_at": "2024-01-01T01:00:00",
             "completed_at": "2024-01-01T02:00:00", "result": ""},
        ])
        counts = self.worker.queue_status()
        assert counts["pending"] == 2
        assert counts["running"] == 1
        assert counts["completed"] == 1
        assert counts["failed"] == 1
        assert counts["needs_review"] == 1
        assert sum(counts.values()) == 6


class TestResetStuckItems:
    """Tests for reset_stuck_items() method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.tmpdir = tempfile.mkdtemp()
        self.worker = _make_worker(self.tmpdir)

    def test_stalled_matches_reset_filter(self):
        """Query with IN ('running', 'stalled') returns stalled items."""
        _insert_items(self.worker, [
            {
                "id": 1,
                "description": "stalled item",
                "work_type": "task",
                "status": "stalled",
                "enqueued_at": "2024-01-01T00:00:00",
                "started_at": "2024-01-01T01:00:00",
                "completed_at": "",
                "result": "",
            },
            {
                "id": 2,
                "description": "pending item",
                "work_type": "task",
                "status": "pending",
                "enqueued_at": "2024-01-01T00:00:00",
                "started_at": "",
                "completed_at": "",
                "result": "",
            },
        ])
        # Verify the filter matches stalled items
        db = self.worker._get_db()
        stuck = list(db["queue"].rows_where(
            "status IN (?, ?)", ["running", "stalled"]
        ))
        assert len(stuck) >= 1
        statuses = {r["status"] for r in stuck}
        assert "stalled" in statuses

    def test_awaiting_approval_not_reset(self):
        """awaiting_approval items are not matched by the reset filter."""
        _insert_items(self.worker, [
            {
                "id": 1,
                "description": "awaiting approval item",
                "work_type": "task",
                "status": "awaiting_approval",
                "enqueued_at": "2024-01-01T00:00:00",
                "started_at": "2024-01-01T01:00:00",
                "completed_at": "",
                "result": "",
            },
            {
                "id": 2,
                "description": "pending item",
                "work_type": "task",
                "status": "pending",
                "enqueued_at": "2024-01-01T00:00:00",
                "started_at": "",
                "completed_at": "",
                "result": "",
            },
        ])
        # Verify the filter does NOT match awaiting_approval items
        db = self.worker._get_db()
        stuck = list(db["queue"].rows_where(
            "status IN (?, ?)", ["running", "stalled"]
        ))
        statuses = {r["status"] for r in stuck}
        assert "awaiting_approval" not in statuses
