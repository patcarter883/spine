"""Tests for list_work() ordering behavior in dispatcher."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.config import SpineConfig
from spine.work.dispatcher import _get_work_db, list_work


class TestListWorkOrdering:
    """Tests for list_work() ordering (-created_at, newest first)."""

    def _setup_work_entries(self, tmpdir: str) -> SpineConfig:
        """Create a SpineConfig with an isolated work_entries.db."""
        config = SpineConfig()
        config.queue_path = str(Path(tmpdir) / "queue.db")  # work_entries.db is parent/queue.db
        config.ensure_dirs()
        db = _get_work_db(config)
        db["work_entries"].insert_all([
            {
                "id": "work-1",
                "description": "oldest work",
                "work_type": "quick",
                "status": "completed",
                "current_phase": "verify",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T01:00:00",
                "result": '{"status": "completed"}',
            },
            {
                "id": "work-2",
                "description": "middle work",
                "work_type": "spec",
                "status": "completed",
                "current_phase": "verify",
                "created_at": "2024-06-15T12:00:00",
                "updated_at": "2024-06-15T13:00:00",
                "result": '{"status": "completed"}',
            },
            {
                "id": "work-3",
                "description": "newest work",
                "work_type": "quick",
                "status": "running",
                "current_phase": "implement",
                "created_at": "2024-12-31T23:59:59",
                "updated_at": "2024-12-31T23:59:59",
                "result": "",
            },
        ])
        return config

    def _setup_one_entry(self, tmpdir: str, status: str = "completed") -> SpineConfig:
        """Create config with a single work entry."""
        config = SpineConfig()
        config.queue_path = str(Path(tmpdir) / "queue.db")
        config.ensure_dirs()
        db = _get_work_db(config)
        db["work_entries"].insert({
            "id": "work-1",
            "description": "only work",
            "work_type": "quick",
            "status": status,
            "current_phase": "verify",
            "created_at": "2024-06-15T12:00:00",
            "updated_at": "2024-06-15T13:00:00",
            "result": '{"status": "completed"}',
        })
        return config

    def test_empty_db_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpineConfig()
            config.queue_path = str(Path(tmpdir) / "queue.db")
            config.ensure_dirs()
            results = list_work(config=config)
            assert results == []

    def test_single_item_returns_that_item(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._setup_one_entry(tmpdir)
            results = list_work(config=config)
            assert len(results) == 1
            assert results[0]["id"] == "work-1"

    def test_multiple_items_ordered_newest_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._setup_work_entries(tmpdir)
            results = list_work(config=config)
            assert len(results) == 3
            timestamps = [item["created_at"] for item in results]
            assert timestamps == sorted(timestamps, reverse=True)
            # Explicit order check
            assert timestamps == [
                "2024-12-31T23:59:59",
                "2024-06-15T12:00:00",
                "2024-01-01T00:00:00",
            ]

    def test_filtered_by_status_returns_newest_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._setup_work_entries(tmpdir)
            results = list_work(status="completed", config=config)
            assert len(results) == 2
            timestamps = [item["created_at"] for item in results]
            assert timestamps == sorted(timestamps, reverse=True)
            assert timestamps == [
                "2024-06-15T12:00:00",
                "2024-01-01T00:00:00",
            ]

    def test_respects_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._setup_work_entries(tmpdir)
            results = list_work(limit=2, config=config)
            assert len(results) == 2
            # Should be the two newest
            assert results[0]["id"] == "work-3"
            assert results[1]["id"] == "work-2"

    def test_no_results_for_nonexistent_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._setup_work_entries(tmpdir)
            results = list_work(status="nonexistent", config=config)
            assert results == []
