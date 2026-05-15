"""Tests for work item ordering on the dashboard.

Verifies that ``list_work()`` returns items in strict newest-first order
with proper tiebreaker and NULL handling behaviour.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spine.config import SpineConfig
from spine.work.dispatcher import _get_work_db, list_work


class TestListWorkOrdering:
    """Ordering edge cases for ``list_work()``."""

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _make_config(tmpdir: str) -> SpineConfig:
        """Create a SpineConfig with an isolated work_entries.db."""
        config = SpineConfig()
        config.queue_path = str(Path(tmpdir) / "queue.db")
        config.ensure_dirs()
        return config

    @staticmethod
    def _insert_entries(
        config: SpineConfig,
        entries: list[dict],
    ) -> None:
        """Insert entries into the work_entries table."""
        db = _get_work_db(config)
        db["work_entries"].insert_all(entries)

    # ── Test cases ────────────────────────────────────────────────────────

    def test_list_work_default_order(self):
        """Items with increasing timestamps are returned newest-first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            self._insert_entries(config, [
                {
                    "id": "work-1",
                    "description": "oldest",
                    "work_type": "quick",
                    "status": "completed",
                    "current_phase": "verify",
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T01:00:00",
                    "result": "{}",
                },
                {
                    "id": "work-2",
                    "description": "middle",
                    "work_type": "spec",
                    "status": "completed",
                    "current_phase": "verify",
                    "created_at": "2024-06-15T12:00:00",
                    "updated_at": "2024-06-15T13:00:00",
                    "result": "{}",
                },
                {
                    "id": "work-3",
                    "description": "newest",
                    "work_type": "quick",
                    "status": "running",
                    "current_phase": "implement",
                    "created_at": "2024-12-31T23:59:59",
                    "updated_at": "2024-12-31T23:59:59",
                    "result": "",
                },
            ])

            results = list_work(config=config)
            assert len(results) == 3
            ids = [r["id"] for r in results]
            assert ids == ["work-3", "work-2", "work-1"], f"Expected newest-first, got {ids}"

    def test_list_work_filtered_order(self):
        """Filtered results are still returned newest-first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            self._insert_entries(config, [
                {
                    "id": "work-1",
                    "description": "completed oldest",
                    "work_type": "quick",
                    "status": "completed",
                    "current_phase": "verify",
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T01:00:00",
                    "result": "{}",
                },
                {
                    "id": "work-2",
                    "description": "running only",
                    "work_type": "spec",
                    "status": "running",
                    "current_phase": "implement",
                    "created_at": "2024-06-15T12:00:00",
                    "updated_at": "2024-06-15T13:00:00",
                    "result": "",
                },
                {
                    "id": "work-3",
                    "description": "completed newest",
                    "work_type": "quick",
                    "status": "completed",
                    "current_phase": "verify",
                    "created_at": "2024-12-31T23:59:59",
                    "updated_at": "2024-12-31T23:59:59",
                    "result": "{}",
                },
            ])

            results = list_work(status="completed", config=config)
            assert len(results) == 2
            ids = [r["id"] for r in results]
            assert ids == ["work-3", "work-1"], f"Expected newest-first, got {ids}"

    def test_list_work_limit(self):
        """Limit returns only the N newest items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            self._insert_entries(config, [
                {
                    "id": f"work-{i}",
                    "description": f"entry {i}",
                    "work_type": "quick",
                    "status": "completed",
                    "current_phase": "verify",
                    "created_at": f"2024-01-{i+1:02d}T00:00:00",
                    "updated_at": f"2024-01-{i+1:02d}T01:00:00",
                    "result": "{}",
                }
                for i in range(1, 11)  # 10 entries, created_at Jan-01 through Jan-10
            ])

            results = list_work(limit=3, config=config)
            assert len(results) == 3
            # Should return the 3 newest: Jan-10, Jan-09, Jan-08
            ids = [r["id"] for r in results]
            assert ids == ["work-10", "work-9", "work-8"], f"Expected 3 newest, got {ids}"

    def test_list_work_same_timestamp(self):
        """Items with identical created_at use rowid as tiebreaker.

        When two items have the same timestamp, the one with the higher
        rowid (inserted later) should come first.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            # Insert in reverse expected order to prove rowid tiebreak
            self._insert_entries(config, [
                {
                    "id": "work-older",
                    "description": "inserted first",
                    "work_type": "quick",
                    "status": "completed",
                    "current_phase": "verify",
                    "created_at": "2024-06-15T12:00:00",
                    "updated_at": "2024-06-15T13:00:00",
                    "result": "{}",
                },
                {
                    "id": "work-newer",
                    "description": "inserted second — higher rowid",
                    "work_type": "spec",
                    "status": "completed",
                    "current_phase": "verify",
                    "created_at": "2024-06-15T12:00:00",  # same timestamp
                    "updated_at": "2024-06-15T13:00:00",
                    "result": "{}",
                },
            ])

            results = list_work(config=config)
            assert len(results) == 2
            ids = [r["id"] for r in results]
            # Higher rowid (inserted second) should sort first
            assert ids == ["work-newer", "work-older"], (
                f"Expected higher-rowid first (tiebreaker), got {ids}"
            )

    def test_list_work_null_created_at(self):
        """Items with NULL created_at appear at the end of the list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            self._insert_entries(config, [
                {
                    "id": "work-dated",
                    "description": "has timestamp",
                    "work_type": "quick",
                    "status": "completed",
                    "current_phase": "verify",
                    "created_at": "2024-06-15T12:00:00",
                    "updated_at": "2024-06-15T13:00:00",
                    "result": "{}",
                },
                {
                    "id": "work-null",
                    "description": "no timestamp",
                    "work_type": "spec",
                    "status": "completed",
                    "current_phase": "verify",
                    "created_at": None,
                    "updated_at": "2024-06-15T13:00:00",
                    "result": "{}",
                },
                {
                    "id": "work-older",
                    "description": "older timestamp",
                    "work_type": "quick",
                    "status": "completed",
                    "current_phase": "verify",
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T01:00:00",
                    "result": "{}",
                },
            ])

            results = list_work(config=config)
            assert len(results) == 3
            ids = [r["id"] for r in results]
            # Items with timestamps sort newest-first; NULL at end
            assert ids == ["work-dated", "work-older", "work-null"], (
                f"Expected NULL at end, got {ids}"
            )

    def test_list_work_empty(self):
        """Empty table returns an empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            results = list_work(config=config)
            assert results == []
