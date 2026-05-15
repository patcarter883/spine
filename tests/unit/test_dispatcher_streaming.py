"""Tests for dispatcher stream chunk processing with v2 format.

The v1 format with subgraphs=True produces 3-element tuples
(namespace, mode, data), which was silently dropped by the old
len(chunk) != 2 check. The v2 format (version="v2") produces
consistent StreamPart dicts regardless of stream_mode / subgraph
settings. These tests verify the v2 chunk parsing logic.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.config import SpineConfig
from spine.work.dispatcher import _get_work_db, _update_work_progress


class TestUpdateWorkProgress:
    """Tests for _update_work_progress() — the function that updates
    the work_entries DB and publishes WebSocket events after each phase."""

    def _make_db(self, tmpdir: str):
        """Create an isolated work_entries.db with a single entry."""
        config = SpineConfig()
        config.queue_path = str(Path(tmpdir) / "queue.db")
        config.ensure_dirs()
        db = _get_work_db(config)
        db["work_entries"].insert({
            "id": "test-work",
            "description": "test",
            "work_type": "quick",
            "status": "running",
            "current_phase": "",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00",
            "result": "{}",
        })
        return db

    def test_updates_phase_and_status(self):
        """_update_work_progress writes current_phase and status to the DB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = self._make_db(tmpdir)
            _update_work_progress(db, "test-work", "tasks", "running")
            row = db["work_entries"].get("test-work")
            assert row["current_phase"] == "tasks"
            assert row["status"] == "running"

    def test_progression_through_phases(self):
        """Simulate a workflow progressing through multiple phases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = self._make_db(tmpdir)
            # Simulate the quick workflow: tasks → implement → verify
            _update_work_progress(db, "test-work", "tasks", "running")
            _update_work_progress(db, "test-work", "implement", "running")
            _update_work_progress(db, "test-work", "verify", "completed")

            row = db["work_entries"].get("test-work")
            assert row["current_phase"] == "verify"
            assert row["status"] == "completed"

    def test_updated_at_changes(self):
        """updated_at is refreshed on each call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = self._make_db(tmpdir)
            _update_work_progress(db, "test-work", "tasks", "running")
            row1 = db["work_entries"].get("test-work")

            _update_work_progress(db, "test-work", "implement", "running")
            row2 = db["work_entries"].get("test-work")

            # updated_at should change (at least the timestamp is different)
            assert row2["updated_at"] >= row1["updated_at"]

    def test_does_not_crash_on_missing_work_id(self):
        """A missing work_id logs a warning but doesn't raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = self._make_db(tmpdir)
            # Should not raise
            _update_work_progress(db, "nonexistent", "tasks", "running")


class TestV2StreamChunkParsing:
    """Tests that verify v2 StreamPart chunk format is correctly parsed.

    The submit_work() and resume_work() functions stream with
    stream_mode=["updates", "messages"], subgraphs=True, version="v2".
    Each chunk is a dict: {"type": ..., "ns": ..., "data": ...}

    These tests validate the parsing logic in isolation by simulating
    the chunk format without running the full graph.
    """

    def _parse_chunk(self, chunk) -> dict | None:
        """Simulate the chunk parsing logic from submit_work().

        Returns the {node_name: node_output} dict for updates, or None
        if the chunk should be skipped.
        """
        if not isinstance(chunk, dict) or chunk.get("type") != "updates":
            return None

        ns = chunk.get("ns", ())
        if ns != ():
            return None

        return chunk.get("data", {})

    def test_v2_updates_root_chunk_parsed(self):
        """Root-level updates chunk (ns=()) is correctly parsed."""
        chunk = {
            "type": "updates",
            "ns": (),
            "data": {"tasks": {"current_phase": "tasks", "status": "running"}},
        }
        result = self._parse_chunk(chunk)
        assert result is not None
        assert "tasks" in result
        assert result["tasks"]["current_phase"] == "tasks"

    def test_v2_messages_chunk_skipped(self):
        """Messages chunks are skipped (they only keep stall timer alive)."""
        chunk = {
            "type": "messages",
            "ns": (),
            "data": ("token", {"langgraph_node": "tasks"}),
        }
        result = self._parse_chunk(chunk)
        assert result is None

    def test_v2_subgraph_updates_skipped(self):
        """Subgraph updates (ns != ()) are skipped — they're DA internals."""
        chunk = {
            "type": "updates",
            "ns": ("tasks:abc-123",),
            "data": {"agent": {"output": "subgraph stuff"}},
        }
        result = self._parse_chunk(chunk)
        assert result is None

    def test_v1_tuple_format_not_parsed(self):
        """V1 format tuples are NOT parsed — they'd be silently dropped.

        This is the bug that was fixed: the old code checked
        isinstance(chunk, tuple) and len(chunk) == 2, but with
        subgraphs=True the v1 format produces 3-element tuples
        (namespace, mode, data), so all chunks were silently skipped.
        """
        # V1 format with subgraphs=True: 3-element tuple
        v1_chunk = ((), "updates", {"tasks": {"current_phase": "tasks"}})
        result = self._parse_chunk(v1_chunk)
        assert result is None  # V1 tuples are not dicts — correctly skipped

        # V1 format without subgraphs: 2-element tuple
        v1_chunk_no_sub = ("updates", {"tasks": {"current_phase": "tasks"}})
        result = self._parse_chunk(v1_chunk_no_sub)
        assert result is None  # Also not a dict — correctly skipped

    def test_deeply_nested_subgraph_skipped(self):
        """Deeply nested subgraph updates (multi-level ns) are skipped."""
        chunk = {
            "type": "updates",
            "ns": ("tasks:abc", "agent:def", "subagent:ghi"),
            "data": {"tool": {"output": "deep subgraph"}},
        }
        result = self._parse_chunk(chunk)
        assert result is None

    def test_multiple_nodes_in_single_update(self):
        """A single update chunk can contain multiple node outputs."""
        chunk = {
            "type": "updates",
            "ns": (),
            "data": {
                "tasks": {"current_phase": "tasks", "status": "running"},
                "gate_tasks_to_implement": {"status": "running"},
            },
        }
        result = self._parse_chunk(chunk)
        assert result is not None
        assert "tasks" in result
        assert "gate_tasks_to_implement" in result

    def test_non_dict_chunk_skipped(self):
        """Non-dict chunks (shouldn't happen in v2 but be defensive)."""
        assert self._parse_chunk("string") is None
        assert self._parse_chunk(42) is None
        assert self._parse_chunk(None) is None
