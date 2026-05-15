"""Tests for the restart_work and reset_stuck_items functionality.

This module covers:
  - restart_work() validates status, purges checkpoints, and re-runs the graph
  - reset_stuck_items() moves running queue items back to pending
  - UIApi.restart_work() and UIApi.reset_stuck_items() delegate correctly
  - Edge cases: missing work items, invalid statuses, artifact clearing
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure the spine package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def tmp_config(tmp_path):
    """Create a minimal SpineConfig for testing."""
    from spine.config import SpineConfig

    config = SpineConfig(
        checkpoint_path=str(tmp_path / "spine.db"),
        artifact_path=str(tmp_path / "artifacts"),
        queue_path=str(tmp_path / "queue.db"),
        workspace_root=str(tmp_path),
        max_critic_retries=3,
    )
    config.ensure_dirs()

    # Patch the config loading
    with patch("spine.config.SpineConfig.load", return_value=config):
        yield config


@pytest.fixture
def work_db(tmp_path):
    """Create a fresh work_entries database for testing."""
    import sqlite_utils

    db_path = tmp_path / "work_entries.db"
    db = sqlite_utils.Database(str(db_path))
    db["work_entries"].create(
        {
            "id": str,
            "description": str,
            "work_type": str,
            "status": str,
            "current_phase": str,
            "created_at": str,
            "updated_at": str,
            "result": str,
        },
        pk="id",
    )
    return db


@pytest.fixture
def queue_db(tmp_path):
    """Create a fresh queue database for testing."""
    import sqlite_utils

    db_path = tmp_path / "queue.db"
    db = sqlite_utils.Database(str(db_path))
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


# ── restart_work tests ──────────────────────────────────────────────


class TestRestartWork:
    """Tests for the restart_work dispatcher function."""

    def test_restart_rejects_completed_work(self, tmp_config, work_db):
        """restart_work raises ValueError for completed items."""
        work_db["work_entries"].insert(
            {
                "id": "done123",
                "description": "done work",
                "work_type": "spec",
                "status": "completed",
                "current_phase": "verify",
                "created_at": "2025-01-01T00:00:00",
                "updated_at": "2025-01-01T00:00:00",
                "result": json.dumps({"artifacts": {}}),
            }
        )

        from spine.work.dispatcher import restart_work

        with pytest.raises(ValueError, match="done123.*completed"):
            import asyncio

            asyncio.run(restart_work("done123", tmp_config))

    def test_restart_rejects_failed_work(self, tmp_config, work_db):
        """restart_work raises ValueError for failed items."""
        work_db["work_entries"].insert(
            {
                "id": "fail123",
                "description": "failed work",
                "work_type": "spec",
                "status": "failed",
                "current_phase": "implement",
                "created_at": "2025-01-01T00:00:00",
                "updated_at": "2025-01-01T00:00:00",
                "result": json.dumps({"error": "boom"}),
            }
        )

        from spine.work.dispatcher import restart_work

        with pytest.raises(ValueError, match="fail123.*failed"):
            import asyncio

            asyncio.run(restart_work("fail123", tmp_config))

    def test_restart_accepts_running_work(self, tmp_config, work_db):
        """restart_work accepts running items (validates, then runs graph)."""
        work_db["work_entries"].insert(
            {
                "id": "run123",
                "description": "running work",
                "work_type": "quick",
                "status": "running",
                "current_phase": "tasks",
                "created_at": "2025-01-01T00:00:00",
                "updated_at": "2025-01-01T00:00:00",
                "result": "{}",
            }
        )

        # We can't fully run the workflow without LLM providers, but we
        # verify that validation passes and the initial state is set up.
        from spine.work.dispatcher import restart_work

        # Mock the graph execution to avoid needing real LLM
        mock_result = {
            "work_id": "run123",
            "status": "completed",
            "work_type": "quick",
            "restarted": True,
        }

        with patch(
            "spine.work.dispatcher._run_workflow_graph",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            import asyncio

            result = asyncio.run(restart_work("run123", tmp_config))

        assert result["status"] == "completed"
        assert result["restarted"] is True

    def test_restart_accepts_stalled_work(self, tmp_config, work_db):
        """restart_work accepts stalled items."""
        work_db["work_entries"].insert(
            {
                "id": "stall123",
                "description": "stalled work",
                "work_type": "quick",
                "status": "stalled",
                "current_phase": "implement",
                "created_at": "2025-01-01T00:00:00",
                "updated_at": "2025-01-01T00:00:00",
                "result": json.dumps({"artifacts": {"implement": {"file.md": "content"}}}),
            }
        )

        from spine.work.dispatcher import restart_work

        mock_result = {
            "work_id": "stall123",
            "status": "completed",
            "work_type": "quick",
            "restarted": True,
        }

        with patch(
            "spine.work.dispatcher._run_workflow_graph",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            import asyncio

            result = asyncio.run(restart_work("stall123", tmp_config))

        assert result["status"] == "completed"
        assert result["restarted"] is True

    def test_restart_accepts_needs_review_work(self, tmp_config, work_db):
        """restart_work accepts needs_review items."""
        work_db["work_entries"].insert(
            {
                "id": "review123",
                "description": "needs review work",
                "work_type": "spec",
                "status": "needs_review",
                "current_phase": "critic_plan",
                "created_at": "2025-01-01T00:00:00",
                "updated_at": "2025-01-01T00:00:00",
                "result": json.dumps({"artifacts": {"plan": {"plan.md": "content"}}}),
            }
        )

        from spine.work.dispatcher import restart_work

        mock_result = {
            "work_id": "review123",
            "status": "completed",
            "work_type": "spec",
            "restarted": True,
        }

        with patch(
            "spine.work.dispatcher._run_workflow_graph",
            new_callable=AsyncMock,
            return_value=mock_result,
        ):
            import asyncio

            result = asyncio.run(restart_work("review123", tmp_config))

        assert result["status"] == "completed"

    def test_restart_nonexistent_work(self, tmp_config, work_db):
        """restart_work raises ValueError for non-existent work items."""
        from spine.work.dispatcher import restart_work

        with pytest.raises(ValueError, match="nonexistent"):
            import asyncio

            asyncio.run(restart_work("nonexistent", tmp_config))

    def test_restart_preserves_artifacts_by_default(self, tmp_config, work_db):
        """By default, restart preserves on-disk artifacts (clear_artifacts=False)."""
        artifact_path = tmp_path / "artifacts" / "run123" / "tasks"
        artifact_path.mkdir(parents=True)
        (artifact_path / "tasks.md").write_text("# Tasks content")

        work_db["work_entries"].insert(
            {
                "id": "run123",
                "description": "test",
                "work_type": "quick",
                "status": "running",
                "current_phase": "",
                "created_at": "2025-01-01T00:00:00",
                "updated_at": "2025-01-01T00:00:00",
                "result": "{}",
            }
        )

        from spine.work.dispatcher import restart_work

        # Mock to avoid real graph execution — but still check artifact clearing
        async def mock_graph(*args, **kwargs):
            # Verify artifact file still exists (not cleared)
            assert (artifact_path / "tasks.md").exists()
            return {
                "work_id": "run123",
                "status": "completed",
                "work_type": "quick",
                "artifacts": {},
                "feedback": [],
                "prompt_request": None,
            }

        with patch(
            "spine.work.dispatcher._run_workflow_graph",
            new_callable=AsyncMock,
            side_effect=mock_graph,
        ):
            import asyncio

            result = asyncio.run(restart_work("run123", tmp_config, clear_artifacts=False))

        assert (artifact_path / "tasks.md").exists()

    def test_restart_clears_artifacts_when_flagged(self, tmp_config, work_db):
        """clear_artifacts=True removes on-disk files before re-running."""
        artifact_path = tmp_path / "artifacts" / "run123" / "tasks"
        artifact_path.mkdir(parents=True)
        (artifact_path / "tasks.md").write_text("# Tasks content")

        work_db["work_entries"].insert(
            {
                "id": "run123",
                "description": "test",
                "work_type": "quick",
                "status": "stalled",
                "current_phase": "",
                "created_at": "2025-01-01T00:00:00",
                "updated_at": "2025-01-01T00:00:00",
                "result": "{}",
            }
        )

        from spine.work.dispatcher import restart_work

        async def mock_graph(*args, **kwargs):
            # Verify artifact file was cleared
            assert not (artifact_path / "tasks.md").exists()
            return {
                "work_id": "run123",
                "status": "completed",
                "work_type": "quick",
                "artifacts": {},
                "feedback": [],
                "prompt_request": None,
            }

        with patch(
            "spine.work.dispatcher._run_workflow_graph",
            new_callable=AsyncMock,
            side_effect=mock_graph,
        ):
            import asyncio

            result = asyncio.run(restart_work("run123", tmp_config, clear_artifacts=True))

        assert result["status"] == "completed"


# ── reset_stuck_items tests ─────────────────────────────────────────


class TestResetStuckItems:
    """Tests for the worker's reset_stuck_items method."""

    def test_reset_stuck_items_empty(self, tmp_config, queue_db):
        """Returns 0 when no running items exist."""
        from spine.work.ralph_worker import RalphLoopWorker

        worker = RalphLoopWorker(tmp_config)
        count = worker.reset_stuck_items()
        assert count == 0

    def test_reset_stuck_items_finds_running(self, tmp_config, queue_db):
        """Resets running items back to pending."""
        queue_db["queue"].insert(
            {
                "id": 1,
                "description": "stuck item",
                "work_type": "spec",
                "status": "running",
                "enqueued_at": "2025-01-01T00:00:00",
                "started_at": "2025-01-01T00:01:00",
                "completed_at": "",
                "result": "",
            }
        )
        queue_db["queue"].insert(
            {
                "id": 2,
                "description": "pending item",
                "work_type": "quick",
                "status": "pending",
                "enqueued_at": "2025-01-01T00:02:00",
                "started_at": "",
                "completed_at": "",
                "result": "",
            }
        )

        from spine.work.ralph_worker import RalphLoopWorker

        worker = RalphLoopWorker(tmp_config)
        count = worker.reset_stuck_items()

        assert count == 1

        # Verify item 1 is now pending
        rows = list(queue_db["queue"].rows_where("id = ?", [1]))
        assert rows[0]["status"] == "pending"
        assert rows[0]["started_at"] == ""

        # Verify item 2 is untouched
        rows = list(queue_db["queue"].rows_where("id = ?", [2]))
        assert rows[0]["status"] == "pending"

    def test_reset_stuck_items_multiple_running(self, tmp_config, queue_db):
        """Resets all running items."""
        for i in range(3):
            queue_db["queue"].insert(
                {
                    "id": i + 1,
                    "description": f"stuck item {i+1}",
                    "work_type": "spec",
                    "status": "running",
                    "enqueued_at": "2025-01-01T00:00:00",
                    "started_at": f"2025-01-01T00:0{i}:00",
                    "completed_at": "",
                    "result": "",
                }
            )

        from spine.work.ralph_worker import RalphLoopWorker

        worker = RalphLoopWorker(tmp_config)
        count = worker.reset_stuck_items()
        assert count == 3

        # All should be pending now
        rows = list(queue_db["queue"].rows_where("status = ?", ["pending"]))
        assert len(rows) == 3


# ── UIApi restart_work tests ────────────────────────────────────────


class TestUIApiRestart:
    """Tests for UIApi restart_work integration."""

    def test_restart_work_returns_running_status(self, tmp_config):
        """UIApi.restart_work returns status='running' immediately."""
        from spine.ui_api import UIApi

        # Need a work entry for the restart to find
        import sqlite_utils

        db_path = tmp_config.queue_path.parent / "work_entries.db"
        db = sqlite_utils.Database(str(db_path))
        if "work_entries" not in db.table_names():
            db["work_entries"].create(
                {
                    "id": str,
                    "description": str,
                    "work_type": str,
                    "status": str,
                    "current_phase": str,
                    "created_at": str,
                    "updated_at": str,
                    "result": str,
                },
                pk="id",
            )
        db["work_entries"].insert(
            {
                "id": "ui-test-1",
                "description": "test restart via UI",
                "work_type": "quick",
                "status": "stalled",
                "current_phase": "implement",
                "created_at": "2025-01-01T00:00:00",
                "updated_at": "2025-01-01T00:00:00",
                "result": "{}",
            }
        )

        api = UIApi(tmp_config)

        # Patch _run_workflow_graph to avoid real execution
        with patch(
            "spine.work.dispatcher._run_workflow_graph",
            new_callable=AsyncMock,
            return_value={
                "work_id": "ui-test-1",
                "status": "completed",
                "work_type": "quick",
                "restarted": True,
            },
        ):
            result = api.restart_work("ui-test-1")

        assert result["status"] == "running"
        assert result["action"] == "restart"


# ── UIApi reset_stuck_items tests ───────────────────────────────────


class TestUIApiResetStuck:
    """Tests for UIApi.reset_stuck_items integration."""

    def test_reset_stuck_items_delegates(self, tmp_config, queue_db):
        """UIApi.reset_stuck_items delegates to worker."""
        queue_db["queue"].insert(
            {
                "id": 1,
                "description": "stuck",
                "work_type": "spec",
                "status": "running",
                "enqueued_at": "2025-01-01T00:00:00",
                "started_at": "2025-01-01T00:01:00",
                "completed_at": "",
                "result": "",
            }
        )

        api = UIApi(tmp_config)
        count = api.reset_stuck_items()

        assert count == 1


# ── status_color_css tests ──────────────────────────────────────────


class TestStatusColorCSS:
    def test_css_colors_for_known_statuses(self):
        from spine.ui.utils import status_color_css

        assert status_color_css("running") == "blue"
        assert status_color_css("stalled") == "orange"
        assert status_color_css("completed") == "green"
        assert status_color_css("failed") == "red"
        assert status_color_css("needs_review") == "yellow"
        assert status_color_css("pending") == "gray"

    def test_css_color_fallback(self):
        from spine.ui.utils import status_color_css

        assert status_color_css("unknown_status") == "white"