"""Tests for restart_from_phase active task check.

This module covers:
  - restart_from_phase() checks if the active task is the same work_id
  - Returns "skipped" when same task is already running
  - Proceeds when no active task or different task is active
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure the spine package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.config import SpineConfig


@pytest.fixture
def tmp_config(tmp_path):
    """Create a minimal SpineConfig for testing.

    ``workspace_root`` is a freshly-initialised (clean) git repo so the
    code-producing restart path's worktree preflight passes. The stores live
    outside that repo so they don't dirty it.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    config = SpineConfig(
        checkpoint_path=str(tmp_path / "spine.db"),
        artifact_path=str(tmp_path / "artifacts"),
        queue_path=str(tmp_path / "queue.db"),
        workspace_root=str(repo),
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


class TestRestartFromPhaseActiveTaskCheck:
    """Tests for the active task check in restart_from_phase."""

    def _setup_work_entry(self, work_db, work_id: str = "test-work-1"):
        """Create a work entry in the database."""
        work_db["work_entries"].insert(
            {
                "id": work_id,
                "description": "test work",
                "work_type": "task",
                "status": "running",
                "current_phase": "implement",
                "created_at": "2024-06-15T12:00:00",
                "updated_at": "2024-06-15T13:00:00",
                "result": "",
            }
        )

    def test_no_active_task_returns_proceeds(self, tmp_config, work_db):
        """When no active task exists, restart_from_phase should proceed."""
        self._setup_work_entry(work_db, "test-work-1")

        with patch("spine.work.ralph_worker.get_worker") as mock_get_worker:
            mock_worker = MagicMock()
            mock_worker.get_active.return_value = None
            mock_get_worker.return_value = mock_worker

            from spine.work.dispatcher import restart_from_phase

            # Mock the graph execution to avoid needing real LLM
            with patch(
                "spine.work.dispatcher._run_workflow_graph",
                return_value={"work_id": "test-work-1", "status": "completed", "work_type": "task"},
            ):
                import asyncio

                result = asyncio.run(restart_from_phase("test-work-1", "implement", tmp_config))

            # Should not be skipped - it should proceed to the workflow
            assert result["status"] != "skipped"

    def test_different_active_task_returns_proceeds(self, tmp_config, work_db):
        """When active task is different work_id, restart_from_phase should proceed."""
        self._setup_work_entry(work_db, "test-work-1")

        with patch("spine.work.ralph_worker.get_worker") as mock_get_worker:
            mock_worker = MagicMock()
            mock_worker.get_active.return_value = {"id": "other-work-2", "status": "running"}
            mock_get_worker.return_value = mock_worker

            from spine.work.dispatcher import restart_from_phase

            with patch(
                "spine.work.dispatcher._run_workflow_graph",
                return_value={"work_id": "test-work-1", "status": "completed", "work_type": "task"},
            ):
                import asyncio

                result = asyncio.run(restart_from_phase("test-work-1", "implement", tmp_config))

            # Should not be skipped - different task is active
            assert result["status"] != "skipped"

    def test_cancelled_work_proceeds(self, tmp_config, work_db):
        """A work item stopped via Stop Work (cancelled) can restart from a phase."""
        work_db["work_entries"].insert(
            {
                "id": "test-work-1",
                "description": "stopped work",
                "work_type": "task",
                "status": "cancelled",
                "current_phase": "implement",
                "created_at": "2024-06-15T12:00:00",
                "updated_at": "2024-06-15T13:00:00",
                "result": "",
            }
        )

        with patch("spine.work.ralph_worker.get_worker") as mock_get_worker:
            mock_worker = MagicMock()
            mock_worker.get_active.return_value = None
            mock_get_worker.return_value = mock_worker

            from spine.work.dispatcher import restart_from_phase

            with patch(
                "spine.work.dispatcher._run_workflow_graph",
                return_value={"work_id": "test-work-1", "status": "completed", "work_type": "task"},
            ):
                import asyncio

                result = asyncio.run(restart_from_phase("test-work-1", "implement", tmp_config))

            assert result["status"] != "skipped"
            assert result["status"] == "completed"

    def test_same_active_task_returns_skipped(self, tmp_config, work_db):
        """When active task matches work_id, restart_from_phase should return skipped."""
        self._setup_work_entry(work_db, "test-work-1")

        with patch("spine.work.ralph_worker.get_worker") as mock_get_worker:
            mock_worker = MagicMock()
            mock_worker.get_active.return_value = {"id": "test-work-1", "status": "running"}
            mock_get_worker.return_value = mock_worker

            from spine.work.dispatcher import restart_from_phase

            import asyncio

            result = asyncio.run(restart_from_phase("test-work-1", "implement", tmp_config))

            assert result["status"] == "skipped"

    def test_dirty_tree_raises_without_mutating_status(self, tmp_config, work_db):
        """A dirty workspace makes the code-producing restart raise *before*
        clearing artifacts / purging the checkpoint / marking running, leaving
        the entry's status untouched so it stays retryable once cleaned."""
        from spine.exceptions import SandboxPreparationError

        self._setup_work_entry(work_db, "test-work-1")  # status=running, type=task
        # Dirty the workspace git tree.
        (Path(tmp_config.workspace_root) / "scratch.txt").write_text("wip", encoding="utf-8")

        with patch("spine.work.ralph_worker.get_worker") as mock_get_worker:
            mock_worker = MagicMock()
            mock_worker.get_active.return_value = None
            mock_get_worker.return_value = mock_worker

            from spine.work.dispatcher import restart_from_phase

            import asyncio

            with patch("spine.work.dispatcher._run_workflow_graph") as run_graph:
                with pytest.raises(SandboxPreparationError, match="not clean"):
                    asyncio.run(restart_from_phase("test-work-1", "implement", tmp_config))
                run_graph.assert_not_called()

        # Status preserved → the restart can be retried after cleaning the tree.
        entry = work_db["work_entries"].get("test-work-1")
        assert entry["status"] == "running"


class TestResumeInterruptedWithoutInterrupt:
    """resume_interrupted_work must not silently complete an autonomous task.

    Autonomous work types (``task`` / ``critical_task``) route needs_review
    to the ``flag_needs_review`` terminal node — their graph thread ends at
    END with no pending interrupt. Issuing ``Command(resume=...)`` against an
    ended thread used to stream zero updates, and the empty result then
    defaulted to "completed" — marking the work done with nothing run
    (trace 019ed36f). The guard must detect the absent interrupt and translate
    the action into a real rerun instead.
    """

    def _setup_work_entry(
        self, work_db, work_id: str = "afd125a6", status: str = "running"
    ):
        work_db["work_entries"].insert(
            {
                "id": work_id,
                "description": "test work",
                "work_type": "task",
                "status": status,
                "current_phase": "implement",
                "created_at": "2024-06-15T12:00:00",
                "updated_at": "2024-06-15T13:00:00",
                "result": "{}",
            }
        )

    def _fake_ended_graph(self, flagged_phase: str = "implement"):
        """A compiled-graph stand-in whose thread is already at END."""
        snapshot = SimpleNamespace(
            next=(),  # no pending nodes → graph reached END
            tasks=[],  # no interrupt tasks awaiting resume
            values={
                "needs_review_phase": flagged_phase,
                "current_phase": flagged_phase,
            },
        )
        graph = MagicMock()
        graph.aget_state = AsyncMock(return_value=snapshot)
        return graph

    @pytest.mark.asyncio
    async def test_rework_without_interrupt_delegates_to_restart(
        self, tmp_config, work_db
    ):
        self._setup_work_entry(work_db)

        captured: dict = {}

        async def fake_restart(work_id, phase, config):
            captured["call"] = (work_id, phase)
            return {"work_id": work_id, "status": "completed", "work_type": "task"}

        with patch(
            "spine.workflow.compose.build_workflow_graph",
            return_value=self._fake_ended_graph("implement"),
        ), patch(
            "spine.work.dispatcher.restart_from_phase", side_effect=fake_restart
        ):
            from spine.work.dispatcher import resume_interrupted_work

            await resume_interrupted_work(
                "afd125a6", "rework", "Redo implementation", tmp_config
            )

        # The flagged phase is re-run rather than no-op resumed.
        assert captured["call"] == ("afd125a6", "implement")

    @pytest.mark.asyncio
    async def test_approve_without_interrupt_advances_to_next_phase(
        self, tmp_config, work_db
    ):
        self._setup_work_entry(work_db)

        captured: dict = {}

        async def fake_restart(work_id, phase, config):
            captured["call"] = (work_id, phase)
            return {"work_id": work_id, "status": "completed", "work_type": "task"}

        with patch(
            "spine.workflow.compose.build_workflow_graph",
            return_value=self._fake_ended_graph("implement"),
        ), patch(
            "spine.work.dispatcher.restart_from_phase", side_effect=fake_restart
        ):
            from spine.work.dispatcher import resume_interrupted_work

            await resume_interrupted_work("afd125a6", "approve", "looks good", tmp_config)

        # Approving implement advances to verify (next non-critic phase).
        assert captured["call"] == ("afd125a6", "verify")

    @pytest.mark.asyncio
    async def test_abort_without_interrupt_cancels(self, tmp_config, work_db):
        self._setup_work_entry(work_db)

        with patch(
            "spine.workflow.compose.build_workflow_graph",
            return_value=self._fake_ended_graph("implement"),
        ), patch(
            "spine.work.dispatcher.restart_from_phase",
            side_effect=AssertionError("restart must not run on abort"),
        ):
            from spine.work.dispatcher import resume_interrupted_work

            result = await resume_interrupted_work(
                "afd125a6", "abort", "stop", tmp_config
            )

        assert result["status"] == "cancelled"
        assert work_db["work_entries"].get("afd125a6")["status"] == "cancelled"
