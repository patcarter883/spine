"""Tests for list_work() ordering behavior in dispatcher and approve_and_spawn()."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest

from spine.config import SpineConfig
from spine.models.enums import TaskStatus
from spine.work.dispatcher import _get_work_db, list_work


class TestListWorkOrdering:
    """Tests for list_work() ordering (-created_at, newest first)."""

    def _setup_work_entries(self, tmpdir: str) -> SpineConfig:
        """Create a SpineConfig with an isolated work_entries.db."""
        config = SpineConfig()
        config.queue_path = str(Path(tmpdir) / "queue.db")  # work_entries.db is parent/queue.db
        config.ensure_dirs()
        db = _get_work_db(config)
        db["work_entries"].insert_all(
            [
                {
                    "id": "work-1",
                    "description": "oldest work",
                    "work_type": "task",
                    "status": "completed",
                    "current_phase": "verify",
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T01:00:00",
                    "result": '{"status": "completed"}',
                },
                {
                    "id": "work-2",
                    "description": "middle work",
                    "work_type": "task",
                    "status": "completed",
                    "current_phase": "verify",
                    "created_at": "2024-06-15T12:00:00",
                    "updated_at": "2024-06-15T13:00:00",
                    "result": '{"status": "completed"}',
                },
                {
                    "id": "work-3",
                    "description": "newest work",
                    "work_type": "task",
                    "status": "running",
                    "current_phase": "implement",
                    "created_at": "2024-12-31T23:59:59",
                    "updated_at": "2024-12-31T23:59:59",
                    "result": "",
                },
            ]
        )
        return config

    def _setup_one_entry(self, tmpdir: str, status: str = "completed") -> SpineConfig:
        """Create config with a single work entry."""
        config = SpineConfig()
        config.queue_path = str(Path(tmpdir) / "queue.db")
        config.ensure_dirs()
        db = _get_work_db(config)
        db["work_entries"].insert(
            {
                "id": "work-1",
                "description": "only work",
                "work_type": "task",
                "status": status,
                "current_phase": "verify",
                "created_at": "2024-06-15T12:00:00",
                "updated_at": "2024-06-15T13:00:00",
                "result": '{"status": "completed"}',
            }
        )
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


class TestApproveAndSpawn:
    """Tests for approve_and_spawn() — plan approval and work spawning."""

    PLAN_ID = "plan-1"
    SPAWNED_IDS = ["spawned-1", "spawned-2"]

    def _make_config(self, tmpdir: str) -> SpineConfig:
        """Create a SpineConfig with isolated work_entries.db.

        ``workspace_root`` points at a freshly-initialised (clean) git repo
        so the worktree preflight in the approval path passes. The queue /
        artifact / checkpoint stores live outside that repo so they don't
        dirty it.
        """
        config = SpineConfig()
        config.queue_path = str(Path(tmpdir) / "queue.db")
        config.artifact_path = str(Path(tmpdir) / "artifacts")
        config.checkpoint_path = str(Path(tmpdir) / "checkpoint.db")
        config.workspace_root = self._init_clean_repo(tmpdir)
        config.ensure_dirs()
        return config

    def _init_clean_repo(self, tmpdir: str) -> str:
        """Create a clean git repo under *tmpdir* and return its path."""
        repo = Path(tmpdir) / "repo"
        repo.mkdir(exist_ok=True)
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        return str(repo)

    def _dirty_repo(self, config: SpineConfig) -> None:
        """Leave an uncommitted change in the workspace git tree."""
        (Path(config.workspace_root) / "scratch.txt").write_text("wip", encoding="utf-8")

    def _insert_plan_entry(
        self,
        config: SpineConfig,
        plan_id: str = PLAN_ID,
        status: str = "awaiting_approval",
        work_type: str = "reviewed_task",
    ) -> None:
        """Insert a work entry for testing."""
        db = _get_work_db(config)
        db["work_entries"].insert(
            {
                "id": plan_id,
                "description": "Test plan",
                "work_type": work_type,
                "status": status,
                "current_phase": "plan",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "result": "{}",
            }
        )

    def _create_plan_artifact(self, config: SpineConfig, plan_id: str = PLAN_ID) -> str:
        """Create a plan.md artifact file on disk and return its content."""
        artifact_dir = Path(config.artifact_path) / plan_id / "plan"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        content = "# Test Plan\n\n## Slice 1\nDo something.\n"
        (artifact_dir / "plan.md").write_text(content, encoding="utf-8")
        return content

    def _artifact_path(self, plan_id: str, phase: str, name: str) -> str:
        """Build the artifact path — mirrors what ArtifactStore.artifact_path should do."""
        return str(self._artifact_base / plan_id / phase / name)

    def _get_db_status(self, config: SpineConfig, plan_id: str = PLAN_ID) -> str:
        """Get the current status from the DB."""
        db = _get_work_db(config)
        return db["work_entries"].get(plan_id)["status"]

    # ── Success path ──────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_approve_continues_from_implement(self):
        """Approve re-keys the SAME work item to 'task' and continues it from
        IMPLEMENT, reusing the approved plan — no fresh spawn, no new work_id.

        (The earlier version mocked ArtifactStore + resolve_plan_to_units and
        asserted the spawn behavior; that masked both an AttributeError and the
        wrong-altitude restart. This asserts the real continuation path.)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            self._insert_plan_entry(config)

            # The reviewed run's checkpoint holds the plan state IMPLEMENT needs.
            saved_state = {
                "execution_waves": [[{"id": "slice-1", "title": "Do it"}]],
                "plan_json": {"feature_slices": [{"id": "slice-1"}]},
                "specification_json": {"title": "Spec"},
                "artifacts": {"plan": {"plan.md": "# Plan"}},
            }
            mock_ckpt = MagicMock()
            mock_ckpt.get_state = AsyncMock(return_value=saved_state)
            mock_ckpt.delete_state = AsyncMock(return_value=True)

            run_graph = AsyncMock(
                return_value={"work_id": self.PLAN_ID, "status": "completed", "work_type": "task"}
            )

            with (
                patch("spine.work.dispatcher.AuditService", MagicMock()),
                patch("spine.persistence.checkpoint.CheckpointStore", return_value=mock_ckpt),
                patch("spine.work.dispatcher._run_workflow_graph", run_graph),
            ):
                from spine.work.dispatcher import approve_and_spawn

                result = await approve_and_spawn(config=config, plan_id=self.PLAN_ID)

            # Continuation, not spawn.
            assert result["status"] == "completed"
            assert result["spawned_ids"] == []
            assert result["continued_from"] == "implement"
            assert result["work_type"] == "task"

            # The SAME work item is re-keyed to the execution work type.
            db = _get_work_db(config)
            assert db["work_entries"].get(self.PLAN_ID)["work_type"] == "task"

            # The execution graph ran from IMPLEMENT, seeded with approved waves.
            run_graph.assert_awaited_once()
            kwargs = run_graph.await_args.kwargs
            assert kwargs["work_type"] == "task"
            assert kwargs["start_from_phase"] == "implement"
            assert kwargs["is_restart"] is True
            assert kwargs["initial_state"]["execution_waves"] == saved_state["execution_waves"]

    # ── Dirty tree stays retryable ────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_approve_on_dirty_tree_raises_and_stays_retryable(self):
        """Approving while the git tree is dirty raises *before* any status
        mutation, so the entry is left awaiting_approval — retryable once the
        tree is cleaned — rather than progressed to running/failed."""
        from spine.exceptions import SandboxPreparationError

        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            self._insert_plan_entry(config)
            self._dirty_repo(config)

            saved_state = {
                "execution_waves": [[{"id": "slice-1", "title": "Do it"}]],
                "plan_json": {"feature_slices": [{"id": "slice-1"}]},
            }
            mock_ckpt = MagicMock()
            mock_ckpt.get_state = AsyncMock(return_value=saved_state)
            mock_ckpt.delete_state = AsyncMock(return_value=True)
            run_graph = AsyncMock()

            with (
                patch("spine.work.dispatcher.AuditService", MagicMock()),
                patch("spine.persistence.checkpoint.CheckpointStore", return_value=mock_ckpt),
                patch("spine.work.dispatcher._run_workflow_graph", run_graph),
            ):
                from spine.work.dispatcher import approve_and_spawn

                with pytest.raises(SandboxPreparationError, match="not clean"):
                    await approve_and_spawn(config=config, plan_id=self.PLAN_ID)

            # Untouched: still awaiting_approval + reviewed_task, graph never ran.
            db = _get_work_db(config)
            entry = db["work_entries"].get(self.PLAN_ID)
            assert entry["status"] == "awaiting_approval"
            assert entry["work_type"] == "reviewed_task"
            run_graph.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_approve_retry_succeeds_after_cleaning_tree(self):
        """After a dirty-tree failure, cleaning the tree and re-approving the
        same (still awaiting_approval) plan succeeds."""
        from spine.exceptions import SandboxPreparationError

        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            self._insert_plan_entry(config)
            self._dirty_repo(config)

            saved_state = {
                "execution_waves": [[{"id": "slice-1", "title": "Do it"}]],
                "plan_json": {"feature_slices": [{"id": "slice-1"}]},
            }
            mock_ckpt = MagicMock()
            mock_ckpt.get_state = AsyncMock(return_value=saved_state)
            mock_ckpt.delete_state = AsyncMock(return_value=True)
            run_graph = AsyncMock(
                return_value={"work_id": self.PLAN_ID, "status": "completed", "work_type": "task"}
            )

            with (
                patch("spine.work.dispatcher.AuditService", MagicMock()),
                patch("spine.persistence.checkpoint.CheckpointStore", return_value=mock_ckpt),
                patch("spine.work.dispatcher._run_workflow_graph", run_graph),
            ):
                from spine.work.dispatcher import approve_and_spawn

                # First attempt fails on the dirty tree.
                with pytest.raises(SandboxPreparationError):
                    await approve_and_spawn(config=config, plan_id=self.PLAN_ID)

                # Clean the tree, then retry — the status guard still allows it.
                (Path(config.workspace_root) / "scratch.txt").unlink()
                result = await approve_and_spawn(config=config, plan_id=self.PLAN_ID)

            assert result["status"] == "completed"
            assert result["continued_from"] == "implement"
            run_graph.assert_awaited_once()

    # ── No recoverable plan state ─────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_approve_without_recoverable_plan_state_raises(self):
        """If neither checkpoint state nor plan.json yields execution_waves,
        approval raises and the work item is NOT advanced or re-keyed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            self._insert_plan_entry(config)

            mock_ckpt = MagicMock()
            mock_ckpt.get_state = AsyncMock(return_value=None)
            mock_ckpt.delete_state = AsyncMock(return_value=True)
            run_graph = AsyncMock()

            with (
                patch("spine.work.dispatcher.AuditService", MagicMock()),
                patch("spine.persistence.checkpoint.CheckpointStore", return_value=mock_ckpt),
                patch("spine.work.dispatcher._run_workflow_graph", run_graph),
            ):
                from spine.work.dispatcher import approve_and_spawn

                with pytest.raises(ValueError, match="execution_waves"):
                    await approve_and_spawn(config=config, plan_id=self.PLAN_ID)

            # Did not advance: status + work_type unchanged, graph never ran.
            db = _get_work_db(config)
            entry = db["work_entries"].get(self.PLAN_ID)
            assert entry["status"] == "awaiting_approval"
            assert entry["work_type"] == "reviewed_task"
            run_graph.assert_not_awaited()

    # ── Reject ────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_reject_plan(self):
        """Reject action returns 'rejected' status and updates DB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            self._insert_plan_entry(config)

            mock_audit = MagicMock()

            with (
                patch("spine.work.dispatcher.ArtifactStore", MagicMock()),
                patch("spine.work.dispatcher.AuditService", return_value=mock_audit),
            ):
                from spine.work.dispatcher import approve_and_spawn

                result = await approve_and_spawn(
                    config=config,
                    plan_id=self.PLAN_ID,
                    action="reject",
                    feedback="Not good enough",
                )

            assert result["status"] == "rejected"
            assert result["spawned_ids"] == []
            assert self._get_db_status(config) == "rejected"
            mock_audit.log_event.assert_called_with(
                self.PLAN_ID, "plan_rejected", "dispatcher", {"feedback": "Not good enough"}
            )

    # ── Request revision ──────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_request_revision(self):
        """Request revision updates status to needs_review."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            self._insert_plan_entry(config)

            # Build a mock graph whose astream yields a valid update chunk
            # that results in final_status="awaiting_approval" (which means
            # needs_review from the feedback check is not triggered, but the
            # DB was pre-set to needs_review before the streaming)
            async def _mock_astream(*args, **kwargs):
                yield {
                    "type": "updates",
                    "ns": (),
                    "data": {
                        "some_node": {
                            "current_phase": "review",
                            "status": "completed",
                            "artifacts": {},
                        }
                    },
                }

            mock_graph = MagicMock()
            mock_graph.astream = _mock_astream

            # Inject mocks into sys.modules so local imports resolve to them
            import sys as _sys

            mock_compose = MagicMock()
            mock_compose.build_workflow_graph.return_value = mock_graph
            _sys.modules["spine.workflow.compose"] = mock_compose

            mock_cp_mod = MagicMock()
            mock_cp = MagicMock()
            mock_cp_mod.CheckpointStore = MagicMock(return_value=mock_cp)
            mock_cp.get_checkpointer = AsyncMock()
            mock_cp.get_state = AsyncMock(return_value=None)
            _sys.modules["spine.persistence.checkpoint"] = mock_cp_mod

            mock_audit = MagicMock()

            try:
                with (
                    patch("spine.work.dispatcher.ArtifactStore", MagicMock()),
                    patch("spine.work.dispatcher.AuditService", return_value=mock_audit),
                    patch.dict("os.environ", {"SPINE_STALL_TIMEOUT": "120"}),
                ):
                    from spine.work.dispatcher import approve_and_spawn

                    result = await approve_and_spawn(
                        config=config,
                        plan_id=self.PLAN_ID,
                        action="request_revision",
                        feedback="Please add more detail",
                    )

                # The graph yields status="completed" with no needs_review feedback,
                # so the code converts it to "awaiting_approval" (plan work_type).
                # The function first sets DB to "needs_review" before streaming,
                # then updates again after streaming to "awaiting_approval".
                assert result["status"] == "awaiting_approval"
                assert result["spawned_ids"] == []
                # Verify the initial needs_review update happened via audit log
                mock_audit.log_event.assert_any_call(
                    self.PLAN_ID,
                    "plan_revision_requested",
                    "dispatcher",
                    {"feedback": "Please add more detail"},
                )
                mock_audit.log_event.assert_any_call(
                    self.PLAN_ID,
                    "plan_revision_requested",
                    "dispatcher",
                    {"feedback": "Please add more detail"},
                )
            finally:
                # Clean up sys.modules to avoid side effects
                _sys.modules.pop("spine.workflow.compose", None)
                _sys.modules.pop("spine.persistence.checkpoint", None)

    # ── Stall detection ───────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_resume_stall_is_per_chunk_not_wall_clock(self):
        """A backend that goes silent for longer than SPINE_STALL_TIMEOUT is
        marked stalled. The stall budget is per-chunk: it must NOT be a flat
        wall-clock cap on the whole stream, which previously killed healthy long
        generations mid-flight (trace 019ed38f cancelled a live critic at 130s)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            self._insert_plan_entry(config)

            # astream that emits one chunk immediately (progress), then goes
            # silent forever — the second __anext__ must trip the per-chunk
            # timeout and mark the run stalled.
            async def _mock_astream(*args, **kwargs):
                yield {
                    "type": "updates",
                    "ns": (),
                    "data": {"some_node": {"current_phase": "review", "status": "running"}},
                }
                await asyncio.sleep(60)  # silence longer than the stall timeout
                yield {"type": "updates", "ns": (), "data": {}}  # never reached

            mock_graph = MagicMock()
            mock_graph.astream = _mock_astream

            import sys as _sys

            mock_compose = MagicMock()
            mock_compose.build_workflow_graph.return_value = mock_graph
            _sys.modules["spine.workflow.compose"] = mock_compose

            mock_cp_mod = MagicMock()
            mock_cp = MagicMock()
            mock_cp_mod.CheckpointStore = MagicMock(return_value=mock_cp)
            mock_cp.get_checkpointer = AsyncMock()
            mock_cp.get_state = AsyncMock(return_value=None)
            _sys.modules["spine.persistence.checkpoint"] = mock_cp_mod

            try:
                with (
                    patch("spine.work.dispatcher.ArtifactStore", MagicMock()),
                    patch("spine.work.dispatcher.AuditService", return_value=MagicMock()),
                    patch.dict("os.environ", {"SPINE_STALL_TIMEOUT": "1"}),
                ):
                    from spine.work.dispatcher import approve_and_spawn

                    result = await approve_and_spawn(
                        config=config,
                        plan_id=self.PLAN_ID,
                        action="request_revision",
                        feedback="go",
                    )

                assert result["status"] == TaskStatus.STALLED.value
                assert result["spawned_ids"] == []
            finally:
                _sys.modules.pop("spine.workflow.compose", None)
                _sys.modules.pop("spine.persistence.checkpoint", None)

    # ── Wrong status ──────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_approve_wrong_status(self):
        """Trying to approve a plan not in 'awaiting_approval' status raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            self._insert_plan_entry(config, status="running")

            with (
                patch("spine.work.dispatcher.ArtifactStore", MagicMock()),
                patch("spine.work.dispatcher.AuditService", MagicMock()),
            ):
                from spine.work.dispatcher import approve_and_spawn

                with pytest.raises(ValueError, match="not 'awaiting_approval'"):
                    await approve_and_spawn(config=config, plan_id=self.PLAN_ID)

    # ── Wrong work type ───────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_approve_wrong_work_type(self):
        """Trying to approve a non-planning work type raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            self._insert_plan_entry(config, work_type="task")

            with (
                patch("spine.work.dispatcher.ArtifactStore", MagicMock()),
                patch("spine.work.dispatcher.AuditService", MagicMock()),
            ):
                from spine.work.dispatcher import approve_and_spawn

                with pytest.raises(ValueError, match="is not a planning work type"):
                    await approve_and_spawn(config=config, plan_id=self.PLAN_ID)

    # ── Plan not found ────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_approve_plan_not_found(self):
        """Trying to approve a non-existent plan raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)

            with (
                patch("spine.work.dispatcher.ArtifactStore", MagicMock()),
                patch("spine.work.dispatcher.AuditService", MagicMock()),
            ):
                from spine.work.dispatcher import approve_and_spawn

                with pytest.raises(ValueError, match="not found"):
                    await approve_and_spawn(config=config, plan_id="nonexistent-plan")


class TestDirtyTreeRetryable:
    """A dirty git tree must fail a code-producing re-run *before* it moves the
    work entry out of the status its own retry path accepts — otherwise the
    entry lands in 'failed' and can never be retried, even after cleaning up."""

    def _make_config(self, tmpdir: str) -> SpineConfig:
        repo = Path(tmpdir) / "repo"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        config = SpineConfig()
        config.queue_path = str(Path(tmpdir) / "queue.db")
        config.artifact_path = str(Path(tmpdir) / "artifacts")
        config.checkpoint_path = str(Path(tmpdir) / "checkpoint.db")
        config.workspace_root = str(repo)
        config.ensure_dirs()
        return config

    def _insert(self, config: SpineConfig, work_id: str, status: str) -> None:
        db = _get_work_db(config)
        db["work_entries"].insert(
            {
                "id": work_id,
                "description": "a job",
                "work_type": "task",  # code-producing → worktree required
                "status": status,
                "current_phase": "implement",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "result": "{}",
            }
        )

    def _dirty(self, config: SpineConfig) -> None:
        (Path(config.workspace_root) / "scratch.txt").write_text("wip", encoding="utf-8")

    @pytest.mark.asyncio
    async def test_resume_on_dirty_tree_stays_needs_review(self):
        """resume_work only accepts 'needs_review'; a dirty tree must not flip
        the entry to 'failed' and lock it out of being resumed again."""
        from spine.exceptions import SandboxPreparationError

        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            self._insert(config, "w1", status="needs_review")
            self._dirty(config)

            with patch("spine.work.dispatcher.AuditService", MagicMock()):
                from spine.work.dispatcher import resume_work

                with pytest.raises(SandboxPreparationError, match="not clean"):
                    await resume_work("w1", "fix it", config=config)

            assert _get_work_db(config)["work_entries"].get("w1")["status"] == "needs_review"

    @pytest.mark.asyncio
    async def test_restart_on_dirty_tree_keeps_restartable_status(self):
        """restart_work's restartable set excludes 'failed'; a dirty tree must
        leave the original (running) status intact so it stays restartable."""
        from spine.exceptions import SandboxPreparationError

        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            self._insert(config, "w1", status="running")
            self._dirty(config)

            with patch("spine.work.dispatcher.AuditService", MagicMock()):
                from spine.work.dispatcher import restart_work

                with pytest.raises(SandboxPreparationError, match="not clean"):
                    await restart_work("w1", config=config)

            assert _get_work_db(config)["work_entries"].get("w1")["status"] == "running"


class TestGhostPrevention:
    """The shared graph runner must finalise a work entry to 'failed' on an
    unhandled error so it never lingers as a phantom 'running' (ghost) job."""

    def _config(self, tmpdir: str) -> SpineConfig:
        config = SpineConfig()
        config.queue_path = str(Path(tmpdir) / "queue.db")
        config.ensure_dirs()
        return config

    def _insert_running(self, config: SpineConfig, work_id: str):
        db = _get_work_db(config)
        db["work_entries"].insert(
            {
                "id": work_id,
                "description": "a job",
                "work_type": "task",
                "status": "running",
                "current_phase": "implement",
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00",
                "result": "{}",
            }
        )
        return db

    @staticmethod
    def _noop_sandbox():
        """A WorktreeSandbox stand-in that performs no real git operations.

        Code-producing work now runs through a mandatory worktree sandbox
        (``_run_workflow_graph`` calls ``WorktreeSandbox.enter()`` before the
        inner runner). These ghost-prevention tests exercise the *inner*
        failure path, so we neutralise the sandbox to avoid touching git.
        """
        sandbox = MagicMock()
        sandbox.enter.side_effect = lambda: sandbox.enter.config
        return sandbox

    def test_run_workflow_graph_finalises_failed_on_error(self):
        import asyncio

        import spine.work.dispatcher as disp

        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir)
            db = self._insert_running(config, "ghost-x")
            boom = RuntimeError("node exploded")

            sandbox = self._noop_sandbox()
            sandbox.enter.config = config

            with patch("spine.git.WorktreeSandbox", return_value=sandbox), patch.object(
                disp, "_run_workflow_graph_inner", new=AsyncMock(side_effect=boom)
            ):
                with pytest.raises(RuntimeError, match="node exploded"):
                    asyncio.run(
                        disp._run_workflow_graph(
                            work_id="ghost-x",
                            work_type="task",
                            config=config,
                            db=db,
                            audit=MagicMock(),
                            artifact_store=MagicMock(),
                            initial_state={},
                            checkpoint_store=MagicMock(),
                        )
                    )

            # The entry must be terminal, not a lingering "running" ghost.
            assert db["work_entries"].get("ghost-x")["status"] == "failed"
            # The sandbox must be rolled back, never merged, on an error.
            sandbox.abort.assert_called_once()
            sandbox.finalize.assert_not_called()

    def test_run_workflow_graph_finalises_failed_on_sandbox_prep_error(self):
        """A worktree-preparation failure must also finalise the entry, not ghost."""
        import asyncio

        import spine.work.dispatcher as disp
        from spine.exceptions import SandboxPreparationError

        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir)
            db = self._insert_running(config, "ghost-prep")

            sandbox = MagicMock()
            sandbox.enter.side_effect = SandboxPreparationError("tree is dirty")

            with patch("spine.git.WorktreeSandbox", return_value=sandbox), patch.object(
                disp, "_run_workflow_graph_inner", new=AsyncMock()
            ) as inner:
                with pytest.raises(SandboxPreparationError, match="dirty"):
                    asyncio.run(
                        disp._run_workflow_graph(
                            work_id="ghost-prep",
                            work_type="task",
                            config=config,
                            db=db,
                            audit=MagicMock(),
                            artifact_store=MagicMock(),
                            initial_state={},
                            checkpoint_store=MagicMock(),
                        )
                    )

            # Inner never ran, but the entry is finalised — no ghost.
            inner.assert_not_called()
            assert db["work_entries"].get("ghost-prep")["status"] == "failed"

    def test_finalise_failed_work_marks_failed_with_error(self):
        from spine.work.dispatcher import _finalise_failed_work

        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._config(tmpdir)
            db = self._insert_running(config, "ghost-y")

            _finalise_failed_work(db, "ghost-y", None, RuntimeError("boom"))

            row = db["work_entries"].get("ghost-y")
            assert row["status"] == "failed"
            assert "boom" in (row["result"] or "")
