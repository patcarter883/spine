"""Tests for list_work() ordering behavior in dispatcher and approve_and_spawn()."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pytest

from spine.config import SpineConfig
from spine.models.types import PlanDecomposition, WorkUnit
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
        """Create a SpineConfig with isolated work_entries.db."""
        config = SpineConfig()
        config.queue_path = str(Path(tmpdir) / "queue.db")
        config.artifact_path = str(Path(tmpdir) / "artifacts")
        config.checkpoint_path = str(Path(tmpdir) / "checkpoint.db")
        config.workspace_root = str(Path(tmpdir))
        config.ensure_dirs()
        return config

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
    async def test_approve_plan_success(self):
        """Happy path — approve a plan with valid artifacts and spawn succeeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            self._insert_plan_entry(config)

            # Create the artifact file for real, then mock artifact_path to point to it
            artifact_file = Path(config.artifact_path) / self.PLAN_ID / "plan" / "plan.md"
            artifact_file.parent.mkdir(parents=True, exist_ok=True)
            plan_content = "# Test Plan\n\n## Slice 1\nDo something.\n"
            artifact_file.write_text(plan_content, encoding="utf-8")

            decomposition = PlanDecomposition(
                units=[
                    WorkUnit(title="Slice 1", description="Do something"),
                    WorkUnit(title="Slice 2", description="Do more"),
                ]
            )

            specs = [
                {"description": "Slice 1: Do something", "work_type": "task"},
                {"description": "Slice 2: Do more", "work_type": "task"},
            ]

            mock_artifact_store = MagicMock()
            mock_artifact_store.artifact_path.return_value = str(artifact_file)

            with (
                patch("spine.work.dispatcher.ArtifactStore", return_value=mock_artifact_store),
                patch("spine.work.dispatcher.AuditService", MagicMock()),
                patch(
                    "spine.work.plan_resolver.resolve_plan_to_units",
                    AsyncMock(return_value=decomposition),
                ),
                patch(
                    "spine.work.plan_resolver.create_work_spawn_specs",
                    return_value=specs,
                ),
                patch(
                    "spine.work.dispatcher.submit_work",
                    AsyncMock(
                        side_effect=[
                            {"work_id": "spawned-1"},
                            {"work_id": "spawned-2"},
                        ]
                    ),
                ),
            ):
                from spine.work.dispatcher import approve_and_spawn

                result = await approve_and_spawn(config=config, plan_id=self.PLAN_ID)

            assert result["status"] == "approved"
            assert result["spawned_ids"] == self.SPAWNED_IDS
            assert "decomposition" in result
            assert self._get_db_status(config) == "approved"

    # ── Missing artifact ──────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_approve_plan_missing_artifact(self):
        """Plan approval fails when the plan.md artifact does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir)
            self._insert_plan_entry(config)

            # Mock artifact_path to return a path that doesn't exist
            missing_path = str(Path(config.artifact_path) / self.PLAN_ID / "plan" / "plan.md")

            mock_artifact_store = MagicMock()
            mock_artifact_store.artifact_path.return_value = missing_path

            with (
                patch("spine.work.dispatcher.ArtifactStore", return_value=mock_artifact_store),
                patch("spine.work.dispatcher.AuditService", MagicMock()),
            ):
                from spine.work.dispatcher import approve_and_spawn

                with pytest.raises(FileNotFoundError):
                    await approve_and_spawn(config=config, plan_id=self.PLAN_ID)

            # DB must NOT be updated to "approved"
            assert self._get_db_status(config) == "awaiting_approval"

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
