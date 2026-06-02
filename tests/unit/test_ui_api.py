"""Tests for UIApi.get_queue_overview() ordering behavior and get_feedback()."""

from __future__ import annotations

import json
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


class TestUIApiStopWork:
    """Stop Work must take effect — mark the work entry cancelled so the job
    drops out of the running active set (and a cancellation-aware run halts)."""

    def test_stop_work_marks_work_entry_cancelled(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpineConfig()
            config.queue_path = str(Path(tmpdir) / "queue.db")
            config.checkpoint_path = str(Path(tmpdir) / "spine.db")
            config.artifact_path = str(Path(tmpdir) / "artifacts")
            config.ensure_dirs()
            _init_queue_db(config)

            # Seed a RUNNING, off-queue work entry (restart/resume shape — no
            # queue row), so stop_work exercises the work-entry + purge path.
            from spine.work.dispatcher import get_work_db

            db = get_work_db(config)
            db["work_entries"].insert(
                {
                    "id": "wk-stop",
                    "description": "x",
                    "work_type": "task",
                    "status": "running",
                    "current_phase": "implement",
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:00:00",
                    "result": "{}",
                },
                pk="id",
            )

            # stop_work submits its work to a background executor; run it inline
            # and skip the real checkpoint purge so the test is deterministic.
            class _InlineExecutor:
                def submit(self, fn, *args, **kwargs):
                    fn(*args, **kwargs)
                    return None

            monkeypatch.setattr(
                RalphLoopWorker, "get_executor", lambda self: _InlineExecutor()
            )
            monkeypatch.setattr(
                RalphLoopWorker, "_purge_checkpoint", lambda self, work_id: None
            )

            api = UIApi(config=config)
            out = api.stop_work("wk-stop")

            # The API reports the requested terminal state...
            assert out["status"] == "cancelled"
            assert out["action"] == "stop"
            # ...and the work entry is actually flipped to cancelled, so the UI
            # active set (running/stalled only) no longer shows it.
            row = get_work_db(config)["work_entries"].get("wk-stop")
            assert row["status"] == "cancelled"


class TestUIApiGetQueueOverviewOrdering:
    """Tests for get_queue_overview() ordering."""

    def test_pending_ordered_newest_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpineConfig()
            config.queue_path = str(Path(tmpdir) / "queue.db")
            config.ensure_dirs()

            worker = _init_queue_db(config)
            db = worker._get_db()
            db["queue"].insert_all(
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
                        "description": "newest",
                        "work_type": "task",
                        "status": "pending",
                        "enqueued_at": "2024-12-31T23:59:59",
                        "started_at": "",
                        "completed_at": "",
                        "result": "",
                    },
                    {
                        "id": 3,
                        "description": "middle",
                        "work_type": "task",
                        "status": "pending",
                        "enqueued_at": "2024-06-15T12:00:00",
                        "started_at": "",
                        "completed_at": "",
                        "result": "",
                    },
                ]
            )

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
            db["queue"].insert_all(
                [
                    {
                        "id": 1,
                        "description": "pending item",
                        "work_type": "task",
                        "status": "pending",
                        "enqueued_at": "2024-06-15T12:00:00",
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
                        "completed_at": "2024-01-01T01:00:00",
                        "result": "",
                    },
                ]
            )

            api = UIApi(config)
            overview = api.get_queue_overview()
            assert len(overview["pending"]) == 1
            assert len(overview["recent"]) == 1


class TestUIApiActiveJobs:
    """active_jobs sourcing: every running path shows up, not just queued ones."""

    def _api(self, tmpdir: str) -> tuple[UIApi, RalphLoopWorker, SpineConfig]:
        config = SpineConfig()
        config.queue_path = str(Path(tmpdir) / "queue.db")
        config.ensure_dirs()
        worker = _init_queue_db(config)
        return UIApi(config), worker, config

    def _insert_running_entry(self, config: SpineConfig, work_id: str, **kw) -> None:
        from spine.work.dispatcher import get_work_db

        db = get_work_db(config)
        db["work_entries"].insert(
            {
                "id": work_id,
                "description": kw.get("description", "a job"),
                "work_type": kw.get("work_type", "task"),
                "status": kw.get("status", "running"),
                "current_phase": kw.get("current_phase", "implement"),
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:01:00",
                "result": json.dumps(kw.get("result", {})),
            }
        )

    def test_restart_job_without_queue_row_shows_as_active(self):
        """A running work entry with no queue row (restart/resume) is active."""
        with tempfile.TemporaryDirectory() as tmpdir:
            api, _worker, config = self._api(tmpdir)
            self._insert_running_entry(config, "restart01", current_phase="plan")

            overview = api.get_queue_overview()
            active_jobs = overview["active_jobs"]

            assert len(active_jobs) == 1
            job = active_jobs[0]
            assert job["work_id"] == "restart01"
            assert job["status"] == "running"
            assert job["current_phase"] == "plan"
            # Back-compat single-active key still populated.
            assert overview["active"]["work_id"] == "restart01"

    def test_onboarding_job_correlates_queue_row_with_work_entry(self):
        """An onboarding job (queue row + work entry) shows queue id + phase/mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            api, worker, config = self._api(tmpdir)
            worker._get_db()["queue"].insert(
                {
                    "id": 7,
                    "description": "onboard repo",
                    "work_type": "onboarding",
                    "status": "running",
                    "enqueued_at": "2024-01-01T00:00:00",
                    "started_at": "2024-01-01T00:00:05",
                    "completed_at": "",
                    "result": "",
                    "work_id": "onb12345",
                }
            )
            self._insert_running_entry(
                config,
                "onb12345",
                work_type="onboarding",
                current_phase="analyze",
                result={"mode": "brownfield"},
            )

            overview = api.get_queue_overview()
            active_jobs = overview["active_jobs"]

            assert len(active_jobs) == 1
            job = active_jobs[0]
            assert job["id"] == 7  # queue id surfaced
            assert job["work_id"] == "onb12345"
            assert job["work_type"] == "onboarding"
            assert job["current_phase"] == "analyze"
            assert job["mode"] == "brownfield"

    def test_queue_running_row_without_work_entry_still_active(self):
        """The hand-off window (queue running, work entry not yet created)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            api, worker, _config = self._api(tmpdir)
            worker._get_db()["queue"].insert(
                {
                    "id": 3,
                    "description": "just started",
                    "work_type": "task",
                    "status": "running",
                    "enqueued_at": "2024-01-01T00:00:00",
                    "started_at": "2024-01-01T00:00:05",
                    "completed_at": "",
                    "result": "",
                    "work_id": "handoff1",
                }
            )

            overview = api.get_queue_overview()
            assert len(overview["active_jobs"]) == 1
            assert overview["active_jobs"][0]["work_id"] == "handoff1"

    def test_no_double_count_when_queue_and_entry_share_work_id(self):
        """A queued job with a matching work entry appears exactly once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            api, worker, config = self._api(tmpdir)
            worker._get_db()["queue"].insert(
                {
                    "id": 1,
                    "description": "dual",
                    "work_type": "task",
                    "status": "running",
                    "enqueued_at": "2024-01-01T00:00:00",
                    "started_at": "2024-01-01T00:00:05",
                    "completed_at": "",
                    "result": "",
                    "work_id": "dup00001",
                }
            )
            self._insert_running_entry(config, "dup00001")

            overview = api.get_queue_overview()
            assert len(overview["active_jobs"]) == 1
            assert overview["active_jobs"][0]["id"] == 1

    def test_running_queue_row_surfaces_stalled_work_status(self):
        """A job whose queue row is still running but whose work entry was
        marked stalled mid-run surfaces work_status='stalled' (drives the
        stalled banner)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            api, worker, config = self._api(tmpdir)
            worker._get_db()["queue"].insert(
                {
                    "id": 9,
                    "description": "slow job",
                    "work_type": "task",
                    "status": "running",
                    "enqueued_at": "2024-01-01T00:00:00",
                    "started_at": "2024-01-01T00:00:05",
                    "completed_at": "",
                    "result": "",
                    "work_id": "stall001",
                }
            )
            self._insert_running_entry(config, "stall001", status="stalled")

            active_jobs = api.get_queue_overview()["active_jobs"]
            assert len(active_jobs) == 1
            # Queue lifecycle status stays "running"; the stalled signal is
            # carried separately for the banner.
            assert active_jobs[0]["status"] == "running"
            assert active_jobs[0]["work_status"] == "stalled"

    def test_reset_clears_ghost_work_entry_of_stuck_queue_item(self):
        """Resetting a stuck queue item also stalls its work entry so it
        stops showing as a ghost active job."""
        with tempfile.TemporaryDirectory() as tmpdir:
            api, worker, config = self._api(tmpdir)
            worker._get_db()["queue"].insert(
                {
                    "id": 5,
                    "description": "crashed mid-run",
                    "work_type": "task",
                    "status": "running",
                    "enqueued_at": "2024-01-01T00:00:00",
                    "started_at": "2024-01-01T00:00:05",
                    "completed_at": "",
                    "result": "",
                    "work_id": "ghost001",
                }
            )
            self._insert_running_entry(config, "ghost001")

            assert len(api.get_queue_overview()["active_jobs"]) == 1
            api.reset_stuck_items()

            entry = api.get_work("ghost001")
            assert entry["status"] == "failed"
            assert api.get_queue_overview()["active_jobs"] == []

    def test_reset_clears_offqueue_ghost_without_queue_row(self):
        """A restart/resume job whose thread died leaves a running work entry
        with NO queue row. Reset must still be able to finalise it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            api, _worker, config = self._api(tmpdir)
            self._insert_running_entry(config, "offq0001")

            assert len(api.get_queue_overview()["active_jobs"]) == 1
            count = api.reset_stuck_items()

            assert count == 1
            assert api.get_work("offq0001")["status"] == "failed"
            assert api.get_queue_overview()["active_jobs"] == []

    def test_offqueue_stalled_entry_still_shows_as_active(self):
        """A stalled off-queue job stays visible (with its stalled signal)
        rather than silently vanishing from the active list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            api, _worker, config = self._api(tmpdir)
            self._insert_running_entry(config, "stallq01", status="stalled")

            active_jobs = api.get_queue_overview()["active_jobs"]
            assert len(active_jobs) == 1
            assert active_jobs[0]["work_id"] == "stallq01"
            assert active_jobs[0]["work_status"] == "stalled"

    def test_no_running_work_means_empty_active(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            api, _worker, _config = self._api(tmpdir)
            overview = api.get_queue_overview()
            assert overview["active_jobs"] == []
            assert overview["active"] is None


class TestUIApiReconcileOrphans:
    """Startup reconcile finalises phantom in-flight entries without touching
    queue rows, and never twice or while a job is genuinely live."""

    def _api(self, tmpdir: str) -> tuple[UIApi, RalphLoopWorker, SpineConfig]:
        config = SpineConfig()
        config.queue_path = str(Path(tmpdir) / "queue.db")
        config.ensure_dirs()
        worker = _init_queue_db(config)
        return UIApi(config), worker, config

    def _insert_running_entry(self, config: SpineConfig, work_id: str, **kw) -> None:
        from spine.work.dispatcher import get_work_db

        db = get_work_db(config)
        db["work_entries"].insert(
            {
                "id": work_id,
                "description": kw.get("description", "a job"),
                "work_type": kw.get("work_type", "onboarding"),
                "status": kw.get("status", "running"),
                "current_phase": kw.get("current_phase", "synthesize"),
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:01:00",
                "result": json.dumps(kw.get("result", {})),
            }
        )

    def test_reconcile_finalises_offqueue_onboarding_ghost(self):
        """The incident shape: a running onboarding entry with NO queue row
        (worker thread killed mid-synthesis) is finalised to failed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            api, _worker, config = self._api(tmpdir)
            self._insert_running_entry(config, "synth001", current_phase="synthesize")

            assert len(api.get_queue_overview()["active_jobs"]) == 1
            count = api.reconcile_orphaned_entries()

            assert count == 1
            assert api.get_work("synth001")["status"] == "failed"
            assert api.get_queue_overview()["active_jobs"] == []

    def test_reconcile_leaves_queue_rows_untouched(self):
        """Unlike reset_stuck_items, reconcile must NOT reset queue rows to
        pending (no surprise auto-reprocessing)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            api, worker, _config = self._api(tmpdir)
            worker._get_db()["queue"].insert(
                {
                    "id": 9,
                    "description": "onboard repo",
                    "work_type": "onboarding",
                    "status": "running",
                    "enqueued_at": "2024-01-01T00:00:00",
                    "started_at": "2024-01-01T00:00:05",
                    "completed_at": "",
                    "result": "",
                    "work_id": "live0001",
                }
            )

            api.reconcile_orphaned_entries()

            # The running queue row is still running — reconcile only finalises
            # work entries, it does not reset the queue.
            assert worker._get_db()["queue"].get(9)["status"] == "running"

    def test_reconcile_excludes_live_queue_backed_job(self):
        """A work entry backed by a live running queue row is NOT finalised."""
        with tempfile.TemporaryDirectory() as tmpdir:
            api, worker, config = self._api(tmpdir)
            worker._get_db()["queue"].insert(
                {
                    "id": 4,
                    "description": "live",
                    "work_type": "onboarding",
                    "status": "running",
                    "enqueued_at": "2024-01-01T00:00:00",
                    "started_at": "2024-01-01T00:00:05",
                    "completed_at": "",
                    "result": "",
                    "work_id": "alive001",
                }
            )
            self._insert_running_entry(config, "alive001")

            count = api.reconcile_orphaned_entries()

            assert count == 0
            assert api.get_work("alive001")["status"] == "running"

    def test_reconcile_once_runs_at_most_once_per_process(self):
        """The process-global guard makes a second call a no-op, so a session
        opened mid-run can never finalise a then-live job."""
        import spine.ui_api.api as api_module

        with tempfile.TemporaryDirectory() as tmpdir:
            api, _worker, config = self._api(tmpdir)
            api_module._ORPHAN_RECONCILE_DONE = False  # reset for a clean assertion

            self._insert_running_entry(config, "first001")
            assert api.reconcile_orphaned_entries_once() == 1
            assert api.get_work("first001")["status"] == "failed"

            # A second entry appearing later must NOT be touched by _once again.
            self._insert_running_entry(config, "second02")
            assert api.reconcile_orphaned_entries_once() == 0
            assert api.get_work("second02")["status"] == "running"


class TestUIApiWorkerStatus:
    """Worker status reflects real thread liveness, not a stale flag."""

    def test_worker_not_started_reports_not_running(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpineConfig()
            config.queue_path = str(Path(tmpdir) / "queue.db")
            config.ensure_dirs()
            _init_queue_db(config)
            api = UIApi(config)
            assert api.get_worker_status()["running"] is False

    def test_stale_running_flag_does_not_fake_liveness(self):
        """A dead thread with a left-over running=True flag reports not running."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpineConfig()
            config.queue_path = str(Path(tmpdir) / "queue.db")
            config.ensure_dirs()
            worker = _init_queue_db(config)
            worker.running = True  # stale flag, but no live thread
            api = UIApi(config)
            assert api.get_worker_status()["running"] is False


class TestUIApiGetFeedback:
    """Tests for get_feedback() method."""

    def test_get_feedback_returns_empty_for_nonexistent_work(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpineConfig()
            config.queue_path = str(Path(tmpdir) / "queue.db")
            config.ensure_dirs()

            _init_queue_db(config)
            api = UIApi(config)

            feedback = api.get_feedback("nonexistent")
            assert feedback == []

    def test_get_feedback_returns_feedback_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpineConfig()
            config.queue_path = str(Path(tmpdir) / "queue.db")
            config.ensure_dirs()

            _init_queue_db(config)
            api = UIApi(config)

            # Create a work entry with feedback in the result
            from spine.work.dispatcher import get_work_db

            work_db = get_work_db(config)
            feedback = [
                {
                    "status": "needs_review",
                    "tier": "structural",
                    "reason": "Test reason for review",
                    "suggestions": ["Suggestion 1", "Suggestion 2"],
                }
            ]
            work_db["work_entries"].insert(
                {
                    "id": "test-work-123",
                    "description": "Test work",
                    "work_type": "task",
                    "status": "needs_review",
                    "current_phase": "implement",
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:00:00",
                    "result": json.dumps({"feedback": feedback}),
                }
            )

            result = api.get_feedback("test-work-123")
            assert len(result) == 1
            assert result[0]["status"] == "needs_review"
            assert result[0]["tier"] == "structural"
            assert result[0]["reason"] == "Test reason for review"
            assert result[0]["suggestions"] == ["Suggestion 1", "Suggestion 2"]

    def test_get_feedback_returns_empty_when_no_feedback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpineConfig()
            config.queue_path = str(Path(tmpdir) / "queue.db")
            config.ensure_dirs()

            _init_queue_db(config)
            api = UIApi(config)

            from spine.work.dispatcher import get_work_db

            work_db = get_work_db(config)
            # Work entry with no feedback in result
            work_db["work_entries"].insert(
                {
                    "id": "test-work-456",
                    "description": "Test work",
                    "work_type": "task",
                    "status": "completed",
                    "current_phase": "verify",
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:00:00",
                    "result": json.dumps({"artifacts": {}}),
                }
            )

            result = api.get_feedback("test-work-456")
            assert result == []


class TestUIApiGetCriticReview:
    """Tests for get_critic_review() method."""

    def test_returns_none_for_nonexistent_work(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpineConfig()
            config.queue_path = str(Path(tmpdir) / "queue.db")
            config.ensure_dirs()

            _init_queue_db(config)
            api = UIApi(config)

            assert api.get_critic_review("nonexistent") is None

    def test_returns_review_for_passed_plan(self):
        """awaiting_approval plans PASS, so feedback is empty — but the
        critic verdict must still be available via last_critic_review."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpineConfig()
            config.queue_path = str(Path(tmpdir) / "queue.db")
            config.ensure_dirs()

            _init_queue_db(config)
            api = UIApi(config)

            from spine.work.dispatcher import get_work_db

            work_db = get_work_db(config)
            review = {
                "phase": "plan",
                "status": "passed",
                "tier": "agent",
                "reason": "Plan is well-structured and traceable.",
                "suggestions": ["Consider adding a rollback step"],
                "attempt": 1,
            }
            work_db["work_entries"].insert(
                {
                    "id": "plan-work-1",
                    "description": "Plan work",
                    "work_type": "reviewed_task",
                    "status": "awaiting_approval",
                    "current_phase": "critic",
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:00:00",
                    "result": json.dumps(
                        {"feedback": [], "last_critic_review": review}
                    ),
                }
            )

            # No needs_review feedback, but the critic verdict is available.
            assert api.get_feedback("plan-work-1") == []
            result = api.get_critic_review("plan-work-1")
            assert result is not None
            assert result["status"] == "passed"
            assert result["reason"] == "Plan is well-structured and traceable."
            assert result["suggestions"] == ["Consider adding a rollback step"]

    def test_returns_none_when_review_missing_or_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = SpineConfig()
            config.queue_path = str(Path(tmpdir) / "queue.db")
            config.ensure_dirs()

            _init_queue_db(config)
            api = UIApi(config)

            from spine.work.dispatcher import get_work_db

            work_db = get_work_db(config)
            work_db["work_entries"].insert(
                {
                    "id": "no-review-1",
                    "description": "Work",
                    "work_type": "task",
                    "status": "completed",
                    "current_phase": "verify",
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:00:00",
                    "result": json.dumps({"last_critic_review": None}),
                }
            )

            assert api.get_critic_review("no-review-1") is None
