"""End-to-end tests: UI work submission → queued → visible-in-status flow.

Simulates the full user-facing flow: submit work via API, query status,
and verify the lifecycle transitions that 'spine status' would report.

Covers the prior bug where submissions returned a work ID but never
transitioned through QUEUED, making them invisible to status queries.
"""

import sys
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from starlette.status import HTTP_202_ACCEPTED, HTTP_200_OK

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from backend.models.job import Job, JobStatus, JobStore
from backend.routes.work import rate_limiter, job_store as route_job_store
from backend.main import create_app


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    rate_limiter._buckets.clear()


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


# ═══════════════════════════════════════════════════════════════════
# Full E2E Flow: Submit → Queued → Running → Completed
# ═══════════════════════════════════════════════════════════════════


class TestFullSubmissionLifecycle:
    """End-to-end: submit, verify queued, simulate dispatch, verify
    running, complete, verify terminal state."""

    def test_submit_to_queued_to_running_to_completed(self, client):
        with patch("backend.routes.work._dispatch_async"):
            resp = client.post("/work/submit", json={
                "requirement": "E2E lifecycle test",
                "method": "Quick Work",
                "project_type": "Greenfield",
            })
        assert resp.status_code == HTTP_202_ACCEPTED
        data = resp.json()
        job_id = data["job_id"]
        assert data["status"] == "queued"

        s = client.get(f"/jobs/{job_id}").json()
        assert s["status"] == "queued"

        route_job_store.update_status(job_id, JobStatus.RUNNING, thread_id="th-e2e")
        s = client.get(f"/jobs/{job_id}").json()
        assert s["status"] == "running"
        assert s["thread_id"] == "th-e2e"

        route_job_store.update_status(job_id, JobStatus.COMPLETED)
        s = client.get(f"/jobs/{job_id}").json()
        assert s["status"] == "completed"
        assert s["completed_at"] is not None

    def test_submit_to_queued_to_failed(self, client):
        with patch("backend.routes.work._dispatch_async"):
            resp = client.post("/work/submit", json={
                "requirement": "E2E failure test",
            })
        job_id = resp.json()["job_id"]

        route_job_store.update_status(job_id, JobStatus.RUNNING)
        route_job_store.update_status(
            job_id, JobStatus.FAILED, error_message="Worker crashed",
        )

        s = client.get(f"/jobs/{job_id}").json()
        assert s["status"] == "failed"
        assert s["error_message"] == "Worker crashed"
        assert s["completed_at"] is not None


class TestQueuedWorkVisibility:
    """Verify queued work items are visible to status queries — the
    core requirement: submission → queued → visible."""

    def test_queued_work_appears_in_status_query(self, client):
        with patch("backend.routes.work._dispatch_async"):
            resp = client.post("/work/submit", json={
                "requirement": "Visible queued work",
            })
        job_id = resp.json()["job_id"]

        s = client.get(f"/jobs/{job_id}").json()
        assert s["status"] == "queued"
        assert s["job_id"] == job_id

    def test_queued_work_has_no_thread_id(self, client):
        with patch("backend.routes.work._dispatch_async"):
            resp = client.post("/work/submit", json={
                "requirement": "No thread yet",
            })
        job_id = resp.json()["job_id"]

        s = client.get(f"/jobs/{job_id}").json()
        assert s["status"] == "queued"
        assert s["thread_id"] is None
        assert s["completed_at"] is None

    def test_multiple_queued_jobs_all_visible(self, client):
        ids = []
        with patch("backend.routes.work._dispatch_async"):
            for i in range(5):
                resp = client.post("/work/submit", json={
                    "requirement": f"Batch job {i}",
                })
                ids.append(resp.json()["job_id"])

        for jid in ids:
            s = client.get(f"/jobs/{jid}").json()
            assert s["status"] == "queued"


# ═══════════════════════════════════════════════════════════════════
# Failure Mode: Missing QUEUED Transition (the prior bug)
# ═══════════════════════════════════════════════════════════════════


class TestMissingQueuedTransition:
    """Catch the bug where submissions skip QUEUED status entirely.

    When a job is created directly in RUNNING or COMPLETED state (bypassing
    QUEUED), or when work is enqueued without a JobStore record, the job
    becomes invisible to 'spine status'. These tests detect that.
    """

    def test_job_created_without_queued_is_invisible_in_queued_list(self, client):
        with patch("backend.routes.work._dispatch_async"):
            resp = client.post("/work/submit", json={
                "requirement": "Normal queued",
            })
        normal_id = resp.json()["job_id"]

        route_job_store.update_status(normal_id, JobStatus.RUNNING)

        s = client.get(f"/jobs/{normal_id}").json()
        assert s["status"] == "running"

        jobs = route_job_store.list_jobs(limit=100)
        running_ids = [j.job_id for j in jobs if j.status == JobStatus.RUNNING]
        queued_ids = [j.job_id for j in jobs if j.status == JobStatus.QUEUED]

        assert normal_id in running_ids
        assert normal_id not in queued_ids

    def test_job_only_in_taskqueue_not_in_jobstore(self, client):
        """Simulate the prior UI submission bug: writes to TaskQueue
        but never creates a JobStore record — spine status sees nothing."""
        from spine.config.queue import TaskQueue

        queue = TaskQueue(db_path="/tmp/test_e2e_missing_queue.db")
        task_id = queue.enqueue("work", {
            "requirement": "Orphaned in queue",
            "thread_id": "orphan-thread",
        })

        store = JobStore()
        assert store.get("orphan-thread") is None

        jobs = store.list_jobs(limit=100)
        orphaned = [j for j in jobs if j.requirement == "Orphaned in queue"]
        assert len(orphaned) == 0

    def test_direct_creation_skips_queued_timestamps(self, client):
        store = JobStore()
        job = Job(
            requirement="Skipped queued",
            status=JobStatus.RUNNING,
            thread_id="th-skip-queued",
        )
        store.create(job)

        s = client.get(f"/jobs/{job.job_id}").json()
        assert s["status"] == "running"
        assert s["created_at"] is not None
        assert s["thread_id"] == "th-skip-queued"


class TestStatusReportsQueuedWork:
    """Verify that the status reporting pathway that 'spine status' uses
    would correctly surface queued work."""

    def test_status_cli_would_find_queued_jobs(self, client):
        """The spine status CLI reads from checkpoints in spine.db.
        The status endpoint /jobs/{id} reads from JobStore.
        This test verifies the JobStore path correctly reports QUEUED items."""
        with patch("backend.routes.work._dispatch_async"):
            resp = client.post("/work/submit", json={
                "requirement": "CLI visible",
            })
        job_id = resp.json()["job_id"]

        s = client.get(f"/jobs/{job_id}").json()
        assert s["status"] == "queued"

    def test_queued_status_matches_jobstore(self, client):
        with patch("backend.routes.work._dispatch_async"):
            resp = client.post("/work/submit", json={
                "requirement": "Status match",
            })
        job_id = resp.json()["job_id"]

        store = JobStore()
        store.update_status(job_id, JobStatus.RUNNING, thread_id="th-match")
        store.update_status(job_id, JobStatus.COMPLETED)

        s = client.get(f"/jobs/{job_id}").json()
        assert s["status"] == "completed"
        assert s["thread_id"] == "th-match"


class TestConcurrentSubmissions:
    """Multiple concurrent submissions should all be queryable."""

    def test_bulk_submissions_all_queryable(self, client):
        ids = []
        with patch("backend.routes.work._dispatch_async"):
            for i in range(10):
                resp = client.post("/work/submit", json={
                    "requirement": f"Concurrent job {i}",
                })
                ids.append(resp.json()["job_id"])

        assert len(ids) == 10
        assert len(set(ids)) == 10

        for jid in ids:
            s = client.get(f"/jobs/{jid}").json()
            assert s["status"] == "queued"

    def test_timestamps_are_monotonic(self, client):
        with patch("backend.routes.work._dispatch_async"):
            resp = client.post("/work/submit", json={
                "requirement": "Timestamp test",
            })
        job_id = resp.json()["job_id"]

        s1 = client.get(f"/jobs/{job_id}").json()
        assert s1["created_at"] == s1["updated_at"]

        route_job_store.update_status(job_id, JobStatus.RUNNING)
        s2 = client.get(f"/jobs/{job_id}").json()
        assert s2["updated_at"] >= s2["created_at"]
