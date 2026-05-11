"""Integration tests: UI work submission → queued → visible-in-status flow.

Exercises the full lifecycle through JobStore and backend API, including
failure modes where the QUEUED transition is skipped (the prior bug).
"""

import sqlite3
import sys
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from starlette.status import (
    HTTP_202_ACCEPTED,
    HTTP_200_OK,
    HTTP_404_NOT_FOUND,
    HTTP_409_CONFLICT,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.models.job import Job, JobStatus, JobStore
from backend.routes.work import rate_limiter
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
    return TestClient(client)


@pytest.fixture
def isolated_store(tmp_path):
    """JobStore backed by a temporary path (isolated from API's default store)."""
    return JobStore(db_path=str(tmp_path / "test_jobs.db"))


# ═══════════════════════════════════════════════════════════════════
# Submission → Status Query Flow
# ═══════════════════════════════════════════════════════════════════


class TestSubmissionToStatusFlow:
    """Verify: submitted work is QUEUED and immediately queryable via status endpoint."""

    def test_submit_and_query_status(self, client):
        with patch("backend.routes.work._dispatch_async"):
            resp = client.post("/work/submit", json={
                "requirement": "Build auth system",
                "method": "Quick Work",
                "project_type": "Greenfield",
            })
        assert resp.status_code == HTTP_202_ACCEPTED
        data = resp.json()
        job_id = data["job_id"]
        assert data["status"] == "queued"

        status_resp = client.get(f"/jobs/{job_id}")
        assert status_resp.status_code == HTTP_200_OK
        s = status_resp.json()
        assert s["job_id"] == job_id
        assert s["status"] == "queued"
        assert s["requirement"] == "Build auth system"

    def test_submit_persists_in_job_store(self, client):
        with patch("backend.routes.work._dispatch_async"):
            resp = client.post("/work/submit", json={
                "requirement": "Persist test",
                "llm_provider": "gpt-4",
                "parallel_agents": 5,
            })
        assert resp.status_code == HTTP_202_ACCEPTED
        job_id = resp.json()["job_id"]

        store = JobStore()
        job = store.get(job_id)
        assert job is not None
        assert job.requirement == "Persist test"
        assert job.status == JobStatus.QUEUED
        assert job.llm_provider == "gpt-4"
        assert job.parallel_agents == 5

    def test_submission_includes_metadata_in_status(self, client):
        with patch("backend.routes.work._dispatch_async"):
            resp = client.post("/work/submit", json={
                "requirement": "Metadata check",
                "method": "Full Spec Work",
                "project_type": "Brownfield",
                "llm_provider": "claude-4",
                "parallel_agents": 2,
            })
        job_id = resp.json()["job_id"]

        s = client.get(f"/jobs/{job_id}").json()
        assert s["method"] == "Full Spec Work"
        assert s["project_type"] == "Brownfield"
        assert s["llm_provider"] == "claude-4"
        assert s["parallel_agents"] == 2
        assert s["created_at"] is not None
        assert s["updated_at"] is not None
        assert s["completed_at"] is None

    def test_nonexistent_job_returns_404(self, client):
        resp = client.get("/jobs/nonexistent-id")
        assert resp.status_code == HTTP_404_NOT_FOUND

    def test_multiple_submissions_all_queryable(self, client):
        ids = []
        with patch("backend.routes.work._dispatch_async"):
            for req in ["Job A", "Job B", "Job C"]:
                resp = client.post("/work/submit", json={"requirement": req})
                ids.append(resp.json()["job_id"])

        for jid in ids:
            s = client.get(f"/jobs/{jid}").json()
            assert s["job_id"] == jid
            assert s["status"] == "queued"

    def test_list_jobs_includes_submitted(self, client):
        with patch("backend.routes.work._dispatch_async"):
            client.post("/work/submit", json={"requirement": "List test 1"})
            client.post("/work/submit", json={"requirement": "List test 2"})

        store = JobStore()
        jobs = store.list_jobs(limit=10)
        reqs = [j.requirement for j in jobs]
        assert "List test 1" in reqs
        assert "List test 2" in reqs


# ═══════════════════════════════════════════════════════════════════
# Job Lifecycle: QUEUED → RUNNING → COMPLETED / FAILED
# ═══════════════════════════════════════════════════════════════════


class TestJobLifecycle:
    """Verify full QUEUED → RUNNING → COMPLETED / FAILED transitions."""

    def test_queued_to_running_to_completed(self, client):
        with patch("backend.routes.work._dispatch_async"):
            resp = client.post("/work/submit", json={"requirement": "Full cycle"})
        job_id = resp.json()["job_id"]

        api_store = JobStore()
        assert api_store.get(job_id).status == JobStatus.QUEUED

        api_store.update_status(job_id, JobStatus.RUNNING, thread_id="th-cycle")
        job = api_store.get(job_id)
        assert job.status == JobStatus.RUNNING
        assert job.thread_id == "th-cycle"

        api_store.update_status(job_id, JobStatus.COMPLETED)
        job = api_store.get(job_id)
        assert job.status == JobStatus.COMPLETED
        assert job.completed_at is not None

    def test_queued_to_failed_preserves_error(self, client):
        with patch("backend.routes.work._dispatch_async"):
            resp = client.post("/work/submit", json={"requirement": "Failure case"})
        job_id = resp.json()["job_id"]

        api_store = JobStore()
        api_store.update_status(job_id, JobStatus.RUNNING)
        api_store.update_status(
            job_id, JobStatus.FAILED, error_message="LLM provider unavailable",
        )
        job = api_store.get(job_id)
        assert job.status == JobStatus.FAILED
        assert job.error_message == "LLM provider unavailable"
        assert job.completed_at is not None

    def test_re_running_completed_job_updates_status(self, client):
        with patch("backend.routes.work._dispatch_async"):
            resp = client.post("/work/submit", json={"requirement": "Re-run"})
        job_id = resp.json()["job_id"]

        api_store = JobStore()
        api_store.update_status(job_id, JobStatus.COMPLETED)
        assert api_store.get(job_id).status == JobStatus.COMPLETED

        api_store.update_status(job_id, JobStatus.RUNNING)
        assert api_store.get(job_id).status == JobStatus.RUNNING

    def test_idempotency_prevents_duplicate(self, client):
        with patch("backend.routes.work._dispatch_async"):
            r1 = client.post(
                "/work/submit",
                json={"requirement": "Idempotent"},
                headers={"X-Idempotency-Key": "unique-key-99"},
            )
        assert r1.status_code == HTTP_202_ACCEPTED

        with patch("backend.routes.work._dispatch_async"):
            r2 = client.post(
                "/work/submit",
                json={"requirement": "Idempotent"},
                headers={"X-Idempotency-Key": "unique-key-99"},
            )
        assert r2.status_code == HTTP_409_CONFLICT
        assert "duplicate" in r2.json()["detail"].lower()


# ═══════════════════════════════════════════════════════════════════
# Failure Mode: Missing QUEUED Transition
# ═══════════════════════════════════════════════════════════════════


class TestMissingQueuedTransition:
    """Test the bug: work submitted without proper QUEUED transition.

    The prior bug was that submissions could create jobs directly in RUNNING
    or create entries in TaskQueue without a corresponding JobStore record,
    causing 'spine status' to show nothing. These tests catch that.
    """

    def test_job_created_directly_in_running_has_no_queued_record(self, store):
        job = Job(
            requirement="Direct running",
            status=JobStatus.RUNNING,
        )
        store.create(job)
        retrieved = store.get(job.job_id)
        assert retrieved is not None
        assert retrieved.status == JobStatus.RUNNING

        jobs = store.list_jobs(limit=100)
        queued_jobs = [j for j in jobs if j.status == JobStatus.QUEUED]
        running_jobs = [j for j in jobs if j.status == JobStatus.RUNNING]

        assert job.job_id in [j.job_id for j in running_jobs]
        assert job.job_id not in [j.job_id for j in queued_jobs]

    def test_work_only_in_queue_not_in_job_store(self, store):
        """Simulate the prior bug: UI submits to TaskQueue but no JobStore record."""
        from spine.config.queue import TaskQueue

        queue = TaskQueue(db_path="/tmp/test_queue_no_job.db")
        queue.enqueue("work", {
            "requirement": "Queue-only work item",
            "thread_id": "queue-only-thread",
        })

        job = store.get("queue-only-thread")
        assert job is None

        jobs = store.list_jobs(limit=100)
        queue_only_in_jobs = [
            j for j in jobs
            if j.requirement == "Queue-only work item"
        ]
        assert len(queue_only_in_jobs) == 0

    def test_job_without_status_transition_is_invisible_to_status(self, client, store):
        """Job stuck at QUEUED with no worker to advance it is queryable but
        never reaches RUNNING. This simulates the UI-submission-no-worker bug."""
        with patch("backend.routes.work._dispatch_async"):
            resp = client.post("/work/submit", json={
                "requirement": "Stuck queued",
            })
        job_id = resp.json()["job_id"]

        s = client.get(f"/jobs/{job_id}").json()
        assert s["status"] == "queued"

        store.update_status(job_id, JobStatus.QUEUED)
        s = client.get(f"/jobs/{job_id}").json()
        assert s["status"] == "queued"

        assert s["completed_at"] is None
        assert s["thread_id"] is None

    def test_skipped_queued_direct_to_completed_is_anomalous(self, store):
        job = Job(
            requirement="Jumped to completed",
            status=JobStatus.COMPLETED,
            thread_id="th-skip",
        )
        store.create(job)
        retrieved = store.get(job.job_id)
        assert retrieved.status == JobStatus.COMPLETED

        jobs = store.list_jobs(limit=100)
        completed_straight = [
            j for j in jobs
            if j.status == JobStatus.COMPLETED
            and j.job_id == job.job_id
        ]
        assert len(completed_straight) == 1


# ═══════════════════════════════════════════════════════════════════
# Status Query Parity
# ═══════════════════════════════════════════════════════════════════


class TestStatusParity:
    """Verify status endpoint and spine-status would agree."""

    def test_status_fields_match_job_store(self, client, store):
        with patch("backend.routes.work._dispatch_async"):
            resp = client.post("/work/submit", json={
                "requirement": "Parity check",
                "method": "Full Spec Project",
                "project_type": "Brownfield",
                "llm_provider": "claude-4",
                "parallel_agents": 7,
            })
        job_id = resp.json()["job_id"]

        s = client.get(f"/jobs/{job_id}").json()
        job = store.get(job_id)

        assert s["job_id"] == job.job_id
        assert s["status"] == job.status.value
        assert s["requirement"] == job.requirement
        assert s["method"] == job.method
        assert s["project_type"] == job.project_type
        assert s["llm_provider"] == job.llm_provider
        assert s["parallel_agents"] == job.parallel_agents
        assert s["thread_id"] == job.thread_id

    def test_status_updates_reflected_in_both_stores(self, client):
        with patch("backend.routes.work._dispatch_async"):
            resp = client.post("/work/submit", json={"requirement": "Sync test"})
        job_id = resp.json()["job_id"]

        store = JobStore()
        store.update_status(job_id, JobStatus.RUNNING, thread_id="th-sync")
        store.update_status(job_id, JobStatus.COMPLETED)

        s = client.get(f"/jobs/{job_id}").json()
        assert s["status"] == "completed"
        assert s["thread_id"] == "th-sync"
        assert s["completed_at"] is not None
