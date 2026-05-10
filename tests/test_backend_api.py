"""Tests for the SPINE Backend REST API."""

import sys
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from starlette.status import (
    HTTP_202_ACCEPTED,
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_409_CONFLICT,
    HTTP_429_TOO_MANY_REQUESTS,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.models.job import Job, JobStatus, JobStore
from backend.schemas.work import WorkSubmitRequest, sanitize_input, ALLOWED_METHODS, ALLOWED_PROJECT_TYPES
from backend.routes.work import TokenBucket, rate_limiter
from backend.main import create_app


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset rate limiter buckets before each test."""
    rate_limiter._buckets.clear()


# ═══════════════════════════════════════════════════════════════════
# Job Model Tests
# ═══════════════════════════════════════════════════════════════════


class TestJobModel:
    def test_job_creation_defaults(self):
        job = Job(requirement="Build auth")
        assert job.job_id is not None
        assert job.status == JobStatus.QUEUED
        assert job.requirement == "Build auth"
        assert job.method == "Quick Work"
        assert job.parallel_agents == 3

    def test_job_to_dict_roundtrip(self):
        original = Job(
            requirement="Test req",
            method="Full Spec Work",
            project_type="Brownfield",
            llm_provider="gpt-4",
            parallel_agents=5,
            idempotency_key="key-123",
        )
        data = original.to_dict()
        restored = Job.from_dict(data)
        assert restored.job_id == original.job_id
        assert restored.idempotency_key == "key-123"
        assert restored.requirement == "Test req"
        assert restored.status == JobStatus.QUEUED

    def test_job_status_transitions(self):
        job = Job(requirement="test")
        assert job.status == JobStatus.QUEUED
        d = job.to_dict()
        d["status"] = "running"
        job2 = Job.from_dict(d)
        assert job2.status == JobStatus.RUNNING


class TestJobStore:
    @pytest.fixture
    def store(self, tmp_path):
        db_path = str(tmp_path / "test_jobs.db")
        return JobStore(db_path=db_path)

    def test_create_and_get(self, store):
        job = Job(requirement="Test")
        store.create(job)
        retrieved = store.get(job.job_id)
        assert retrieved is not None
        assert retrieved.job_id == job.job_id
        assert retrieved.requirement == "Test"

    def test_get_not_found(self, store):
        assert store.get("nonexistent") is None

    def test_idempotency_key_lookup(self, store):
        job = Job(requirement="Test", idempotency_key="abc-123")
        store.create(job)
        found = store.get_by_idempotency_key("abc-123")
        assert found is not None
        assert found.job_id == job.job_id
        assert store.get_by_idempotency_key("nope") is None

    def test_update_status(self, store):
        job = Job(requirement="Test")
        store.create(job)
        updated = store.update_status(job.job_id, JobStatus.RUNNING, thread_id="th-1")
        assert updated is not None
        assert updated.status == JobStatus.RUNNING
        assert updated.thread_id == "th-1"

    def test_update_status_failed(self, store):
        job = Job(requirement="Test")
        store.create(job)
        updated = store.update_status(job.job_id, JobStatus.FAILED, error_message="boom")
        assert updated is not None
        assert updated.status == JobStatus.FAILED
        assert updated.error_message == "boom"
        assert updated.completed_at is not None

    def test_list_jobs(self, store):
        for i in range(5):
            store.create(Job(requirement=f"Job {i}"))
        jobs = store.list_jobs(limit=3)
        assert len(jobs) == 3

    def test_duplicate_idempotency_key_raises(self, store):
        job1 = Job(requirement="Test", idempotency_key="dup-key")
        store.create(job1)
        job2 = Job(requirement="Test2", idempotency_key="dup-key")
        with pytest.raises(Exception):
            store.create(job2)


# ═══════════════════════════════════════════════════════════════════
# Schema Validation Tests
# ═══════════════════════════════════════════════════════════════════


class TestWorkSubmitRequest:
    def test_valid_payload(self):
        req = WorkSubmitRequest(requirement=" Build my app  ")
        assert req.requirement == "Build my app"

    def test_empty_requirement(self):
        with pytest.raises(Exception):
            WorkSubmitRequest(requirement="")

    def test_whitespace_only(self):
        with pytest.raises(Exception):
            WorkSubmitRequest(requirement="   ")

    def test_invalid_method(self):
        with pytest.raises(Exception):
            WorkSubmitRequest(requirement="test", method="Bad Method")

    def test_invalid_project_type(self):
        with pytest.raises(Exception):
            WorkSubmitRequest(requirement="test", project_type="Unknown")

    def test_parallel_agents_out_of_range(self):
        with pytest.raises(Exception):
            WorkSubmitRequest(requirement="test", parallel_agents=0)
        with pytest.raises(Exception):
            WorkSubmitRequest(requirement="test", parallel_agents=100)

    def test_requirement_too_long(self):
        with pytest.raises(Exception):
            WorkSubmitRequest(requirement="x" * 10001)


class TestSanitizeInput:
    def test_removes_html_chars(self):
        assert sanitize_input("<script>alert(1)</script>") == "scriptalert1/script"
        assert sanitize_input("a&b") == "ab"
        assert sanitize_input("\"'") == ""

    def test_allows_safe_chars(self):
        assert sanitize_input("hello world 123") == "hello world 123"
        assert sanitize_input("safe-text_with.dots") == "safe-text_with.dots"


# ═══════════════════════════════════════════════════════════════════
# Token Bucket Rate Limiter Tests
# ═══════════════════════════════════════════════════════════════════


class TestTokenBucket:
    def test_initial_tokens_available(self):
        bucket = TokenBucket(capacity=5, refill_rate=1.0)
        for _ in range(5):
            assert bucket.consume("test") is True
        assert bucket.consume("test") is False

    def test_refill_over_time(self):
        bucket = TokenBucket(capacity=2, refill_rate=10.0)
        assert bucket.consume("test") is True
        assert bucket.consume("test") is True
        assert bucket.consume("test") is False
        time.sleep(0.15)
        assert bucket.consume("test") is True

    def test_independent_buckets(self):
        bucket = TokenBucket(capacity=2, refill_rate=1.0)
        assert bucket.consume("alice") is True
        assert bucket.consume("alice") is True
        assert bucket.consume("alice") is False
        assert bucket.consume("bob") is True


# ═══════════════════════════════════════════════════════════════════
# API Endpoint Tests
# ═══════════════════════════════════════════════════════════════════


class TestPostWorkSubmit:
    def test_submit_success(self, client):
        with patch("backend.routes.work._dispatch_async"):
            response = client.post("/work/submit", json={
                "requirement": "Build authentication system",
                "method": "Quick Work",
                "project_type": "Greenfield",
                "llm_provider": "ollama",
                "parallel_agents": 3,
            })
        assert response.status_code == HTTP_202_ACCEPTED
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "queued"

    def test_submit_invalid_empty_requirement(self, client):
        response = client.post("/work/submit", json={
            "requirement": "",
        })
        assert response.status_code == HTTP_400_BAD_REQUEST

    def test_submit_invalid_method(self, client):
        response = client.post("/work/submit", json={
            "requirement": "test",
            "method": "INVALID_METHOD",
        })
        assert response.status_code == HTTP_400_BAD_REQUEST

    def test_submit_invalid_project_type(self, client):
        response = client.post("/work/submit", json={
            "requirement": "test",
            "project_type": "INVALID",
        })
        assert response.status_code == HTTP_400_BAD_REQUEST

    def test_submit_missing_requirement(self, client):
        response = client.post("/work/submit", json={})
        assert response.status_code == HTTP_400_BAD_REQUEST

    def test_submit_injection_sanitized(self, client):
        with patch("backend.routes.work._dispatch_async"):
            response = client.post("/work/submit", json={
                "requirement": "<script>malicious</script>",
            })
        assert response.status_code == HTTP_202_ACCEPTED
        data = response.json()
        assert data["status"] == "queued"

    def test_idempotency_key_returns_same_job(self, client):
        with patch("backend.routes.work._dispatch_async"):
            response1 = client.post(
                "/work/submit",
                json={"requirement": "Build auth"},
                headers={"X-Idempotency-Key": "idem-001"},
            )
        assert response1.status_code == HTTP_202_ACCEPTED

        with patch("backend.routes.work._dispatch_async"):
            response2 = client.post(
                "/work/submit",
                json={"requirement": "Build auth"},
                headers={"X-Idempotency-Key": "idem-001"},
            )
        assert response2.status_code == HTTP_409_CONFLICT
        data2 = response2.json()
        assert "duplicate" in data2["detail"].lower()

    def test_rate_limiting(self, client):
        rate_limiter.capacity = 2
        rate_limiter._buckets.clear()

        with patch("backend.routes.work._dispatch_async"):
            for _ in range(2):
                response = client.post(
                    "/work/submit",
                    json={"requirement": f"test {_}"},
                )
                assert response.status_code == HTTP_202_ACCEPTED

        response = client.post(
            "/work/submit",
            json={"requirement": "rate limited"},
        )
        assert response.status_code == HTTP_429_TOO_MANY_REQUESTS

        rate_limiter.capacity = 10

    def test_submit_creates_job_record(self, client):
        with patch("backend.routes.work._dispatch_async"):
            response = client.post("/work/submit", json={
                "requirement": "Persist test",
            })
        assert response.status_code == HTTP_202_ACCEPTED
        job_id = response.json()["job_id"]

        from backend.models.job import JobStore
        store = JobStore()
        job = store.get(job_id)
        assert job is not None
        assert job.requirement == "Persist test"
        assert job.status == JobStatus.QUEUED


class TestGetJobStatus:
    def test_get_existing_job(self, client):
        with patch("backend.routes.work._dispatch_async"):
            create_resp = client.post("/work/submit", json={
                "requirement": "Status check",
            })
        job_id = create_resp.json()["job_id"]

        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "queued"
        assert data["requirement"] == "Status check"

    def test_get_nonexistent_job(self, client):
        response = client.get("/jobs/nonexistent-id")
        assert response.status_code == HTTP_404_NOT_FOUND

    def test_job_includes_metadata(self, client):
        with patch("backend.routes.work._dispatch_async"):
            create_resp = client.post("/work/submit", json={
                "requirement": "Full check",
                "method": "Full Spec Work",
                "project_type": "Brownfield",
                "llm_provider": "gpt-4",
                "parallel_agents": 5,
            })
        job_id = create_resp.json()["job_id"]

        response = client.get(f"/jobs/{job_id}")
        data = response.json()
        assert data["method"] == "Full Spec Work"
        assert data["project_type"] == "Brownfield"
        assert data["llm_provider"] == "gpt-4"
        assert data["parallel_agents"] == 5
        assert data["created_at"] is not None
        assert data["updated_at"] is not None


class TestHealthEndpoint:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
