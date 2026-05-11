"""Tests for the GET /audit REST endpoint."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from starlette.status import (
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
    HTTP_422_UNPROCESSABLE_ENTITY,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.main import create_app
from spine.models.work_entry import WorkEntry, WorkEntryStore


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def store(tmp_path: Path) -> WorkEntryStore:
    return WorkEntryStore(db_path=str(tmp_path / "test_audit.db"))


@pytest.fixture
def app(store: WorkEntryStore):
    with patch("backend.routes.audit.store", store):
        yield create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def seed_entries(store: WorkEntryStore):
    entries = [
        WorkEntry(thread_id="thread-a", action="login", details={"user": "alice"}),
        WorkEntry(thread_id="thread-a", action="logout", details={"user": "alice"}),
        WorkEntry(thread_id="thread-b", action="login", details={"user": "bob"}),
        WorkEntry(thread_id="thread-b", action="view", details={"page": "dashboard"}),
        WorkEntry(thread_id="thread-c", action="login", details={"user": "carol"}),
        WorkEntry(thread_id="thread-c", action="delete", details={"item": "report"}),
    ]
    for e in entries:
        store.upsert(e)
    return entries


# ═══════════════════════════════════════════════════════════════════
# Audit Query Tests
# ═══════════════════════════════════════════════════════════════════


class TestAuditQuery:
    def test_no_filters_returns_all(self, client, seed_entries):
        response = client.get("/audit")
        assert response.status_code == HTTP_200_OK
        data = response.json()
        assert data["total"] == 6
        assert len(data["entries"]) == 6
        assert data["limit"] == 50
        assert data["offset"] == 0

    def test_filter_by_thread_id(self, client, seed_entries):
        response = client.get("/audit", params={"thread_id": "thread-a"})
        assert response.status_code == HTTP_200_OK
        data = response.json()
        assert data["total"] == 2
        assert all(e["thread_id"] == "thread-a" for e in data["entries"])

    def test_filter_by_action(self, client, seed_entries):
        response = client.get("/audit", params={"action": "login"})
        assert response.status_code == HTTP_200_OK
        data = response.json()
        assert data["total"] == 3
        assert all(e["action"] == "login" for e in data["entries"])

    def test_filter_by_thread_and_action(self, client, seed_entries):
        response = client.get("/audit", params={"thread_id": "thread-a", "action": "login"})
        assert response.status_code == HTTP_200_OK
        data = response.json()
        assert data["total"] == 1
        assert data["entries"][0]["action"] == "login"
        assert data["entries"][0]["thread_id"] == "thread-a"

    def test_filter_by_timestamp_range(self, client, seed_entries, store):
        entry = WorkEntry(
            thread_id="thread-z",
            action="early",
            timestamp="2020-01-01T00:00:00",
        )
        store.upsert(entry)

        response = client.get("/audit", params={
            "timestamp-from": "2019-01-01T00:00:00",
            "timestamp-to": "2021-01-01T00:00:00",
        })
        assert response.status_code == HTTP_200_OK
        data = response.json()
        assert data["total"] == 1
        assert data["entries"][0]["action"] == "early"

    def test_pagination_limit(self, client, seed_entries):
        response = client.get("/audit", params={"limit": 2})
        assert response.status_code == HTTP_200_OK
        data = response.json()
        assert len(data["entries"]) == 2
        assert data["total"] == 6
        assert data["limit"] == 2

    def test_pagination_offset(self, client, seed_entries):
        response = client.get("/audit", params={"limit": 2, "offset": 2})
        assert response.status_code == HTTP_200_OK
        data = response.json()
        assert len(data["entries"]) == 2
        assert data["total"] == 6
        assert data["offset"] == 2

    def test_empty_result(self, client):
        response = client.get("/audit", params={"thread_id": "nonexistent"})
        assert response.status_code == HTTP_200_OK
        data = response.json()
        assert data["total"] == 0
        assert data["entries"] == []

    def test_response_shape(self, client, seed_entries):
        response = client.get("/audit", params={"limit": 1})
        data = response.json()
        entry = data["entries"][0]
        assert "entry_id" in entry
        assert "thread_id" in entry
        assert "action" in entry
        assert "details" in entry
        assert "timestamp" in entry
        assert "created_at" in entry


# ═══════════════════════════════════════════════════════════════════
# Error Handling Tests
# ═══════════════════════════════════════════════════════════════════


class TestAuditQueryErrors:
    def test_invalid_limit_below_one(self, client):
        response = client.get("/audit", params={"limit": 0})
        assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

    def test_invalid_limit_above_max(self, client):
        response = client.get("/audit", params={"limit": 1001})
        assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

    def test_invalid_negative_offset(self, client):
        response = client.get("/audit", params={"offset": -1})
        assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY

    def test_invalid_limit_type(self, client):
        response = client.get("/audit", params={"limit": "abc"})
        assert response.status_code == HTTP_422_UNPROCESSABLE_ENTITY
