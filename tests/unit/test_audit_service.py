"""Tests for AuditService.query_events() ordering behavior."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.services.audit_service import AuditService


class TestAuditServiceQueryEventsOrdering:
    """Tests for query_events() ordering (-timestamp, newest first)."""

    def _make_service(self, tmpdir: str) -> AuditService:
        db_path = str(Path(tmpdir) / "audit.db")
        return AuditService(db_path)

    def _insert_events(self, svc: AuditService, events: list[dict]) -> None:
        """Insert events directly into the database."""
        db = svc._get_db()
        for event in events:
            db["audit_events"].insert(event)

    def test_empty_db_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = self._make_service(tmpdir)
            events = svc.query_events()
            assert events == []

    def test_single_event_returns_that_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = self._make_service(tmpdir)
            self._insert_events(svc, [
                {
                    "id": 1,
                    "work_id": "work-1",
                    "event_type": "phase_start",
                    "phase": "specify",
                    "details": json.dumps({}),
                    "timestamp": "2024-06-15T12:00:00",
                },
            ])
            events = svc.query_events()
            assert len(events) == 1
            assert events[0]["id"] == 1
            assert events[0]["details"] == {}

    def test_multiple_events_ordered_newest_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = self._make_service(tmpdir)
            self._insert_events(svc, [
                {
                    "id": 1,
                    "work_id": "work-1",
                    "event_type": "phase_start",
                    "phase": "specify",
                    "details": json.dumps({"detail": "oldest"}),
                    "timestamp": "2024-01-01T00:00:00",
                },
                {
                    "id": 2,
                    "work_id": "work-1",
                    "event_type": "phase_complete",
                    "phase": "specify",
                    "details": json.dumps({"detail": "middle"}),
                    "timestamp": "2024-06-15T12:00:00",
                },
                {
                    "id": 3,
                    "work_id": "work-1",
                    "event_type": "phase_start",
                    "phase": "plan",
                    "details": json.dumps({"detail": "newest"}),
                    "timestamp": "2024-12-31T23:59:59",
                },
            ])
            events = svc.query_events()
            assert len(events) == 3
            timestamps = [e["timestamp"] for e in events]
            assert timestamps == sorted(timestamps, reverse=True)
            assert timestamps == [
                "2024-12-31T23:59:59",
                "2024-06-15T12:00:00",
                "2024-01-01T00:00:00",
            ]

    def test_filtered_by_work_id_returns_newest_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = self._make_service(tmpdir)
            self._insert_events(svc, [
                {
                    "id": 1,
                    "work_id": "work-1",
                    "event_type": "phase_start",
                    "phase": "specify",
                    "details": json.dumps({}),
                    "timestamp": "2024-01-01T00:00:00",
                },
                {
                    "id": 2,
                    "work_id": "work-1",
                    "event_type": "phase_complete",
                    "phase": "specify",
                    "details": json.dumps({}),
                    "timestamp": "2024-06-15T12:00:00",
                },
                {
                    "id": 3,
                    "work_id": "work-2",
                    "event_type": "phase_start",
                    "phase": "plan",
                    "details": json.dumps({}),
                    "timestamp": "2024-12-31T23:59:59",
                },
            ])
            events = svc.query_events(work_id="work-1")
            assert len(events) == 2
            timestamps = [e["timestamp"] for e in events]
            assert timestamps == sorted(timestamps, reverse=True)
            assert timestamps == [
                "2024-06-15T12:00:00",
                "2024-01-01T00:00:00",
            ]

    def test_filtered_by_event_type_returns_newest_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = self._make_service(tmpdir)
            self._insert_events(svc, [
                {
                    "id": 1,
                    "work_id": "work-1",
                    "event_type": "phase_start",
                    "phase": "specify",
                    "details": json.dumps({}),
                    "timestamp": "2024-01-01T00:00:00",
                },
                {
                    "id": 2,
                    "work_id": "work-1",
                    "event_type": "phase_complete",
                    "phase": "specify",
                    "details": json.dumps({}),
                    "timestamp": "2024-06-15T12:00:00",
                },
                {
                    "id": 3,
                    "work_id": "work-1",
                    "event_type": "phase_start",
                    "phase": "plan",
                    "details": json.dumps({}),
                    "timestamp": "2024-12-31T23:59:59",
                },
            ])
            events = svc.query_events(event_type="phase_start")
            assert len(events) == 2
            timestamps = [e["timestamp"] for e in events]
            assert timestamps == sorted(timestamps, reverse=True)
            assert timestamps == [
                "2024-12-31T23:59:59",
                "2024-01-01T00:00:00",
            ]

    def test_respects_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = self._make_service(tmpdir)
            self._insert_events(svc, [
                {
                    "id": i,
                    "work_id": "work-1",
                    "event_type": "phase_start",
                    "phase": "specify",
                    "details": json.dumps({}),
                    "timestamp": f"2024-06-{i:02d}T12:00:00",
                }
                for i in range(1, 11)
            ])
            events = svc.query_events(limit=3)
            assert len(events) == 3

    def test_events_have_details_parsed(self):
        """Verify details JSON is parsed back to dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = self._make_service(tmpdir)
            self._insert_events(svc, [
                {
                    "id": 1,
                    "work_id": "work-1",
                    "event_type": "critic_review",
                    "phase": "critic",
                    "details": json.dumps({"status": "passed", "score": 0.95}),
                    "timestamp": "2024-06-15T12:00:00",
                },
            ])
            events = svc.query_events()
            assert len(events) == 1
            assert events[0]["details"] == {"status": "passed", "score": 0.95}
