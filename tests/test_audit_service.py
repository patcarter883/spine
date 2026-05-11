"""Tests for the AuditService and thread context propagation."""

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spine.models.work_entry import WorkEntry, WorkEntryStore
from spine.services.audit_service import AuditService
from spine.utils.thread_context import (
    get_current_thread_id,
    set_current_thread_id,
    reset_current_thread_id,
    clear_current_thread_id,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test_work_entries.db")


@pytest.fixture
def store(db_path: str) -> WorkEntryStore:
    return WorkEntryStore(db_path=db_path)


@pytest.fixture
def audit_service(store: WorkEntryStore) -> AuditService:
    return AuditService(store=store)


@pytest.fixture
def sample_thread_id() -> str:
    return "test-thread-uuid-1234"


# ═══════════════════════════════════════════════════════════════════
# Thread Context Tests
# ═══════════════════════════════════════════════════════════════════


class TestThreadContext:
    def test_default_is_none(self):
        assert get_current_thread_id() is None

    def test_set_and_get(self):
        token = set_current_thread_id("abc-123")
        assert get_current_thread_id() == "abc-123"
        reset_current_thread_id(token)
        assert get_current_thread_id() is None

    def test_clear(self):
        set_current_thread_id("abc-123")
        clear_current_thread_id()
        assert get_current_thread_id() is None

    def test_reset_restores_previous(self):
        inner_token = set_current_thread_id("outer")
        token = set_current_thread_id("inner")
        assert get_current_thread_id() == "inner"
        reset_current_thread_id(token)
        assert get_current_thread_id() == "outer"
        reset_current_thread_id(inner_token)
        assert get_current_thread_id() is None


# ═══════════════════════════════════════════════════════════════════
# WorkEntry Model Tests
# ═══════════════════════════════════════════════════════════════════


class TestWorkEntryModel:
    def test_create_entry(self, sample_thread_id: str):
        entry = WorkEntry(thread_id=sample_thread_id, action="test_action")
        assert entry.thread_id == sample_thread_id
        assert entry.action == "test_action"
        assert entry.entry_id is not None

    def test_to_dict_roundtrip(self, sample_thread_id: str):
        entry = WorkEntry(
            thread_id=sample_thread_id,
            action="roundtrip",
            details={"key": "value"},
        )
        d = entry.to_dict()
        restored = WorkEntry.from_dict(d)
        assert restored.entry_id == entry.entry_id
        assert restored.thread_id == entry.thread_id
        assert restored.action == entry.action
        assert restored.details == entry.details

    def test_default_details_is_empty_dict(self, sample_thread_id: str):
        entry = WorkEntry(thread_id=sample_thread_id, action="no_details")
        assert entry.details == {}


# ═══════════════════════════════════════════════════════════════════
# AuditService Tests
# ═══════════════════════════════════════════════════════════════════


class TestAuditServiceRecordAction:
    def test_record_with_explicit_thread_id(self, audit_service: AuditService):
        entry = audit_service.record_action(
            action="test_action",
            details={"step": 1},
            thread_id="explicit-thread",
        )
        assert entry.thread_id == "explicit-thread"
        assert entry.action == "test_action"
        assert entry.details == {"step": 1}

    def test_record_from_thread_context(self, audit_service: AuditService):
        token = set_current_thread_id("context-thread")
        try:
            entry = audit_service.record_action(
                action="context_action",
                details={"from": "context"},
            )
            assert entry.thread_id == "context-thread"
            assert entry.action == "context_action"
        finally:
            reset_current_thread_id(token)

    def test_record_raises_without_thread_id(self, audit_service: AuditService):
        clear_current_thread_id()
        with pytest.raises(ValueError, match="thread_id is required"):
            audit_service.record_action(action="no_thread")

    def test_idempotent_upsert(self, audit_service: AuditService):
        entry1 = audit_service.record_action(
            action="idempotent_action",
            details={"data": "first"},
            thread_id="idempotent-thread",
        )
        entry2 = audit_service.record_action(
            action="idempotent_action",
            details={"data": "first"},
            thread_id="idempotent-thread",
        )
        assert entry1.entry_id != entry2.entry_id

        entries = audit_service.get_entries(thread_id="idempotent-thread")
        matching = [e for e in entries if e.action == "idempotent_action"]
        assert len(matching) == 2

    def test_record_persists_to_store(self, audit_service: AuditService, store: WorkEntryStore):
        audit_service.record_action(
            action="persist_test",
            details={"check": True},
            thread_id="persist-thread",
        )
        entries = store.get_by_thread("persist-thread")
        assert len(entries) == 1
        assert entries[0].action == "persist_test"
        assert entries[0].details == {"check": True}

    def test_record_with_details(self, audit_service: AuditService):
        details = {"key": "value", "nested": {"a": 1}}
        entry = audit_service.record_action(
            action="detailed",
            details=details,
            thread_id="detail-thread",
        )
        assert entry.details == details


class TestAuditServiceGetEntries:
    def test_get_entries_by_thread(self, audit_service: AuditService):
        audit_service.record_action(action="a1", thread_id="thread-A")
        audit_service.record_action(action="a2", thread_id="thread-A")
        audit_service.record_action(action="b1", thread_id="thread-B")

        entries_a = audit_service.get_entries(thread_id="thread-A")
        assert len(entries_a) == 2
        assert all(e.thread_id == "thread-A" for e in entries_a)

    def test_get_entries_all(self, audit_service: AuditService):
        audit_service.record_action(action="x", thread_id="t1")
        audit_service.record_action(action="y", thread_id="t2")
        entries = audit_service.get_entries()
        assert len(entries) == 2

    def test_get_entries_by_action(self, audit_service: AuditService):
        audit_service.record_action(action="login", thread_id="t1")
        audit_service.record_action(action="login", thread_id="t2")
        audit_service.record_action(action="logout", thread_id="t1")

        entries = audit_service.get_entries_by_action("login")
        assert len(entries) == 2
        assert all(e.action == "login" for e in entries)


class TestAuditServiceRetries:
    def test_retry_on_transient_failure(self, store: WorkEntryStore):
        original_upsert = store.upsert
        call_count = [0]

        def flaky_upsert(entry):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("transient db error")
            return original_upsert(entry)

        store.upsert = flaky_upsert
        service = AuditService(store=store, max_retries=2, retry_delay=0.01)

        entry = service.record_action(
            action="flaky",
            thread_id="retry-thread",
        )
        assert entry.action == "flaky"
        assert entry.thread_id == "retry-thread"
        assert call_count[0] == 2

    def test_exhausts_retries(self, store: WorkEntryStore):
        def always_fail(entry):
            raise RuntimeError("persistent error")

        store.upsert = always_fail
        service = AuditService(store=store, max_retries=1, retry_delay=0.01)

        with pytest.raises(RuntimeError, match="Failed to record action"):
            service.record_action(
                action="doomed",
                thread_id="fail-thread",
            )


class TestWorkEntryStore:
    def test_upsert_and_retrieve(self, store: WorkEntryStore, sample_thread_id: str):
        entry = WorkEntry(thread_id=sample_thread_id, action="store_test")
        store.upsert(entry)
        retrieved = store.get(entry.entry_id)
        assert retrieved is not None
        assert retrieved.entry_id == entry.entry_id
        assert retrieved.action == "store_test"

    def test_get_by_thread(self, store: WorkEntryStore):
        store.upsert(WorkEntry(thread_id="tid1", action="a"))
        store.upsert(WorkEntry(thread_id="tid1", action="b"))
        store.upsert(WorkEntry(thread_id="tid2", action="c"))

        entries = store.get_by_thread("tid1")
        assert len(entries) == 2

    def test_count(self, store: WorkEntryStore):
        assert store.count() == 0
        store.upsert(WorkEntry(thread_id="t", action="a"))
        assert store.count() == 1

    def test_list_entries_pagination(self, store: WorkEntryStore):
        for i in range(10):
            store.upsert(WorkEntry(thread_id="t", action=f"a{i}"))
        all_entries = store.list_entries(limit=10)
        assert len(all_entries) == 10
        page1 = store.list_entries(limit=5, offset=0)
        page2 = store.list_entries(limit=5, offset=5)
        assert len(page1) == 5
        assert len(page2) == 5
        assert page1[0].entry_id != page2[0].entry_id
