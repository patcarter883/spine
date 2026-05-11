"""Unit tests for thread-ID propagation and unique constraint compliance.

Verifies:
- Thread-ID is propagated correctly through audit service (explicit + context)
- Unique constraint (entry_id PK) prevents truly duplicate entries
- INSERT OR REPLACE provides idempotent upsert semantics
"""

import threading
import uuid
from pathlib import Path

import pytest

from spine.models.work_entry import WorkEntry, WorkEntryStore
from spine.services.audit_service import AuditService
from spine.utils.thread_context import (
    get_current_thread_id,
    set_current_thread_id,
    reset_thread_id,
    generate_thread_id,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test_audit.db")


@pytest.fixture
def store(db_path: str) -> WorkEntryStore:
    return WorkEntryStore(db_path=db_path)


@pytest.fixture
def audit(store: WorkEntryStore) -> AuditService:
    return AuditService(store=store)


# ═══════════════════════════════════════════════════════════════════
# Thread-ID Propagation
# ═══════════════════════════════════════════════════════════════════


class TestThreadIdPropagation:
    """record_action must propagate thread_id from explicit arg or context."""

    def test_explicit_thread_id_used(self, audit: AuditService):
        tid = "explicit-abc-123"
        entry = audit.record_action(action="test", thread_id=tid)
        assert entry.thread_id == tid

    def test_context_thread_id_used_when_no_explicit(self, audit: AuditService):
        tid = "ctx-thread-456"
        token = set_current_thread_id(tid)
        try:
            entry = audit.record_action(action="ctx_test")
            assert entry.thread_id == tid
        finally:
            reset_thread_id(token)

    def test_differs_from_other_context(self, audit: AuditService):
        t1 = "thread-one"
        t2 = "thread-two"
        token1 = set_current_thread_id(t1)
        e1 = audit.record_action("a1")
        reset_thread_id(token1)
        token2 = set_current_thread_id(t2)
        e2 = audit.record_action("a2")
        reset_thread_id(token2)
        assert e1.thread_id == t1
        assert e2.thread_id == t2
        assert e1.thread_id != e2.thread_id

    def test_parallel_threads_isolated(self, audit: AuditService):
        results = []

        def worker(wid: str):
            token = set_current_thread_id(f"par-{wid}")
            try:
                e = audit.record_action(action="parallel")
                results.append(e.thread_id)
            finally:
                reset_thread_id(token)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert len(set(results)) == 10

    def test_get_entries_respects_thread_id(self, audit: AuditService):
        audit.record_action("a", thread_id="tid-a")
        audit.record_action("b", thread_id="tid-a")
        audit.record_action("c", thread_id="tid-b")

        entries_a = audit.get_entries(thread_id="tid-a")
        assert len(entries_a) == 2
        assert all(e.thread_id == "tid-a" for e in entries_a)

        entries_b = audit.get_entries(thread_id="tid-b")
        assert len(entries_b) == 1
        assert entries_b[0].thread_id == "tid-b"

    def test_get_entries_from_context(self, audit: AuditService):
        tid = "ctx-get-test"
        token = set_current_thread_id(tid)
        try:
            audit.record_action("ctx_a")
            audit.record_action("ctx_b")
            entries = audit.get_entries()
            assert len(entries) == 2
            assert all(e.thread_id == tid for e in entries)
        finally:
            reset_thread_id(token)

    def test_entry_ids_always_unique(self, audit: AuditService):
        ids = set()
        for i in range(50):
            e = audit.record_action(action=f"bulk_{i}", thread_id="bulk-thread")
            ids.add(e.entry_id)
        assert len(ids) == 50


# ═══════════════════════════════════════════════════════════════════
# Unique Constraint Compliance
# ═══════════════════════════════════════════════════════════════════


class TestUniqueConstraintCompliance:
    """entry_id is the PK; INSERT OR REPLACE governs idempotency."""

    def test_same_entry_id_replaced(self, store: WorkEntryStore):
        eid = str(uuid.uuid4())
        e1 = WorkEntry(entry_id=eid, thread_id="t", action="first", details={"v": 1})
        e2 = WorkEntry(entry_id=eid, thread_id="t", action="second", details={"v": 2})
        store.upsert(e1)
        store.upsert(e2)
        got = store.get(eid)
        assert got is not None
        assert got.action == "second"
        assert got.details == {"v": 2}

    def test_multiple_entries_same_thread_allowed(self, store: WorkEntryStore):
        store.upsert(WorkEntry(thread_id="shared", action="a"))
        store.upsert(WorkEntry(thread_id="shared", action="b"))
        store.upsert(WorkEntry(thread_id="shared", action="c"))
        assert store.count() == 3
        assert len(store.get_by_thread("shared")) == 3

    def test_different_entry_ids_same_data_allowed(self, store: WorkEntryStore):
        store.upsert(WorkEntry(thread_id="dup", action="ping"))
        store.upsert(WorkEntry(thread_id="dup", action="ping"))
        assert store.count() == 2

    def test_repeated_upsert_same_entry_id_idempotent(self, store: WorkEntryStore):
        eid = str(uuid.uuid4())
        e = WorkEntry(entry_id=eid, thread_id="t", action="x")
        for _ in range(5):
            store.upsert(e)
        assert store.count() == 1
        assert store.get(eid) is not None

    def test_pk_enforces_no_duplicate_entry_ids(self, store: WorkEntryStore):
        eid = str(uuid.uuid4())
        e1 = WorkEntry(entry_id=eid, thread_id="t1", action="a")
        e2 = WorkEntry(entry_id=eid, thread_id="t2", action="b")
        store.upsert(e1)
        store.upsert(e2)
        got = store.get(eid)
        assert got.thread_id == "t2"
        assert got.action == "b"

    def test_get_by_action_returns_all_matching(self, store: WorkEntryStore):
        for i in range(5):
            store.upsert(WorkEntry(thread_id="t", action="searchable"))
        store.upsert(WorkEntry(thread_id="t", action="other"))
        assert len(store.get_by_action("searchable")) == 5

    def test_get_by_thread_ordered_by_timestamp(self, store: WorkEntryStore):
        for i in range(3):
            store.upsert(WorkEntry(thread_id="order-test", action=f"step_{i}"))
        got = store.get_by_thread("order-test")
        timestamps = [g.timestamp for g in got]
        assert timestamps == sorted(timestamps, reverse=True)
        assert len(got) == 3


__all__ = []
