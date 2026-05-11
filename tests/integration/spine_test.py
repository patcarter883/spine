"""Integration tests for audit-write persistence and duplicate-entry handling.

Verifies:
- Write persistence through store close/reopen (simulating process restart)
- Handling of duplicate entries (idempotent upsert semantics)
"""

import os
import uuid
from pathlib import Path

import pytest

from spine.models.work_entry import WorkEntry, WorkEntryStore
from spine.services.audit_service import AuditService


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "persist.db")


@pytest.fixture
def store(db_path: str) -> WorkEntryStore:
    return WorkEntryStore(db_path=db_path)


# ═══════════════════════════════════════════════════════════════════
# Write Persistence Through Store Re-open
# ═══════════════════════════════════════════════════════════════════


class TestPersistenceThroughRestart:
    """Entries survive store close / re-open (process-restart simulation)."""

    def test_entries_survive_close_reopen(self, db_path: str):
        store1 = WorkEntryStore(db_path=db_path)
        e1 = WorkEntry(thread_id="t1", action="restart_a", details={"seq": 1})
        e2 = WorkEntry(thread_id="t1", action="restart_b", details={"seq": 2})
        store1.upsert(e1)
        store1.upsert(e2)
        del store1

        store2 = WorkEntryStore(db_path=db_path)
        assert store2.count() == 2
        got1 = store2.get(e1.entry_id)
        got2 = store2.get(e2.entry_id)
        assert got1 is not None
        assert got1.action == "restart_a"
        assert got2 is not None
        assert got2.action == "restart_b"

    def test_db_file_created(self, db_path: str):
        assert not os.path.exists(db_path)
        store = WorkEntryStore(db_path=db_path)
        assert os.path.exists(db_path)
        store.upsert(WorkEntry(thread_id="t", action="marker"))
        assert os.path.getsize(db_path) > 0

    def test_multiple_open_close_cycles(self, db_path: str):
        entries = []
        for cycle in range(5):
            store = WorkEntryStore(db_path=db_path)
            e = WorkEntry(thread_id="cycle", action=f"cycle_{cycle}")
            store.upsert(e)
            entries.append(e.entry_id)
            del store

        final = WorkEntryStore(db_path=db_path)
        assert final.count() == 5
        for eid in entries:
            assert final.get(eid) is not None

    def test_audit_service_persists_across_restart(self, db_path: str):
        svc1 = AuditService(db_path=db_path)
        svc1.record_action(action="init", thread_id="svc-restart")
        svc1.record_action(action="process", thread_id="svc-restart", details={"p": 1})
        del svc1

        svc2 = AuditService(db_path=db_path)
        entries = svc2.get_entries(thread_id="svc-restart")
        assert len(entries) == 2
        actions = {e.action for e in entries}
        assert actions == {"init", "process"}

    def test_wal_does_not_interfere_with_restart(self, db_path: str):
        store1 = WorkEntryStore(db_path=db_path)
        store1.upsert(WorkEntry(thread_id="wal", action="write"))
        store1.upsert(WorkEntry(thread_id="wal", action="write2"))
        del store1

        store2 = WorkEntryStore(db_path=db_path)
        assert store2.count() == 2
        assert len(store2.get_by_thread("wal")) == 2

    def test_large_batch_survives_restart(self, db_path: str):
        store1 = WorkEntryStore(db_path=db_path)
        eids = []
        for i in range(100):
            e = WorkEntry(thread_id="bulk", action=f"item_{i}")
            store1.upsert(e)
            eids.append(e.entry_id)
        del store1

        store2 = WorkEntryStore(db_path=db_path)
        assert store2.count() == 100
        assert len(store2.get_by_thread("bulk")) == 100

    def test_different_threads_persisted(self, db_path: str):
        store1 = WorkEntryStore(db_path=db_path)
        threads = [f"thread-{i}" for i in range(5)]
        for tid in threads:
            store1.upsert(WorkEntry(thread_id=tid, action="start"))
        del store1

        store2 = WorkEntryStore(db_path=db_path)
        assert store2.count() == 5
        for tid in threads:
            assert len(store2.get_by_thread(tid)) == 1


# ═══════════════════════════════════════════════════════════════════
# Duplicate Entry Handling
# ═══════════════════════════════════════════════════════════════════


class TestDuplicateEntryHandling:
    """INSERT OR REPLACE semantics for idempotent upsert and dedup."""

    def test_upsert_same_entry_id_is_idempotent(self, db_path: str):
        store = WorkEntryStore(db_path=db_path)
        eid = str(uuid.uuid4())

        e1 = WorkEntry(entry_id=eid, thread_id="dup", action="first", details={"v": 1})
        e2 = WorkEntry(entry_id=eid, thread_id="dup", action="second", details={"v": 2})

        store.upsert(e1)
        assert store.count() == 1

        store.upsert(e2)
        assert store.count() == 1

        got = store.get(eid)
        assert got.action == "second"
        assert got.details == {"v": 2}

    def test_upsert_with_same_fields_keeps_single(self, db_path: str):
        store = WorkEntryStore(db_path=db_path)
        e = WorkEntry(thread_id="idempotent", action="same")
        eid = e.entry_id

        for _ in range(3):
            store.upsert(e)

        assert store.count() == 1
        got = store.get(eid)
        assert got.action == "same"

    def test_similar_entries_with_diff_ids_all_kept(self, db_path: str):
        store = WorkEntryStore(db_path=db_path)
        e1 = WorkEntry(thread_id="similar", action="dup")
        e2 = WorkEntry(thread_id="similar", action="dup")
        e3 = WorkEntry(thread_id="similar", action="dup")

        store.upsert(e1)
        store.upsert(e2)
        store.upsert(e3)

        assert store.count() == 3
        assert len(store.get_by_thread("similar")) == 3

    def test_audit_service_idempotent_same_thread_action(self, db_path: str):
        svc = AuditService(db_path=db_path)
        e1 = svc.record_action(action="idempotent", thread_id="dup-thread")
        e2 = svc.record_action(action="idempotent", thread_id="dup-thread")

        assert e1.entry_id != e2.entry_id
        entries = svc.get_entries(thread_id="dup-thread")
        matching = [e for e in entries if e.action == "idempotent"]
        assert len(matching) == 2

    def test_upsert_replace_updates_existing(self, db_path: str):
        store = WorkEntryStore(db_path=db_path)
        eid = str(uuid.uuid4())

        e_orig = WorkEntry(entry_id=eid, thread_id="t", action="original", details={"x": 1})
        store.upsert(e_orig)

        e_repl = WorkEntry(entry_id=eid, thread_id="t", action="replaced", details={"x": 2})
        store.upsert(e_repl)

        got = store.get(eid)
        assert got.action == "replaced"
        assert got.details == {"x": 2}

    def test_upsert_does_not_affect_other_entries(self, db_path: str):
        store = WorkEntryStore(db_path=db_path)
        e1 = WorkEntry(thread_id="t", action="a")
        e2 = WorkEntry(thread_id="t", action="b")
        e3 = WorkEntry(thread_id="t", action="c")

        store.upsert(e1)
        store.upsert(e2)
        store.upsert(e3)
        assert store.count() == 3

        store.upsert(e2)
        assert store.count() == 3
        assert store.get(e2.entry_id).action == "b"

    def test_stress_duplicate_upserts(self, db_path: str):
        store = WorkEntryStore(db_path=db_path)
        eid = str(uuid.uuid4())
        e = WorkEntry(entry_id=eid, thread_id="stress", action="stress_test")

        for _ in range(100):
            store.upsert(e)

        assert store.count() == 1
        got = store.get(eid)
        assert got.thread_id == "stress"


__all__ = []
