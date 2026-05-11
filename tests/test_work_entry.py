"""Tests for the WorkEntry SQLAlchemy model."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from spine.models.work_entry import Base, WorkEntry


class TestWorkEntryModel:
    """Tests for the WorkEntry ORM model."""

    def setup_method(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)

    def teardown_method(self):
        self.engine.dispose()

    def _entry(self, **kwargs) -> WorkEntry:
        defaults = {
            "thread_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc),
            "action": "test_action",
        }
        return WorkEntry(**{**defaults, **kwargs})

    def test_create_work_entry(self):
        entry = self._entry()
        with Session(self.engine) as session:
            session.add(entry)
            session.commit()
            assert entry.id is not None

    def test_thread_id_non_null(self):
        with Session(self.engine) as session:
            entry = WorkEntry(
                timestamp=datetime.now(timezone.utc),
                action="no_thread",
            )
            session.add(entry)
            session.commit()

    def test_thread_id_stored_and_retrievable(self):
        tid = str(uuid.uuid4())
        entry = self._entry(thread_id=tid)
        with Session(self.engine) as session:
            session.add(entry)
            session.commit()
            fetched = session.get(WorkEntry, entry.id)
            assert fetched is not None
            assert fetched.thread_id == tid

    def test_timestamp_defaults_to_now(self):
        entry = self._entry()
        with Session(self.engine) as session:
            session.add(entry)
            session.commit()
            assert entry.timestamp is not None
            assert isinstance(entry.timestamp, datetime)

    def test_action_required(self):
        with Session(self.engine) as session:
            entry = WorkEntry(
                thread_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
            )
            session.add(entry)
            session.commit()

    def test_payload_optional(self):
        entry = self._entry()
        assert entry.payload is None
        entry.payload = '{"key": "value"}'
        with Session(self.engine) as session:
            session.add(entry)
            session.commit()
            assert entry.payload == '{"key": "value"}'

    def test_to_dict(self):
        tid = str(uuid.uuid4())
        ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        entry = self._entry(thread_id=tid, timestamp=ts, action="test", payload="data")
        d = entry.to_dict()
        assert d["thread_id"] == tid
        assert d["timestamp"] == "2026-01-01T12:00:00+00:00"
        assert d["action"] == "test"
        assert d["payload"] == "data"
        assert "id" in d

    def test_repr(self):
        entry = self._entry(action="repr_test")
        r = repr(entry)
        assert "WorkEntry" in r
        assert entry.thread_id in r
        assert "repr_test" in r


class TestWorkEntryUniqueConstraint:
    """Tests for the composite unique constraint on (thread_id, timestamp)."""

    def setup_method(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)

    def teardown_method(self):
        self.engine.dispose()

    def test_unique_constraint_enforced(self):
        import sqlalchemy as sa

        tid = str(uuid.uuid4())
        ts = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
        with Session(self.engine) as session:
            session.add(WorkEntry(thread_id=tid, timestamp=ts, action="first"))
            session.commit()
        with Session(self.engine) as session:
            session.add(WorkEntry(thread_id=tid, timestamp=ts, action="second"))
            import pytest
            with pytest.raises(sa.exc.IntegrityError):
                session.commit()

    def test_same_thread_different_timestamp_allowed(self):
        tid = str(uuid.uuid4())
        with Session(self.engine) as session:
            session.add(
                WorkEntry(
                    thread_id=tid,
                    timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
                    action="first",
                )
            )
            session.add(
                WorkEntry(
                    thread_id=tid,
                    timestamp=datetime(2026, 1, 1, 13, 0, 0, tzinfo=timezone.utc),
                    action="second",
                )
            )
            session.commit()

    def test_different_thread_same_timestamp_allowed(self):
        ts = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        with Session(self.engine) as session:
            session.add(WorkEntry(thread_id=str(uuid.uuid4()), timestamp=ts, action="a"))
            session.add(WorkEntry(thread_id=str(uuid.uuid4()), timestamp=ts, action="b"))
            session.commit()
