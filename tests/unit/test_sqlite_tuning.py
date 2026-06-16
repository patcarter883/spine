"""Tests for SQLite connection tuning (WAL + busy_timeout).

These lock in the fix for ``sqlite3.OperationalError: database is locked`` on
work submission: SPINE opens independent connections to ``queue.db`` /
``work_entries.db`` from the UI thread and the Ralph worker, and under the
default rollback journal a held read blocks a concurrent writer until the
busy_timeout expires. WAL removes that contention.
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import sqlite3

import sqlite_utils

from spine.persistence import sqlite_tuning
from spine.persistence.sqlite_tuning import retry_on_locked, tune_connection


def _LockedError() -> sqlite3.OperationalError:
    return sqlite3.OperationalError("database is locked")


class _FakeDb:
    """Minimal stand-in whose ``conn.execute`` replays a scripted sequence.

    Each ``PRAGMA journal_mode=WAL`` consumes the next item: a string is the
    journal mode returned (silent success/failure), an ``Exception`` is raised.
    Lets ``_enable_wal`` be tested deterministically — the real C-level
    ``sqlite3.Connection.execute`` is read-only and cannot be monkeypatched.
    """

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0
        self.conn = self

    def execute(self, sql, *args, **kwargs):
        self.calls += 1
        outcome = self._script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome

        class _Cur:
            def fetchone(self_inner):
                return (outcome,)

        return _Cur()


class TestTuneConnection:
    """``tune_connection`` applies WAL + a generous busy_timeout."""

    def test_enables_wal_and_busy_timeout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = tune_connection(sqlite_utils.Database(str(Path(tmpdir) / "t.db")))
            db["t"].create({"id": int}, pk="id")  # materialise the file
            assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            assert db.execute("PRAGMA busy_timeout").fetchone()[0] >= 30_000

    def test_returns_same_handle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = sqlite_utils.Database(str(Path(tmpdir) / "t.db"))
            assert tune_connection(db) is db

    def test_enable_wal_retries_transient_failure(self, monkeypatch):
        """A transient WAL-transition failure must be retried, not surrendered.

        On a freshly created ``queue.db`` the rollback-journal → WAL switch can
        silently return the old mode (or raise ``database is locked``) while
        another connection holds the database — e.g. the worker's dequeue poll
        racing the UI thread. Degrading to rollback-journal mode there is what
        surfaces 'database is locked' on the first onboarding insert, so the
        transition is retried until it lands.
        """
        monkeypatch.setattr(sqlite_tuning, "_WAL_RETRY_BASE_DELAY_S", 0.0)
        db = _FakeDb(["delete", _LockedError(), "wal"])
        assert sqlite_tuning._enable_wal(db) is True
        assert db.calls == 3, "expected two failed WAL attempts then a success"

    def test_enable_wal_gives_up_after_attempts(self, monkeypatch):
        """If WAL never takes, report failure rather than spinning forever."""
        monkeypatch.setattr(sqlite_tuning, "_WAL_RETRY_BASE_DELAY_S", 0.0)
        db = _FakeDb(["delete"] * 20)
        assert sqlite_tuning._enable_wal(db) is False
        assert db.calls == sqlite_tuning._WAL_RETRY_ATTEMPTS

    def test_enable_wal_propagates_non_lock_errors(self, monkeypatch):
        """A genuine SQL error during the pragma must not be swallowed."""
        monkeypatch.setattr(sqlite_tuning, "_WAL_RETRY_BASE_DELAY_S", 0.0)
        db = _FakeDb([sqlite3.OperationalError("disk I/O error")])
        try:
            sqlite_tuning._enable_wal(db)
        except sqlite3.OperationalError as exc:
            assert "disk I/O error" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected the non-lock error to propagate")

    def test_tune_connection_degrades_gracefully_with_warning(self, monkeypatch, caplog):
        """If WAL can never take, warn but still return a usable connection."""
        import logging

        monkeypatch.setattr(sqlite_tuning, "_WAL_RETRY_BASE_DELAY_S", 0.0)
        monkeypatch.setattr(sqlite_tuning, "_enable_wal", lambda db: False)
        with tempfile.TemporaryDirectory() as tmpdir:
            db = sqlite_utils.Database(str(Path(tmpdir) / "t.db"))
            with caplog.at_level(logging.WARNING):
                assert tune_connection(db) is db
            assert any("did not take" in r.message for r in caplog.records)

    def test_held_read_does_not_block_writer(self):
        """A reader holding a transaction must not lock out a concurrent writer.

        Under the default ``delete`` journal this writer would wait out the
        busy_timeout and raise ``database is locked``; WAL lets it through.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "t.db")
            seed = sqlite_utils.Database(path)
            seed["t"].create({"id": int, "v": int}, pk="id")
            seed["t"].insert({"id": 1, "v": 0})

            outcome: dict[str, object] = {}

            def reader() -> None:
                db = tune_connection(sqlite_utils.Database(path))
                db.execute("BEGIN")
                db.execute("SELECT * FROM t").fetchall()
                time.sleep(1.0)
                db.execute("COMMIT")

            def writer() -> None:
                db = tune_connection(sqlite_utils.Database(path))
                time.sleep(0.2)  # let the reader take its lock first
                try:
                    db["t"].insert({"id": None, "v": 99}, pk="id")
                    outcome["ok"] = True
                except Exception as exc:  # pragma: no cover - failure path
                    outcome["err"] = repr(exc)

            tr, tw = threading.Thread(target=reader), threading.Thread(target=writer)
            tr.start(), tw.start()
            tr.join(), tw.join()

            assert outcome.get("ok") is True, outcome.get("err")


class TestRetryOnLocked:
    """``retry_on_locked`` covers the lock-upgrade BUSY busy_timeout cannot."""

    def test_returns_value_when_no_error(self):
        assert retry_on_locked(lambda: 42) == 42

    def test_retries_then_succeeds(self):
        calls = {"n": 0}

        def flaky() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise sqlite3.OperationalError("database is locked")
            return "ok"

        result = retry_on_locked(flaky, base_delay_s=0.0)
        assert result == "ok"
        assert calls["n"] == 3

    def test_raises_after_exhausting_attempts(self):
        calls = {"n": 0}

        def always_locked() -> None:
            calls["n"] += 1
            raise sqlite3.OperationalError("database is locked")

        try:
            retry_on_locked(always_locked, attempts=3, base_delay_s=0.0)
        except sqlite3.OperationalError:
            pass
        else:  # pragma: no cover
            raise AssertionError("expected OperationalError to propagate")
        assert calls["n"] == 3

    def test_other_operational_errors_propagate_immediately(self):
        calls = {"n": 0}

        def bad_sql() -> None:
            calls["n"] += 1
            raise sqlite3.OperationalError("no such table: nope")

        try:
            retry_on_locked(bad_sql, base_delay_s=0.0)
        except sqlite3.OperationalError as exc:
            assert "no such table" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected OperationalError to propagate")
        assert calls["n"] == 1, "non-lock errors must not be retried"
