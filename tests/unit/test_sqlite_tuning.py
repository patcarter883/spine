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

from spine.persistence.sqlite_tuning import retry_on_locked, tune_connection


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
