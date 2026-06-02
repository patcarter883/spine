"""Shared SQLite connection tuning for SPINE's ``sqlite_utils`` databases.

SPINE opens several independent connections to the same on-disk databases
(``.spine/queue.db`` and ``.spine/work_entries.db``): the Streamlit UI thread,
the ``RalphLoopWorker`` daemon thread, and that worker's ``ralph-oob`` thread
pool each call their respective ``_get_db`` factory, and every call creates a
*fresh* ``sqlite3.Connection``. For SQLite's locking model, connections in the
same process behave like separate processes.

Under the default rollback journal (``journal_mode=delete``) a writer takes an
exclusive lock on the whole file, and — critically — when two connections each
hold a read lock and one tries to upgrade to a write, SQLite returns
``SQLITE_BUSY`` *immediately, without invoking the busy handler*, to avoid a
deadlock. That is the ``database is locked`` error users hit on submit, and a
``busy_timeout`` alone cannot fix it.

``tune_connection`` switches every connection to **WAL** mode (readers never
block writers; that lock-upgrade deadlock does not occur) and sets a generous
``busy_timeout`` so the single permitted writer waits its turn instead of
failing fast. WAL is persisted in the database header, so enabling it on any
connection makes it stick for all of them; ``busy_timeout`` is per-connection
and must be set every time.
"""

from __future__ import annotations

import sqlite_utils

# Generous wait for the one permitted WAL writer before surfacing
# ``database is locked``. A submit/dequeue write is sub-millisecond, so 30s is
# only ever approached under pathological contention.
_BUSY_TIMEOUT_MS = 30_000


def tune_connection(db: sqlite_utils.Database) -> sqlite_utils.Database:
    """Apply WAL + ``busy_timeout`` to a ``sqlite_utils`` database, in place.

    Idempotent and safe to call on every connection open. Returns the same
    ``db`` handle for convenient chaining.
    """
    # WAL is recorded in the file header; enabling it once makes it stick for
    # all connections, but the call is cheap and idempotent.
    db.conn.execute("PRAGMA journal_mode=WAL")
    # Per-connection; must be set on every fresh connection.
    db.conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    return db
