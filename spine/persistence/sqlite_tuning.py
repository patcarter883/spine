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

import logging
import sqlite3
import time
from typing import Callable, TypeVar

import sqlite_utils

logger = logging.getLogger(__name__)

# Generous wait for the one permitted WAL writer before surfacing
# ``database is locked``. A submit/dequeue write is sub-millisecond, so 30s is
# only ever approached under pathological contention.
_BUSY_TIMEOUT_MS = 30_000

# Bounded retry for the one ``database is locked`` case ``busy_timeout`` cannot
# cover: a writer-vs-writer lock upgrade, where SQLite returns SQLITE_BUSY
# *immediately* (bypassing the busy handler) to break a potential deadlock.
# Exponential backoff from 50ms tops out around 3.2s of cumulative wait.
_RETRY_ATTEMPTS = 7
_RETRY_BASE_DELAY_S = 0.05

T = TypeVar("T")


def tune_connection(db: sqlite_utils.Database) -> sqlite_utils.Database:
    """Apply WAL + ``busy_timeout`` to a ``sqlite_utils`` database, in place.

    Idempotent and safe to call on every connection open. Returns the same
    ``db`` handle for convenient chaining.
    """
    # WAL is recorded in the file header; enabling it once makes it stick for
    # all connections, but the call is cheap and idempotent. ``PRAGMA
    # journal_mode`` *silently* fails to switch (returning the unchanged mode)
    # if another connection holds the database at this instant, so verify the
    # result rather than trusting it — a silent regression to rollback-journal
    # mode reintroduces the lock-upgrade deadlock this module exists to avoid.
    mode = db.conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    if str(mode).lower() != "wal":
        logger.warning(
            "PRAGMA journal_mode=WAL did not take (got %r); connection is in "
            "rollback-journal mode and may hit 'database is locked'.",
            mode,
        )
    # Per-connection; must be set on every fresh connection.
    db.conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    return db


def retry_on_locked(
    fn: Callable[[], T],
    *,
    attempts: int = _RETRY_ATTEMPTS,
    base_delay_s: float = _RETRY_BASE_DELAY_S,
) -> T:
    """Run ``fn``, retrying on ``database is locked`` with exponential backoff.

    WAL + ``busy_timeout`` make a writer wait its turn, but SQLite returns
    SQLITE_BUSY *immediately* — bypassing ``busy_timeout`` — when a connection
    must upgrade a read lock to a write lock while another writer holds the
    lock (deadlock avoidance). That transient is exactly what this retries;
    every other ``OperationalError`` propagates unchanged. ``fn`` must be
    idempotent on retry (e.g. an atomic ``UPDATE … WHERE``), since it may run
    more than once.
    """
    delay = base_delay_s
    for attempt in range(attempts):
        try:
            return fn()
        except sqlite3.OperationalError as exc:
            if "database is locked" not in str(exc).lower() or attempt == attempts - 1:
                raise
            logger.debug(
                "database is locked (attempt %d/%d); backing off %.0fms",
                attempt + 1,
                attempts,
                delay * 1000,
            )
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")  # pragma: no cover
