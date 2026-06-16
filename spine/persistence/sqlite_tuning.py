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

# The rollback-journal → WAL transition needs a momentary exclusive lock, so it
# fails (silently returning the old mode, or raising SQLITE_BUSY) whenever
# *another* connection holds the database at that instant — e.g. the Ralph
# worker's 5s dequeue poll racing the UI thread's first connection on a freshly
# created ``queue.db``. The failure is transient: retrying lets the transition
# land as soon as the contending connection releases. Same backoff envelope as
# ``_RETRY_ATTEMPTS`` (≈3.2s cumulative).
_WAL_RETRY_ATTEMPTS = 7
_WAL_RETRY_BASE_DELAY_S = 0.05

T = TypeVar("T")


def _enable_wal(db: sqlite_utils.Database) -> bool:
    """Switch ``db`` to WAL, retrying the transient contention failures.

    Returns ``True`` once the connection is in WAL mode. ``PRAGMA
    journal_mode=WAL`` is unreliable under concurrent access: it can *silently*
    return the unchanged mode, or raise ``database is locked``, when another
    connection holds the database while the transition tries to take its brief
    exclusive lock. Both are transient — once WAL is recorded in the file header
    it sticks for every connection — so we retry rather than degrade to
    rollback-journal mode (which reintroduces the lock-upgrade deadlock this
    module exists to avoid).
    """
    delay = _WAL_RETRY_BASE_DELAY_S
    for attempt in range(_WAL_RETRY_ATTEMPTS):
        try:
            mode = db.conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if str(mode).lower() == "wal":
                return True
        except sqlite3.OperationalError as exc:
            if "database is locked" not in str(exc).lower():
                raise
        if attempt < _WAL_RETRY_ATTEMPTS - 1:
            time.sleep(delay)
            delay *= 2
    return False


def tune_connection(db: sqlite_utils.Database) -> sqlite_utils.Database:
    """Apply WAL + ``busy_timeout`` to a ``sqlite_utils`` database, in place.

    Idempotent and safe to call on every connection open. Returns the same
    ``db`` handle for convenient chaining.
    """
    # WAL is recorded in the file header; enabling it once makes it stick for
    # all connections, but the call is cheap and idempotent. The transition can
    # fail transiently under concurrent access, so ``_enable_wal`` retries it —
    # a silent regression to rollback-journal mode reintroduces the lock-upgrade
    # deadlock this module exists to avoid.
    if not _enable_wal(db):
        mode = db.conn.execute("PRAGMA journal_mode").fetchone()[0]
        logger.warning(
            "PRAGMA journal_mode=WAL did not take after %d attempts (got %r); "
            "connection is in rollback-journal mode and may hit 'database is "
            "locked'.",
            _WAL_RETRY_ATTEMPTS,
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
