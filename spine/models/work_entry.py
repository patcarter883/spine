"""Work entry model with SQLite persistence."""

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class WorkEntry:
    """Represents a single auditable work entry in a thread."""

    def __init__(
        self,
        thread_id: str,
        action: str,
        details: Optional[dict] = None,
        entry_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        created_at: Optional[str] = None,
    ):
        self.entry_id = entry_id or str(uuid.uuid4())
        self.thread_id = thread_id
        self.action = action
        self.details = details or {}
        now = datetime.now(timezone.utc).isoformat()
        self.timestamp = timestamp or now
        self.created_at = created_at or now

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "thread_id": self.thread_id,
            "action": self.action,
            "details": json.dumps(self.details),
            "timestamp": self.timestamp,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkEntry":
        details_raw = data.get("details", "{}")
        if isinstance(details_raw, str):
            details = json.loads(details_raw) if details_raw else {}
        else:
            details = details_raw or {}
        return cls(
            entry_id=data.get("entry_id"),
            thread_id=data["thread_id"],
            action=data["action"],
            details=details,
            timestamp=data.get("timestamp"),
            created_at=data.get("created_at"),
        )


class WorkEntryStore:
    """SQLite-backed store for work entries with thread-safe operations."""

    def __init__(self, db_path: str = ".spine/work_entries.db"):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS work_entries (
                    entry_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    details TEXT DEFAULT '{}',
                    timestamp TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_work_entries_thread
                ON work_entries(thread_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_work_entries_action
                ON work_entries(action)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_work_entries_timestamp
                ON work_entries(timestamp)
            """)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def upsert(self, entry: WorkEntry) -> WorkEntry:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO work_entries
                       (entry_id, thread_id, action, details, timestamp, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        entry.entry_id,
                        entry.thread_id,
                        entry.action,
                        json.dumps(entry.details),
                        entry.timestamp,
                        entry.created_at,
                    ),
                )
                conn.commit()
            return entry

    def get(self, entry_id: str) -> Optional[WorkEntry]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM work_entries WHERE entry_id = ?", (entry_id,)
            ).fetchone()
            if row is None:
                return None
            return WorkEntry.from_dict(dict(row))

    def get_by_thread(self, thread_id: str, limit: int = 100) -> list[WorkEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM work_entries WHERE thread_id = ? ORDER BY timestamp DESC LIMIT ?",
                (thread_id, limit),
            ).fetchall()
            return [WorkEntry.from_dict(dict(r)) for r in rows]

    def get_by_action(self, action: str, limit: int = 100) -> list[WorkEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM work_entries WHERE action = ? ORDER BY timestamp DESC LIMIT ?",
                (action, limit),
            ).fetchall()
            return [WorkEntry.from_dict(dict(r)) for r in rows]

    def list_entries(self, limit: int = 50, offset: int = 0) -> list[WorkEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM work_entries ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return [WorkEntry.from_dict(dict(r)) for r in rows]

    def query_entries(
        self,
        thread_id: Optional[str] = None,
        action: Optional[str] = None,
        timestamp_from: Optional[str] = None,
        timestamp_to: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[WorkEntry], int]:
        conditions = []
        params: list[str] = []

        if thread_id:
            conditions.append("thread_id = ?")
            params.append(thread_id)
        if action:
            conditions.append("action = ?")
            params.append(action)
        if timestamp_from:
            conditions.append("timestamp >= ?")
            params.append(timestamp_from)
        if timestamp_to:
            conditions.append("timestamp <= ?")
            params.append(timestamp_to)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        with self._connect() as conn:
            count_row = conn.execute(
                f"SELECT COUNT(*) as cnt FROM work_entries {where_clause}", params
            ).fetchone()
            total = count_row["cnt"] if count_row else 0

            rows = conn.execute(
                f"SELECT * FROM work_entries {where_clause} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                [*params, str(limit), str(offset)],
            ).fetchall()

        return [WorkEntry.from_dict(dict(r)) for r in rows], total

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM work_entries").fetchone()
            return row["cnt"] if row else 0


__all__ = ["WorkEntry", "WorkEntryStore"]
