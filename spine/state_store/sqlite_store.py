"""SQLite-backed state store for LangGraph checkpoint data."""

import json
import sqlite3
from pathlib import Path
from typing import Optional

from .base import StateStore, StateStoreError


class SqliteStateStore(StateStore):
    """Reads thread state from LangGraph's SqliteSaver checkpoint database.

    This store is read-only for the checkpoint data written by LangGraph.
    It queries the underlying SQLite tables to discover thread IDs and
    checkpoint state without going through the LangGraph API.
    """

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            if not self._db_path.exists():
                raise StateStoreError(
                    f"Checkpoint database not found: {self._db_path}"
                )
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def get_thread_ids(self) -> list[str]:
        if not self._db_path.exists():
            return []
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            thread_ids: list[str] = []
            for table in tables:
                try:
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = [col[1] for col in cursor.fetchall()]
                    if "thread_id" in columns:
                        cursor.execute(f"SELECT DISTINCT thread_id FROM {table}")
                        for row in cursor.fetchall():
                            tid = row[0]
                            if tid and tid not in thread_ids:
                                thread_ids.append(tid)
                except Exception:
                    continue
            return thread_ids
        except Exception:
            return []

    def get_state(self, thread_id: str) -> Optional[dict]:
        if not self._db_path.exists():
            return None
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            for table in ["checkpoint_blobs", "checkpoints"]:
                if table not in tables:
                    continue
                cursor.execute(
                    f"SELECT parent_id, ts, blob "
                    f"FROM {table} WHERE thread_id=? ORDER BY ts DESC LIMIT 1",
                    (thread_id,),
                )
                row = cursor.fetchone()
                if row:
                    blob_data = row[2]
                    if blob_data:
                        try:
                            if isinstance(blob_data, str):
                                return json.loads(blob_data)
                            return json.loads(blob_data)
                        except (json.JSONDecodeError, TypeError):
                            try:
                                import orjson
                                return orjson.loads(blob_data)
                            except Exception:
                                return {"raw": str(blob_data)}

            for table in tables:
                try:
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = [col[1] for col in cursor.fetchall()]
                    if "thread_id" in columns:
                        cursor.execute(
                            f"SELECT blob FROM {table} WHERE thread_id=? ORDER BY rowid DESC LIMIT 1",
                            (thread_id,),
                        )
                        row = cursor.fetchone()
                        if row:
                            blob_data = row[0]
                            if blob_data:
                                try:
                                    if isinstance(blob_data, str):
                                        return json.loads(blob_data)
                                    return json.loads(blob_data)
                                except Exception:
                                    try:
                                        import orjson
                                        return orjson.loads(blob_data)
                                    except Exception:
                                        return {"table": table, "raw": str(blob_data)}
                except Exception:
                    continue
            return None
        except Exception:
            return None

    def thread_exists(self, thread_id: str) -> bool:
        return thread_id in self.get_thread_ids()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
