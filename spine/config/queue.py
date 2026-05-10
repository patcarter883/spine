"""Queue/broker connection configuration for background task processing.

Provides Redis-backed queue with connection pooling for concurrent
submission handling, with SQLite fallback when Redis is unavailable.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class QueueStatus(str, Enum):
    """Status values for queued tasks."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QueueTask:
    """A task in the queue."""
    id: str
    status: QueueStatus
    task_type: str
    payload: dict[str, Any]
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    attempts: int = 0
    max_attempts: int = 3


class QueueBackend:
    """Abstract queue backend — subclass to implement."""

    def enqueue(self, task_type: str, payload: dict) -> str:
        raise NotImplementedError

    def dequeue(self, task_types: list[str] | None = None) -> QueueTask | None:
        raise NotImplementedError

    def acknowledge(self, task_id: str, result: dict | None = None) -> None:
        raise NotImplementedError

    def fail(self, task_id: str, error: str) -> None:
        raise NotImplementedError

    def get_status(self, task_id: str) -> QueueStatus | None:
        raise NotImplementedError


class RedisQueueBackend(QueueBackend):
    """Redis-backed queue with connection pooling."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self._available = False
        try:
            import redis as redis_mod

            self._pool = redis_mod.ConnectionPool.from_url(
                redis_url,
                max_connections=10,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            self._redis = redis_mod.Redis(connection_pool=self._pool)
            self._redis.ping()
            self._available = True
        except Exception:
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def enqueue(self, task_type: str, payload: dict) -> str:
        task_id = str(uuid.uuid4())
        task = {
            "id": task_id,
            "status": QueueStatus.PENDING.value,
            "task_type": task_type,
            "payload": payload,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "attempts": 0,
            "max_attempts": 3,
        }
        self._redis.lpush(f"queue:{task_type}", json.dumps(task))
        self._redis.hset("tasks:metadata", task_id, json.dumps(task))
        return task_id

    def dequeue(self, task_types: list[str] | None = None) -> QueueTask | None:
        types = task_types or ["default"]
        for t in types:
            data = self._redis.rpop(f"queue:{t}")
            if data:
                task_data = json.loads(data)
                task = QueueTask(
                    id=task_data["id"],
                    status=QueueStatus.RUNNING,
                    task_type=task_data["task_type"],
                    payload=task_data["payload"],
                    created_at=task_data["created_at"],
                    started_at=datetime.now(timezone.utc).isoformat(),
                    attempts=task_data.get("attempts", 0) + 1,
                    max_attempts=task_data.get("max_attempts", 3),
                )
                self._redis.hset("tasks:metadata", task.id, json.dumps(task_data))
                return task
        return None

    def acknowledge(self, task_id: str, result: dict | None = None) -> None:
        data = self._redis.hget("tasks:metadata", task_id)
        if data:
            task_data = json.loads(data)
            task_data["status"] = QueueStatus.SUCCESS.value
            task_data["completed_at"] = datetime.now(timezone.utc).isoformat()
            task_data["result"] = result
            self._redis.hset("tasks:metadata", task_id, json.dumps(task_data))

    def fail(self, task_id: str, error: str) -> None:
        data = self._redis.hget("tasks:metadata", task_id)
        if data:
            task_data = json.loads(data)
            task_data["status"] = QueueStatus.FAILED.value
            task_data["completed_at"] = datetime.now(timezone.utc).isoformat()
            task_data["error"] = error
            self._redis.hset("tasks:metadata", task_id, json.dumps(task_data))

    def get_status(self, task_id: str) -> QueueStatus | None:
        data = self._redis.hget("tasks:metadata", task_id)
        if data:
            return QueueStatus(json.loads(data)["status"])
        return None


class SqliteQueueBackend(QueueBackend):
    """SQLite-backed queue for environments without Redis."""

    def __init__(self, db_path: str = ".spine/queue.db"):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS queue_tasks (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'pending',
                    task_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    result TEXT,
                    error TEXT,
                    attempts INTEGER DEFAULT 0,
                    max_attempts INTEGER DEFAULT 3
                )
            """)
            conn.commit()
            conn.close()

    def enqueue(self, task_type: str, payload: dict) -> str:
        task_id = str(uuid.uuid4())
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                """INSERT INTO queue_tasks
                   (id, status, task_type, payload, created_at, attempts, max_attempts)
                   VALUES (?, ?, ?, ?, ?, 0, 3)""",
                (task_id, QueueStatus.PENDING.value, task_type,
                 json.dumps(payload), datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            conn.close()
        return task_id

    def dequeue(self, task_types: list[str] | None = None) -> QueueTask | None:
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            if task_types:
                placeholders = ",".join("?" * len(task_types))
                cursor = conn.execute(
                    f"""SELECT id, status, task_type, payload, created_at,
                               started_at, completed_at, result, error,
                               attempts, max_attempts
                        FROM queue_tasks
                        WHERE status = ? AND task_type IN ({placeholders})
                        ORDER BY created_at ASC LIMIT 1""",
                    [QueueStatus.PENDING.value] + task_types,
                )
            else:
                cursor = conn.execute(
                    """SELECT id, status, task_type, payload, created_at,
                              started_at, completed_at, result, error,
                              attempts, max_attempts
                       FROM queue_tasks
                       WHERE status = ?
                       ORDER BY created_at ASC LIMIT 1""",
                    [QueueStatus.PENDING.value],
                )
            row = cursor.fetchone()
            if row:
                task = QueueTask(
                    id=row[0], status=QueueStatus.RUNNING, task_type=row[2],
                    payload=json.loads(row[3]), created_at=row[4],
                    started_at=datetime.now(timezone.utc).isoformat(),
                    completed_at=row[6], result=json.loads(row[7]) if row[7] else None,
                    error=row[8], attempts=row[9] + 1, max_attempts=row[10],
                )
                conn.execute(
                    "UPDATE queue_tasks SET status = ?, started_at = ?, attempts = ? WHERE id = ?",
                    (QueueStatus.RUNNING.value, task.started_at, task.attempts, task.id),
                )
                conn.commit()
                conn.close()
                return task
            conn.close()
            return None

    def acknowledge(self, task_id: str, result: dict | None = None) -> None:
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "UPDATE queue_tasks SET status = ?, completed_at = ?, result = ? WHERE id = ?",
                (QueueStatus.SUCCESS.value, datetime.now(timezone.utc).isoformat(),
                 json.dumps(result) if result else None, task_id),
            )
            conn.commit()
            conn.close()

    def fail(self, task_id: str, error: str) -> None:
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.execute(
                "UPDATE queue_tasks SET status = ?, completed_at = ?, error = ? WHERE id = ?",
                (QueueStatus.FAILED.value, datetime.now(timezone.utc).isoformat(), error, task_id),
            )
            conn.commit()
            conn.close()

    def get_status(self, task_id: str) -> QueueStatus | None:
        conn = sqlite3.connect(self._db_path)
        cursor = conn.execute("SELECT status FROM queue_tasks WHERE id = ?", (task_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return QueueStatus(row[0])
        return None


class TaskQueue:
    """Task queue with automatic backend selection.

    Attempts Redis first, falls back to SQLite.
    Provides connection pooling for concurrent submission handling.
    """

    def __init__(self, redis_url: str | None = None, db_path: str = ".spine/queue.db"):
        self._redis_url = redis_url
        self._db_path = db_path
        self._backend: QueueBackend = self._init_backend()

    def _init_backend(self) -> QueueBackend:
        if self._redis_url:
            backend = RedisQueueBackend(self._redis_url)
            if backend.available:
                return backend
        return SqliteQueueBackend(self._db_path)

    @property
    def backend_type(self) -> str:
        return "redis" if isinstance(self._backend, RedisQueueBackend) else "sqlite"

    def enqueue(self, task_type: str, payload: dict) -> str:
        """Enqueue a task and return its ID."""
        return self._backend.enqueue(task_type, payload)

    def dequeue(self, task_types: list[str] | None = None) -> QueueTask | None:
        """Dequeue the next pending task."""
        return self._backend.dequeue(task_types)

    def acknowledge(self, task_id: str, result: dict | None = None) -> None:
        """Mark a task as successfully completed."""
        self._backend.acknowledge(task_id, result)

    def fail(self, task_id: str, error: str) -> None:
        """Mark a task as failed with an error message."""
        self._backend.fail(task_id, error)

    def get_status(self, task_id: str) -> QueueStatus | None:
        """Get the current status of a task."""
        return self._backend.get_status(task_id)


__all__ = [
    "QueueStatus",
    "QueueTask",
    "QueueBackend",
    "RedisQueueBackend",
    "SqliteQueueBackend",
    "TaskQueue",
]
