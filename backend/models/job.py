"""Job data model with SQLite persistence."""

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Job:
    """Represents a submitted work job."""

    def __init__(
        self,
        requirement: str,
        method: str = "Quick Work",
        project_type: str = "Greenfield",
        llm_provider: str = "ollama",
        parallel_agents: int = 3,
        idempotency_key: Optional[str] = None,
        job_id: Optional[str] = None,
        status: JobStatus = JobStatus.QUEUED,
        thread_id: Optional[str] = None,
        error_message: Optional[str] = None,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
        completed_at: Optional[str] = None,
    ):
        self.job_id = job_id or str(uuid.uuid4())
        self.idempotency_key = idempotency_key
        self.requirement = requirement
        self.method = method
        self.project_type = project_type
        self.llm_provider = llm_provider
        self.parallel_agents = parallel_agents
        self.status = status
        self.thread_id = thread_id
        self.error_message = error_message
        now = datetime.now(timezone.utc).isoformat()
        self.created_at = created_at or now
        self.updated_at = updated_at or now
        self.completed_at = completed_at

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "idempotency_key": self.idempotency_key,
            "requirement": self.requirement,
            "method": self.method,
            "project_type": self.project_type,
            "llm_provider": self.llm_provider,
            "parallel_agents": self.parallel_agents,
            "status": self.status.value,
            "thread_id": self.thread_id,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Job":
        return cls(
            job_id=data.get("job_id"),
            idempotency_key=data.get("idempotency_key"),
            requirement=data["requirement"],
            method=data.get("method", "Quick Work"),
            project_type=data.get("project_type", "Greenfield"),
            llm_provider=data.get("llm_provider", "ollama"),
            parallel_agents=data.get("parallel_agents", 3),
            status=JobStatus(data["status"]),
            thread_id=data.get("thread_id"),
            error_message=data.get("error_message"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            completed_at=data.get("completed_at"),
        )


class JobStore:
    """SQLite-backed job store with thread-safe operations."""

    def __init__(self, db_path: str = ".spine/jobs.db"):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    idempotency_key TEXT UNIQUE,
                    requirement TEXT NOT NULL,
                    method TEXT NOT NULL DEFAULT 'Quick Work',
                    project_type TEXT NOT NULL DEFAULT 'Greenfield',
                    llm_provider TEXT NOT NULL DEFAULT 'ollama',
                    parallel_agents INTEGER NOT NULL DEFAULT 3,
                    status TEXT NOT NULL DEFAULT 'queued',
                    thread_id TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_jobs_idempotency ON jobs(idempotency_key)
            """)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def create(self, job: Job) -> Job:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO jobs
                       (job_id, idempotency_key, requirement, method, project_type,
                        llm_provider, parallel_agents, status, thread_id,
                        error_message, created_at, updated_at, completed_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        job.job_id,
                        job.idempotency_key,
                        job.requirement,
                        job.method,
                        job.project_type,
                        job.llm_provider,
                        job.parallel_agents,
                        job.status.value,
                        job.thread_id,
                        job.error_message,
                        job.created_at,
                        job.updated_at,
                        job.completed_at,
                    ),
                )
                conn.commit()
            return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return None
            return Job.from_dict(dict(row))

    def get_by_idempotency_key(self, key: str) -> Optional[Job]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE idempotency_key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            return Job.from_dict(dict(row))

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        thread_id: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Optional[Job]:
        with self._lock:
            with self._connect() as conn:
                now = datetime.now(timezone.utc).isoformat()
                completed_at = now if status in (JobStatus.COMPLETED, JobStatus.FAILED) else None
                conn.execute(
                    """UPDATE jobs
                       SET status = ?, thread_id = COALESCE(?, thread_id),
                           error_message = ?, updated_at = ?,
                           completed_at = COALESCE(?, completed_at)
                       WHERE job_id = ?""",
                    (status.value, thread_id, error_message, now, completed_at, job_id),
                )
                conn.commit()
            return self.get(job_id)

    def list_jobs(self, limit: int = 50, offset: int = 0) -> list[Job]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            return [Job.from_dict(dict(r)) for r in rows]
