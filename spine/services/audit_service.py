"""SPINE audit service — event logging for workflow execution."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import sqlite_utils


class AuditService:
    """Records workflow events to a SQLite database for auditing.

    Table schema:
        - id: INTEGER PRIMARY KEY
        - work_id: TEXT
        - event_type: TEXT (e.g. "phase_start", "phase_complete", "critic_review")
        - phase: TEXT
        - details: TEXT (JSON)
        - timestamp: TEXT (ISO 8601)
    """

    def __init__(self, db_path: str = ".spine/audit.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_table()

    def _get_db(self) -> sqlite_utils.Database:
        """Return a fresh connection (safe across threads)."""
        return sqlite_utils.Database(str(self._db_path))

    def _ensure_table(self) -> None:
        """Create the audit_events table if it doesn't exist."""
        db = self._get_db()
        if "audit_events" not in db.table_names():
            db["audit_events"].create(
                {
                    "id": int,
                    "work_id": str,
                    "event_type": str,
                    "phase": str,
                    "details": str,
                    "timestamp": str,
                },
                pk="id",
            )

    def log_event(
        self,
        work_id: str,
        event_type: str,
        phase: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record an audit event.

        Args:
            work_id: The work item ID.
            event_type: Type of event (e.g. "phase_start", "critic_review").
            phase: The workflow phase name.
            details: Optional dict of additional details (stored as JSON).
        """
        db = self._get_db()
        db["audit_events"].insert(
            {
                "work_id": work_id,
                "event_type": event_type,
                "phase": phase,
                "details": json.dumps(details or {}),
                "timestamp": datetime.now().isoformat(),
            }
        )

    def query_events(
        self,
        work_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query audit events with optional filters.

        Args:
            work_id: Filter by work item ID.
            event_type: Filter by event type.
            limit: Maximum number of events to return.

        Returns:
            A list of event dicts, newest first.
        """
        db = self._get_db()
        table = db["audit_events"]
        where_clauses: list[str] = []
        where_args: list[Any] = []

        if work_id is not None:
            where_clauses.append("work_id = ?")
            where_args.append(work_id)
        if event_type is not None:
            where_clauses.append("event_type = ?")
            where_args.append(event_type)

        where = " AND ".join(where_clauses) if where_clauses else None
        results = table.rows_where(
            where=where,
            where_args=where_args if where_args else None,
            order_by="-timestamp",
            limit=limit,
        )
        # Parse details JSON back to dict
        events = []
        for row in results:
            row["details"] = json.loads(row.get("details", "{}"))
            events.append(row)
        return events
