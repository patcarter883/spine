"""Audit service for recording and querying work entries.

Uses context-propagated thread_id and handles retries for transient failures.
"""

import logging
import time
from typing import Optional

from spine.models.work_entry import WorkEntry, WorkEntryStore
from spine.utils.thread_context import get_current_thread_id

logger = logging.getLogger(__name__)


class AuditService:
    """Service for recording auditable work entries with thread context.

    Each record_action call persists a WorkEntry using the thread_id from the
    current execution context (or an explicitly provided one). Transient SQLite
    failures are retried up to a configurable limit.
    """

    def __init__(
        self,
        store: Optional[WorkEntryStore] = None,
        db_path: str = ".spine/work_entries.db",
        max_retries: int = 3,
        retry_delay: float = 0.1,
    ):
        self._store = store or WorkEntryStore(db_path=db_path)
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    @property
    def store(self) -> WorkEntryStore:
        return self._store

    def record_action(
        self,
        action: str,
        details: Optional[dict] = None,
        thread_id: Optional[str] = None,
    ) -> WorkEntry:
        """Record a work entry for the given action.

        If thread_id is not provided, the value from the current thread
        context is used. If no context value is set, a ValueError is raised.

        Retries on transient errors (locked database, timeout).
        """
        resolved_thread_id = thread_id or get_current_thread_id()
        if not resolved_thread_id:
            raise ValueError(
                "thread_id is required either explicitly or via thread context"
            )

        entry = WorkEntry(
            thread_id=resolved_thread_id,
            action=action,
            details=details,
        )

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                self._store.upsert(entry)
                return entry
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "record_action attempt %d/%d failed: %s",
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )
                if attempt < self._max_retries:
                    time.sleep(self._retry_delay * (2 ** attempt))

        raise RuntimeError(
            f"Failed to record action '{action}' after {self._max_retries + 1} attempts"
        ) from last_error

    def get_entries(
        self,
        thread_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[WorkEntry]:
        """Retrieve work entries, optionally filtered by thread_id."""
        resolved = thread_id or get_current_thread_id()
        if resolved:
            return self._store.get_by_thread(resolved, limit=limit)
        return self._store.list_entries(limit=limit)

    def get_entries_by_action(self, action: str, limit: int = 100) -> list[WorkEntry]:
        return self._store.get_by_action(action, limit=limit)

    def query_entries(
        self,
        thread_id: Optional[str] = None,
        action: Optional[str] = None,
        timestamp_from: Optional[str] = None,
        timestamp_to: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[WorkEntry], int]:
        return self._store.query_entries(
            thread_id=thread_id,
            action=action,
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            limit=limit,
            offset=offset,
        )


__all__ = ["AuditService"]
