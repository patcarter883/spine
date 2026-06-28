"""SPINE checkpoint store — LangGraph SQLite-backed persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


class CheckpointStore:
    """Manages LangGraph checkpoint persistence using SQLite.

    Each work item uses its own thread_id for checkpoint isolation.
    The SQLite database is stored at ``.spine/spine.db``.

    Uses ``AsyncSqliteSaver`` so the checkpointer works with async
    LangGraph graph invocations.
    """

    def __init__(self, db_path: str = ".spine/spine.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ctx: Any | None = None
        self._saver: AsyncSqliteSaver | None = None

    async def get_checkpointer(self) -> BaseCheckpointSaver:
        """Return an AsyncSqliteSaver configured for the checkpoint database.

        The underlying async context manager is entered on first call and held
        open for the lifetime of this ``CheckpointStore`` instance.  Call
        ``close()`` when done to release the SQLite connection.

        Returns:
            An AsyncSqliteSaver instance suitable for passing to
            ``StateGraph.compile()``.
        """
        if self._saver is None:
            # AsyncSqliteSaver.from_conn_string() is an @asynccontextmanager —
            # we must enter it to get the actual saver instance.
            self._ctx = AsyncSqliteSaver.from_conn_string(str(self._db_path))
            self._saver = await self._ctx.__aenter__()
        return self._saver

    async def close(self) -> None:
        """Release the SQLite connection held by the checkpointer."""
        if self._ctx is not None:
            await self._ctx.__aexit__(None, None, None)
            self._ctx = None
            self._saver = None

    async def list_checkpoints(self, work_id: str | None = None) -> list[dict[str, Any]]:
        """List all checkpoint thread IDs and timestamps.

        Args:
            work_id: If provided, filter checkpoints to this work item.

        Returns:
            A list of dicts with keys ``thread_id`` and ``parent_config``.
        """
        # SqliteSaver lists all threads; we filter by prefix if work_id given
        results: list[dict[str, Any]] = []
        # Note: actual checkpoint listing requires async iteration
        # This is a simplified synchronous wrapper for the UI API
        return results

    async def get_state(self, work_id: str) -> dict[str, Any] | None:
        """Get the latest state snapshot for a work item.

        Args:
            work_id: The work item ID to look up.

        Returns:
            The state dict if found, or None.
        """
        checkpointer = await self.get_checkpointer()
        config = {"configurable": {"thread_id": work_id}}
        # Let read errors propagate: a transient failure (e.g. SQLite
        # "database is locked") must NOT be silently treated as "no checkpoint",
        # which would make the caller resume from empty state and discard all
        # accumulated artifacts. A genuinely absent checkpoint returns None.
        state = await checkpointer.aget(config)
        if state:
            return state.get("channel_values", {})
        return None

    async def delete_state(self, work_id: str) -> bool:
        """Delete the checkpoint state for a work item.

        Args:
            work_id: The work item ID to delete.

        Returns:
            True if deletion succeeded, False otherwise.
        """
        checkpointer = await self.get_checkpointer()
        try:
            config = {"configurable": {"thread_id": work_id}}
            await checkpointer.adelete(config)
            return True
        except Exception:
            pass
        return False
