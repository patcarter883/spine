"""SPINE WebSocket event bus — in-process pub/sub for state changes.

Provides a singleton async event bus that the dispatcher, worker, and
state machine publish to whenever work state changes.  Streamlit UI
clients connect via the WebSocket endpoint (``spine.ui.ws_server``) and
receive filtered push notifications instead of polling.

Bus lifecycle
-------------
- ``get_bus()`` returns the process-level singleton (created on first call).
- Publishers call ``bus.publish(event_type, payload)`` — non-blocking.
- Subscribers are asyncio.Queue instances drained by the WebSocket server.
- The bus is safe to call from sync code (uses ``asyncio.run_coroutine_threadsafe``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """A single state-change event."""

    event_type: str
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(
            {
                "event_type": self.event_type,
                "payload": self.payload,
                "timestamp": self.timestamp,
            }
        )


class WSEventBus:
    """Async pub/sub event bus for SPINE state changes.

    Thread-safe for publishing (``publish`` works from sync or async
    contexts).  Subscribers are asyncio.Queue instances; each gets a copy
    of every published event.
    """

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[Event]] = []
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    # ── Publishing ──

    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        """Publish an event to all subscribers."""
        event = Event(event_type=event_type, payload=payload)
        async with self._lock:
            dead: list[asyncio.Queue[Event]] = []
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    # Drop oldest and push (bounded queue).
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        q.put_nowait(event)
                    except asyncio.QueueFull:
                        dead.append(q)
            for d in dead:
                self._subscribers.remove(d)
                logger.warning("Removed stuck subscriber from event bus")

    def publish_sync(self, event_type: str, payload: dict[str, Any]) -> None:
        """Publish from a synchronous context (thread-safe).

        Uses ``asyncio.run_coroutine_threadsafe`` to schedule the
        publish on the running event loop.  No-op if no loop is running.
        """
        if self._loop is None:
            # Try to grab the running loop.
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.debug("No running event loop — event dropped: %s", event_type)
                return

        if self._loop.is_closed():
            self._loop = None
            return

        asyncio.run_coroutine_threadsafe(self.publish(event_type, payload), self._loop)

    # ── Subscribing ──

    async def subscribe(self, maxsize: int = 256) -> asyncio.Queue[Event]:
        """Create a new subscriber queue."""
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        async with self._lock:
            self._subscribers.append(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue[Event]) -> None:
        """Remove a subscriber queue."""
        async with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


# ── Singleton ──

_BUS: WSEventBus | None = None


def get_bus() -> WSEventBus:
    """Return the process-level event bus singleton."""
    global _BUS
    if _BUS is None:
        _BUS = WSEventBus()
    return _BUS
