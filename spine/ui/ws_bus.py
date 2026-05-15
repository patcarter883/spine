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
- Events published before the event loop is bound are buffered and flushed
  automatically once :meth:`set_loop` or :meth:`publish_sync` binds a loop.

Thread safety
-------------
The bus is constructed from the main thread before any async loop exists,
so ``_lock`` is created lazily on first async use (not at ``__init__``
time) to avoid ``asyncio.Lock`` being bound to the wrong event loop.
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

    Events published before a ``_loop`` is bound are buffered in
    ``_pending_buffer`` and flushed automatically when the event loop
    becomes available via :meth:`set_loop` or :meth:`publish_sync`.
    """

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[Event]] = []
        # Created lazily on first async use to avoid binding to the wrong
        # event loop (the bus may be instantiated from the main thread
        # before the WS server thread starts its own loop).
        self._lock: asyncio.Lock | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # Buffer for events published before the event loop is bound.
        self._pending_buffer: list[tuple[str, dict[str, Any]]] = []

    def _get_lock(self) -> asyncio.Lock:
        """Return the lock, creating it lazily on the running event loop."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    # ── Publishing ──

    async def publish(self, event_type: str, payload: dict[str, Any]) -> None:
        """Publish an event to all subscribers."""
        event = Event(event_type=event_type, payload=payload)
        async with self._get_lock():
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
        publish on the running event loop.  If the event loop is not
        yet bound, the event is buffered and will be flushed once a
        loop becomes available via :meth:`set_loop`.

        This is safe to call from worker threads (e.g.
        ``RalphLoopWorker._loop``) even before the WebSocket server
        thread has started its event loop — events are never silently
        dropped.
        """
        if self._loop is not None and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.publish(event_type, payload), self._loop)
            return

        # Loop is None or closed — try to grab the running loop from the
        # current thread (e.g. if called inside an async context).
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

        if self._loop is not None and not self._loop.is_closed():
            asyncio.run_coroutine_threadsafe(self.publish(event_type, payload), self._loop)
            # Flush any events that were buffered while the loop was absent.
            self._flush_pending()
            return

        # Still no loop — buffer the event so it's not silently dropped.
        self._pending_buffer.append((event_type, payload))
        logger.debug(
            "No event loop available — buffered event: %s (buffer size now %d)",
            event_type,
            len(self._pending_buffer),
        )

    def _flush_pending(self) -> None:
        """Publish all buffered events now that a loop is available."""
        if not self._pending_buffer:
            return
        if self._loop is None or self._loop.is_closed():
            return
        batch = self._pending_buffer[:]
        self._pending_buffer.clear()
        for event_type, payload in batch:
            asyncio.run_coroutine_threadsafe(self.publish(event_type, payload), self._loop)
        logger.info("Flushed %d buffered events to event bus", len(batch))

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the bus to an event loop and flush any buffered events.

        Called by the WebSocket server thread once its event loop is
        running, so that ``publish_sync()`` calls from worker threads
        can deliver events via the correct loop.

        Args:
            loop: The running event loop to bind to.
        """
        self._loop = loop
        self._flush_pending()

    # ── Subscribing ──

    async def subscribe(self, maxsize: int = 256) -> asyncio.Queue[Event]:
        """Create a new subscriber queue."""
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        async with self._get_lock():
            self._subscribers.append(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue[Event]) -> None:
        """Remove a subscriber queue."""
        async with self._get_lock():
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
