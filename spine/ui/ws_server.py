"""SPINE WebSocket server — lightweight async WS endpoint for live UI updates.

Runs alongside the Streamlit app (on a separate port).  Clients connect,
optionally filter by ``work_id``, and receive JSON push events from
``spine.ui.ws_bus`` whenever work state changes.

Usage
-----
The server is started automatically by the Streamlit app (see
``spine/ui/app.py``) on port 8765 by default.  To start manually::

    python -m spine.ui.ws_server

The ``SPINE_WS_PORT`` env var overrides the default port.
"""

from __future__ import annotations

import asyncio
import logging
import os

import websockets

from spine.ui.ws_bus import Event, get_bus

logger = logging.getLogger(__name__)

DEFAULT_WS_PORT = 8765


async def _client_handler(websocket: websockets.ServerConnection) -> None:
    """Handle a single WebSocket client connection.

    Reads optional filter messages from the client (JSON with
    ``work_id`` key) and pushes events from the bus.
    """
    bus = get_bus()
    q = await bus.subscribe()
    try:
        # Drain events from the subscription queue and push to client.
        while True:
            event: Event = await q.get()
            if event is None:
                break
            try:
                await websocket.send(event.to_json())
            except websockets.ConnectionClosed:
                break
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.debug("WebSocket client error", exc_info=True)
    finally:
        await bus.unsubscribe(q)


async def run_server(host: str = "0.0.0.0", port: int | None = None) -> None:
    """Start the WebSocket server (blocking).

    Args:
        host: Bind address.
        port: Port number.  Falls back to ``SPINE_WS_PORT`` env var,
            then ``DEFAULT_WS_PORT``.
    """
    port = port or int(os.getenv("SPINE_WS_PORT", str(DEFAULT_WS_PORT)))
    bus = get_bus()
    # Bind the bus to this event loop so publish_sync works from threads.
    bus._loop = asyncio.get_running_loop()

    async with websockets.serve(_client_handler, host, port):
        logger.info("SPINE WebSocket server listening on ws://%s:%d", host, port)
        await asyncio.Future()  # run forever


def start_ws_server(port: int | None = None) -> None:
    """Start the WS server in a daemon thread (non-blocking).

    Called once at Streamlit app startup.
    """
    import threading

    def _run() -> None:
        asyncio.run(run_server(port=port))

    t = threading.Thread(target=_run, name="spine-ws-server", daemon=True)
    t.start()
    logger.info("SPINE WebSocket server thread started")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(run_server())
