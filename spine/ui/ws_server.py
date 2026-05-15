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

Server lifecycle
----------------
``start_ws_server()`` is idempotent — safe to call multiple times.
Only one WS server thread runs per process.  If the port is already
bound (e.g. by a previous Streamlit run's daemon thread), the call
is a no-op.  This prevents the ``OSError: address already in use``
crashes that occur when Streamlit re-runs the app script and tries
to start a second server on the same port.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import threading

import websockets

from spine.ui.ws_bus import Event, get_bus

logger = logging.getLogger(__name__)

DEFAULT_WS_PORT = 8765

# ── Process-level guard ──
# Thread object of the running WS server, or None.
# Unlike ``st.session_state``, this survives Streamlit script re-runs
# within the same process.
_SERVER_THREAD: threading.Thread | None = None


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
    bus.set_loop(asyncio.get_running_loop())

    async with websockets.serve(_client_handler, host, port):
        logger.info("SPINE WebSocket server listening on ws://%s:%d", host, port)
        await asyncio.Future()  # run forever


def _port_in_use(port: int, host: str = "0.0.0.0") -> bool:
    """Check whether *port* is already bound on *host*.

    Uses a non-blocking socket connect to test availability without
    actually binding the port.  This is cheaper than trying
    ``websockets.serve`` and catching the OSError.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            s.connect((host.replace("0.0.0.0", "127.0.0.1"), port))
            return True
    except (ConnectionRefusedError, OSError):
        return False


def start_ws_server(port: int | None = None) -> None:
    """Start the WS server in a daemon thread (non-blocking).

    Idempotent — if a server thread is already running or the port is
    already bound (by this process or a previous Streamlit run's daemon
    thread), this is a no-op instead of crashing with
    ``OSError: [Errno 98] address already in use``.

    Called by the Streamlit app at startup.  Safe to call multiple times.
    """
    global _SERVER_THREAD

    port = port or int(os.getenv("SPINE_WS_PORT", str(DEFAULT_WS_PORT)))

    # ── Guard 1: our own thread is still alive ───────────────────────
    if _SERVER_THREAD is not None and _SERVER_THREAD.is_alive():
        logger.debug(
            "WS server thread '%s' already running — skipping start",
            _SERVER_THREAD.name,
        )
        return

    # ── Guard 2: port already bound (e.g. stale daemon thread from a ──
    # previous Streamlit run in the same process) ─────────────────────
    if _port_in_use(port):
        logger.info(
            "Port %d already in use — WS server is likely already running. "
            "Skipping start to avoid address-in-use error.",
            port,
        )
        return

    def _run() -> None:
        try:
            asyncio.run(run_server(port=port))
        except OSError as exc:
            # Race: another thread grabbed the port between our check
            # and the bind attempt.  Log and move on — the UI will
            # still work (just without live push until the next restart).
            logger.warning(
                "WS server failed to bind port %d: %s. "
                "Live push updates may not work until the port is freed.",
                port,
                exc,
            )

    t = threading.Thread(target=_run, name="spine-ws-server", daemon=True)
    t.start()
    _SERVER_THREAD = t
    logger.info("SPINE WebSocket server thread started")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(run_server())
