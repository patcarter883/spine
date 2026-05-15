"""SPINE WebSocket status indicator for Streamlit.

Embeds a tiny invisible HTML iframe that connects to the SPINE WebSocket
server and displays a connection status indicator (green/red dot).

Previously, this component triggered full-page reruns via URL manipulation
(``window.parent.location.replace``) whenever a WS event arrived. This
caused form inputs (like the Submit Work description field) to be cleared
mid-typing. That behaviour has been removed — pages that need live data
refreshes now use ``@st.fragment(run_every=...)`` for isolated re-renders
that preserve widget state.
"""

from __future__ import annotations

import os

import streamlit as st

DEFAULT_WS_PORT = int(os.getenv("SPINE_WS_PORT", "8765"))

# Minimal iframe: shows WS connection status (green dot / red dot) only.
# No page-rerun logic — pages use @st.fragment(run_every=...) instead.
_WS_HTML = """<html><body>
<div id="s" style="font:9px monospace;color:#ccc">ws…</div>
<script>
(function(){{
  var p={port},s=document.getElementById('s'),ws,d=500,D=15000;
  function c(){{
    try{{ws=new WebSocket('ws://localhost:'+p)}}catch(e){{return r()}}
    ws.onopen=function(){{s.textContent='🟢';d=500}};
    ws.onclose=function(){{s.textContent='🔴';r()}};
    ws.onerror=function(){{try{{ws.close()}}catch(e){{}}}};
  }}
  function r(){{setTimeout(function(){{d=Math.min(d*2,D);c()}},d)}}
  c();
}})();
</script></body></html>"""


def render_ws_client(ws_port: int | None = None) -> None:
    """Render a WebSocket connection status indicator in the sidebar.

    The embedded iframe maintains a persistent WS connection and shows
    a green dot when connected, red dot when disconnected. It does NOT
    trigger page reruns — use ``@st.fragment(run_every=...)`` on pages
    that need live data updates.

    Args:
        ws_port: WebSocket server port.  Defaults to ``SPINE_WS_PORT``
            env var or 8765.
    """
    port = ws_port or DEFAULT_WS_PORT
    st.iframe(_WS_HTML.format(port=port), height=20)
