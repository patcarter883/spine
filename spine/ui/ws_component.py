"""SPINE WebSocket live-update bridge for Streamlit.

Embeds a tiny invisible HTML iframe that connects to the SPINE WebSocket
server. When a state-change event arrives, it updates the ``_ws_ts``
query parameter in the parent frame, which triggers a Streamlit rerun
with fresh data — replacing the old ``<meta http-equiv=refresh>`` hack.

A guard timestamp prevents rerun loops: the iframe only triggers a
reload if the event's timestamp is newer than the last processed event.
"""

from __future__ import annotations

import os

import streamlit as st

DEFAULT_WS_PORT = int(os.getenv("SPINE_WS_PORT", "8765"))

_WS_HTML = """<html><body>
<div id="s" style="font:9px monospace;color:#ccc">ws…</div>
<script>
(function(){{
  var p={port},s=document.getElementById('s'),ws,d=500,D=15000,last=0;
  function c(){{
    try{{ws=new WebSocket('ws://localhost:'+p)}}catch(e){{return r()}}
    ws.onopen=function(){{s.textContent='🟢';d=500}};
    ws.onmessage=function(e){{
      try{{
        var v=JSON.parse(e.data);
        if(v.event_type && v.timestamp && v.timestamp>last){{
          last=v.timestamp;
          var u=new URL(window.parent.location.href);
          u.searchParams.set('_ws_ts',v.timestamp.toString());
          window.parent.location.replace(u.toString());
        }}
      }}catch(x){{}}
    }};
    ws.onclose=function(){{s.textContent='🔴';r()}};
    ws.onerror=function(){{try{{ws.close()}}catch(e){{}}}};
  }}
  function r(){{setTimeout(function(){{d=Math.min(d*2,D);c()}},d)}}
  c();
}})();
</script></body></html>"""


def render_ws_client(ws_port: int | None = None) -> None:
    """Render an invisible WebSocket bridge that triggers reruns on events.

    The embedded iframe maintains a persistent WS connection.  When a
    push event arrives (``work_progress``, ``work_completed``,
    ``work_failed``, etc.), it navigates the parent frame to the same
    URL with an updated ``_ws_ts`` query parameter — Streamlit detects
    the change and reruns the app.

    The ``last`` guard ensures that stale or duplicate events don't
    cause rerun loops.

    Args:
        ws_port: WebSocket server port.  Defaults to ``SPINE_WS_PORT``
            env var or 8765.
    """
    port = ws_port or DEFAULT_WS_PORT
    st.iframe(_WS_HTML.format(port=port), height=20)
