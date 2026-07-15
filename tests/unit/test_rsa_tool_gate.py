"""RSA is gated off tool-bound requests; schema/plain requests keep it.

Zaya-serve guidance (2026-07-15): one-shot tool calls never benefit from
recursive self-aggregation. Probe evidence pinned the gate to tools-only —
plain json_schema on the current serve build is unreliable, so
response_format calls keep their configured rsa.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from spine.agents.helpers import _build_local_model, _gate_rsa_for_tool_calls


class TestGateFunction:
    def _payload(self, **over):
        base = {
            "model": "zaya",
            "messages": [],
            "extra_body": {"top_k": 40, "rsa": {"n": 2, "k": 1, "t": 1}},
        }
        base.update(over)
        return base

    def test_tools_present_disables_rsa(self):
        out = _gate_rsa_for_tool_calls(self._payload(tools=[{"type": "function"}]))
        assert out["extra_body"]["rsa"] is False
        assert out["extra_body"]["top_k"] == 40  # other extras untouched

    def test_no_tools_keeps_rsa(self):
        out = _gate_rsa_for_tool_calls(self._payload())
        assert out["extra_body"]["rsa"] == {"n": 2, "k": 1, "t": 1}

    def test_response_format_alone_keeps_rsa(self):
        out = _gate_rsa_for_tool_calls(
            self._payload(response_format={"type": "json_schema"})
        )
        assert out["extra_body"]["rsa"] == {"n": 2, "k": 1, "t": 1}

    def test_empty_tools_list_keeps_rsa(self):
        out = _gate_rsa_for_tool_calls(self._payload(tools=[]))
        assert out["extra_body"]["rsa"] == {"n": 2, "k": 1, "t": 1}

    def test_no_rsa_configured_is_noop(self):
        p = {"model": "zaya", "tools": [{"type": "function"}], "extra_body": {"top_k": 40}}
        out = _gate_rsa_for_tool_calls(p)
        assert out is p  # untouched

    def test_input_payload_not_mutated(self):
        p = self._payload(tools=[{"type": "function"}])
        _gate_rsa_for_tool_calls(p)
        assert p["extra_body"]["rsa"] == {"n": 2, "k": 1, "t": 1}


@pytest.fixture
def wire_sink():
    captured = []

    class Sink(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            captured.append(json.loads(self.rfile.read(n)))
            resp = json.dumps({
                "id": "x", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                              "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Sink)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1", captured
    srv.shutdown()


def _zaya_cfg(base_url):
    return {
        "base_url": base_url,
        "api_key": "x",
        "context_window": 60000,
        "rsa": {"n": 2, "k": 1, "t": 1, "max_tokens": 2048,
                "think_budget": 512, "agg_max_tokens": 768},
    }


def test_wire_tool_call_sends_rsa_false(wire_sink):
    base_url, captured = wire_sink
    from langchain_core.tools import tool

    @tool
    def write_specification(title: str) -> str:
        """Write the spec."""
        return "ok"

    model = _build_local_model("openai:/models/test", _zaya_cfg(base_url))
    asyncio.run(model.bind_tools([write_specification], tool_choice="required").ainvoke("hi"))
    body = captured[-1]
    assert body["rsa"] is False
    assert body["tools"]


def test_wire_plain_call_keeps_rsa_dict(wire_sink):
    base_url, captured = wire_sink
    model = _build_local_model("openai:/models/test", _zaya_cfg(base_url))
    asyncio.run(model.ainvoke("hi"))
    body = captured[-1]
    assert body["rsa"] == {"n": 2, "k": 1, "t": 1, "max_tokens": 2048,
                           "think_budget": 512, "agg_max_tokens": 768}
