"""Tests for SPECIFY structured-output contract robustness.

Covers the three-layer defense against a synthesizer that emits the spec as
fenced JSON text instead of calling ``write_specification`` (trace
``019eaa90-2b02-7761-938b-69dd575f2cf6``):

1. ``ForceToolUntilCalledMiddleware`` — forces a tool call until the write
   succeeds (cross-provider via ``tool_choice``).
2. ``salvage_specification_from_text`` — recovers the spec from model text.
3. ``make_subgraph_node`` — auto-retries a structural contract failure once
   on a fresh thread before escalating to human review.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.agents.specify_tools import (
    _coerce_str_list,
    _extract_json_object,
    salvage_specification_from_text,
)
from spine.agents.tool_forcing import ForceToolUntilCalledMiddleware, _RELEASE, _tool_name
from spine.exceptions import CriticalContractFailure


# ── Layer 2: salvage ───────────────────────────────────────────────────────


class TestSalvage:
    def _spec_files(self, root: str, work_id: str) -> tuple[Path, Path]:
        base = Path(root) / ".spine" / "artifacts" / work_id / "specify"
        return base / "specification.json", base / "specification.md"

    def test_salvage_fenced_json_block(self, tmp_path):
        text = (
            "Here is the spec:\n```json\n"
            '{"title": "T", "summary": "S", '
            '"requirements": ["r1", "r2"], "scope_exclusions": ["none"]}\n'
            "```\nDone."
        )
        assert salvage_specification_from_text(text, str(tmp_path), "w1") is True
        js, md = self._spec_files(str(tmp_path), "w1")
        assert js.exists() and md.exists()
        data = json.loads(js.read_text())
        assert data["title"] == "T"
        assert data["requirements"] == ["r1", "r2"]
        # Markdown is tool-rendered, not the raw model text.
        assert md.read_text().startswith("# T")

    def test_salvage_bare_object_without_fence(self, tmp_path):
        text = 'noise {"title":"T","summary":"S","requirements":["r1"]} trailing'
        assert salvage_specification_from_text(text, str(tmp_path), "w2") is True

    def test_salvage_rejects_missing_requirements(self, tmp_path):
        text = '```json\n{"title":"T","summary":"S"}\n```'
        assert salvage_specification_from_text(text, str(tmp_path), "w3") is False
        js, _ = self._spec_files(str(tmp_path), "w3")
        assert not js.exists()

    def test_salvage_rejects_prose(self, tmp_path):
        assert salvage_specification_from_text("just some prose", str(tmp_path), "w4") is False

    def test_salvage_rejects_empty(self, tmp_path):
        assert salvage_specification_from_text("", str(tmp_path), "w5") is False

    def test_extract_prefers_fenced_over_bare(self):
        text = '{"a": 1}\n```json\n{"title": "real"}\n```'
        assert _extract_json_object(text) == {"title": "real"}

    def test_coerce_str_list_filters_none_and_blanks(self):
        assert _coerce_str_list(["a", " b ", "", None]) == ["a", "b"]
        assert _coerce_str_list("solo") == ["solo"]
        assert _coerce_str_list(None) == []


# ── Layer 1: forcing middleware ─────────────────────────────────────────────


class _FakeRequest:
    def __init__(self, model, messages):
        self.model = model
        self.messages = messages
        self.tool_choice = None

    def override(self, **kw):
        r = _FakeRequest(self.model, self.messages)
        r.tool_choice = self.tool_choice
        for k, v in kw.items():
            setattr(r, k, v)
        return r


def _openai_model():
    from langchain_openai import ChatOpenAI

    # Construction makes no network call; isinstance is what the middleware
    # checks. No base_url → treated as cloud OpenAI, where a named tool pin
    # is supported on the wire.
    return ChatOpenAI(model="gpt-4o-mini", api_key="x")


def _local_model():
    from langchain_openai import ChatOpenAI

    # Custom base_url marks a local OpenAI-compatible server (e.g. llama.cpp),
    # which rejects the object-form tool_choice a named pin serializes to.
    return ChatOpenAI(model="gemma", api_key="x", base_url="http://localhost:1/v1")


class TestForceToolMiddleware:
    GATE = [
        HumanMessage("do it"),
        AIMessage("", tool_calls=[{"name": "read_work_context", "args": {}, "id": "g"}]),
        ToolMessage("ctx", tool_call_id="g", name="read_work_context"),
    ]

    def test_fresh_turn_forces_any(self):
        mw = ForceToolUntilCalledMiddleware("write_specification", gate_tool="read_work_context")
        assert mw._decide(_FakeRequest(_openai_model(), [HumanMessage("x")])) == "any"

    def test_after_gate_pins_final_tool(self):
        mw = ForceToolUntilCalledMiddleware("write_specification", gate_tool="read_work_context")
        assert mw._decide(_FakeRequest(_openai_model(), self.GATE)) == "write_specification"

    def test_local_server_demotes_pin_to_any(self):
        # llama.cpp only parses string tool_choice values; a named pin would be
        # sent as the object form and silently dropped ("Wrong type supplied
        # for parameter 'tool_choice'"). Local models keep forcing "any".
        mw = ForceToolUntilCalledMiddleware("write_specification", gate_tool="read_work_context")
        assert mw._decide(_FakeRequest(_local_model(), self.GATE)) == "any"

    def test_no_gate_variant_never_pins(self):
        mw = ForceToolUntilCalledMiddleware("write_specification")
        # Orchestrator variant keeps forcing "any" so it can still use `recall`.
        assert mw._decide(_FakeRequest(_openai_model(), self.GATE)) == "any"

    def test_failed_write_keeps_forcing(self):
        mw = ForceToolUntilCalledMiddleware("write_specification", gate_tool="read_work_context")
        msgs = self.GATE + [
            AIMessage("", tool_calls=[{"name": "write_specification", "args": {}, "id": "w"}]),
            ToolMessage(
                "VALIDATION_ERROR: specification rejected before writing.",
                tool_call_id="w",
                name="write_specification",
            ),
        ]
        assert mw._decide(_FakeRequest(_openai_model(), msgs)) == "write_specification"

    def test_successful_write_releases(self):
        mw = ForceToolUntilCalledMiddleware("write_specification", gate_tool="read_work_context")
        msgs = self.GATE + [
            AIMessage("", tool_calls=[{"name": "write_specification", "args": {}, "id": "w"}]),
            ToolMessage(
                "specification.md (10 chars) and specification.json (20 chars) "
                "written to .spine/artifacts/w/specify/.",
                tool_call_id="w",
                name="write_specification",
            ),
        ]
        assert mw._decide(_FakeRequest(_openai_model(), msgs)) is _RELEASE

    def test_non_openai_model_is_noop(self):
        mw = ForceToolUntilCalledMiddleware("write_specification")

        class NotOpenAI:
            pass

        assert mw._decide(_FakeRequest(NotOpenAI(), [HumanMessage("x")])) is _RELEASE

    def test_apply_sets_tool_choice(self):
        mw = ForceToolUntilCalledMiddleware("write_specification", gate_tool="read_work_context")
        out = mw._apply(_FakeRequest(_openai_model(), self.GATE))
        assert out.tool_choice == "write_specification"

    # ── gate-drop backstop (trace 019ec913) ────────────────────────────────
    # A weak model can ignore the tool_choice pin and re-call the gate tool
    # forever, re-sending the heavy evidence prompt each turn. Once the gate
    # has fired we remove it from the tool surface so it physically can't.

    TOOLS = [
        {"type": "function", "function": {"name": "read_work_context"}},
        {"type": "function", "function": {"name": "write_specification"}},
    ]

    def _req_with_tools(self, model, messages, tools):
        req = _FakeRequest(model, messages)
        req.tools = list(tools)
        return req

    def test_gate_dropped_after_called(self):
        mw = ForceToolUntilCalledMiddleware("write_specification", gate_tool="read_work_context")
        kept = mw._gate_dropped_tools(self._req_with_tools(_openai_model(), self.GATE, self.TOOLS))
        assert [_tool_name(t) for t in kept] == ["write_specification"]

    def test_gate_not_dropped_before_called(self):
        mw = ForceToolUntilCalledMiddleware("write_specification", gate_tool="read_work_context")
        req = self._req_with_tools(_openai_model(), [HumanMessage("x")], self.TOOLS)
        assert mw._gate_dropped_tools(req) is None

    def test_apply_drops_gate_and_pins(self):
        mw = ForceToolUntilCalledMiddleware("write_specification", gate_tool="read_work_context")
        out = mw._apply(self._req_with_tools(_openai_model(), self.GATE, self.TOOLS))
        assert [_tool_name(t) for t in out.tools] == ["write_specification"]
        assert out.tool_choice == "write_specification"

    def test_local_server_drops_gate_so_any_forces_write(self):
        # The named pin degrades to "any" on a local server, but with the gate
        # removed "any" can only resolve to the single remaining write tool.
        mw = ForceToolUntilCalledMiddleware("write_specification", gate_tool="read_work_context")
        out = mw._apply(self._req_with_tools(_local_model(), self.GATE, self.TOOLS))
        assert [_tool_name(t) for t in out.tools] == ["write_specification"]
        assert out.tool_choice == "any"

    def test_no_gate_variant_keeps_all_tools(self):
        # The orchestrator (no gate_tool) legitimately re-uses `recall`, so its
        # tool surface must never be trimmed.
        mw = ForceToolUntilCalledMiddleware("write_specification")
        assert mw._gate_dropped_tools(self._req_with_tools(_openai_model(), self.GATE, self.TOOLS)) is None

    def test_gate_not_dropped_after_successful_write(self):
        mw = ForceToolUntilCalledMiddleware("write_specification", gate_tool="read_work_context")
        msgs = self.GATE + [
            AIMessage("", tool_calls=[{"name": "write_specification", "args": {}, "id": "w"}]),
            ToolMessage(
                "specification.md (10 chars) written to .spine/artifacts/w/specify/.",
                tool_call_id="w",
                name="write_specification",
            ),
        ]
        assert mw._gate_dropped_tools(self._req_with_tools(_openai_model(), msgs, self.TOOLS)) is None

    # ── end-after-final-write short-circuit (trace 019ec997) ────────────────
    # Once the structured write lands the artifact is already on disk, so the
    # trailing model call is pure waste — and a cancellation window in which
    # completed work can be discarded. ``before_model`` jumps straight to end.

    _DONE = [
        HumanMessage("do it"),
        AIMessage("", tool_calls=[{"name": "write_specification", "args": {}, "id": "w"}]),
        ToolMessage(
            "specification.md (10 chars) and specification.json (20 chars) "
            "written to .spine/artifacts/w/specify/.",
            tool_call_id="w",
            name="write_specification",
        ),
    ]

    def test_before_model_no_jump_before_write(self):
        mw = ForceToolUntilCalledMiddleware("write_specification")
        assert mw._should_end({"messages": [HumanMessage("x")]}) is None

    def test_before_model_ends_after_successful_write(self):
        mw = ForceToolUntilCalledMiddleware("write_specification")
        assert mw._should_end({"messages": self._DONE}) == {"jump_to": "end"}

    def test_before_model_does_not_end_on_failed_write(self):
        mw = ForceToolUntilCalledMiddleware("write_specification")
        msgs = [
            HumanMessage("do it"),
            AIMessage("", tool_calls=[{"name": "write_specification", "args": {}, "id": "w"}]),
            ToolMessage(
                "VALIDATION_ERROR: rejected.", tool_call_id="w", name="write_specification"
            ),
        ]
        assert mw._should_end({"messages": msgs}) is None

    def test_before_model_short_circuit_is_provider_agnostic(self):
        # Unlike tool_choice forcing (OpenAI-only), ending after the write is
        # correct for every provider — the wasted/cancellation-prone trailing
        # call happens regardless of model.
        mw = ForceToolUntilCalledMiddleware("write_specification")
        # _should_end never inspects the model, so no model is needed at all.
        assert mw._should_end({"messages": self._DONE}) == {"jump_to": "end"}

    def test_before_model_hooks_declare_end_jump(self):
        # The factory wires the conditional "→ end" edge only if the hook
        # advertises can_jump_to=["end"] via __can_jump_to__.
        assert getattr(ForceToolUntilCalledMiddleware.before_model, "__can_jump_to__", None) == [
            "end"
        ]
        assert getattr(ForceToolUntilCalledMiddleware.abefore_model, "__can_jump_to__", None) == [
            "end"
        ]

    @pytest.mark.asyncio
    async def test_abefore_model_ends_after_successful_write(self):
        mw = ForceToolUntilCalledMiddleware("write_specification")
        assert await mw.abefore_model({"messages": self._DONE}, None) == {"jump_to": "end"}


# ── Layer 3: structural retry in the subgraph wrapper ───────────────────────


class TestStructuralRetry:
    @staticmethod
    def _node(mock_subgraph):
        from spine.workflow.subgraph_wrapper import make_subgraph_node

        return make_subgraph_node(
            mock_subgraph,
            "specify",
            lambda p, c: {"phase": "specify", "work_id": p.get("work_id", "")},
            lambda r, p: {"current_phase": "specify", "status": "running"},
        )

    @pytest.mark.asyncio
    async def test_retries_once_then_succeeds(self):
        mock_subgraph = AsyncMock()
        mock_subgraph.ainvoke.side_effect = [
            CriticalContractFailure(phase="specify", reason="specification.json does not exist"),
            {"artifacts_output": {"specification.md": "ok"}, "phase_status": "success"},
        ]
        result = await self._node(mock_subgraph)({"work_id": "abc", "status": "running"}, None)

        assert result["status"] == "running"  # second attempt succeeded
        assert mock_subgraph.ainvoke.call_count == 2
        # The retry must use a *distinct* thread_id so it re-runs from START.
        threads = [
            call.args[1]["configurable"]["thread_id"]
            for call in mock_subgraph.ainvoke.call_args_list
        ]
        assert threads[0] == "abc_specify"
        assert threads[1] == "abc_specify_retry1"
        assert threads[0] != threads[1]

    @pytest.mark.asyncio
    async def test_persistent_failure_escalates(self):
        mock_subgraph = AsyncMock()
        mock_subgraph.ainvoke.side_effect = CriticalContractFailure(
            phase="specify", reason="specification.json does not exist"
        )
        result = await self._node(mock_subgraph)({"work_id": "abc", "status": "running"}, None)

        # Exhausted retries → escalate (error status surfaces as needs_review).
        assert result["status"] == "needs_review"
        assert result["phase_results"]["specify"]["status"] == "error"
        assert mock_subgraph.ainvoke.call_count == 2  # 1 initial + 1 retry

    @pytest.mark.asyncio
    async def test_non_contract_error_not_retried(self):
        mock_subgraph = AsyncMock()
        mock_subgraph.ainvoke.side_effect = RuntimeError("boom")
        result = await self._node(mock_subgraph)({"work_id": "abc", "status": "running"}, None)

        assert result["status"] == "needs_review"
        assert mock_subgraph.ainvoke.call_count == 1  # generic errors are not retried
