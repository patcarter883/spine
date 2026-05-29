"""Schemas + supervisor + worker for the researcher micro-loop.

Covers :mod:`spine.agents.researcher_supervisor` in isolation — the
``run_explore_do_node`` orchestrator is tested separately in
``test_explore_supervisor_loop.py``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from spine.agents import researcher_supervisor as rs
from spine.agents.researcher_supervisor import (
    FindingStatus,
    StructuredFinding,
    SupervisorDirective,
    ToolClass,
    TOOL_CLASS_TO_TOOLNAMES,
    _enforce_directive_contract,
    _extract_anchors_from_tool_payload,
    _extract_finding_from_worker_messages,
    _initialization_directive,
    _terminating_directive,
    _validate_directive_response,
    filter_extra_tools_for_class,
    render_history_as_evidence,
    run_supervisor_node,
    run_worker_node,
)


# ── Tool-class filtering ───────────────────────────────────────────────


class _FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name


def test_tool_class_taxonomy_covers_all_researcher_tools():
    """Every callable on the researcher surface (codebase_query,
    search_codebase, ast_extract_symbol) must appear in at least one
    tool class — otherwise that tool can never be invoked.
    """
    all_allowed = set()
    for names in TOOL_CLASS_TO_TOOLNAMES.values():
        all_allowed.update(names)
    assert {"codebase_query", "search_codebase", "ast_extract_symbol"} <= all_allowed


def test_filter_extra_tools_for_class_picks_only_allowed():
    tools = [
        _FakeTool("codebase_query"),
        _FakeTool("search_codebase"),
        _FakeTool("ast_extract_symbol"),
        _FakeTool("execute"),  # off-class, should never be exposed
    ]
    assert {t.name for t in filter_extra_tools_for_class(tools, ToolClass.SEARCH)} == {
        "codebase_query",
        "search_codebase",
    }
    assert {t.name for t in filter_extra_tools_for_class(tools, ToolClass.READ_SOURCE)} == {
        "codebase_query",
        "ast_extract_symbol",
    }
    assert {t.name for t in filter_extra_tools_for_class(tools, ToolClass.TRACE_DEPS)} == {
        "codebase_query",
    }


def test_filter_extra_tools_drops_unknown_tools():
    """A tool object without a ``.name`` attribute must be dropped."""
    weird = object()  # no .name
    out = filter_extra_tools_for_class([weird], ToolClass.SEARCH)
    assert out == []


# ── Directive contract enforcement ─────────────────────────────────────


def test_initialization_directive_seeds_search_class():
    d = _initialization_directive("how does X work?")
    assert d.is_complete is False
    assert d.allowed_tool_class == ToolClass.SEARCH
    assert "how does X work?" in d.next_directive


def test_terminating_directive_is_complete():
    d = _terminating_directive("supervisor unavailable")
    assert d.is_complete is True
    assert d.allowed_tool_class is None
    assert "supervisor unavailable" in d.analysis_and_reasoning


def test_enforce_contract_terminates_when_tool_class_missing():
    """If is_complete=False but allowed_tool_class is None, the worker
    can't dispatch — treat as terminating rather than guessing a class.
    """
    bad = SupervisorDirective(
        analysis_and_reasoning="r",
        is_complete=False,
        next_directive="do it",
        allowed_tool_class=None,
    )
    fixed = _enforce_directive_contract(bad)
    assert fixed.is_complete is True


def test_enforce_contract_passes_valid_continue():
    good = SupervisorDirective(
        analysis_and_reasoning="r",
        is_complete=False,
        next_directive="get_source for Foo",
        allowed_tool_class=ToolClass.READ_SOURCE,
    )
    assert _enforce_directive_contract(good) is good


def test_enforce_contract_passes_valid_terminate():
    done = SupervisorDirective(
        analysis_and_reasoning="enough evidence",
        is_complete=True,
    )
    assert _enforce_directive_contract(done) is done


# ── _validate_directive_response shape tolerance ───────────────────────


def test_validate_response_accepts_pydantic_instance():
    d = SupervisorDirective(
        analysis_and_reasoning="r", is_complete=True
    )
    assert _validate_directive_response(d, "wid").is_complete is True


def test_validate_response_accepts_dict():
    raw = {
        "analysis_and_reasoning": "r",
        "is_complete": False,
        "next_directive": "find_symbol Foo",
        "allowed_tool_class": "find_symbol",
    }
    out = _validate_directive_response(raw, "wid")
    assert out.is_complete is False
    assert out.allowed_tool_class == ToolClass.FIND_SYMBOL


def test_validate_response_accepts_json_string():
    raw = json.dumps(
        {
            "analysis_and_reasoning": "r",
            "is_complete": False,
            "next_directive": "search",
            "allowed_tool_class": "search",
        }
    )
    out = _validate_directive_response(raw, "wid")
    assert out.allowed_tool_class == ToolClass.SEARCH


def test_validate_response_terminates_on_garbage():
    out = _validate_directive_response(object(), "wid")
    assert out.is_complete is True


def test_validate_response_terminates_on_invalid_dict():
    out = _validate_directive_response({"not": "a directive"}, "wid")
    assert out.is_complete is True


# ── run_supervisor_node ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_supervisor_cycle_zero_skips_llm_call(monkeypatch):
    """Cycle 0 with no history must return the seed directive without
    touching the model — saves a turn when there's nothing to evaluate.
    """

    def _boom(*args, **kwargs):
        raise AssertionError("resolve_model must not be called on cycle 0")

    monkeypatch.setattr(rs, "resolve_model", _boom)

    d = await run_supervisor_node(
        state={"work_id": "w"},
        config=None,
        phase_path="specify/subagents/researcher/supervisor",
        global_goal="topic A",
        latest_finding=None,
        evaluation_history=[],
        cycle_idx=0,
        max_cycles=6,
    )
    assert d.is_complete is False
    assert d.allowed_tool_class == ToolClass.SEARCH


@pytest.mark.asyncio
async def test_supervisor_happy_path_dict_response(monkeypatch):
    """When the model emits a valid dict, the supervisor returns the
    parsed SupervisorDirective.
    """
    captured: dict[str, Any] = {}

    class _Structured:
        async def ainvoke(self, messages, **kw):
            captured["messages"] = messages
            return {
                "analysis_and_reasoning": "found a symbol; now trace callers",
                "is_complete": False,
                "next_directive": "get_dependents of SpineConfig",
                "allowed_tool_class": "trace_deps",
            }

    class _Model:
        def with_structured_output(self, schema):
            return _Structured()

    monkeypatch.setattr(rs, "resolve_model", lambda *a, **kw: _Model())

    finding = StructuredFinding(
        tool_name="codebase_query",
        tool_class=ToolClass.FIND_SYMBOL,
        status=FindingStatus.SUCCESS,
        target_path="spine/config.py",
        matched_symbols=["SpineConfig"],
        structured_code_block="class SpineConfig:",
    )
    d = await run_supervisor_node(
        state={"work_id": "w"},
        config=None,
        phase_path="plan/subagents/researcher/supervisor",
        global_goal="how does config thread through?",
        latest_finding=finding,
        evaluation_history=[finding],
        cycle_idx=1,
        max_cycles=6,
    )
    assert d.is_complete is False
    assert d.allowed_tool_class == ToolClass.TRACE_DEPS
    # Supervisor prompt must include the topic and the latest finding
    user_msg = captured["messages"][1].content
    assert "how does config thread through?" in user_msg
    assert "SpineConfig" in user_msg
    assert "spine/config.py" in user_msg


@pytest.mark.asyncio
async def test_supervisor_terminates_when_resolve_model_fails(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("no model")

    monkeypatch.setattr(rs, "resolve_model", _boom)

    d = await run_supervisor_node(
        state={"work_id": "w"},
        config=None,
        phase_path="specify/subagents/researcher/supervisor",
        global_goal="topic",
        latest_finding=StructuredFinding(
            tool_name="t", tool_class=ToolClass.SEARCH, status=FindingStatus.SUCCESS
        ),
        evaluation_history=[],
        cycle_idx=1,
        max_cycles=4,
    )
    assert d.is_complete is True


@pytest.mark.asyncio
async def test_supervisor_terminates_when_structured_output_unsupported(monkeypatch):
    class _Model:
        def with_structured_output(self, schema):
            raise NotImplementedError("not supported by this model")

    monkeypatch.setattr(rs, "resolve_model", lambda *a, **kw: _Model())

    d = await run_supervisor_node(
        state={"work_id": "w"},
        config=None,
        phase_path="specify/subagents/researcher/supervisor",
        global_goal="topic",
        latest_finding=StructuredFinding(
            tool_name="t", tool_class=ToolClass.SEARCH, status=FindingStatus.SUCCESS
        ),
        evaluation_history=[],
        cycle_idx=1,
        max_cycles=4,
    )
    assert d.is_complete is True


# ── Worker: message → StructuredFinding extraction ─────────────────────


def test_extract_finding_from_successful_tool_message():
    """Worker that issued one tool call returning a JSON payload with
    file_path + symbol_name yields a SUCCESS finding with anchors set.
    """
    payload = json.dumps(
        {"file_path": "spine/agents/factory.py", "symbol_name": "build_phase_agent"}
    )
    messages = [
        HumanMessage(content="dispatch"),
        AIMessage(content="calling tool"),
        ToolMessage(content=payload, name="codebase_query", tool_call_id="t1"),
        AIMessage(content="Found build_phase_agent at line 297."),
    ]
    f = _extract_finding_from_worker_messages(
        messages=messages, tool_class=ToolClass.FIND_SYMBOL
    )
    assert f.status == FindingStatus.SUCCESS
    assert f.tool_name == "codebase_query"
    assert f.target_path == "spine/agents/factory.py"
    assert "build_phase_agent" in f.matched_symbols


def test_extract_finding_from_results_list_payload():
    payload = json.dumps(
        {
            "results": [
                {"file_path": "spine/a.py", "symbol_name": "AlphaTool"},
                {"file_path": "spine/b.py", "symbol_name": "BetaTool"},
            ]
        }
    )
    messages = [
        ToolMessage(content=payload, name="search_codebase", tool_call_id="t1"),
    ]
    f = _extract_finding_from_worker_messages(
        messages=messages, tool_class=ToolClass.SEARCH
    )
    assert f.target_path == "spine/a.py"
    assert "AlphaTool" in f.matched_symbols and "BetaTool" in f.matched_symbols


def test_extract_finding_from_error_tool_message():
    """A ToolMessage with status=error becomes a FindingStatus.ERROR
    finding carrying the error body in ``execution_error_details``.
    """
    err_msg = ToolMessage(
        content="ToolValidationError: missing 'name' arg",
        name="codebase_query",
        tool_call_id="t1",
    )
    err_msg.status = "error"
    f = _extract_finding_from_worker_messages(
        messages=[err_msg], tool_class=ToolClass.READ_SOURCE
    )
    assert f.status == FindingStatus.ERROR
    assert "missing 'name'" in f.execution_error_details


def test_extract_finding_from_no_tool_call_is_empty():
    """If the worker produced text but no tool call, we record EMPTY
    rather than guessing a SUCCESS from the narrative."""
    messages = [
        HumanMessage(content="dispatch"),
        AIMessage(content="I think the answer is X."),
    ]
    f = _extract_finding_from_worker_messages(
        messages=messages, tool_class=ToolClass.TRACE_DEPS
    )
    assert f.status == FindingStatus.EMPTY
    assert "I think the answer" in f.structured_code_block


def test_extract_finding_caps_long_body():
    """A multi-KB tool result must be truncated in the finding."""
    big = "Z" * 20_000
    messages = [ToolMessage(content=big, name="codebase_query", tool_call_id="t1")]
    f = _extract_finding_from_worker_messages(
        messages=messages, tool_class=ToolClass.READ_SOURCE
    )
    # Cap is _FINDING_SNIPPET_CHAR_CAP (~2 KB) with an ellipsis marker.
    assert len(f.structured_code_block) < 2_100


def test_extract_anchors_from_non_json_returns_empty():
    target, syms = _extract_anchors_from_tool_payload("not json at all")
    assert target == ""
    assert syms == []


# ── run_worker_node ────────────────────────────────────────────────────


class _StubBoundModel:
    """Stand-in for ``model.bind_tools(scoped_tools)`` — returns a
    pre-canned AIMessage with the configured tool_calls on .ainvoke().
    """

    def __init__(self, ai_msg: AIMessage) -> None:
        self._ai_msg = ai_msg

    async def ainvoke(self, messages, **kwargs):
        # Capture the messages on the instance so tests can inspect them.
        self.last_messages = list(messages)
        return self._ai_msg


class _StubTool:
    """Stand-in for a LangChain BaseTool — exposes .name and async .ainvoke."""

    def __init__(self, name: str, result):
        self.name = name
        self._result = result
        self.calls: list = []

    async def ainvoke(self, args):
        self.calls.append(args)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


@pytest.mark.asyncio
async def test_worker_returns_error_finding_when_no_bound_model_for_class():
    """If the supervisor picked a tool class the loop hasn't bound a
    model for, return an ERROR finding rather than crashing — the
    supervisor sees it next cycle and can pick a different class.
    """
    directive = SupervisorDirective(
        analysis_and_reasoning="r",
        is_complete=False,
        next_directive="do it",
        allowed_tool_class=ToolClass.TRACE_DEPS,
    )
    f = await run_worker_node(
        state={"work_id": "w"},
        config=None,
        topic="x",
        directive=directive,
        bound_models={},  # empty — no model bound for TRACE_DEPS
        system_prompt="role",
    )
    assert f.status == FindingStatus.ERROR
    assert f.tool_class == ToolClass.TRACE_DEPS


@pytest.mark.asyncio
async def test_worker_invokes_correct_bound_model_and_executes_tool_once():
    """Happy path: supervisor picks READ_SOURCE → worker invokes that
    class's bound model ONCE → executes the FIRST tool call manually →
    returns a SUCCESS finding distilled from the AIMessage + ToolMessage.

    This pins the one-shot semantics that fix the agent-loop bloat: the
    worker MUST NOT call the model a second time after the tool result.
    """
    payload = json.dumps({"file_path": "spine/x.py", "symbol_name": "X"})

    # Build a stub AIMessage that the bound model returns — it carries a
    # single tool_call which the worker must execute exactly once.
    ai_msg = AIMessage(
        content="calling tool",
        tool_calls=[
            {
                "name": "codebase_query",
                "args": {"action": "get_source", "name": "X"},
                "id": "tc-1",
                "type": "tool_call",
            }
        ],
    )
    bound = _StubBoundModel(ai_msg)
    tool = _StubTool("codebase_query", payload)

    # Wrong class entry — must NOT be touched.
    other_bound = _StubBoundModel(AIMessage(content="wrong class"))
    other_tool = _StubTool("codebase_query", "should not run")

    bound_models = {
        ToolClass.SEARCH: (other_bound, [other_tool]),
        ToolClass.READ_SOURCE: (bound, [tool]),
    }

    directive = SupervisorDirective(
        analysis_and_reasoning="r",
        is_complete=False,
        next_directive="get_source for X",
        allowed_tool_class=ToolClass.READ_SOURCE,
    )
    f = await run_worker_node(
        state={"work_id": "w"},
        config=None,
        topic="how does X work?",
        directive=directive,
        bound_models=bound_models,
        system_prompt="researcher-role",
    )

    # Right class chosen, wrong-class entries untouched.
    assert len(tool.calls) == 1, "expected exactly ONE tool execution"
    assert tool.calls[0] == {"action": "get_source", "name": "X"}
    assert other_tool.calls == [], "wrong-class tool must not run"

    # Finding distilled correctly.
    assert f.status == FindingStatus.SUCCESS
    assert f.tool_class == ToolClass.READ_SOURCE
    assert f.target_path == "spine/x.py"
    assert "X" in f.matched_symbols

    # Prompt structure: system + user, with directive + topic in user msg.
    sys_msg, user_msg = bound.last_messages
    assert sys_msg.content == "researcher-role"
    user_text = user_msg.content
    assert "get_source for X" in user_text  # directive flows in
    assert "how does X work?" in user_text  # topic flows in
    assert "read_source" in user_text.lower()  # action hint


@pytest.mark.asyncio
async def test_worker_returns_empty_finding_when_no_tool_call_emitted():
    """If the model declines to call a tool (returns a plain AIMessage),
    record EMPTY with the narration — don't fabricate a tool result.
    """
    ai_msg = AIMessage(content="I think the answer is X.")
    bound = _StubBoundModel(ai_msg)
    tool = _StubTool("codebase_query", "should not run")

    directive = SupervisorDirective(
        analysis_and_reasoning="r",
        is_complete=False,
        next_directive="x",
        allowed_tool_class=ToolClass.SEARCH,
    )
    f = await run_worker_node(
        state={"work_id": "w"},
        config=None,
        topic="x",
        directive=directive,
        bound_models={ToolClass.SEARCH: (bound, [tool])},
        system_prompt="role",
    )
    assert f.status == FindingStatus.EMPTY
    assert tool.calls == []
    assert "I think the answer" in f.structured_code_block


@pytest.mark.asyncio
async def test_worker_returns_error_finding_when_model_raises():
    """Bound-model invocation failure surfaces as an ERROR finding."""

    class _BoomBound:
        async def ainvoke(self, *a, **kw):
            raise RuntimeError("network down")

    directive = SupervisorDirective(
        analysis_and_reasoning="r",
        is_complete=False,
        next_directive="x",
        allowed_tool_class=ToolClass.SEARCH,
    )
    f = await run_worker_node(
        state={"work_id": "w"},
        config=None,
        topic="x",
        directive=directive,
        bound_models={ToolClass.SEARCH: (_BoomBound(), [])},
        system_prompt="role",
    )
    assert f.status == FindingStatus.ERROR
    assert "RuntimeError" in f.execution_error_details
    assert "network down" in f.execution_error_details


@pytest.mark.asyncio
async def test_worker_returns_error_when_tool_execution_raises():
    """Tool-execution failure surfaces as an ERROR finding."""
    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "codebase_query", "args": {"action": "search"}, "id": "t1", "type": "tool_call"}
        ],
    )
    bound = _StubBoundModel(ai_msg)
    tool = _StubTool("codebase_query", RuntimeError("MCP socket closed"))

    directive = SupervisorDirective(
        analysis_and_reasoning="r",
        is_complete=False,
        next_directive="x",
        allowed_tool_class=ToolClass.SEARCH,
    )
    f = await run_worker_node(
        state={"work_id": "w"},
        config=None,
        topic="x",
        directive=directive,
        bound_models={ToolClass.SEARCH: (bound, [tool])},
        system_prompt="role",
    )
    assert f.status == FindingStatus.ERROR
    assert "RuntimeError" in f.execution_error_details
    assert "MCP socket closed" in f.execution_error_details


@pytest.mark.asyncio
async def test_worker_returns_error_when_tool_name_not_in_scoped_surface():
    """Model hallucinated a tool name not in the scoped set → ERROR
    (do not silently swallow; the supervisor sees this and adapts)."""
    ai_msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "ghost_tool", "args": {}, "id": "t1", "type": "tool_call"}
        ],
    )
    bound = _StubBoundModel(ai_msg)
    tool = _StubTool("codebase_query", "won't run")

    directive = SupervisorDirective(
        analysis_and_reasoning="r",
        is_complete=False,
        next_directive="x",
        allowed_tool_class=ToolClass.SEARCH,
    )
    f = await run_worker_node(
        state={"work_id": "w"},
        config=None,
        topic="x",
        directive=directive,
        bound_models={ToolClass.SEARCH: (bound, [tool])},
        system_prompt="role",
    )
    assert f.status == FindingStatus.ERROR
    assert "ghost_tool" in f.execution_error_details
    assert tool.calls == []


# ── render_history_as_evidence ─────────────────────────────────────────


def test_render_history_drops_error_and_empty_findings():
    """Only SUCCESS findings render into the evidence dossier — ERROR /
    EMPTY findings would otherwise become hallucination fuel for the
    summarise node.
    """
    history = [
        StructuredFinding(
            tool_name="codebase_query",
            tool_class=ToolClass.READ_SOURCE,
            status=FindingStatus.SUCCESS,
            target_path="spine/config.py",
            structured_code_block="class SpineConfig: ...",
        ),
        StructuredFinding(
            tool_name="codebase_query",
            tool_class=ToolClass.TRACE_DEPS,
            status=FindingStatus.ERROR,
            execution_error_details="boom",
        ),
        StructuredFinding(
            tool_name="codebase_query",
            tool_class=ToolClass.FIND_SYMBOL,
            status=FindingStatus.EMPTY,
        ),
    ]
    out = render_history_as_evidence(history)
    assert "spine/config.py" in out
    assert "class SpineConfig" in out
    assert "boom" not in out  # error narration excluded


def test_render_history_empty_returns_empty_string():
    assert render_history_as_evidence([]) == ""
