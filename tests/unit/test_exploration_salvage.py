"""Tests for the researcher-salvage path in :mod:`spine.agents.exploration_agents`.

When the researcher subagent hits LangGraph's recursion cap mid-investigation,
the salvage path must either (a) return real research content extracted from
the tool results that did accumulate, or (b) return ``None`` so the caller
falls through to the error sentinel. It must NOT return a hallucinated summary
that just paraphrases the research topic — that's the regression these tests
guard against.
"""
from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from spine.agents.exploration_agents import (
    _SALVAGE_MIN_EVIDENCE_CHARS,
    _SALVAGE_SECTION_CHAR_CAP,
    _attempt_research_salvage,
    _collect_salvage_evidence,
    _extract_findings,
    _finalize_research_findings,
    _finalize_research_findings_from_evidence,
)


# ── _collect_salvage_evidence ─────────────────────────────────────────────


def test_collect_evidence_empty_messages_returns_empty():
    assert _collect_salvage_evidence([]) == ""


def test_collect_evidence_skips_human_and_system_messages():
    # The topic prompt lives in HumanMessage. The whole point of the salvage
    # is to NOT echo this back as findings.
    msgs = [
        SystemMessage(content="You are a researcher."),
        HumanMessage(content="## Research Topic\nInvestigate auth flow"),
    ]
    assert _collect_salvage_evidence(msgs) == ""


def test_collect_evidence_skips_ai_messages_with_no_content():
    # AIMessage at recursion cap often has empty .content (only tool_calls).
    msgs = [AIMessage(content="")]
    assert _collect_salvage_evidence(msgs) == ""


def test_collect_evidence_includes_tool_messages():
    msgs = [
        HumanMessage(content="topic"),
        ToolMessage(
            content="def login(): ...",
            tool_call_id="t1",
            name="mcp_codebase-index_get_function_source",
        ),
    ]
    out = _collect_salvage_evidence(msgs)
    assert "Tool result: mcp_codebase-index_get_function_source" in out
    assert "def login()" in out
    # HumanMessage must not leak in.
    assert "topic" not in out


def test_collect_evidence_includes_ai_synthesis():
    msgs = [
        AIMessage(content="So far I've found that auth lives in spine/auth.py"),
    ]
    out = _collect_salvage_evidence(msgs)
    assert "Intermediate synthesis" in out
    assert "spine/auth.py" in out


def test_collect_evidence_truncates_huge_tool_results():
    huge = "x" * (_SALVAGE_SECTION_CHAR_CAP + 5000)
    msgs = [ToolMessage(content=huge, tool_call_id="t1", name="search_codebase")]
    out = _collect_salvage_evidence(msgs)
    # Truncated to roughly the cap (allow header + ellipsis overhead).
    assert len(out) < _SALVAGE_SECTION_CHAR_CAP + 200
    assert out.endswith("…")


def test_collect_evidence_concatenates_in_order():
    msgs = [
        ToolMessage(content="result A", tool_call_id="t1", name="tool_a"),
        AIMessage(content="thinking step"),
        ToolMessage(content="result B", tool_call_id="t2", name="tool_b"),
    ]
    out = _collect_salvage_evidence(msgs)
    # Order preserved: A, thinking, B.
    assert out.index("result A") < out.index("thinking step") < out.index("result B")


# ── _finalize_research_findings_from_evidence ─────────────────────────────


class _FakeStructuredModel:
    """Mimics .with_structured_output(...).ainvoke(...) returning a ResearchFindings."""

    def __init__(self, response: Any, captured: dict[str, Any] | None = None):
        self._response = response
        self._captured = captured

    async def ainvoke(self, messages):
        if self._captured is not None:
            self._captured["prompt"] = messages[0].content
        return self._response


class _FakeModel:
    def __init__(self, response: Any, captured: dict[str, Any] | None = None):
        self._response = response
        self._captured = captured

    def with_structured_output(self, _schema):
        return _FakeStructuredModel(self._response, self._captured)


@pytest.mark.asyncio
async def test_finalize_from_evidence_none_model_is_noop():
    result: dict[str, Any] = {}
    await _finalize_research_findings_from_evidence(result, None, "some evidence")
    assert "structured_response" not in result


@pytest.mark.asyncio
async def test_finalize_from_evidence_empty_evidence_is_noop():
    result: dict[str, Any] = {}
    model = _FakeModel(response=object())
    await _finalize_research_findings_from_evidence(result, model, "")
    assert "structured_response" not in result


@pytest.mark.asyncio
async def test_finalize_from_evidence_sets_structured_response():
    from spine.agents.subagents import ResearchFindings

    expected = ResearchFindings(
        summary="Found auth in spine/auth.py.",
        patterns=["jwt-based"],
        file_map={"spine/auth.py": "auth entrypoint"},
        dependencies=["pyjwt"],
    )
    captured: dict[str, Any] = {}
    result: dict[str, Any] = {}
    await _finalize_research_findings_from_evidence(
        result, _FakeModel(expected, captured), "evidence body"
    )
    assert result["structured_response"] is expected
    # Prompt must frame this as partial evidence, not a finished report,
    # and explicitly forbid using the research topic to fill fields.
    prompt = captured["prompt"]
    assert "ran out of steps" in prompt
    assert "Do NOT use the research topic" in prompt
    assert "evidence body" in prompt


# ── _attempt_research_salvage ────────────────────────────────────────────


def _tool_msg(name: str, content: str, *, tcid: str = "t") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=tcid, name=name)


@pytest.mark.asyncio
async def test_salvage_skipped_when_only_topic_in_history():
    # Cap fired before any tool returned useful content.
    partial_state = {
        "messages": [
            HumanMessage(content="## Research Topic\nInvestigate auth flow"),
            AIMessage(content=""),  # tool_calls only, no text
        ]
    }
    out = await _attempt_research_salvage(
        partial_state, _FakeModel(response=object()), "auth flow", "work1"
    )
    assert out is None


@pytest.mark.asyncio
async def test_salvage_skipped_when_evidence_below_threshold():
    # One tool result, but tiny — under the minimum char threshold.
    short = "x" * (_SALVAGE_MIN_EVIDENCE_CHARS - 100)
    partial_state = {
        "messages": [_tool_msg("search_codebase", short)],
    }
    out = await _attempt_research_salvage(
        partial_state, _FakeModel(response=object()), "topic", "work1"
    )
    assert out is None


@pytest.mark.asyncio
async def test_salvage_rejected_when_all_structured_fields_empty():
    # The model dutifully returns a ResearchFindings but every concrete
    # field is empty — that's the hallucinated-paraphrase failure mode.
    from spine.agents.subagents import ResearchFindings

    big_evidence_body = "lots of tool output " * 50  # > min threshold
    partial_state = {
        "messages": [_tool_msg("search_codebase", big_evidence_body)],
    }
    empty_findings = ResearchFindings(
        summary="The agent was investigating the topic.",
        patterns=[],
        file_map={},
        dependencies=[],
    )
    out = await _attempt_research_salvage(
        partial_state, _FakeModel(empty_findings), "topic", "work1"
    )
    assert out is None


@pytest.mark.asyncio
async def test_salvage_accepted_when_structured_response_has_file_map():
    from spine.agents.subagents import ResearchFindings

    big_evidence_body = "lots of tool output " * 50
    partial_state = {
        "messages": [
            _tool_msg("mcp_codebase-index_find_symbol", big_evidence_body),
        ],
    }
    real_findings = ResearchFindings(
        summary="Auth lives in spine/auth.py with JWT validation.",
        patterns=[],
        file_map={"spine/auth.py": "JWT validation entrypoint"},
        dependencies=[],
    )
    out = await _attempt_research_salvage(
        partial_state, _FakeModel(real_findings), "auth flow", "work1"
    )
    assert out is not None
    assert len(out) == 1
    finding = out[0]
    assert finding["topic"] == "auth flow"
    assert finding["partial"] is True
    assert finding["file_map"] == {"spine/auth.py": "JWT validation entrypoint"}


@pytest.mark.asyncio
async def test_salvage_accepted_when_only_patterns_populated():
    # Any one of {file_map, patterns, dependencies} counts as concrete.
    from spine.agents.subagents import ResearchFindings

    big_evidence_body = "lots of tool output " * 50
    partial_state = {
        "messages": [_tool_msg("search_codebase", big_evidence_body)],
    }
    findings = ResearchFindings(
        summary="Discovered repeated retry-with-backoff usage.",
        patterns=["exponential-backoff", "circuit-breaker"],
        file_map={},
        dependencies=[],
    )
    out = await _attempt_research_salvage(
        partial_state, _FakeModel(findings), "retry handling", "work1"
    )
    assert out is not None
    assert out[0]["patterns"] == ["exponential-backoff", "circuit-breaker"]
    assert out[0]["partial"] is True


@pytest.mark.asyncio
async def test_salvage_returns_none_when_finalize_call_raises():
    class _ExplodingModel:
        def with_structured_output(self, _schema):
            class _S:
                async def ainvoke(self, _msgs):
                    raise RuntimeError("LLM call failed")

            return _S()

    big_evidence_body = "lots of tool output " * 50
    partial_state = {
        "messages": [_tool_msg("search_codebase", big_evidence_body)],
    }
    # _finalize_research_findings_from_evidence swallows the error and
    # leaves structured_response unset. _extract_findings then falls back
    # to the last AIMessage; with only a ToolMessage in history, it
    # returns the "(no findings)" sentinel — which the all-empty check
    # in _attempt_research_salvage rejects.
    out = await _attempt_research_salvage(
        partial_state, _ExplodingModel(), "topic", "work1"
    )
    assert out is None


# ── Error-status ToolMessage filtering (regression guards) ────────────────
#
# When a tool call raises, ToolSchemaValidator wraps the exception as
# ToolMessage(status="error", content="Tool 'X' execution failed: ...").
# Three code paths used to splice that error string into findings passed
# back to the model. These tests pin the fix.


def test_collect_evidence_skips_error_status_tool_messages():
    msgs = [
        ToolMessage(
            content="Tool 'mcp_find_symbol' execution failed: timeout. Check the arguments and retry.",
            tool_call_id="t1",
            name="mcp_find_symbol",
            status="error",
        ),
        ToolMessage(
            content="def login(): ...",
            tool_call_id="t2",
            name="mcp_get_function_source",
        ),
    ]
    out = _collect_salvage_evidence(msgs)
    assert "def login()" in out
    assert "execution failed" not in out
    assert "mcp_find_symbol" not in out


def test_collect_evidence_returns_empty_when_only_errors():
    msgs = [
        ToolMessage(
            content="Tool 'search_codebase' execution failed: connection refused.",
            tool_call_id="t1",
            name="search_codebase",
            status="error",
        ),
        ToolMessage(
            content="Tool 'fetch_url' execution failed: HTTP 503.",
            tool_call_id="t2",
            name="fetch_url",
            status="error",
        ),
    ]
    assert _collect_salvage_evidence(msgs) == ""


@pytest.mark.asyncio
async def test_finalize_research_findings_ignores_trailing_error_tool_message():
    # Researcher loop ended on a failed tool call. Without the fix, the
    # reverse scan would pick up the error ToolMessage and feed
    # "Tool 'X' execution failed: ..." into the structured-output prompt.
    captured: dict[str, Any] = {}
    result: dict[str, Any] = {
        "messages": [
            AIMessage(content="Auth lives in spine/auth.py with JWT validation."),
            AIMessage(content=""),  # final assistant turn — tool_calls only
            ToolMessage(
                content="Tool 'mcp_find_symbol' execution failed: file not found. Check the arguments and retry.",
                tool_call_id="t1",
                name="mcp_find_symbol",
                status="error",
            ),
        ]
    }

    from spine.agents.subagents import ResearchFindings

    fake_findings = ResearchFindings(
        summary="Auth lives in spine/auth.py with JWT validation.",
        patterns=[],
        file_map={"spine/auth.py": "auth"},
        dependencies=[],
    )
    await _finalize_research_findings(result, _FakeModel(fake_findings, captured))

    assert result["structured_response"] is fake_findings
    assert "execution failed" not in captured["prompt"]
    assert "spine/auth.py" in captured["prompt"]


@pytest.mark.asyncio
async def test_finalize_research_findings_noop_when_only_error_tool_messages():
    # No AIMessage with content at all — nothing legitimate to coerce.
    # Must NOT pull from the error ToolMessage.
    result: dict[str, Any] = {
        "messages": [
            ToolMessage(
                content="Tool 'X' execution failed: boom.",
                tool_call_id="t1",
                name="X",
                status="error",
            ),
        ]
    }
    captured: dict[str, Any] = {}
    await _finalize_research_findings(
        result, _FakeModel(response=object(), captured=captured)
    )
    assert "structured_response" not in result
    assert "prompt" not in captured  # ainvoke must not have been called


def test_extract_findings_returns_sentinel_when_only_error_tool_messages():
    result = {
        "messages": [
            ToolMessage(
                content="Tool 'search_codebase' execution failed: timeout.",
                tool_call_id="t1",
                name="search_codebase",
                status="error",
            ),
        ]
    }
    out = _extract_findings(result)
    assert out == [
        {"summary": "(no findings)", "patterns": [], "file_map": {}, "dependencies": []}
    ]


def test_extract_findings_picks_last_ai_message_over_trailing_tool_message():
    # AIMessage with the real synthesis exists, but a ToolMessage trails it.
    # The fallback must pick the AIMessage, not wrap the ToolMessage body.
    result = {
        "messages": [
            AIMessage(content="Found auth in spine/auth.py."),
            ToolMessage(
                content="def login(): ...",  # successful tool result
                tool_call_id="t1",
                name="get_source",
            ),
        ]
    }
    out = _extract_findings(result)
    assert out[0]["summary"] == "Found auth in spine/auth.py."


def test_extract_findings_skips_error_tool_message_and_uses_earlier_ai_message():
    result = {
        "messages": [
            AIMessage(content="Earlier synthesis: pattern X observed."),
            ToolMessage(
                content="Tool 'Y' execution failed: timeout.",
                tool_call_id="t1",
                name="Y",
                status="error",
            ),
        ]
    }
    out = _extract_findings(result)
    assert out[0]["summary"] == "Earlier synthesis: pattern X observed."
    assert "execution failed" not in out[0]["summary"]
