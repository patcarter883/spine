"""Tests for the explore_do → summarise split in the exploration subgraph.

The legacy ``run_explore_node`` ran tool-using research AND structured
finalisation in one node. Smaller models could not switch cognitive modes
between those two jobs, so the node is now bisected:

- :func:`run_explore_do_node` runs the researcher loop and returns an
  evidence dossier.
- :func:`run_summarise_node` consumes the dossier (no tools attached) and
  emits a :class:`ResearchFindings` aligned to the original topic.

These tests pin the contract between the two nodes without standing up a
real LangGraph runtime.
"""
from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from spine.agents import exploration_agents
from spine.agents.exploration_agents import (
    _ainvoke_explore_collecting,
    _empty_research_finding,
    collect_exploration_evidence,
    run_summarise_node,
    summarise_evidence,
)


class _FakeStructured:
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
        return _FakeStructured(self._response, self._captured)


def test_collect_exploration_evidence_is_alias():
    """The public name and the legacy private name must point at the same code."""
    msgs = [
        HumanMessage(content="topic"),
        ToolMessage(
            content="found function login()",
            tool_call_id="t1",
            name="codebase_query",
        ),
    ]
    out_public = collect_exploration_evidence(msgs)
    out_legacy = exploration_agents._collect_salvage_evidence(msgs)
    assert out_public == out_legacy
    assert "login()" in out_public


def test_empty_research_finding_shape_matches_error_sentinel():
    """The sentinel emitted by summarise MUST match what the filter chain drops.

    Regression guard for the memory-pinned rule
    ``feedback_no_error_text_in_research_results`` — the filter chain in
    ``_summarize_findings`` / ``_format_findings`` keys on
    ``finding.get("error") is True``.
    """
    sentinel = _empty_research_finding("auth flow", "GraphRecursionError")
    assert sentinel["error"] is True
    assert sentinel["error_class"] == "GraphRecursionError"
    assert sentinel["topic"] == "auth flow"
    # Summary must be neutral — never embed raw exception text.
    assert "GraphRecursion" not in sentinel["summary"]
    assert "Exception" not in sentinel["summary"]
    assert "Traceback" not in sentinel["summary"]


@pytest.mark.asyncio
async def test_summarise_evidence_uses_only_evidence_in_prompt():
    """The summarise prompt is built via the XML-tagged hostage layout
    (see ``spine.agents.prompt_format``). The forbid-hallucination
    constraints live in a ``<constraints>`` block; the topic in
    ``<objective>``; the evidence in ``<findings>``. The directive sits
    after every closing tag.
    """
    from spine.agents.prompt_format import (
        Tag,
        assert_has_tags,
        assert_hostage_layout,
        get_block,
    )
    from spine.agents.subagents import ResearchFindings

    expected = ResearchFindings(
        summary="x", patterns=[], file_map={}, dependencies=[]
    )
    captured: dict[str, Any] = {}
    out = await summarise_evidence(
        model=_FakeModel(expected, captured),
        topic="auth flow",
        evidence_text="### Tool result: codebase_query\nfound login()",
        narrative="",
    )
    assert out is not None
    assert out["summary"] == "x"
    prompt = captured["prompt"]
    # Structural invariants — the data/instruction split is enforced by
    # the prompt-format helpers, not by literal substring matching.
    assert_hostage_layout(prompt)
    assert_has_tags(prompt, Tag.OBJECTIVE, Tag.CONSTRAINTS, Tag.FINDINGS)
    # Per-block semantic checks: the constraint text and the topic / evidence
    # land in the correct tagged blocks.
    constraints = get_block(prompt, Tag.CONSTRAINTS)
    assert "Use ONLY information present" in constraints
    assert "Do NOT use the topic name" in constraints
    assert get_block(prompt, Tag.OBJECTIVE) == "auth flow"
    assert "found login()" in get_block(prompt, Tag.FINDINGS)


@pytest.mark.asyncio
async def test_summarise_evidence_returns_none_when_model_lacks_structured_output():
    class _NoStructured:
        def with_structured_output(self, _schema):
            raise NotImplementedError

    out = await summarise_evidence(
        model=_NoStructured(),
        topic="x",
        evidence_text="some evidence",
    )
    assert out is None


@pytest.mark.asyncio
async def test_summarise_node_empty_evidence_emits_sentinel():
    """No usable evidence → drop into the error-sentinel path."""
    state = {
        "work_id": "w1",
        "phase": "specify",
        "exploration_evidence": {
            "topic": "auth flow",
            "tool_results_text": "",
            "narrative": "",
            "recursion_capped": False,
            "error_class": None,
        },
    }
    out = await run_summarise_node(state, None)
    findings = out["findings"]
    assert len(findings) == 1
    f = findings[0]
    assert f["error"] is True
    assert f["topic"] == "auth flow"


@pytest.mark.asyncio
async def test_summarise_node_recursion_capped_with_thin_evidence_emits_sentinel():
    state = {
        "work_id": "w1",
        "phase": "plan",
        "exploration_evidence": {
            "topic": "X",
            "tool_results_text": "tiny",  # below _SUMMARISE_MIN_EVIDENCE_CHARS
            "narrative": "",
            "recursion_capped": True,
            "error_class": "GraphRecursionError",
        },
    }
    out = await run_summarise_node(state, None)
    f = out["findings"][0]
    assert f["error"] is True
    assert f.get("error_class") == "GraphRecursionError"


@pytest.mark.asyncio
async def test_explore_do_node_returns_command_with_send_to_summarise(monkeypatch):
    """Regression for the InvalidUpdateError crash.

    With N parallel ``Send("explore_do", ...)`` branches, each branch
    must dispatch its own ``Send("summarise", ...)`` rather than write
    to a shared ``exploration_evidence`` channel. Otherwise apply_writes
    sees N writes to a LastValue channel and raises.
    """
    from langgraph.types import Command, Send
    from spine.workflow.subgraphs.exploration_subgraph import _explore_do_node

    async def _fake_do(state, config, *, topic):
        return {
            "exploration_evidence": {
                "topic": topic,
                "tool_results_text": "evidence body",
                "narrative": "",
                "recursion_capped": False,
                "error_class": None,
            },
            "read_cache": {"some": "cache"},
        }

    monkeypatch.setattr(
        "spine.agents.exploration_agents.run_explore_do_node", _fake_do
    )

    state = {
        "phase": "specify",
        "work_id": "w1",
        "work_type": "task",
        "workspace_root": "/tmp",
        "topic": "auth flow",
    }
    out = await _explore_do_node(state, None)
    assert isinstance(out, Command)
    # read_cache (reducer-backed) is safe to ship via update; evidence is NOT.
    assert "read_cache" in out.update
    assert "exploration_evidence" not in out.update
    assert isinstance(out.goto, Send)
    assert out.goto.node == "summarise"
    assert out.goto.arg["exploration_evidence"]["topic"] == "auth flow"
    assert out.goto.arg["topic"] == "auth flow"
    assert out.goto.arg["phase"] == "specify"


@pytest.mark.asyncio
async def test_summarise_node_happy_path(monkeypatch):
    """Evidence present → calls summarise_evidence and returns a real finding."""
    from spine.agents.subagents import ResearchFindings

    captured: dict[str, Any] = {}

    async def _fake_summarise_evidence(
        *, model, topic, evidence_text, narrative, recursion_capped
    ):
        captured["topic"] = topic
        captured["evidence"] = evidence_text
        captured["narrative"] = narrative
        captured["capped"] = recursion_capped
        return ResearchFindings(
            summary="found stuff",
            patterns=["pattern1"],
            file_map={"spine/auth.py": "auth entry"},
            dependencies=["pyjwt"],
        ).model_dump()

    # build_subagent_spec is the only heavy dependency in run_summarise_node.
    # Stub it so we don't have to spin up the full agent infrastructure.
    def _fake_build_subagent_spec(*args, **kwargs):
        return {"model": object()}  # any object — summarise_evidence is stubbed

    monkeypatch.setattr(exploration_agents, "summarise_evidence", _fake_summarise_evidence)
    monkeypatch.setattr(
        "spine.agents.subagents.build_subagent_spec", _fake_build_subagent_spec
    )

    state = {
        "work_id": "w1",
        "phase": "specify",
        "exploration_evidence": {
            "topic": "auth flow",
            "tool_results_text": "### Tool result: codebase_query\n" + "X" * 500,
            "narrative": "Researcher found login() in spine/auth.py.",
            "recursion_capped": False,
            "error_class": None,
        },
    }
    out = await run_summarise_node(state, None)
    findings = out["findings"]
    assert len(findings) == 1
    f = findings[0]
    assert f.get("error") is not True
    assert f["topic"] == "auth flow"
    assert "spine/auth.py" in f["file_map"]
    # The summariser saw both the evidence and the narrative.
    assert captured["topic"] == "auth flow"
    assert "Researcher found" in captured["narrative"]
    assert captured["capped"] is False


# ── Generalised salvage: partial_state on any terminal exception ───────────


class _StreamingAgent:
    """Minimal agent stub that yields N chunks via astream, then raises."""

    def __init__(self, chunks, exc):
        self._chunks = chunks
        self._exc = exc

    def astream(self, _input, *, stream_mode, **_kwargs):
        chunks = self._chunks
        exc = self._exc

        async def _gen():
            for ch in chunks:
                yield ch
            raise exc

        return _gen()


@pytest.mark.asyncio
async def test_ainvoke_collecting_attaches_partial_state_on_badrequest_error():
    """Regression for trace 019e6e53.

    Before this change, ``_ainvoke_explore_collecting`` only attached
    ``partial_state`` to ``GraphRecursionError``. ``BadRequestError`` on
    80K context overflow (and any other non-transient exception) lost
    the accumulated stream state, so the explore_do salvage path saw
    ``message_count=0`` and emitted a sentinel even though the
    researcher had built up real investigation history before the
    provider rejected the call.
    """
    # Imitate openai.BadRequestError shape — _is_transient_error returns
    # False for this so the loop re-raises immediately.
    class FakeBadRequest(Exception):
        pass

    accumulated = {"messages": ["m1", "m2", "m3"], "topic": "auth"}
    agent = _StreamingAgent(chunks=[accumulated], exc=FakeBadRequest("80K window exceeded"))

    with pytest.raises(FakeBadRequest) as exc_info:
        await _ainvoke_explore_collecting(
            agent,
            {"messages": []},
            work_id="w1",
            context=None,
            config={"recursion_limit": 50},
        )

    # The exception must carry the partial state so the caller can salvage.
    assert getattr(exc_info.value, "partial_state", None) == accumulated


@pytest.mark.asyncio
async def test_ainvoke_collecting_skips_partial_state_when_no_chunks():
    """Don't attach an empty partial_state — keeps the attribute meaningful."""

    class FakeBadRequest(Exception):
        pass

    agent = _StreamingAgent(chunks=[], exc=FakeBadRequest("immediate failure"))

    with pytest.raises(FakeBadRequest) as exc_info:
        await _ainvoke_explore_collecting(
            agent,
            {"messages": []},
            work_id="w1",
            context=None,
            config={"recursion_limit": 50},
        )

    assert not hasattr(exc_info.value, "partial_state")
