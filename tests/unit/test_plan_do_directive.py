"""Tests for the plan-before-do helper (:mod:`spine.agents.plan_do`).

The plan node is the no-tool half of the plan→do split applied to every
non-explore subagent node. It must:

1. Return a typed :class:`SubagentDirective` on the happy path.
2. Fall back to :func:`empty_directive` on every failure mode so the do
   node still runs.
3. Render directives into a prompt block the do node can prepend.
"""
from __future__ import annotations

from typing import Any

import pytest

from spine.agents import plan_do
from spine.agents.plan_do import (
    SubagentDirective,
    directive_from_state,
    empty_directive,
    format_directive_for_prompt,
    run_plan_node,
)


class _FakeStructured:
    def __init__(self, response: Any):
        self._response = response

    async def ainvoke(self, _messages):
        return self._response


class _FakeChatModel:
    def __init__(self, response: Any):
        self._response = response

    def with_structured_output(self, _schema):
        return _FakeStructured(self._response)


def test_format_directive_renders_all_fields():
    d = SubagentDirective(
        approach="Investigate auth flow, then map symbols.",
        target_files=["spine/auth.py", "spine/config.py"],
        tool_calls_to_make=["codebase_query find_symbol login"],
        acceptance=["List of auth-related symbols", "Map of file paths"],
        notes="Watch out for jwt_secret env var.",
    )
    out = format_directive_for_prompt(d)
    assert "Plan Directive" in out
    assert "Investigate auth flow" in out
    assert "spine/auth.py" in out
    assert "codebase_query find_symbol login" in out
    assert "Map of file paths" in out
    assert "jwt_secret" in out


def test_format_directive_handles_dict_round_trip():
    """Directives round-trip through LangGraph state as dicts."""
    d = SubagentDirective(approach="x", target_files=["a.py"])
    as_dict = d.model_dump()
    out = format_directive_for_prompt(as_dict)
    assert "x" in out
    assert "a.py" in out


def test_format_directive_empty_object_still_includes_header():
    out = format_directive_for_prompt(empty_directive("test reason"))
    assert "Plan Directive" in out
    # The approach line is rendered even for the stub.
    assert "no directive produced" in out


def test_empty_directive_includes_reason():
    d = empty_directive("model lacks structured output")
    assert "model lacks structured output" in d.approach
    assert d.target_files == []
    assert d.acceptance == []


def test_directive_from_state_handles_missing_key():
    out = directive_from_state({}, "plan_directive")
    assert isinstance(out, SubagentDirective)
    assert "no directive" in out.approach.lower()


def test_directive_from_state_returns_dict_unchanged():
    raw = {"approach": "x", "target_files": ["a.py"]}
    out = directive_from_state({"plan_directive": raw}, "plan_directive")
    # Caller doesn't care whether it's a dict or a model — format_directive
    # accepts both. Just verify we got the right payload back.
    assert isinstance(out, dict)
    assert out["approach"] == "x"


@pytest.mark.asyncio
async def test_run_plan_node_returns_directive_on_happy_path(monkeypatch):
    expected = SubagentDirective(
        approach="Edit spine/auth.py to add JWT validation.",
        target_files=["spine/auth.py"],
        acceptance=["JWT validation present", "Tests pass"],
    )
    monkeypatch.setattr(plan_do, "resolve_model", lambda *a, **kw: _FakeChatModel(expected))
    out = await run_plan_node(
        state={"work_id": "w1"},
        config=None,
        phase_path="implement",
        task_description="Add JWT validation to spine/auth.py",
        role_hint="slice-implementer",
    )
    assert isinstance(out, SubagentDirective)
    assert out.target_files == ["spine/auth.py"]


@pytest.mark.asyncio
async def test_run_plan_node_returns_empty_when_resolve_fails(monkeypatch):
    def _bad_resolve(*args, **kwargs):
        raise RuntimeError("config missing")

    monkeypatch.setattr(plan_do, "resolve_model", _bad_resolve)
    out = await run_plan_node(
        state={"work_id": "w1"},
        config=None,
        phase_path="implement",
        task_description="...",
    )
    assert isinstance(out, SubagentDirective)
    assert "resolve_model failed" in out.approach


@pytest.mark.asyncio
async def test_run_plan_node_returns_empty_when_model_lacks_structured_output(monkeypatch):
    class _NoStructured:
        def with_structured_output(self, _schema):
            raise NotImplementedError

    monkeypatch.setattr(plan_do, "resolve_model", lambda *a, **kw: _NoStructured())
    out = await run_plan_node(
        state={"work_id": "w1"},
        config=None,
        phase_path="implement",
        task_description="...",
    )
    assert "structured output unsupported" in out.approach


@pytest.mark.asyncio
async def test_run_plan_node_returns_empty_when_invocation_raises(monkeypatch):
    class _Boom:
        async def ainvoke(self, _msgs):
            raise RuntimeError("timeout")

    class _Model:
        def with_structured_output(self, _schema):
            return _Boom()

    monkeypatch.setattr(plan_do, "resolve_model", lambda *a, **kw: _Model())
    out = await run_plan_node(
        state={"work_id": "w1"},
        config=None,
        phase_path="implement",
        task_description="...",
    )
    assert "plan invocation failed" in out.approach


@pytest.mark.asyncio
async def test_run_plan_node_accepts_dict_response(monkeypatch):
    """Some providers return a dict instead of the pydantic instance."""
    monkeypatch.setattr(
        plan_do,
        "resolve_model",
        lambda *a, **kw: _FakeChatModel({"approach": "ok", "target_files": ["a.py"]}),
    )
    out = await run_plan_node(
        state={"work_id": "w1"},
        config=None,
        phase_path="implement",
        task_description="...",
    )
    assert isinstance(out, SubagentDirective)
    assert out.approach == "ok"
    assert out.target_files == ["a.py"]
