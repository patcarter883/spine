"""Loop orchestration in :func:`run_explore_do_node`.

Verifies the supervisor↔worker micro-loop terminates on
``is_complete=True``, hits the per-phase cycle cap when it doesn't, and
exposes the supervisor's choice of tool class to the worker.
"""

from __future__ import annotations

from typing import Any

import pytest

from spine.agents import exploration_agents, researcher_supervisor
from spine.agents.researcher_supervisor import (
    FindingStatus,
    StructuredFinding,
    SupervisorDirective,
    ToolClass,
)


def _stub_subagent_spec(monkeypatch) -> dict:
    """Replace ``build_subagent_spec`` with a no-op returning the minimal
    shape ``run_explore_do_node`` reads (tools, system_prompt).
    """

    class _FakeTool:
        def __init__(self, name: str) -> None:
            self.name = name

    spec = {
        "name": "researcher",
        "system_prompt": "scout-system",
        "model": object(),
        "tools": [
            _FakeTool("codebase_query"),
            _FakeTool("search_codebase"),
            _FakeTool("ast_extract_symbol"),
        ],
        "response_format": None,
    }
    monkeypatch.setattr(
        "spine.agents.subagents.build_subagent_spec",
        lambda **kw: spec,
    )
    return spec


def _stub_build_phase_agent(monkeypatch, *, calls: list[dict]):
    """Replace ``build_phase_agent`` with a fake that records the
    ``extra_tools`` it was handed and returns a dummy agent the tests
    don't actually invoke (the supervisor/worker calls are also stubbed).
    """

    def _fake(**kw):
        calls.append(
            {
                "extra_tools": [getattr(t, "name", "?") for t in (kw.get("extra_tools") or [])],
            }
        )
        return object()

    monkeypatch.setattr("spine.agents.factory.build_phase_agent", _fake)


def _stub_context(monkeypatch) -> object:
    """Replace ``build_context`` with a marker we can identify in assertions."""

    class _Ctx:
        read_cache = {"sentinel": "ctx"}

    ctx = _Ctx()
    monkeypatch.setattr(
        "spine.agents.context.build_context", lambda state, phase: ctx
    )
    return ctx


@pytest.mark.asyncio
async def test_loop_terminates_when_supervisor_marks_complete(monkeypatch):
    """First cycle (initialization) seeds a SEARCH directive; one worker
    turn returns a success finding; supervisor then says is_complete=True
    and the loop exits. Evidence carries the worker's tool output.
    """
    _stub_subagent_spec(monkeypatch)
    build_calls: list[dict] = []
    _stub_build_phase_agent(monkeypatch, calls=build_calls)
    _stub_context(monkeypatch)

    supervisor_calls = 0

    async def _fake_supervisor(**kw):
        nonlocal supervisor_calls
        supervisor_calls += 1
        # Cycle 0 returns seed (SEARCH); cycle 1 returns complete.
        if kw["cycle_idx"] == 0:
            return SupervisorDirective(
                analysis_and_reasoning="seed",
                is_complete=False,
                next_directive="search the topic",
                allowed_tool_class=ToolClass.SEARCH,
            )
        return SupervisorDirective(
            analysis_and_reasoning="enough evidence — terminating",
            is_complete=True,
        )

    async def _fake_worker(**kw):
        return StructuredFinding(
            tool_name="search_codebase",
            tool_class=ToolClass.SEARCH,
            status=FindingStatus.SUCCESS,
            target_path="spine/agents/factory.py",
            matched_symbols=["build_phase_agent"],
            structured_code_block="def build_phase_agent(...): ...",
        )

    monkeypatch.setattr(researcher_supervisor, "run_supervisor_node", _fake_supervisor)
    monkeypatch.setattr(researcher_supervisor, "run_worker_node", _fake_worker)

    out = await exploration_agents.run_explore_do_node(
        {
            "work_id": "w1",
            "phase": "specify",
            "workspace_root": "/tmp",
        },
        None,
        topic="how does build_phase_agent assemble middleware?",
    )

    evidence = out["exploration_evidence"]
    assert evidence["recursion_capped"] is False
    assert evidence["supervisor_cycles"] == 1  # one worker turn ran
    assert "build_phase_agent" in evidence["tool_results_text"]
    assert "terminating" in evidence["narrative"]
    # Built exactly one worker (SEARCH) — the second cycle was
    # is_complete=True so no further worker built.
    assert len(build_calls) == 1
    assert set(build_calls[0]["extra_tools"]) == {"codebase_query", "search_codebase"}
    # 2 supervisor calls: cycle 0 (seed shortcut still invokes the function)
    # and cycle 1 (the terminating decision).
    assert supervisor_calls == 2


@pytest.mark.asyncio
async def test_loop_hits_cap_when_supervisor_never_completes(monkeypatch):
    """If the supervisor never says is_complete=True, the loop must
    terminate at the per-phase cap (PLAN: 6 by default) and mark
    ``recursion_capped=True`` so the summarise sentinel path kicks in.
    """
    _stub_subagent_spec(monkeypatch)
    _stub_build_phase_agent(monkeypatch, calls=[])
    _stub_context(monkeypatch)

    async def _fake_supervisor(**kw):
        # Always continue with READ_SOURCE — never terminate.
        return SupervisorDirective(
            analysis_and_reasoning="more please",
            is_complete=False,
            next_directive="get_source for something",
            allowed_tool_class=ToolClass.READ_SOURCE,
        )

    worker_calls = 0

    async def _fake_worker(**kw):
        nonlocal worker_calls
        worker_calls += 1
        return StructuredFinding(
            tool_name="codebase_query",
            tool_class=ToolClass.READ_SOURCE,
            status=FindingStatus.SUCCESS,
            target_path="spine/x.py",
            structured_code_block="content",
        )

    monkeypatch.setattr(researcher_supervisor, "run_supervisor_node", _fake_supervisor)
    monkeypatch.setattr(researcher_supervisor, "run_worker_node", _fake_worker)

    out = await exploration_agents.run_explore_do_node(
        {
            "work_id": "w1",
            "phase": "plan",
            "workspace_root": "/tmp",
            "spec_path": "",
        },
        None,
        topic="some plan topic",
    )

    evidence = out["exploration_evidence"]
    assert evidence["recursion_capped"] is True
    # PLAN default cap is 6 — the worker ran the cap number of times.
    assert evidence["supervisor_cycles"] == 6
    assert worker_calls == 6


@pytest.mark.asyncio
async def test_loop_builds_separate_worker_per_tool_class(monkeypatch):
    """When the supervisor picks distinct tool classes across cycles,
    the loop lazy-builds one worker agent per class with that class's
    filtered tool surface — and reuses the cached agent on revisit.
    """
    _stub_subagent_spec(monkeypatch)
    build_calls: list[dict] = []
    _stub_build_phase_agent(monkeypatch, calls=build_calls)
    _stub_context(monkeypatch)

    # Plan the supervisor's choices: SEARCH → READ_SOURCE → SEARCH → COMPLETE.
    plan = iter(
        [
            ToolClass.SEARCH,
            ToolClass.READ_SOURCE,
            ToolClass.SEARCH,  # revisit — should hit the cache, no new build
            None,  # complete
        ]
    )

    async def _fake_supervisor(**kw):
        nxt = next(plan)
        if nxt is None:
            return SupervisorDirective(
                analysis_and_reasoning="done", is_complete=True
            )
        return SupervisorDirective(
            analysis_and_reasoning="continue",
            is_complete=False,
            next_directive=f"use {nxt.value}",
            allowed_tool_class=nxt,
        )

    async def _fake_worker(**kw):
        return StructuredFinding(
            tool_name="codebase_query",
            tool_class=kw["directive"].allowed_tool_class,
            status=FindingStatus.SUCCESS,
            target_path="spine/x.py",
            structured_code_block="body",
        )

    monkeypatch.setattr(researcher_supervisor, "run_supervisor_node", _fake_supervisor)
    monkeypatch.setattr(researcher_supervisor, "run_worker_node", _fake_worker)

    out = await exploration_agents.run_explore_do_node(
        {
            "work_id": "w1",
            "phase": "specify",
            "workspace_root": "/tmp",
        },
        None,
        topic="t",
    )

    assert out["exploration_evidence"]["supervisor_cycles"] == 3  # 3 worker turns
    # 2 build_phase_agent calls — one for SEARCH, one for READ_SOURCE.
    # The second SEARCH cycle hits the cache.
    assert len(build_calls) == 2
    tool_sets = [set(c["extra_tools"]) for c in build_calls]
    assert {"codebase_query", "search_codebase"} in tool_sets   # SEARCH
    assert {"codebase_query", "ast_extract_symbol"} in tool_sets  # READ_SOURCE


@pytest.mark.asyncio
async def test_loop_passes_shared_context_to_worker(monkeypatch):
    """The same SpineContext built once at loop entry must be threaded
    into every worker invocation so ReadCacheMiddleware dedupes across
    cycles.
    """
    _stub_subagent_spec(monkeypatch)
    _stub_build_phase_agent(monkeypatch, calls=[])
    ctx = _stub_context(monkeypatch)

    seen_contexts: list[Any] = []

    async def _fake_supervisor(**kw):
        if kw["cycle_idx"] >= 2:
            return SupervisorDirective(
                analysis_and_reasoning="done", is_complete=True
            )
        return SupervisorDirective(
            analysis_and_reasoning="continue",
            is_complete=False,
            next_directive="x",
            allowed_tool_class=ToolClass.SEARCH,
        )

    async def _fake_worker(**kw):
        seen_contexts.append(kw.get("context"))
        return StructuredFinding(
            tool_name="search_codebase",
            tool_class=ToolClass.SEARCH,
            status=FindingStatus.SUCCESS,
            target_path="spine/y.py",
            structured_code_block="body",
        )

    monkeypatch.setattr(researcher_supervisor, "run_supervisor_node", _fake_supervisor)
    monkeypatch.setattr(researcher_supervisor, "run_worker_node", _fake_worker)

    out = await exploration_agents.run_explore_do_node(
        {
            "work_id": "w1",
            "phase": "specify",
            "workspace_root": "/tmp",
        },
        None,
        topic="t",
    )

    # 2 worker calls, both got the same context object.
    assert len(seen_contexts) == 2
    assert all(c is ctx for c in seen_contexts)
    # And the loop bubbled the ctx.read_cache back into state.
    assert out.get("read_cache") == {"sentinel": "ctx"}


@pytest.mark.asyncio
async def test_loop_terminates_immediately_when_supervisor_seeds_complete(
    monkeypatch,
):
    """Defensive: if a future supervisor implementation marks
    is_complete=True on the very first cycle (skipping all work), the
    loop must exit cleanly with zero worker turns and empty evidence.
    """
    _stub_subagent_spec(monkeypatch)
    _stub_build_phase_agent(monkeypatch, calls=[])
    _stub_context(monkeypatch)

    async def _fake_supervisor(**kw):
        return SupervisorDirective(
            analysis_and_reasoning="topic doesn't need research",
            is_complete=True,
        )

    async def _fake_worker(**kw):
        raise AssertionError("worker must not run when is_complete=True on cycle 0")

    monkeypatch.setattr(researcher_supervisor, "run_supervisor_node", _fake_supervisor)
    monkeypatch.setattr(researcher_supervisor, "run_worker_node", _fake_worker)

    out = await exploration_agents.run_explore_do_node(
        {
            "work_id": "w1",
            "phase": "specify",
            "workspace_root": "/tmp",
        },
        None,
        topic="trivial",
    )
    ev = out["exploration_evidence"]
    assert ev["supervisor_cycles"] == 0
    assert ev["tool_results_text"] == ""
    assert ev["recursion_capped"] is False
