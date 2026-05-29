"""``build_phase_agent`` must honour ``skip_default_mcp_injection``.

The supervisor↔worker loop in ``run_explore_do_node`` curates a per-
ToolClass scoped tool surface upstream. Without ``skip_default_mcp_injection``,
``build_phase_agent`` re-loads the full MCP catalog and appends it to
``extra_tools`` — silently undoing the upstream filter. Trace
019e7164 (audited 2026-05-29) showed a 226:1 prompt:completion ratio
caused by this exact failure: every worker turn carried ~21 tool
definitions where it should have had 1-3.

These tests pin the contract.
"""

from __future__ import annotations

from typing import Any

import pytest

from spine.agents.factory import build_phase_agent
from spine.models.enums import PhaseName


class _FakeTool:
    """Stand-in for a BaseTool with just enough surface to be counted."""

    def __init__(self, name: str) -> None:
        self.name = name


def _stub_state(workspace_root: str = "/tmp/spine-test") -> dict:
    return {
        "work_id": "wid",
        "workspace_root": workspace_root,
        "description": "",
        "feedback": [],
        "last_critic_review": None,
        "messages": [],
        "artifacts_output": {},
        "phase_status": "",
        "read_cache": {},
    }


def _stub_factory_dependencies(monkeypatch, mcp_count: int = 18) -> dict[str, Any]:
    """Monkeypatch every external dep ``build_phase_agent`` reaches for so
    the test can assert against the tool list that flows into ``create_agent``.
    Returns the dict captured by the fake ``create_agent``.
    """
    captured: dict[str, Any] = {}

    # Fake MCP catalog: ``mcp_count`` mcp_* tools so we can verify
    # they are (or are NOT) appended.
    fake_mcp_tools = [_FakeTool(f"mcp_codebase-index_tool_{i}") for i in range(mcp_count)]

    monkeypatch.setattr(
        "spine.mcp.client.get_mcp_tools",
        lambda *a, **kw: list(fake_mcp_tools),
    )

    # Fake model resolution — return a string spec the factory can accept.
    monkeypatch.setattr(
        "spine.agents.factory.resolve_model",
        lambda *a, **kw: "openai:gpt-4o-mini",
    )
    monkeypatch.setattr(
        "spine.agents.factory._resolve_model_for_profile",
        lambda model: model,
    )
    monkeypatch.setattr(
        "spine.agents.factory._resolve_profile",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "spine.agents.factory._apply_profile_prompt",
        lambda profile, base: base,
    )
    monkeypatch.setattr(
        "spine.agents.factory._get_tool_description_overrides",
        lambda profile: {},
    )
    # These are imported INSIDE build_phase_agent, so we patch the source
    # modules rather than the factory module's namespace.
    monkeypatch.setattr(
        "spine.agents.backend.build_backend",
        lambda workspace_root: object(),
    )
    monkeypatch.setattr(
        "spine.agents.artifacts.materialize_artifacts",
        lambda state, workspace_root, work_id=None: None,
    )
    monkeypatch.setattr(
        "spine.agents.skills_resolver.resolve_skills",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        "spine.agents.skills_resolver.resolve_memory",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        "spine.agents.factory._build_middleware_stack",
        lambda **kw: [],
    )

    # Capture what gets passed into create_agent — that's the tool list
    # the model actually sees. ``create_agent`` is called with ``model``
    # positional + ``system_prompt`` / ``tools`` / ``middleware`` /
    # ``response_format`` / ``context_schema`` / ``debug`` / ``name`` as
    # kwargs (see factory.py:497). The return value must be chainable via
    # ``.with_config(...)`` because the caller appends one.
    class _FakeCompiled:
        def with_config(self, _cfg):
            return self

    def _fake_create_agent(model, *args, **kwargs):
        captured["tools"] = list(kwargs.get("tools") or [])
        captured["middleware"] = list(kwargs.get("middleware") or [])
        return _FakeCompiled()

    monkeypatch.setattr("spine.agents.factory.create_agent", _fake_create_agent)
    return captured


def test_default_behavior_appends_mcp_catalog(monkeypatch):
    """Baseline: without the new flag, build_phase_agent loads and appends
    every MCP tool to the agent's tool list. This is the pre-fix behaviour
    that causes the 226:1 prompt:completion bloat — pinned here so anyone
    flipping the default sees the test fail and reads this comment.
    """
    captured = _stub_factory_dependencies(monkeypatch, mcp_count=18)
    scoped_tool = _FakeTool("codebase_query")

    build_phase_agent(
        state=_stub_state(),  # type: ignore[arg-type]
        config=None,
        phase=PhaseName.SPECIFY,
        system_prompt="role",
        is_subagent=True,
        extra_tools=[scoped_tool],
        skip_filesystem_middleware=True,
    )

    names = [getattr(t, "name", "?") for t in captured["tools"]]
    assert "codebase_query" in names
    mcp_appended = [n for n in names if n.startswith("mcp_codebase-index_")]
    assert len(mcp_appended) == 18, (
        f"baseline must inject all MCP tools; got {mcp_appended}"
    )


def test_skip_default_mcp_injection_omits_full_catalog(monkeypatch):
    """With skip_default_mcp_injection=True, build_phase_agent MUST NOT
    append the MCP catalog — the agent's tool list contains EXACTLY the
    caller-provided extra_tools.

    This is the fix for trace 019e7164's tool-schema bloat: workers in the
    supervisor↔worker loop have their tool surface curated upstream, so
    re-injection at the factory layer is pure prefix overhead.
    """
    captured = _stub_factory_dependencies(monkeypatch, mcp_count=18)
    scoped_tools = [_FakeTool("codebase_query"), _FakeTool("search_codebase")]

    build_phase_agent(
        state=_stub_state(),  # type: ignore[arg-type]
        config=None,
        phase=PhaseName.SPECIFY,
        system_prompt="role",
        is_subagent=True,
        extra_tools=scoped_tools,
        skip_filesystem_middleware=True,
        skip_default_mcp_injection=True,
    )

    names = [getattr(t, "name", "?") for t in captured["tools"]]
    assert names == ["codebase_query", "search_codebase"], (
        f"skip_default_mcp_injection=True must leave extra_tools intact; "
        f"got {names}"
    )
    mcp_appended = [n for n in names if n.startswith("mcp_codebase-index_")]
    assert mcp_appended == [], (
        f"no MCP wrappers may be appended when skip flag is True; "
        f"got {mcp_appended}"
    )


def test_skip_flag_also_avoids_mcp_load_call(monkeypatch):
    """When the flag is True, ``get_mcp_tools`` MUST NOT be called at all.
    This guards against a future regression where someone keeps the load
    (paying network / cache lookup) but conditionally skips the append.
    """
    captured = _stub_factory_dependencies(monkeypatch, mcp_count=5)
    get_mcp_calls = []

    def _spy_get_mcp_tools(*a, **kw):
        get_mcp_calls.append((a, kw))
        return []

    monkeypatch.setattr("spine.mcp.client.get_mcp_tools", _spy_get_mcp_tools)

    build_phase_agent(
        state=_stub_state(),  # type: ignore[arg-type]
        config=None,
        phase=PhaseName.SPECIFY,
        system_prompt="role",
        is_subagent=True,
        extra_tools=[_FakeTool("codebase_query")],
        skip_filesystem_middleware=True,
        skip_default_mcp_injection=True,
    )

    assert get_mcp_calls == [], (
        f"get_mcp_tools must not be called when skip flag is True; "
        f"got {len(get_mcp_calls)} calls"
    )


@pytest.mark.asyncio
async def test_explore_do_node_does_not_build_phase_agent_for_workers(
    monkeypatch, tmp_path
):
    """After audit #2 (trace 019e71b4), workers bypass build_phase_agent
    entirely. They go through ``model.bind_tools(...).ainvoke(...)`` per
    turn — no agent loop, no middleware stack. This test pins that
    contract: ``run_explore_do_node`` MUST NOT call ``build_phase_agent``
    in the worker code path.

    The fix shipped in 7a26454 (skip_default_mcp_injection) is now
    redundant for the worker — kept as a guard for any future caller
    that DOES want a curated tool set without the MCP catalog.
    """
    from spine.agents import exploration_agents, researcher_supervisor
    from spine.agents.researcher_supervisor import (
        FindingStatus,
        StructuredFinding,
        SupervisorDirective,
        ToolClass,
    )

    captured_builds: list[dict[str, Any]] = []
    bind_log: list[list[str]] = []

    def _fake_build_phase_agent(**kw):
        captured_builds.append(
            {
                "extra_tools": [getattr(t, "name", "?") for t in (kw.get("extra_tools") or [])],
            }
        )
        return object()

    class _StubBaseModel:
        def bind_tools(self, tools):
            bind_log.append([getattr(t, "name", "?") for t in tools])
            return object()

    def _fake_subagent_spec(**kw):
        return {
            "system_prompt": "role",
            "tools": [_FakeTool("codebase_query"), _FakeTool("search_codebase")],
            "model": _StubBaseModel(),
        }

    monkeypatch.setattr(
        "spine.agents.factory.build_phase_agent", _fake_build_phase_agent
    )
    monkeypatch.setattr(
        "spine.agents.subagents.build_subagent_spec", _fake_subagent_spec
    )
    monkeypatch.setattr(
        "spine.agents.context.build_context", lambda state, phase: None
    )

    async def _fake_supervisor(**kw):
        if kw["cycle_idx"] == 0:
            return SupervisorDirective(
                analysis_and_reasoning="r",
                is_complete=False,
                next_directive="search",
                allowed_tool_class=ToolClass.SEARCH,
            )
        return SupervisorDirective(
            analysis_and_reasoning="done", is_complete=True
        )

    async def _fake_worker(**kw):
        return StructuredFinding(
            tool_name="search_codebase",
            tool_class=ToolClass.SEARCH,
            status=FindingStatus.SUCCESS,
            target_path="spine/x.py",
            structured_code_block="body",
        )

    monkeypatch.setattr(researcher_supervisor, "run_supervisor_node", _fake_supervisor)
    monkeypatch.setattr(researcher_supervisor, "run_worker_node", _fake_worker)

    await exploration_agents.run_explore_do_node(
        {
            "work_id": "w1",
            "phase": "specify",
            "workspace_root": str(tmp_path),
        },
        None,
        topic="test topic",
    )

    # The headline contract: build_phase_agent MUST NOT be called for
    # the worker path. The supervisor doesn't go through the factory
    # either — it uses resolve_model directly via run_supervisor_node.
    assert captured_builds == [], (
        f"build_phase_agent must NOT be called for workers; got {captured_builds}"
    )
    # And the per-class bind DOES happen with the scoped tool set.
    assert len(bind_log) == 1
    assert set(bind_log[0]) == {"codebase_query", "search_codebase"}
