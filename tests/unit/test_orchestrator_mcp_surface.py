"""Every phase orchestrator MUST opt out of build_phase_agent's
default MCP injection.

LangSmith trace 019e721d (audit #3) showed the PLAN synthesizer (and
by symmetry every other orchestrator built via build_phase_agent) was
receiving the full ~17-tool MCP wrapper catalog despite having a
prompt that only named 2-3 phase-specific tools. The MCP catalog was
pure prefix overhead AND a hallucination magnet — the synthesizer was
observed calling ``mcp_codebase-index_get_function_source(file_path=
'spine/ui/_pages/work_submit.py', max_lines=100)`` (missing the
required ``name`` arg) because too many similar tools were available.

The fix: every orchestrator call site passes
``skip_default_mcp_injection=True``. The slice-implementer dispatcher
in implement_subgraph.py does the same — its restricted_tools list
already contains the curated MCP tools it actually needs (filtered
from the subagent_spec), so the factory-level re-injection would just
duplicate the catalog.

These tests pin the contract by stubbing ``build_phase_agent`` and
asserting every orchestrator builder passes the flag.
"""

from __future__ import annotations

from typing import Any

import pytest


def _stub_factory_and_helpers(monkeypatch) -> list[dict[str, Any]]:
    """Replace ``build_phase_agent`` with a fake that records its kwargs,
    plus stub every external helper the orchestrator builders reach for.

    Returns the captured-kwargs list so each test can assert on it.
    """
    captured: list[dict[str, Any]] = []

    def _fake_build(**kwargs):
        captured.append(kwargs)
        return object()

    # Patch the factory symbol at EVERY import site — the orchestrators
    # each have ``from spine.agents.factory import build_phase_agent``
    # at module top, so the binding lives in the orchestrator's module.
    for mod in (
        "spine.agents.specify_agent",
        "spine.agents.plan_agent",
        "spine.agents.tasks_agent",
        "spine.agents.verify_agent",
        "spine.agents.gap_plan_agent",
        "spine.critic.agent",
        "spine.workflow.subgraphs.implement_subgraph",
    ):
        try:
            monkeypatch.setattr(f"{mod}.build_phase_agent", _fake_build)
        except AttributeError:
            # Module may not have it imported at top level (e.g. local
            # import inside a function). The fake will still apply via
            # the source module patch below.
            pass

    # Also patch the source so any local imports pick it up.
    monkeypatch.setattr("spine.agents.factory.build_phase_agent", _fake_build)

    # Stub external resources the builders touch.
    monkeypatch.setattr(
        "spine.agents.skills_resolver.resolve_memory", lambda *a, **kw: []
    )
    monkeypatch.setattr(
        "spine.agents.skills_resolver.resolve_skills", lambda *a, **kw: []
    )

    return captured


def _base_state() -> dict[str, Any]:
    return {
        "work_id": "wid-test",
        "work_type": "reviewed_task",
        "workspace_root": "/tmp/spine-test",
        "description": "test description",
        "feedback": [],
        "last_critic_review": None,
        "artifacts": {},
        "messages": [],
        "phase_status": "",
        "read_cache": {},
    }


def _assert_skip_flag_in(captured: list[dict[str, Any]], orchestrator: str) -> None:
    assert captured, f"{orchestrator}: build_phase_agent was not called"
    for call in captured:
        flag = call.get("skip_default_mcp_injection")
        assert flag is True, (
            f"{orchestrator}: build_phase_agent called without "
            f"skip_default_mcp_injection=True (got {flag!r}). The full MCP "
            f"catalog would be re-injected — see trace 019e721d audit."
        )


def test_specify_orchestrator_skips_default_mcp(monkeypatch):
    """build_specify_agent (full SPECIFY orchestrator) opts out."""
    captured = _stub_factory_and_helpers(monkeypatch)
    from spine.agents.specify_agent import build_specify_agent

    build_specify_agent(_base_state(), config=None)
    _assert_skip_flag_in(captured, "build_specify_agent")


def test_specify_synthesizer_skips_default_mcp(monkeypatch):
    """build_specify_synthesizer (used by exploration_subgraph._synthesize_specify)."""
    captured = _stub_factory_and_helpers(monkeypatch)
    from spine.agents.specify_agent import build_specify_synthesizer

    build_specify_synthesizer(_base_state(), config=None)
    _assert_skip_flag_in(captured, "build_specify_synthesizer")


def test_plan_orchestrator_skips_default_mcp(monkeypatch):
    captured = _stub_factory_and_helpers(monkeypatch)
    from spine.agents.plan_agent import build_plan_agent

    build_plan_agent(_base_state(), config=None)
    _assert_skip_flag_in(captured, "build_plan_agent")


def test_plan_synthesizer_skips_default_mcp(monkeypatch):
    captured = _stub_factory_and_helpers(monkeypatch)
    from spine.agents.plan_agent import build_plan_synthesizer

    build_plan_synthesizer(_base_state(), config=None)
    _assert_skip_flag_in(captured, "build_plan_synthesizer")


def test_tasks_orchestrator_skips_default_mcp(monkeypatch):
    captured = _stub_factory_and_helpers(monkeypatch)
    from spine.agents.tasks_agent import build_tasks_agent

    build_tasks_agent(_base_state(), config=None)
    _assert_skip_flag_in(captured, "build_tasks_agent")


def test_verify_orchestrator_skips_default_mcp(monkeypatch):
    captured = _stub_factory_and_helpers(monkeypatch)
    from spine.agents.verify_agent import build_verify_agent

    build_verify_agent(_base_state(), config=None)
    _assert_skip_flag_in(captured, "build_verify_agent")


def test_gap_plan_orchestrator_skips_default_mcp(monkeypatch):
    captured = _stub_factory_and_helpers(monkeypatch)
    from spine.agents.gap_plan_agent import build_gap_plan_agent

    state = _base_state()
    # gap_plan typically runs after verify, so seed minimal context.
    state["feedback"] = ["needs a retry"]
    build_gap_plan_agent(state, config=None)
    _assert_skip_flag_in(captured, "build_gap_plan_agent")


def test_critic_skips_default_mcp(monkeypatch):
    captured = _stub_factory_and_helpers(monkeypatch)
    from spine.critic.agent import build_critic_agent

    state = _base_state()
    # ``_get_reviewed_phase`` keys on ``critic_reviewing`` first.
    state["critic_reviewing"] = "plan"
    build_critic_agent(state=state, config=None)
    _assert_skip_flag_in(captured, "build_critic_agent")


@pytest.mark.asyncio
async def test_slice_implementer_dispatcher_skips_default_mcp(monkeypatch, tmp_path):
    """The implement subgraph's slice_implementer node curates its own
    MCP-tool subset in restricted_tools — the factory must not re-inject."""
    captured = _stub_factory_and_helpers(monkeypatch)

    class _FakeTool:
        def __init__(self, name: str) -> None:
            self.name = name

    monkeypatch.setattr(
        "spine.agents.subagents.build_subagent_spec",
        lambda **kw: {
            "system_prompt": "implementer",
            "tools": [
                _FakeTool("mcp_codebase-index_find_symbol"),
                _FakeTool("mcp_codebase-index_get_function_source"),
            ],
            "model": object(),
            "response_format": None,
        },
    )

    # Drive one slice through the implementer node.
    from spine.workflow.subgraphs.implement_subgraph import _slice_implementer_node

    state: dict[str, Any] = {
        **_base_state(),
        "phase": "implement",
        "active_slice": {
            "id": "slice-1",
            "title": "test slice",
            "description": "do the thing",
            "target_files": ["spine/x.py"],
            "acceptance_criteria": ["X happens"],
        },
        "pending_slices": [],
        "completed_slices": [],
        "failed_slices": [],
    }
    try:
        await _slice_implementer_node(state, config=None)
    except Exception:
        # The fake build returns a plain ``object()`` so the downstream
        # agent.ainvoke will fail. That's fine — we only care that
        # build_phase_agent was called with the right flag, which
        # happens BEFORE the invocation attempt.
        pass

    _assert_skip_flag_in(captured, "slice_implementer_node")
