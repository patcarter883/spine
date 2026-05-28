"""Prior-phase research findings must be injected into PLAN researcher
and PLAN research-manager prompts.

SPECIFY persists its findings to ``research_log.json``. PLAN's state
mapper carries them across via the ``prior_phase_findings`` state field.
Both the per-topic Blueprint Scout (``run_explore_do_node``) and the
PLAN research manager (``run_research_manager``) must surface that block
inside the prompt sent to the LLM, so the model doesn't re-discover what
SPECIFY already mapped.
"""

from __future__ import annotations

from typing import Any

import pytest

from spine.agents import exploration_agents


_SPECIFY_FINDINGS = [
    {
        "topic": "How does the configuration loader resolve workspace roots?",
        "summary": "Workspace root resolution happens in SpineConfig.load().",
        "patterns": ["lru_cache singleton pattern"],
        "file_map": {
            "spine/config.py": "main config loader",
            "spine/cli/__init__.py": "passes --workspace through",
        },
        "dependencies": ["pyyaml"],
    },
    {
        "topic": "How are CLI subcommands wired through to the agent factory?",
        "summary": "CLI entrypoint imports build_phase_agent from factory.",
        "patterns": ["click.group + click.command"],
        "file_map": {"spine/agents/factory.py": "phase agent factory"},
        "dependencies": ["click"],
    },
]


# ── Per-topic researcher prompt (run_explore_do_node) ──────────────────


@pytest.mark.asyncio
async def test_plan_researcher_prompt_includes_prior_specify_findings(
    monkeypatch, tmp_path
):
    """When `phase=="plan"` and `prior_phase_findings` is set, the PLAN
    Blueprint Scout's user prompt must contain a "Prior SPECIFY Research"
    section with the rendered finding blocks and don't-re-investigate
    framing.
    """
    captured: dict[str, Any] = {}

    async def _fake_invoke(agent, input_, *, work_id, context, config):
        captured["input"] = input_
        return {"messages": []}

    monkeypatch.setattr(
        exploration_agents,
        "_ainvoke_explore_collecting",
        _fake_invoke,
    )

    # Stub the heavy subagent / agent factory wiring — we only care about
    # the prompt string that flows into the agent invocation.
    def _fake_build_subagent_spec(*args, **kwargs):
        return {
            "system_prompt": "scout-system",
            "tools": [],
            "model": object(),
        }

    def _fake_build_phase_agent(**kwargs):
        return object()

    monkeypatch.setattr(
        "spine.agents.subagents.build_subagent_spec", _fake_build_subagent_spec
    )
    monkeypatch.setattr(
        "spine.agents.factory.build_phase_agent", _fake_build_phase_agent
    )

    state = {
        "work_id": "w1",
        "phase": "plan",
        "workspace_root": str(tmp_path),
        "spec_path": "",  # leave empty so we take the non-spec branch
        "prior_phase_findings": _SPECIFY_FINDINGS,
    }
    await exploration_agents.run_explore_do_node(state, None, topic="touch points for verbose flag")

    prompt = captured["input"]["messages"][0]["content"]
    assert "## Prior SPECIFY Research" in prompt
    assert "don't re-investigate" in prompt
    assert "spine/config.py" in prompt
    assert "spine/agents/factory.py" in prompt
    # Topic still flows through normally
    assert "touch points for verbose flag" in prompt


@pytest.mark.asyncio
async def test_specify_researcher_prompt_does_not_include_prior_section(
    monkeypatch, tmp_path
):
    """The prior-phase inject is PLAN-only. A SPECIFY researcher must not
    receive a "Prior SPECIFY Research" section even if (theoretically)
    the state contained one.
    """
    captured: dict[str, Any] = {}

    async def _fake_invoke(agent, input_, *, work_id, context, config):
        captured["input"] = input_
        return {"messages": []}

    monkeypatch.setattr(
        exploration_agents,
        "_ainvoke_explore_collecting",
        _fake_invoke,
    )

    def _fake_build_subagent_spec(*args, **kwargs):
        return {"system_prompt": "scout-system", "tools": [], "model": object()}

    def _fake_build_phase_agent(**kwargs):
        return object()

    monkeypatch.setattr(
        "spine.agents.subagents.build_subagent_spec", _fake_build_subagent_spec
    )
    monkeypatch.setattr(
        "spine.agents.factory.build_phase_agent", _fake_build_phase_agent
    )

    state = {
        "work_id": "w1",
        "phase": "specify",
        "workspace_root": str(tmp_path),
        # Even if a caller mistakenly seeded this on SPECIFY, the renderer
        # must ignore it (the cross-phase contract is one-way SPECIFY → PLAN).
        "prior_phase_findings": _SPECIFY_FINDINGS,
    }
    await exploration_agents.run_explore_do_node(state, None, topic="boundary of cli")

    prompt = captured["input"]["messages"][0]["content"]
    assert "## Prior SPECIFY Research" not in prompt


@pytest.mark.asyncio
async def test_plan_researcher_prompt_omits_section_when_field_absent(
    monkeypatch, tmp_path
):
    """The quick-workflow / no-prior-research path must render unchanged."""
    captured: dict[str, Any] = {}

    async def _fake_invoke(agent, input_, *, work_id, context, config):
        captured["input"] = input_
        return {"messages": []}

    monkeypatch.setattr(
        exploration_agents, "_ainvoke_explore_collecting", _fake_invoke
    )

    def _fake_build_subagent_spec(*args, **kwargs):
        return {"system_prompt": "scout-system", "tools": [], "model": object()}

    def _fake_build_phase_agent(**kwargs):
        return object()

    monkeypatch.setattr(
        "spine.agents.subagents.build_subagent_spec", _fake_build_subagent_spec
    )
    monkeypatch.setattr(
        "spine.agents.factory.build_phase_agent", _fake_build_phase_agent
    )

    state = {
        "work_id": "w1",
        "phase": "plan",
        "workspace_root": str(tmp_path),
        "spec_path": "",
        # No prior_phase_findings key at all
    }
    await exploration_agents.run_explore_do_node(state, None, topic="some plan topic")

    prompt = captured["input"]["messages"][0]["content"]
    assert "## Prior SPECIFY Research" not in prompt


# ── Research-manager prompt (run_research_manager) ─────────────────────


@pytest.mark.asyncio
async def test_plan_research_manager_context_includes_prior_findings(monkeypatch):
    """The PLAN research-manager's human-message context must contain a
    "Prior SPECIFY Research" block positioned AFTER the (optional) spec
    section and BEFORE the prior-round / rework / findings sections, so
    its topic-selection prompt sees what SPECIFY already mapped.
    """
    captured: dict[str, Any] = {}

    class _StructuredStub:
        async def ainvoke(self, messages, **kwargs):
            # messages = [SystemMessage, HumanMessage]
            captured["system"] = messages[0].content
            captured["context"] = messages[1].content
            return exploration_agents.ResearchManagerDecision(
                decision="done", topics=[]
            )

    class _ModelStub:
        def with_structured_output(self, schema):
            return _StructuredStub()

    monkeypatch.setattr(
        exploration_agents, "resolve_model", lambda *a, **kw: _ModelStub()
    )

    state = {
        "work_id": "w1",
        "phase": "plan",
        "description": "Add a --verbose flag to spine CLI.",
        "workspace_root": ".",
        "spec_path": "",  # skip spec read
        "research_round": 0,
        "max_rounds": 3,
        "topics": [],
        "findings": [],
        "prior_phase_findings": _SPECIFY_FINDINGS,
    }
    out = await exploration_agents.run_research_manager(state, None)
    assert out["manager_decision"] == "done"

    ctx = captured["context"]
    assert "## Prior SPECIFY Research" in ctx
    assert "spine/config.py" in ctx
    # Architectural-map framing — manager should not re-map these files
    assert "do not re-map" in ctx

    # System prompt picked the PLAN variant and references the new rule
    system = captured["system"]
    assert "Change Surface Research Manager" in system
    assert "Prior SPECIFY Research" in system


@pytest.mark.asyncio
async def test_specify_research_manager_context_excludes_prior_findings(monkeypatch):
    """A SPECIFY manager call must never render the PLAN-only prior section."""
    captured: dict[str, Any] = {}

    class _StructuredStub:
        async def ainvoke(self, messages, **kwargs):
            captured["system"] = messages[0].content
            captured["context"] = messages[1].content
            return exploration_agents.ResearchManagerDecision(
                decision="done", topics=[]
            )

    class _ModelStub:
        def with_structured_output(self, schema):
            return _StructuredStub()

    monkeypatch.setattr(
        exploration_agents, "resolve_model", lambda *a, **kw: _ModelStub()
    )

    state = {
        "work_id": "w1",
        "phase": "specify",
        "description": "Add a --verbose flag to spine CLI.",
        "workspace_root": ".",
        "research_round": 0,
        "max_rounds": 3,
        "topics": [],
        "findings": [],
        # Even if mistakenly seeded, must not render on SPECIFY phase.
        "prior_phase_findings": _SPECIFY_FINDINGS,
    }
    await exploration_agents.run_research_manager(state, None)

    ctx = captured["context"]
    assert "## Prior SPECIFY Research" not in ctx
    # System prompt is the SPECIFY variant
    assert "Architectural Research Manager" in captured["system"]
