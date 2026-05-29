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
#
# The supervisor↔worker refactor moved the per-cycle prompt construction
# into run_worker_node. The prior-SPECIFY-findings inject now flows
# through the enriched topic string the loop passes as ``global_goal`` to
# the supervisor and as ``topic`` to each worker turn. These tests stub
# the supervisor/worker so they don't fire an LLM, and capture what the
# loop passed in.


def _stub_subagent_and_agent(monkeypatch) -> None:
    """Shared scaffolding: stub subagent spec + context build so
    ``run_explore_do_node`` can drive a single loop iteration without
    touching the real model / MCP layer.

    The model returned by the stubbed subagent spec exposes
    ``bind_tools(...)`` (returning a bound stand-in) — required after
    the worker direct-bind refactor that replaced the agent loop.
    """

    class _FakeTool:
        def __init__(self, name: str) -> None:
            self.name = name

    class _StubBaseModel:
        def bind_tools(self, tools):
            return object()

    monkeypatch.setattr(
        "spine.agents.subagents.build_subagent_spec",
        lambda **kw: {
            "system_prompt": "scout-system",
            "tools": [_FakeTool("codebase_query"), _FakeTool("search_codebase")],
            "model": _StubBaseModel(),
        },
    )
    monkeypatch.setattr(
        "spine.agents.context.build_context", lambda state, phase: None
    )


@pytest.mark.asyncio
async def test_plan_researcher_prompt_includes_prior_specify_findings(
    monkeypatch, tmp_path
):
    """When ``phase=="plan"`` and ``prior_phase_findings`` is set, the
    enriched topic the loop hands to both supervisor and worker must
    contain a "Prior SPECIFY Research" block with the don't-re-investigate
    framing and the actual file paths from the findings.
    """
    _stub_subagent_and_agent(monkeypatch)

    captured: dict[str, Any] = {}

    from spine.agents import researcher_supervisor as rs
    from spine.agents.researcher_supervisor import (
        SupervisorDirective,
        ToolClass,
        StructuredFinding,
        FindingStatus,
    )

    async def _fake_supervisor(**kw):
        captured.setdefault("supervisor_global_goal", kw["global_goal"])
        # Cycle 0 returns continue with SEARCH; cycle 1 terminates.
        if kw["cycle_idx"] == 0:
            return SupervisorDirective(
                analysis_and_reasoning="r",
                is_complete=False,
                next_directive="search",
                allowed_tool_class=ToolClass.SEARCH,
            )
        return SupervisorDirective(analysis_and_reasoning="done", is_complete=True)

    async def _fake_worker(**kw):
        captured["worker_topic"] = kw["topic"]
        return StructuredFinding(
            tool_name="search_codebase",
            tool_class=ToolClass.SEARCH,
            status=FindingStatus.SUCCESS,
            target_path="spine/x.py",
            structured_code_block="body",
        )

    monkeypatch.setattr(rs, "run_supervisor_node", _fake_supervisor)
    monkeypatch.setattr(rs, "run_worker_node", _fake_worker)

    state = {
        "work_id": "w1",
        "phase": "plan",
        "workspace_root": str(tmp_path),
        "spec_path": "",  # leave empty so the spec section is omitted
        "prior_phase_findings": _SPECIFY_FINDINGS,
    }
    await exploration_agents.run_explore_do_node(
        state, None, topic="touch points for verbose flag"
    )

    # Both the supervisor's global_goal and the worker's topic carry the
    # enriched payload — they're the same string under the new design.
    # The enriched payload is XML-tagged (see spine.agents.prompt_format):
    # <objective> holds the bare topic, <prior_research> carries the
    # rendered SPECIFY findings.
    from spine.agents.prompt_format import Tag, assert_has_tags, get_block

    for label in ("supervisor_global_goal", "worker_topic"):
        text = captured[label]
        assert_has_tags(text, Tag.OBJECTIVE, Tag.PRIOR_RESEARCH)
        assert get_block(text, Tag.OBJECTIVE) == "touch points for verbose flag"
        prior = get_block(text, Tag.PRIOR_RESEARCH)
        assert "spine/config.py" in prior
        assert "spine/agents/factory.py" in prior


@pytest.mark.asyncio
async def test_specify_researcher_prompt_does_not_include_prior_section(
    monkeypatch, tmp_path
):
    """The prior-phase inject is PLAN-only. A SPECIFY topic must not see
    the prior-findings block even if the state mistakenly carries one.
    """
    _stub_subagent_and_agent(monkeypatch)

    captured: dict[str, Any] = {}

    from spine.agents import researcher_supervisor as rs
    from spine.agents.researcher_supervisor import (
        SupervisorDirective,
        ToolClass,
        StructuredFinding,
        FindingStatus,
    )

    async def _fake_supervisor(**kw):
        captured.setdefault("global_goal", kw["global_goal"])
        if kw["cycle_idx"] == 0:
            return SupervisorDirective(
                analysis_and_reasoning="r",
                is_complete=False,
                next_directive="search",
                allowed_tool_class=ToolClass.SEARCH,
            )
        return SupervisorDirective(analysis_and_reasoning="done", is_complete=True)

    async def _fake_worker(**kw):
        return StructuredFinding(
            tool_name="search_codebase",
            tool_class=ToolClass.SEARCH,
            status=FindingStatus.SUCCESS,
        )

    monkeypatch.setattr(rs, "run_supervisor_node", _fake_supervisor)
    monkeypatch.setattr(rs, "run_worker_node", _fake_worker)

    state = {
        "work_id": "w1",
        "phase": "specify",
        "workspace_root": str(tmp_path),
        "prior_phase_findings": _SPECIFY_FINDINGS,  # mistakenly seeded
    }
    await exploration_agents.run_explore_do_node(
        state, None, topic="boundary of cli"
    )
    from spine.agents.prompt_format import Tag, parse_tags

    tags = {name for name, _ in parse_tags(captured["global_goal"])}
    assert Tag.PRIOR_RESEARCH.value not in tags


@pytest.mark.asyncio
async def test_plan_researcher_prompt_omits_section_when_field_absent(
    monkeypatch, tmp_path
):
    """The quick-workflow / no-prior-research path renders the topic
    unchanged — no "Prior SPECIFY Research" block, no spec block.
    """
    _stub_subagent_and_agent(monkeypatch)

    captured: dict[str, Any] = {}

    from spine.agents import researcher_supervisor as rs
    from spine.agents.researcher_supervisor import (
        SupervisorDirective,
        ToolClass,
        StructuredFinding,
        FindingStatus,
    )

    async def _fake_supervisor(**kw):
        captured.setdefault("global_goal", kw["global_goal"])
        if kw["cycle_idx"] == 0:
            return SupervisorDirective(
                analysis_and_reasoning="r",
                is_complete=False,
                next_directive="search",
                allowed_tool_class=ToolClass.SEARCH,
            )
        return SupervisorDirective(analysis_and_reasoning="done", is_complete=True)

    async def _fake_worker(**kw):
        return StructuredFinding(
            tool_name="search_codebase",
            tool_class=ToolClass.SEARCH,
            status=FindingStatus.SUCCESS,
        )

    monkeypatch.setattr(rs, "run_supervisor_node", _fake_supervisor)
    monkeypatch.setattr(rs, "run_worker_node", _fake_worker)

    state = {
        "work_id": "w1",
        "phase": "plan",
        "workspace_root": str(tmp_path),
        "spec_path": "",
        # No prior_phase_findings key at all
    }
    await exploration_agents.run_explore_do_node(
        state, None, topic="some plan topic"
    )
    from spine.agents.prompt_format import Tag, parse_tags

    tags = {name for name, _ in parse_tags(captured["global_goal"])}
    assert Tag.PRIOR_RESEARCH.value not in tags


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

    from spine.agents.prompt_format import (
        Tag,
        assert_has_tags,
        assert_hostage_layout,
        assert_tag_order,
        get_block,
    )

    ctx = captured["context"]
    # Structural: prior_research sits between objective and findings, the
    # directive sentence is the hostage tail.
    assert_hostage_layout(ctx)
    assert_has_tags(ctx, Tag.OBJECTIVE, Tag.PRIOR_RESEARCH, Tag.FINDINGS)
    assert_tag_order(ctx, Tag.OBJECTIVE, Tag.PRIOR_RESEARCH, Tag.FINDINGS)
    # Semantic: the SPECIFY map's file paths land in the prior_research block.
    assert "spine/config.py" in get_block(ctx, Tag.PRIOR_RESEARCH)

    # System prompt picked the PLAN variant and references the prior-research
    # rule inside its <constraints> block.
    system = captured["system"]
    assert "Change Surface Research Manager" in get_block(system, Tag.ROLE)
    assert "<prior_research>" in get_block(system, Tag.CONSTRAINTS)


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

    from spine.agents.prompt_format import Tag, get_block, parse_tags

    ctx = captured["context"]
    # No prior_research tag emitted on SPECIFY even when the field is set.
    ctx_tags = {name for name, _ in parse_tags(ctx)}
    assert Tag.PRIOR_RESEARCH.value not in ctx_tags
    # System prompt is the SPECIFY variant (role mentions Architectural).
    assert "Architectural Research Manager" in get_block(
        captured["system"], Tag.ROLE
    )
