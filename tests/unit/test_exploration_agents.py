"""Unit tests for the research-manager prompt-building.

Verifies that the prompt assembled by ``run_research_manager``
correctly:
- injects the Retrieved Symbol Summaries section on every round
  (not only round 0 — the previous behaviour caused symbol-name
  hallucination on rounds 1+).
- injects the classification block when ``task_category`` is on state.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


def _captured_prompt():
    """Helper: returns the captured HumanMessage content from the latest run."""
    return _captured_prompt._value  # type: ignore[attr-defined]


_captured_prompt._value = ""  # type: ignore[attr-defined]


def _make_fake_model():
    """Build a fake model whose ``with_structured_output().ainvoke`` captures
    the HumanMessage content and returns a 'done' decision."""
    from spine.agents.exploration_agents import ResearchManagerDecision

    async def ainvoke(messages):
        # messages = [SystemMessage, HumanMessage]
        _captured_prompt._value = messages[-1].content  # type: ignore[attr-defined]
        return ResearchManagerDecision(decision="done", topics=[])

    structured = SimpleNamespace(ainvoke=ainvoke)
    return SimpleNamespace(with_structured_output=lambda _schema: structured)


@pytest.mark.asyncio
async def test_recall_section_present_on_every_round():
    """The Retrieved Symbol Summaries section must appear on rounds ≥1.

    Regression: previously gated on ``round_num == 0``, which left the
    research manager with the system-prompt instruction to cite symbols
    but no actual symbol list — driving hallucinations.
    """
    from spine.agents import exploration_agents

    state = {
        "description": "Refactor classifier output",
        "phase": "specify",
        "work_id": "wk-test",
        "workspace_root": ".",
        "research_round": 2,
        "max_rounds": 3,
        "task_category": "Backend/API",
        "classification_confidence": 0.9,
        "retrieved_context": [
            {
                "symbol_name": "classify_task",
                "symbol_type": "function",
                "file_path": "spine/agents/classification.py",
                "enriched_summary": "Classifies a work description.",
            }
        ],
    }

    with patch.object(
        exploration_agents, "resolve_model", return_value=_make_fake_model()
    ):
        await exploration_agents.run_research_manager(state, config=None)

    assert "Retrieved Symbol Summaries" in _captured_prompt()
    assert "classify_task" in _captured_prompt()


@pytest.mark.asyncio
async def test_classification_block_injected_when_category_present():
    """The ``## Task Classification`` block must be in the prompt when
    ``task_category`` is set on state — the exploration path was
    previously skipping this entirely."""
    from spine.agents import exploration_agents

    state = {
        "description": "Add a new tool to factory",
        "phase": "specify",
        "work_id": "wk-test",
        "workspace_root": ".",
        "research_round": 0,
        "max_rounds": 3,
        "task_category": "Frontend/UI",
        "classification_reasoning": "User is asking about UI components.",
        "retrieved_context": [],
    }

    with patch.object(
        exploration_agents, "resolve_model", return_value=_make_fake_model()
    ):
        await exploration_agents.run_research_manager(state, config=None)

    prompt = _captured_prompt()
    assert "## Task Classification" in prompt
    assert "Frontend/UI" in prompt
    assert "User is asking about UI components." in prompt


@pytest.mark.asyncio
async def test_classification_block_omitted_when_no_category():
    """When classification failed (no ``task_category`` on state), the
    classification block must be absent — must not render
    ``Category: None``."""
    from spine.agents import exploration_agents

    state = {
        "description": "Some work",
        "phase": "specify",
        "work_id": "wk-test",
        "workspace_root": ".",
        "research_round": 0,
        "max_rounds": 3,
        "retrieved_context": [],
    }

    with patch.object(
        exploration_agents, "resolve_model", return_value=_make_fake_model()
    ):
        await exploration_agents.run_research_manager(state, config=None)

    prompt = _captured_prompt()
    assert "## Task Classification" not in prompt
    assert "Category: None" not in prompt


def test_format_classification_block_helper():
    """The shared helper must return empty when category is missing,
    and a proper block with reasoning when present."""
    from spine.agents.helpers import format_classification_block

    assert format_classification_block(None, "anything") == ""
    assert format_classification_block("", "anything") == ""

    block = format_classification_block("Backend/API", "Because it touches a route.")
    assert block.startswith("## Task Classification\n")
    assert "Category: Backend/API" in block
    assert "Because it touches a route." in block
    assert block.endswith("\n\n")

    # Reasoning may be empty/None — block should still render category.
    block_noreason = format_classification_block("Database", "")
    assert "Category: Database" in block_noreason
    assert "## Task Classification" in block_noreason
