"""Coerce ``decision=explore + topics=[]`` to ``done`` in run_research_manager.

Regression for trace 019e72bc-199d-7f02-bac7-e053ddd60c22: a trivial work item
("Add a --verbose flag to the CLI entrypoint") + empty retrieved_context made
the local vLLM return ``{"decision": "explore", "topics": []}`` — structurally
valid Pydantic but semantically a non-decision. The downstream
``_research_router`` raised ``CriticalContractFailure`` and the whole SPECIFY
phase crashed before the synthesiser ran.

The manager now treats explore+empty as the model declining further research
and downgrades to ``done`` so synthesis can proceed.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from spine.agents.exploration_agents import (
    ResearchManagerDecision,
    run_research_manager,
)


class _FakeStructuredModel:
    """Mimics ``model.with_structured_output(ResearchManagerDecision)``."""

    def __init__(self, decision: str, topics: list[str]):
        self._payload = ResearchManagerDecision(
            reasoning="stub reasoning", decision=decision, topics=topics
        )

    async def ainvoke(self, _messages):
        return self._payload


class _FakeModel:
    def __init__(self, decision: str, topics: list[str]):
        self._decision = decision
        self._topics = topics

    def with_structured_output(self, schema):  # noqa: ARG002
        return _FakeStructuredModel(self._decision, self._topics)


def _base_state() -> dict:
    return {
        "phase": "specify",
        "work_id": "wk-empty-topics",
        "description": "Add a --verbose flag to the CLI entrypoint",
        "topics": [],
        "findings": [],
        "research_round": 0,
        "max_rounds": 3,
        "retry_count": 0,
        "last_critic_review": {},
        "workspace_root": "/tmp",
        "retrieved_context": [],
    }


def test_explore_with_empty_topics_coerced_to_done(monkeypatch, caplog):
    """The actual trace 019e72bc shape: model picks explore but populates
    an empty topics list. Manager must downgrade to 'done' with a warning."""

    monkeypatch.setattr(
        "spine.agents.exploration_agents.resolve_chat_model",
        lambda *a, **kw: _FakeModel("explore", []),  # noqa: ARG005
    )

    caplog.set_level(logging.WARNING, logger="spine.agents.exploration_agents")
    result = asyncio.run(run_research_manager(_base_state(), None))

    assert result["manager_decision"] == "done"
    assert result["topics"] == []
    # The warning is the breadcrumb operators rely on to spot lazy-model
    # downgrades during trace audits.
    coercion_records = [
        r for r in caplog.records if "explore with empty topics" in r.getMessage()
    ]
    assert coercion_records, (
        "Expected a warning mentioning 'explore with empty topics' but got: "
        f"{[r.getMessage() for r in caplog.records]}"
    )


def test_explore_with_topics_passes_through(monkeypatch):
    """Healthy multi-topic responses must not be touched by the coercion."""

    topics = [
        "How is CLI verbosity configured and threaded through to subcommands?",
        "What logging conventions are used across the agent layer?",
    ]
    monkeypatch.setattr(
        "spine.agents.exploration_agents.resolve_chat_model",
        lambda *a, **kw: _FakeModel("explore", topics),  # noqa: ARG005
    )

    result = asyncio.run(run_research_manager(_base_state(), None))
    assert result["manager_decision"] == "explore"
    assert result["topics"] == topics


def test_done_with_empty_topics_passes_through(monkeypatch, caplog):
    """When the model legitimately decides 'done', empty topics is the
    expected shape and must not produce a coercion warning."""

    monkeypatch.setattr(
        "spine.agents.exploration_agents.resolve_chat_model",
        lambda *a, **kw: _FakeModel("done", []),  # noqa: ARG005
    )

    caplog.set_level(logging.WARNING, logger="spine.agents.exploration_agents")
    result = asyncio.run(run_research_manager(_base_state(), None))

    assert result["manager_decision"] == "done"
    assert result["topics"] == []
    assert not [
        r for r in caplog.records if "explore with empty topics" in r.getMessage()
    ]


def test_directive_forbids_empty_topics_explore():
    """The hostage-layout tail directive must explicitly forbid the
    empty-topics-explore shape — prompt-level reinforcement of the
    runtime coercion."""

    # The directive is built inside run_research_manager. Rather than
    # mocking the full pipeline, we drive the manager once and inspect
    # the message passed to the structured model.
    captured: dict = {}

    class _CaptureStructured(_FakeStructuredModel):
        def __init__(self):
            super().__init__("done", [])

        async def ainvoke(self, messages):
            captured["messages"] = messages
            return self._payload

    class _CaptureModel:
        def with_structured_output(self, schema):  # noqa: ARG002
            return _CaptureStructured()

    import spine.agents.exploration_agents as ea

    original = ea.resolve_chat_model
    ea.resolve_chat_model = lambda *a, **kw: _CaptureModel()  # noqa: ARG005
    try:
        asyncio.run(run_research_manager(_base_state(), None))
    finally:
        ea.resolve_chat_model = original

    human_msg = captured["messages"][1]
    text = getattr(human_msg, "content", str(human_msg))
    assert "empty topics list is invalid" in text.lower() or (
        "empty topics" in text.lower() and "invalid" in text.lower()
    ), f"Directive missing the empty-topics forbidance; got: {text[-600:]}"
