"""Placeholder/duplicate topic guard in run_research_manager.

Regression for work d8bc459c attempt 7 (trace 019f5e97): the manager returned
``{"decision": "explore", "topics": ["... topic ...", "... topic ...",
"... topic ...", "... topic ..."]}`` — literal template filler pattern-copied
from its own reasoning — and ``topic_lookup`` vector-searched the placeholder
string four times, attaching junk hits as research context.

The manager now sanitizes topics (placeholder/ellipsis/bracket/near-empty +
intra-round duplicates), retries ONCE with the rejection named in-conversation,
and mechanically forces ``done`` if the retry is garbage too.
"""

from __future__ import annotations

import asyncio
import logging

from spine.agents.exploration_agents import (
    ResearchManagerDecision,
    _is_placeholder_topic,
    _sanitize_topics,
    run_research_manager,
)


class _SequencedStructuredModel:
    """Yields one ResearchManagerDecision per ainvoke call, records messages."""

    def __init__(self, payloads: list[ResearchManagerDecision]):
        self._payloads = list(payloads)
        self.calls: list[list] = []

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        return self._payloads.pop(0)


class _SequencedModel:
    def __init__(self, payloads: list[ResearchManagerDecision]):
        self.structured = _SequencedStructuredModel(payloads)

    def with_structured_output(self, schema):  # noqa: ARG002
        return self.structured


def _base_state() -> dict:
    return {
        "phase": "plan",
        "work_id": "wk-topic-guard",
        "description": "Add farm-scoped RainGauge and Rainfall entities",
        "topics": [],
        "findings": [],
        "research_round": 0,
        "max_rounds": 3,
        "retry_count": 0,
        "last_critic_review": {},
        "workspace_root": "/tmp",
        "retrieved_context": [],
    }


def _decision(decision: str, topics: list[str]) -> ResearchManagerDecision:
    return ResearchManagerDecision(reasoning="stub", decision=decision, topics=topics)


class TestPlaceholderDetection:
    def test_literal_attempt7_placeholder(self):
        assert _is_placeholder_topic("... topic ...")

    def test_unicode_ellipsis(self):
        assert _is_placeholder_topic("… topic …")

    def test_bracket_placeholder(self):
        assert _is_placeholder_topic("<topic>")
        assert _is_placeholder_topic("[research question]")

    def test_near_empty(self):
        assert _is_placeholder_topic("")
        assert _is_placeholder_topic("   ")
        assert _is_placeholder_topic("the API")

    def test_real_topic_passes(self):
        assert not _is_placeholder_topic(
            "How are farm-scoped CRUD routes gated by the farm-focus middleware?"
        )


class TestSanitize:
    def test_intra_round_duplicates_collapse(self):
        kept, rejected = _sanitize_topics(
            [
                "How is farm scoping enforced in the HTTP request layer?",
                "How is farm scoping enforced in the HTTP request layer?",
                "What conventions do domain entity migrations follow here?",
            ]
        )
        assert len(kept) == 2
        assert rejected == [
            ("How is farm scoping enforced in the HTTP request layer?", "intra-round duplicate")
        ]

    def test_all_placeholders_rejected(self):
        kept, rejected = _sanitize_topics(["... topic ..."] * 4)
        assert kept == []
        assert all(why == "placeholder" for _, why in rejected)


def test_placeholder_topics_trigger_retry_then_success(monkeypatch, caplog):
    """Attempt-7 shape first, a healthy response on the retry."""
    good = [
        "How are farm-scoped CRUD routes structured and authorized?",
        "What migration conventions exist for uuid-keyed domain tables?",
    ]
    model = _SequencedModel(
        [_decision("explore", ["... topic ..."] * 4), _decision("explore", good)]
    )
    monkeypatch.setattr(
        "spine.agents.exploration_agents.resolve_chat_model",
        lambda *a, **kw: model,  # noqa: ARG005
    )

    caplog.set_level(logging.WARNING, logger="spine.agents.exploration_agents")
    result = asyncio.run(run_research_manager(_base_state(), None))

    assert result["manager_decision"] == "explore"
    assert result["topics"] == good
    assert len(model.structured.calls) == 2
    # The retry conversation must name the rejection so the model sees WHY.
    retry_messages = model.structured.calls[1]
    assert any("REJECTED" in getattr(m, "content", "") for m in retry_messages)
    assert [r for r in caplog.records if "rejected" in r.getMessage()]


def test_placeholder_topics_twice_forces_done(monkeypatch, caplog):
    """Garbage on both attempts → mechanical 'done', never explore-on-filler."""
    model = _SequencedModel(
        [
            _decision("explore", ["... topic ..."] * 4),
            _decision("explore", ["<topic>", "…"]),
        ]
    )
    monkeypatch.setattr(
        "spine.agents.exploration_agents.resolve_chat_model",
        lambda *a, **kw: model,  # noqa: ARG005
    )

    caplog.set_level(logging.WARNING, logger="spine.agents.exploration_agents")
    result = asyncio.run(run_research_manager(_base_state(), None))

    assert result["manager_decision"] == "done"
    assert result["topics"] == []
    assert len(model.structured.calls) == 2
    assert [r for r in caplog.records if "forcing 'done'" in r.getMessage()]


def test_partial_garbage_keeps_good_topics_no_retry(monkeypatch):
    """Mixed response: filler dropped, good topics kept, single call only."""
    model = _SequencedModel(
        [
            _decision(
                "explore",
                [
                    "... topic ...",
                    "How does request validation reject duplicate rainfall entries?",
                ],
            )
        ]
    )
    monkeypatch.setattr(
        "spine.agents.exploration_agents.resolve_chat_model",
        lambda *a, **kw: model,  # noqa: ARG005
    )

    result = asyncio.run(run_research_manager(_base_state(), None))

    assert result["manager_decision"] == "explore"
    assert result["topics"] == [
        "How does request validation reject duplicate rainfall entries?"
    ]
    assert len(model.structured.calls) == 1


def test_healthy_response_untouched(monkeypatch):
    topics = [
        "How is CLI verbosity configured and threaded through to subcommands?",
        "What logging conventions are used across the agent layer?",
    ]
    model = _SequencedModel([_decision("explore", topics)])
    monkeypatch.setattr(
        "spine.agents.exploration_agents.resolve_chat_model",
        lambda *a, **kw: model,  # noqa: ARG005
    )

    result = asyncio.run(run_research_manager(_base_state(), None))
    assert result["manager_decision"] == "explore"
    assert result["topics"] == topics
    assert len(model.structured.calls) == 1
