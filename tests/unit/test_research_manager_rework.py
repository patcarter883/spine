"""Tests for the critic-rework guard in ``run_research_manager``.

Regression for trace 019e6974: PLAN reworked 3 times because critic_plan
rejected slice metadata (missing target_files). Each rework re-fired the
research manager which emitted near-duplicate topics, burning ~3.3M PLAN
prompt tokens to re-learn what the prior research_log already contained.
The guard forces ``manager_decision="done"`` when prior findings exist
and the critic's feedback isn't about a research gap, so synthesis can
take another swing without re-exploring."""

from __future__ import annotations

import asyncio

from spine.agents.exploration_agents import (
    _critic_wants_more_research,
    run_research_manager,
)


class TestCriticWantsMoreResearch:
    def test_no_review_means_no_research_demand(self):
        assert _critic_wants_more_research({}) is False

    def test_slice_metadata_complaint_does_not_demand_research(self):
        review = {
            "status": "needs_revision",
            "reason": (
                "Slice 'add-verbose-flag' does not include any target files "
                "in scope_inclusions. The target_files array is missing the "
                "required file path."
            ),
            "suggestions": [
                "Add the actual target file path 'spine/cli/__init__.py' "
                "to the target_files array in the slice",
            ],
        }
        assert _critic_wants_more_research(review) is False

    def test_explicit_research_request_triggers_demand(self):
        review = {
            "status": "needs_revision",
            "reason": "Need to investigate how the existing logging setup is wired before planning changes.",
            "suggestions": [],
        }
        assert _critic_wants_more_research(review) is True

    def test_missing_context_in_suggestion_triggers_demand(self):
        review = {
            "status": "needs_revision",
            "reason": "Plan is incomplete.",
            "suggestions": ["Explore the existing CLI module structure first."],
        }
        assert _critic_wants_more_research(review) is True


def test_rework_with_findings_skips_exploration_on_format_critic(monkeypatch):
    """Trace 019e6974's PLAN attempt-2 scenario: prior research_log was
    seeded (24 findings), critic rejected on missing target_files.
    Manager must force done — no new exploration."""
    state = {
        "phase": "plan",
        "work_id": "wk1",
        "description": "Add a --verbose flag to the CLI entrypoint",
        "topics": ["how is CLI dispatch organised"],
        "findings": [
            {"topic": "how is CLI dispatch organised",
             "summary": "Click @main.group() in spine/cli/__init__.py",
             "file_map": {"spine/cli/__init__.py": "main group"}},
        ],
        "research_round": 0,
        "max_rounds": 3,
        "retry_count": 1,
        "last_critic_review": {
            "status": "needs_revision",
            "reason": (
                "Slice 'add-verbose-flag' does not include any target files "
                "in scope_inclusions."
            ),
            "suggestions": [
                "Add the actual target file path 'spine/cli/__init__.py' "
                "to the target_files array in the slice",
            ],
            "attempt": 2,
            "tier": "agent",
        },
        "workspace_root": "/tmp",
    }

    # If the guard fires we should never reach the LLM. Patch resolve_model
    # so a model call would blow up loudly if it were attempted.
    def _fail_resolve_model(*args, **kwargs):  # noqa: ARG001
        raise AssertionError(
            "Manager should have force-skipped exploration on rework, "
            "but it tried to call the model"
        )

    monkeypatch.setattr(
        "spine.agents.exploration_agents.resolve_chat_model",
        _fail_resolve_model,
    )

    result = asyncio.run(run_research_manager(state, None))
    assert result == {"manager_decision": "done", "topics": []}


def test_rework_without_prior_findings_still_explores(monkeypatch):
    """If no prior findings are seeded (research_log was lost or this is
    the first attempt), the guard should NOT fire — the manager should
    still get to decide whether exploration is needed."""
    state = {
        "phase": "plan",
        "work_id": "wk2",
        "description": "Add a --verbose flag",
        "topics": [],
        "findings": [],
        "research_round": 0,
        "max_rounds": 3,
        "retry_count": 1,
        "last_critic_review": {
            "status": "needs_revision",
            "reason": "Slice metadata missing.",
            "suggestions": [],
            "attempt": 2,
            "tier": "agent",
        },
        "workspace_root": "/tmp",
    }

    sentinel = {"called": False}

    def _fake_resolve_model(*args, **kwargs):  # noqa: ARG001
        sentinel["called"] = True
        # Return a string so init_chat_model is exercised; the model call
        # itself will fail because we're not stubbing the inner LLM, but
        # we only care that resolve_model was reached.
        raise RuntimeError("ok — reached the model path")

    monkeypatch.setattr(
        "spine.agents.exploration_agents.resolve_chat_model",
        _fake_resolve_model,
    )

    try:
        asyncio.run(run_research_manager(state, None))
    except RuntimeError as exc:
        assert "reached the model path" in str(exc)
    assert sentinel["called"] is True


def test_rework_skipped_when_critic_demands_more_research(monkeypatch):
    """If the critic's reason mentions research/investigation, the guard
    must NOT short-circuit — the manager has to propose new topics."""
    state = {
        "phase": "plan",
        "work_id": "wk3",
        "description": "Add a --verbose flag",
        "topics": ["X"],
        "findings": [
            {"topic": "X", "summary": "...", "file_map": {}},
        ],
        "research_round": 0,
        "max_rounds": 3,
        "retry_count": 1,
        "last_critic_review": {
            "status": "needs_revision",
            "reason": (
                "Plan does not account for the logging configuration. "
                "Investigate how loggers are currently wired before "
                "rewriting the slice."
            ),
            "suggestions": [],
            "attempt": 2,
            "tier": "agent",
        },
        "workspace_root": "/tmp",
    }

    sentinel = {"called": False}

    def _fake_resolve_model(*args, **kwargs):  # noqa: ARG001
        sentinel["called"] = True
        raise RuntimeError("reached the model path")

    monkeypatch.setattr(
        "spine.agents.exploration_agents.resolve_chat_model",
        _fake_resolve_model,
    )

    try:
        asyncio.run(run_research_manager(state, None))
    except RuntimeError:
        pass
    assert sentinel["called"] is True
