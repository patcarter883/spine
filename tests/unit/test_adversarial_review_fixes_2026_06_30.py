"""Regression tests for the 2026-06-30 adversarial-review fixes.

Each test pins a specific finding from the graph/prompt adversarial review so
the bug class cannot silently regress. Findings are referenced by their review
IDs (#N) in the docstrings.
"""

from __future__ import annotations

import re

import pytest


# ── #1 retry: `re` is imported and transient-error detection works ──────────


def test_retry_transient_error_with_code_in_message():
    """`re` must be imported so an HTTP code embedded in the message is matched
    without raising NameError, while a code-like substring is not a false hit."""
    from spine.agents.retry import _is_transient_error

    assert _is_transient_error(Exception("upstream error with code 503")) is True
    assert _is_transient_error(Exception("rejected: max 12500 tokens")) is False


# ── #3 specify/plan prompts render real newlines, not literal `\n` ──────────


@pytest.mark.parametrize(
    "builder_path",
    [
        ("spine.agents.specify_agent", "_build_specify_prompt"),
        ("spine.agents.specify_agent", "_build_specify_synthesizer_prompt"),
        ("spine.agents.plan_agent", "_build_plan_prompt"),
        ("spine.agents.plan_agent", "_build_plan_synthesizer_prompt"),
    ],
)
def test_phase_prompts_have_no_literal_backslash_n(builder_path):
    import importlib

    mod = importlib.import_module(builder_path[0])
    prompt = getattr(mod, builder_path[1])()
    assert "\n" in prompt
    assert "\\n" not in prompt


# ── #5b slice extractor never fabricates success ────────────────────────────


def test_normalize_status_unknown_falls_through_to_blocked():
    from spine.workflow.subgraphs.implement_subgraph import _normalize_status

    assert _normalize_status("implemented") == "implemented"
    assert _normalize_status("done") == "implemented"
    assert _normalize_status("failed") == "blocked"
    # Unknown / non-string must NOT be recorded as implemented.
    assert _normalize_status("totally-unknown") == "blocked"
    assert _normalize_status(None) == "blocked"  # type: ignore[arg-type]


def test_slice_extractor_unparseable_and_empty_are_blocked():
    from spine.workflow.subgraphs.implement_subgraph import _extract_slice_result

    class _Msg:
        def __init__(self, content):
            self.content = content

    # Non-JSON final message → blocked, not implemented.
    res = _extract_slice_result({"messages": [_Msg("I changed some files.")]}, "slice-1")
    assert res["status"] == "blocked"

    # No output at all → blocked.
    res = _extract_slice_result({"messages": []}, "slice-2")
    assert res["status"] == "blocked"


# ── #7 merge reducer preserves prior artifacts on empty update ──────────────


def test_merge_artifacts_empty_update_preserves_prior():
    from spine.models.state import _merge_artifacts

    left = {"specify": {"specification.md": "content"}}
    assert _merge_artifacts(left, {"specify": {}}) == left


# ── #8 verify negation guard fails closed on hedged/negated verdicts ────────


@pytest.mark.parametrize(
    "text",
    [
        "not verified",
        "NOT VERIFIED",
        "not passed",
        "not fully verified",
        "could not be verified",
        "cannot be verified",
        "never verified",
        "Slice 1 passed but slice 2 not yet implemented",
        "unverified",
        "incomplete",
    ],
)
def test_verify_negated_verdicts_are_not_pass(text):
    from spine.phases.verify import _verdict_is_pass

    assert _verdict_is_pass(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "VERIFIED",
        "all slices verified successfully",
        "passed: all acceptance criteria met",
    ],
)
def test_verify_affirmative_verdicts_pass(text):
    from spine.phases.verify import _verdict_is_pass

    assert _verdict_is_pass(text) is True


# ── #10 base prompt is capability-conditional ───────────────────────────────


def test_base_prompt_no_fs_variant_drops_filesystem_guidance():
    from spine.agents.profile import SPINE_BASE_PROMPT, SPINE_BASE_PROMPT_NO_FS

    # Full variant keeps the tool guidance.
    assert "Test after write" in SPINE_BASE_PROMPT
    assert "read_file" in SPINE_BASE_PROMPT
    # No-fs variant drops tool guidance + the free-text status line, but keeps
    # the role and the no-follow-up workflow guidance.
    assert "Test after write" not in SPINE_BASE_PROMPT_NO_FS
    assert "read_file" not in SPINE_BASE_PROMPT_NO_FS
    assert "status indicator" not in SPINE_BASE_PROMPT_NO_FS
    assert "phase executor" in SPINE_BASE_PROMPT_NO_FS
    assert "follow-up" in SPINE_BASE_PROMPT_NO_FS


# ── #12 verify result extractor tolerates a fenced-JSON verdict ─────────────


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"verdict": "VERIFIED", "checklist": []}\n```',
        '```\n{"verdict": "VERIFIED"}\n```',
        '{"verdict": "VERIFIED"}',
    ],
)
def test_verify_extractor_parses_fenced_json(raw):
    from spine.workflow.subgraphs.verify_subgraph import _extract_verification_result

    class _Msg:
        def __init__(self, content):
            self.content = content

    out = _extract_verification_result({"messages": [_Msg(raw)]}, "slice-1")
    assert out["verdict"] == "VERIFIED"
    assert out["slice_name"] == "slice-1"


# ── #4 deterministic plan-validation is a hard floor over the agent vote ────


@pytest.mark.asyncio
async def test_plan_validation_failure_overrides_agent_pass(monkeypatch):
    """A PASSED agent verdict must NOT upgrade a failed deterministic plan
    validation (e.g. dependency cycle). The structural failure wins and its
    reason is folded into the result so the rework prompt sees it."""
    from spine.models.enums import ReviewStatus
    from spine.workflow.subgraphs import critic_subgraph

    async def _fake_agent_critic_check(state, reviewed_phase, config=None):
        return {
            "status": ReviewStatus.PASSED.value,
            "tier": "agent",
            "reason": "Looks fine to me.",
            "suggestions": [],
        }

    monkeypatch.setattr(
        critic_subgraph, "agent_critic_check", _fake_agent_critic_check
    )

    state = {
        "reviewed_phase": "plan",
        "work_id": "w1",
        "validation_result": {
            "status": ReviewStatus.NEEDS_REVISION.value,
            "tier": "structural",
            "reason": "Dependency cycle detected among feature slices",
            "suggestions": ["Remove circular dependencies between slices"],
        },
    }
    out = await critic_subgraph._agent_check_node(state, None)
    assert out["phase_status"] == ReviewStatus.NEEDS_REVISION.value
    assert out["agent_result"]["status"] == ReviewStatus.NEEDS_REVISION.value
    assert "Dependency cycle" in out["agent_result"]["reason"]
    assert "Remove circular dependencies between slices" in out["agent_result"]["suggestions"]


@pytest.mark.asyncio
async def test_plan_validation_pass_keeps_agent_verdict(monkeypatch):
    """When deterministic validation passes, the agent verdict stands."""
    from spine.models.enums import ReviewStatus
    from spine.workflow.subgraphs import critic_subgraph

    async def _fake_agent_critic_check(state, reviewed_phase, config=None):
        return {
            "status": ReviewStatus.PASSED.value,
            "tier": "agent",
            "reason": "ok",
            "suggestions": [],
        }

    monkeypatch.setattr(
        critic_subgraph, "agent_critic_check", _fake_agent_critic_check
    )
    state = {
        "reviewed_phase": "plan",
        "work_id": "w1",
        "validation_result": {"status": ReviewStatus.PASSED.value},
    }
    out = await critic_subgraph._agent_check_node(state, None)
    assert out["phase_status"] == ReviewStatus.PASSED.value


# ── #5 implement deadlock-breaker detects cycles regardless of failed set ───


def test_route_slices_breaks_dependency_cycle_with_failed_present():
    """Two mutually-dependent pending slices (a cycle) must be dispatched to
    break the deadlock even when a failed slice is also present — previously
    the breaker was gated on an empty failed set and churned to the ceiling."""
    from langgraph.types import Send
    from spine.workflow.subgraphs.implement_subgraph import _route_slices

    state = {
        "work_id": "w1",
        "pending_slices": [
            {"id": "A", "dependencies": ["B"]},
            {"id": "B", "dependencies": ["A"]},
        ],
        # An unrelated failed slice in flight must not suppress cycle-breaking.
        "failed_slices": [{"id": "C", "slice_name": "C"}],
        "completed_slices": [],
        "slice_dispatch_count": 0,
    }
    routed = _route_slices(state)
    # Cycle slices are dispatched (not parked) → we get Send objects, not the
    # synthesis terminal string.
    assert isinstance(routed, list)
    assert any(isinstance(s, Send) for s in routed)


# ── #9 feedback reducer is bounded ──────────────────────────────────────────


def test_feedback_reducer_caps_growth():
    from spine.models.state import _MAX_FEEDBACK_ENTRIES, _append_capped_feedback

    left = [{"i": i} for i in range(_MAX_FEEDBACK_ENTRIES)]
    right = [{"i": "new1"}, {"i": "new2"}]
    merged = _append_capped_feedback(left, right)
    assert len(merged) == _MAX_FEEDBACK_ENTRIES
    # The most recent entries are retained (tail), including the newest.
    assert merged[-1] == {"i": "new2"}
    assert merged[-2] == {"i": "new1"}
