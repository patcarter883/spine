"""Critic scope-claim refutation + terminal-verdict corroboration.

Regression target: trace 019ed849, where a weak critic model flagged embedding
and reranker provider config — both listed in the spec's scope_inclusions — as
"scope creep" and fired a terminal NEEDS_REVIEW that halted the run at
human_review.

Three guards are covered:
  - Rec 2: CriticReview carries a `cited_exclusions` field, propagated by the
    parser, and the PLAN prompt demands a verbatim citation.
  - Rec 1: `_validate_scope_claim` overturns an uncited scope-exclusion
    rejection (terminal → revision) and never touches a cited or
    spec_contradiction verdict.
  - Rec 3: `_corroborate_terminal_verdict` downgrades a single-vote terminal
    escalation that a second, independent review does not confirm.
"""

from __future__ import annotations

import asyncio
import json

from spine.models.enums import PhaseName, ReviewStatus
from spine.models.types import CriticReview
from spine.workflow.critic_review import (
    _corroborate_terminal_verdict,
    _parse_agent_review,
    _validate_scope_claim,
)
from spine.critic.agent import _PLAN_REVIEW_INSTRUCTIONS

PLAN = PhaseName.PLAN.value
REVISION = ReviewStatus.NEEDS_REVISION.value
REVIEW = ReviewStatus.NEEDS_REVIEW.value

# The trace's spec: embedding/reranker are INCLUSIONS, not exclusions.
_SPEC = json.dumps(
    {
        "title": "Config UI",
        "scope_inclusions": [
            "Embedding provider configuration UI",
            "Reranker provider configuration UI",
        ],
        "scope_exclusions": [
            "Database schema modifications",
            "Network connectivity or authentication changes",
        ],
    }
)


def _verdict(status, *, reason="r", suggestions=None, blocker=None, citations=None):
    return {
        "status": status,
        "tier": "agent",
        "reason": reason,
        "suggestions": suggestions or [],
        "blocker_category": blocker,
        "cited_exclusions": citations or [],
    }


# ── Rec 2: schema + parser + prompt ──────────────────────────────────────────


def test_critic_review_has_cited_exclusions_field():
    cr = CriticReview(status="NEEDS_REVIEW", tier="agent", reason="x")
    assert cr.cited_exclusions == []
    cr2 = CriticReview(
        status="NEEDS_REVISION", tier="agent", reason="x",
        cited_exclusions=["Database schema modifications"],
    )
    assert cr2.cited_exclusions == ["Database schema modifications"]


def test_parser_propagates_cited_exclusions():
    structured = CriticReview(
        status="NEEDS_REVISION", tier="agent", reason="violates scope",
        cited_exclusions=["Database schema modifications"],
    )
    parsed = _parse_agent_review({"structured_response": structured}, PLAN)
    assert parsed["cited_exclusions"] == ["Database schema modifications"]


def test_plan_prompt_demands_citation_and_protects_inclusions():
    assert "cited_exclusions" in _PLAN_REVIEW_INSTRUCTIONS
    # Inclusions must never be flagged as scope creep — the exact confusion
    # that broke the trace.
    assert "scope_inclusions" in _PLAN_REVIEW_INSTRUCTIONS
    assert "IN scope" in _PLAN_REVIEW_INSTRUCTIONS


# ── Rec 1: _validate_scope_claim ─────────────────────────────────────────────


def test_uncited_terminal_scope_claim_is_overturned_to_revision():
    """The exact trace failure: in-scope work called scope creep → terminal."""
    v = _verdict(
        REVIEW,
        reason=(
            "The plan violates scope by including embedding and reranker provider "
            "configuration, which are excluded. This is scope creep."
        ),
        suggestions=["Remove embedding and reranker provider slices (out of scope)"],
    )
    out = _validate_scope_claim(v, _SPEC, PLAN, "w1")
    assert out["status"] == REVISION  # demoted, run no longer halts
    assert out["blocker_category"] is None
    assert "overturned" in out["reason"].lower()
    # The "remove the in-scope work" suggestion is dropped.
    assert out["suggestions"] == []


def test_validly_cited_scope_violation_is_preserved():
    v = _verdict(
        REVIEW,
        reason="Slice touches the database schema, which is excluded.",
        citations=["Database schema modifications"],
    )
    out = _validate_scope_claim(v, _SPEC, PLAN, "w1")
    assert out["status"] == REVIEW  # genuine, cited → left terminal
    assert out is v or out["reason"] == v["reason"]


def test_spec_contradiction_verdict_is_untouched():
    v = _verdict(
        REVIEW,
        reason="The spec excludes backend config that the requirement needs.",
        blocker="spec_contradiction",
    )
    out = _validate_scope_claim(v, _SPEC, PLAN, "w1")
    assert out["status"] == REVIEW
    assert out["blocker_category"] == "spec_contradiction"


def test_non_scope_terminal_verdict_is_untouched():
    v = _verdict(REVIEW, reason="This design tradeoff needs human judgment.")
    out = _validate_scope_claim(v, _SPEC, PLAN, "w1")
    assert out["status"] == REVIEW


def test_uncited_nonterminal_scope_claim_is_only_annotated():
    v = _verdict(
        REVISION,
        reason="Possible scope creep in slice 2.",
        suggestions=["Tighten slice 2 to the stated targets"],
    )
    out = _validate_scope_claim(v, _SPEC, PLAN, "w1")
    assert out["status"] == REVISION  # unchanged
    assert out["suggestions"] == v["suggestions"]  # preserved
    assert "must not be removed" in out["reason"]


def test_no_spec_means_no_validation():
    v = _verdict(REVIEW, reason="scope creep everywhere")
    out = _validate_scope_claim(v, None, PLAN, "w1")
    assert out is v


def test_non_plan_phase_is_skipped():
    v = _verdict(REVIEW, reason="scope creep")
    out = _validate_scope_claim(v, _SPEC, PhaseName.SPECIFY.value, "w1")
    assert out is v


# ── Rec 3: _corroborate_terminal_verdict ─────────────────────────────────────


def _corroborate(parsed, second_or_exc):
    calls = {"n": 0}

    async def run_once(_prompt):
        calls["n"] += 1
        if isinstance(second_or_exc, Exception):
            raise second_or_exc
        return second_or_exc

    out = asyncio.run(
        _corroborate_terminal_verdict(
            parsed,
            run_once,
            reviewed_phase=PLAN,
            structured_payload="{}",
            spec_payload=_SPEC,
            description="d",
            work_id="w1",
        )
    )
    return out, calls["n"]


def test_terminal_verdict_downgraded_when_second_disagrees():
    first = _verdict(REVIEW, reason="needs human review")
    second = _verdict(ReviewStatus.PASSED.value, reason="looks fine")
    out, n = _corroborate(first, second)
    assert n == 1
    assert out["status"] == REVISION
    assert "not corroborated" in out["reason"].lower()


def test_terminal_verdict_kept_when_second_confirms():
    first = _verdict(REVIEW, reason="needs human review")
    second = _verdict(REVIEW, reason="agreed, halt")
    out, n = _corroborate(first, second)
    assert n == 1
    assert out["status"] == REVIEW


def test_spec_contradiction_kept_when_second_confirms_contradiction():
    first = _verdict(REVISION, reason="spec gap", blocker="spec_contradiction")
    second = _verdict(REVISION, reason="yes, spec gap", blocker="spec_contradiction")
    out, n = _corroborate(first, second)
    assert n == 1  # blocker makes it terminal despite REVISION status
    assert out["blocker_category"] == "spec_contradiction"


def test_spec_contradiction_downgraded_when_second_says_revision():
    first = _verdict(REVISION, reason="spec gap", blocker="spec_contradiction")
    second = _verdict(REVISION, reason="just rework it")
    out, n = _corroborate(first, second)
    assert n == 1
    assert out["status"] == REVISION
    assert out["blocker_category"] is None


def test_non_terminal_verdict_skips_corroboration():
    first = _verdict(REVISION, reason="rework")
    out, n = _corroborate(first, _verdict(ReviewStatus.PASSED.value))
    assert n == 0  # no second LLM call
    assert out is first


def test_corroboration_failure_keeps_first_verdict():
    first = _verdict(REVIEW, reason="halt")
    out, n = _corroborate(first, RuntimeError("boom"))
    assert n == 1
    assert out["status"] == REVIEW  # best-effort: keep the original
