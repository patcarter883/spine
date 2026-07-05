"""Tests for spine.workflow.critic_convergence — repeat/stagnation detection."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.workflow import critic_convergence as cc


# Modelled on trace 019ed383: the critic's three core asks recurred verbatim
# across all three plan attempts and were never addressed.
_PRIOR = {
    "reason": "plan has gaps",
    "suggestions": [
        "Add get_embedding_providers and get_reranker_providers methods",
        "Split the large config-ui-extensions slice into two",
        "Resolve phase_max_retries vs max_critic_retries schema mismatch",
    ],
}
_REPEAT = {
    "reason": "critical gaps remain",
    "suggestions": [
        "Add get_embedding_providers() and get_reranker_providers() to ui-api slice",
        "Split the config-ui-extensions slice into at least two slices",
        "Resolve the phase_max_retries schema assumption vs max_critic_retries",
    ],
}
_DIFFERENT = {
    "reason": "tests are missing",
    "suggestions": ["Add unit tests for input validation and CRUD operations"],
}


class TestReviewPoints:
    def test_prefers_suggestions(self):
        assert cc.review_points({"reason": "r", "suggestions": ["a", "b"]}) == ["a", "b"]

    def test_falls_back_to_reason(self):
        assert cc.review_points({"reason": "only reason", "suggestions": []}) == [
            "only reason"
        ]

    def test_drops_blank_entries(self):
        assert cc.review_points({"suggestions": ["a", "  ", ""]}) == ["a"]

    def test_empty_for_none(self):
        assert cc.review_points(None) == []


class TestIsRepeatVerdict:
    def test_recurring_asks_are_a_repeat(self):
        assert cc.is_repeat_verdict(_PRIOR, _REPEAT) is True

    def test_different_verdict_is_not_a_repeat(self):
        assert cc.is_repeat_verdict(_PRIOR, _DIFFERENT) is False

    def test_no_prior_is_never_repeat(self):
        # First revision round has nothing to repeat.
        assert cc.is_repeat_verdict({}, _REPEAT) is False
        assert cc.is_repeat_verdict(None, _REPEAT) is False

    def test_partial_overlap_below_ratio(self):
        # Only one of three prior asks recurs → below the 0.5 repeat ratio.
        partial = {"suggestions": ["Split the config-ui-extensions slice into two"]}
        assert cc.is_repeat_verdict(_PRIOR, partial) is False


class TestUnaddressedPoints:
    def test_all_three_asks_flagged(self):
        out = cc.unaddressed_points(_PRIOR, _REPEAT)
        assert out == _PRIOR["suggestions"]

    def test_shared_identifier_matches_despite_extra_words(self):
        prior = {"suggestions": ["Add get_embedding_providers method"]}
        current = {
            "suggestions": [
                "the ui-api slice must add get_embedding_providers() with error handling"
            ]
        }
        assert cc.unaddressed_points(prior, current) == [
            "Add get_embedding_providers method"
        ]

    def test_nothing_unaddressed_when_verdict_changes(self):
        assert cc.unaddressed_points(_PRIOR, _DIFFERENT) == []

    def test_reason_counts_as_candidate(self):
        # An ask that moved from a suggestion into the headline reason still
        # counts as recurring.
        prior = {"suggestions": ["Resolve phase_max_retries schema"]}
        current = {
            "reason": "the phase_max_retries schema is still unresolved",
            "suggestions": [],
        }
        assert cc.unaddressed_points(prior, current) == [
            "Resolve phase_max_retries schema"
        ]


class TestStagnationStreak:
    def test_increments_on_repeat(self):
        prior = {**_PRIOR, "stagnation_streak": 0}
        assert cc.next_stagnation_streak(prior, _REPEAT) == 1

    def test_accumulates_across_rounds(self):
        prior = {**_PRIOR, "stagnation_streak": 1}
        assert cc.next_stagnation_streak(prior, _REPEAT) == 2

    def test_resets_on_progress(self):
        prior = {**_PRIOR, "stagnation_streak": 2}
        assert cc.next_stagnation_streak(prior, _DIFFERENT) == 0

    def test_stagnation_limit_is_two(self):
        # Documents the escalation threshold used by the mapper.
        assert cc.STAGNATION_LIMIT == 2


class TestIsGoalpostShift:
    def test_new_asks_against_prior_is_a_shift(self):
        # The trace 019f01c2 failure: each round raises fresh asks.
        assert cc.is_goalpost_shift(_PRIOR, _DIFFERENT) is True

    def test_repeat_is_not_a_shift(self):
        # Repeats are stagnation, owned by the stagnation streak — not churn.
        assert cc.is_goalpost_shift(_PRIOR, _REPEAT) is False

    def test_no_prior_is_never_a_shift(self):
        assert cc.is_goalpost_shift({}, _DIFFERENT) is False
        assert cc.is_goalpost_shift(None, _DIFFERENT) is False


class TestChurnStreak:
    def test_increments_on_shift(self):
        prior = {**_PRIOR, "churn_streak": 0}
        assert cc.next_churn_streak(prior, _DIFFERENT) == 1

    def test_accumulates_across_rounds(self):
        prior = {**_DIFFERENT, "churn_streak": 1}
        # A second, again-different verdict keeps moving the goalposts.
        another = {"suggestions": ["Add docstrings to every new public method"]}
        assert cc.next_churn_streak(prior, another) == 2

    def test_resets_on_repeat(self):
        # A repeat round is stagnation, not churn → churn streak resets.
        prior = {**_PRIOR, "churn_streak": 2}
        assert cc.next_churn_streak(prior, _REPEAT) == 0

    def test_no_prior_is_zero(self):
        assert cc.next_churn_streak({}, _DIFFERENT) == 0

    def test_churn_limit_is_two(self):
        # Documents the escalation threshold used by the mapper.
        assert cc.CHURN_LIMIT == 2


class TestTokenization:
    def test_stopwords_do_not_create_false_matches(self):
        a = {"suggestions": ["you should add the thing to the file"]}
        b = {"suggestions": ["it must be in the config and on the page"]}
        # No shared content words → not a repeat.
        assert cc.is_repeat_verdict(a, b) is False


# Verdicts modelled on trace 019f260c: agent asks → truncation-guard notice →
# reference-gate findings, three sources in three consecutive rounds.
_GUARD = {
    "status": "needs_revision",
    "verdict_source": "guard",
    "reason": (
        "Critic response was truncated at the token limit (finish_reason="
        "length) without a structured verdict — treating as NEEDS_REVISION"
    ),
    "suggestions": [],
}
_GATE_FINDINGS = {
    "status": "needs_revision",
    "reason": (
        "Deterministic reference-symbol gate: 2 symbol contract violations: "
        "slice declares 'SpineConfig.resolve_embedding_provider' in provides, "
        "but that symbol ALREADY EXISTS"
    ),
    "suggestions": [
        "Move SpineConfig.resolve_embedding_provider to reference_symbols",
        "Move SpineConfig.resolve_reranker_provider to reference_symbols",
    ],
}
_GATE_VERDICT = {
    **_GATE_FINDINGS,
    "verdict_source": "gate",
    "agent_status": "passed",
}


class TestComputeStreaksSourceAware:
    def test_first_agent_round_sets_baseline(self):
        out = cc.compute_streaks({}, {**_PRIOR, "status": "needs_revision"})
        assert out["stagnation_streak"] == 0
        assert out["churn_streak"] == 0
        assert out["streak_baseline"]["suggestions"] == _PRIOR["suggestions"]

    def test_guard_round_freezes_streaks_and_carries_baseline(self):
        prior = {
            **_PRIOR,
            "status": "needs_revision",
            "stagnation_streak": 0,
            "churn_streak": 1,
        }
        out = cc.compute_streaks(prior, _GUARD)
        assert out["churn_streak"] == 1  # frozen, NOT incremented to 2
        assert out["stagnation_streak"] == 0
        assert out["unaddressed_points"] == []
        assert out["streak_baseline"]["suggestions"] == _PRIOR["suggestions"]

    def test_first_time_gate_finding_freezes_streaks(self):
        prior_guard_lcr = {
            **_GUARD,
            "stagnation_streak": 0,
            "churn_streak": 0,
            "streak_baseline": {"reason": _PRIOR["reason"], "suggestions": _PRIOR["suggestions"]},
            "reference_gate": {},
        }
        out = cc.compute_streaks(prior_guard_lcr, _GATE_VERDICT, current_gate=_GATE_FINDINGS)
        assert out["churn_streak"] == 0
        assert out["stagnation_streak"] == 0

    def test_trace_019f260c_sequence_does_not_escalate(self):
        # Round 1: agent asks. Round 2: truncation guard. Round 3: gate.
        # Under cross-source comparison this hit CHURN_LIMIT=2 and parked a
        # converging plan; source-aware accounting must keep both streaks at 0.
        r1 = cc.compute_streaks({}, {**_PRIOR, "status": "needs_revision"})
        lcr1 = {**_PRIOR, "status": "needs_revision", **r1}
        r2 = cc.compute_streaks(lcr1, _GUARD)
        lcr2 = {**_GUARD, **r2, "reference_gate": {}}
        r3 = cc.compute_streaks(lcr2, _GATE_VERDICT, current_gate=_GATE_FINDINGS)
        assert r3["churn_streak"] < cc.CHURN_LIMIT
        assert r3["stagnation_streak"] < cc.STAGNATION_LIMIT

    def test_recurring_gate_violations_stagnate(self):
        prior_gate_lcr = {
            **_GATE_VERDICT,
            "stagnation_streak": 0,
            "churn_streak": 0,
            "streak_baseline": {},
            "reference_gate": _GATE_FINDINGS,
        }
        out = cc.compute_streaks(prior_gate_lcr, _GATE_VERDICT, current_gate=_GATE_FINDINGS)
        assert out["stagnation_streak"] == 1
        assert out["churn_streak"] == 0

    def test_shifting_gate_violations_churn(self):
        prior_gate_lcr = {
            **_GATE_VERDICT,
            "stagnation_streak": 0,
            "churn_streak": 0,
            "streak_baseline": {},
            "reference_gate": _GATE_FINDINGS,
        }
        other_gate = {
            "status": "needs_revision",
            "reason": "Dependency cycle detected among feature slices",
            "suggestions": ["Reorder slices into a proper DAG"],
        }
        out = cc.compute_streaks(
            prior_gate_lcr,
            {**other_gate, "verdict_source": "gate"},
            current_gate=other_gate,
        )
        assert out["churn_streak"] == 1

    def test_agent_chain_resumes_across_guard_round(self):
        # agent asks → guard round → agent repeats the same asks: the repeat
        # must be detected against the ROUND-1 baseline, not the guard notice.
        r1 = cc.compute_streaks({}, {**_PRIOR, "status": "needs_revision"})
        lcr1 = {**_PRIOR, "status": "needs_revision", **r1}
        r2 = cc.compute_streaks(lcr1, _GUARD)
        lcr2 = {**_GUARD, **r2, "reference_gate": {}}
        r3 = cc.compute_streaks(lcr2, {**_REPEAT, "status": "needs_revision"})
        assert r3["stagnation_streak"] == 1
        assert r3["unaddressed_points"] == _PRIOR["suggestions"]

    def test_gate_override_of_agent_pass_clears_baseline(self):
        prior = {
            **_PRIOR,
            "status": "needs_revision",
            "stagnation_streak": 0,
            "churn_streak": 0,
        }
        out = cc.compute_streaks(prior, _GATE_VERDICT, current_gate=_GATE_FINDINGS)
        # agent_status == "passed" → the agent ask-chain converged.
        assert out["streak_baseline"] == {}

    def test_plain_agent_rounds_unchanged(self):
        # Legacy verdicts without verdict_source behave exactly as before.
        prior = {
            **_PRIOR,
            "status": "needs_revision",
            "stagnation_streak": 0,
            "churn_streak": 0,
        }
        assert cc.compute_streaks(prior, _REPEAT)["stagnation_streak"] == 1
        assert cc.compute_streaks(prior, _DIFFERENT)["churn_streak"] == 1
