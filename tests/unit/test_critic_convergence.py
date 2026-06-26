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
