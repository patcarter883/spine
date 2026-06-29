"""Regression tests for needs_review detail surfacing.

A restarted task that ended in ``needs_review`` used to show the reviewer
"needs review" with zero explanation, because:

1. ``dispatcher._completion_result_payload`` (formerly inline in each finalize
   path) persisted only ``feedback_count`` on the resume/restart paths,
   dropping the ``feedback`` list and ``last_critic_review`` the UI reads.
2. ``compose._verify_result_mapper`` emitted a hardcoded generic reason that
   pointed at a ``verification.md`` the restart had cleared.

These tests pin the fixed behaviour.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.work.dispatcher import _completion_result_payload
from spine.workflow.compose import (
    _verify_failure_reason,
    _verify_failure_suggestions,
)


class TestCompletionResultPayload:
    """The persisted result blob must carry the review reason every time."""

    def test_persists_needs_review_feedback_and_critic_review(self) -> None:
        result = {
            "artifacts": {"verify": {"verification.md": "..."}},
            "feedback": [
                {"status": "passed", "tier": "critic"},
                {"status": "needs_review", "tier": "verify", "reason": "boom"},
            ],
            "last_critic_review": {"status": "FAIL", "reason": "nope"},
            "prompt_request": None,
        }
        payload = _completion_result_payload(result)

        # The list (not just a count) survives, filtered to needs_review.
        assert payload["feedback"] == [
            {"status": "needs_review", "tier": "verify", "reason": "boom"}
        ]
        assert payload["feedback_count"] == 2
        assert payload["last_critic_review"] == {"status": "FAIL", "reason": "nope"}
        assert payload["artifacts"] == {"verify": ["verification.md"]}

    def test_extra_keys_merged(self) -> None:
        payload = _completion_result_payload({}, extra={"restarted": True})
        assert payload["restarted"] is True
        # Defaults are still present and safe on an empty result.
        assert payload["feedback"] == []
        assert payload["feedback_count"] == 0
        assert payload["last_critic_review"] is None

    def test_non_list_feedback_is_tolerated(self) -> None:
        payload = _completion_result_payload({"feedback": "garbage"})
        assert payload["feedback"] == []
        assert payload["feedback_count"] == 0


class TestVerifyFailureReason:
    """The needs_review reason must name concrete gaps, not a dangling pointer."""

    def test_reason_from_slice_gaps(self) -> None:
        subgraph_result = {
            "verification_findings": [
                {"slice_name": "slice-a.md", "verdict": "VERIFIED", "gaps": []},
                {
                    "slice_name": "slice-b.md",
                    "verdict": "NOT_VERIFIED",
                    "gaps": ["missing error handling", "no tests"],
                },
            ]
        }
        reason = _verify_failure_reason(subgraph_result)
        assert "1 slice(s) did not pass" in reason
        assert "slice-b.md" in reason
        assert "missing error handling" in reason
        # The passing slice is not named.
        assert "slice-a.md" not in reason

    def test_falls_back_to_summary(self) -> None:
        subgraph_result = {
            "verification_findings": [],
            "artifacts_output": {"verification.md": "2 slices failed: foo, bar"},
        }
        assert _verify_failure_reason(subgraph_result) == "2 slices failed: foo, bar"

    def test_generic_fallback_when_nothing_available(self) -> None:
        reason = _verify_failure_reason({})
        assert "Manual review required" in reason

    def test_suggestions_aggregate_recommendations(self) -> None:
        subgraph_result = {
            "verification_findings": [
                {"slice_name": "a", "verdict": "NOT_VERIFIED", "recommendations": ["add tests"]},
                {
                    "slice_name": "b",
                    "verdict": "NOT_VERIFIED",
                    "recommendations": ["add tests", "handle nulls"],
                },
            ]
        }
        suggestions = _verify_failure_suggestions(subgraph_result)
        # Deduped, order-preserving.
        assert suggestions == ["add tests", "handle nulls"]
