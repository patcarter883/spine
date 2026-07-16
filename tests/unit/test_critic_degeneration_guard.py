"""Critic degeneration guards (trace 019ef7fd).

A structured-output critic that loops into free text and hits the token ceiling
must NOT be salvaged into a PASS, and its runaway text must not be stored
verbatim as the review reason.
"""

from __future__ import annotations

from types import SimpleNamespace

from spine.models.enums import ReviewStatus
from spine.workflow.critic_review import _clip_reason, _parse_agent_review_fallback


def _result(content: str, finish_reason: str) -> dict:
    msg = SimpleNamespace(
        content=content,
        response_metadata={"finish_reason": finish_reason},
    )
    return {"messages": [msg]}


def test_clip_reason_collapses_and_caps():
    blob = ("PASSED line\n" * 5000) + ("x" * 50_000)
    out = _clip_reason(blob)
    assert len(out) < 4200  # backstop cap, not the 50K pathological blob
    assert "truncated" in out


def test_truncated_critic_is_not_a_pass():
    # 126K-char repetition that keyword-matches PASSED but hit the length cap.
    content = "PASSED\nThe plan meets all quality standards.\n" * 4000
    parsed = _parse_agent_review_fallback(_result(content, "length"), "plan")
    assert parsed["status"] == ReviewStatus.NEEDS_REVISION.value
    assert "finish_reason=length" in parsed["reason"]
    assert len(parsed["reason"]) < 4300  # not the 126K blob


def test_clean_text_pass_still_works():
    parsed = _parse_agent_review_fallback(
        _result("PASSED — the plan looks good.", "stop"), "plan"
    )
    assert parsed["status"] == ReviewStatus.PASSED.value
    assert len(parsed["reason"]) < 1200


def test_empty_response_is_guard_verdict_not_agent():
    # Run 5646d24c round 4: an EMPTY critic response became an agent-sourced
    # needs_revision — charged an attempt with zero actionable feedback and
    # broke the gate-verdict comparison chain. Empty = harness noise = guard.
    for content in ("", "   \n\t  "):
        parsed = _parse_agent_review_fallback(_result(content, "stop"), "plan")
        assert parsed["status"] == ReviewStatus.NEEDS_REVISION.value
        assert parsed["verdict_source"] == "guard"
        assert parsed["suggestions"] == []
        assert "empty response" in parsed["reason"].lower()


def test_nonempty_unclear_text_stays_agent_sourced():
    # A real (if unparseable) opinion is still an agent verdict.
    parsed = _parse_agent_review_fallback(
        _result("The plan has some issues around config handling.", "stop"), "plan"
    )
    assert parsed["status"] == ReviewStatus.NEEDS_REVISION.value
    assert parsed.get("verdict_source") is None or parsed["verdict_source"] == "agent"
