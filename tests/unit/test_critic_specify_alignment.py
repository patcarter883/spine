"""Critic SPECIFY phase must compare spec against the user's description."""
from __future__ import annotations

from spine.critic.agent import _SPECIFY_REVIEW_INSTRUCTIONS
from spine.workflow.critic_review import _build_review_prompt


def test_specify_review_instructions_cover_traceability_and_proportionality():
    text = _SPECIFY_REVIEW_INSTRUCTIONS.lower()
    assert "traceability" in text
    assert "scope creep" in text
    assert "trivial" in text
    assert "scope_creep:" in _SPECIFY_REVIEW_INSTRUCTIONS


def test_review_prompt_inlines_user_description():
    prompt = _build_review_prompt(
        reviewed_phase="specify",
        structured_payload='{"requirements": []}',
        description="Add a --verbose flag to the CLI entrypoint",
    )
    assert "## Original User Description" in prompt
    assert "Add a --verbose flag" in prompt
    assert "## Structured Output Under Review" in prompt
    # Description must appear before the structured payload so the critic
    # treats it as source-of-truth context.
    assert prompt.index("Original User Description") < prompt.index(
        "Structured Output Under Review"
    )


def test_review_prompt_omits_description_section_when_blank():
    prompt = _build_review_prompt(
        reviewed_phase="plan",
        structured_payload='{"feature_slices": []}',
        description="   ",
    )
    assert "## Original User Description" not in prompt
    assert "## Structured Output Under Review" in prompt
