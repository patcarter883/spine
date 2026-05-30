"""Critic SPECIFY phase must compare spec against the user's description."""
from __future__ import annotations

from spine.agents.prompt_format import Tag, assert_hostage_layout, get_block, parse_tags
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
    # Description lives in the <objective> block, structured payload in
    # <findings>. The directive sits OUTSIDE every closing tag.
    assert "Add a --verbose flag" in get_block(prompt, Tag.OBJECTIVE)
    findings = get_block(prompt, Tag.FINDINGS)
    assert '"requirements"' in findings
    assert_hostage_layout(prompt)
    # Objective must precede findings so the critic reads the source-of-truth
    # before the payload it is reviewing.
    found = [name for name, _ in parse_tags(prompt)]
    assert found.index(Tag.OBJECTIVE.value) < found.index(Tag.FINDINGS.value)


def test_review_prompt_omits_description_block_when_blank():
    prompt = _build_review_prompt(
        reviewed_phase="plan",
        structured_payload='{"feature_slices": []}',
        description="   ",
    )
    tags = {name for name, _ in parse_tags(prompt)}
    assert Tag.OBJECTIVE.value not in tags
    assert Tag.FINDINGS.value in tags
