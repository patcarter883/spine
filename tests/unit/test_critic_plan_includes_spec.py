"""PLAN critic must receive the specification it is instructed to compare against.

Lever A from `.spine/reviews/plan-quality-fixes-proposal.md`: the critic's
PLAN-review prompt explicitly references `scope_inclusions` and
`scope_exclusions`, which live on the SPEC, not the plan. Threading
`specification_json` through the user message via the `<specification>`
hostage-layout block lets the critic actually check what its system prompt
already tells it to check.
"""
from __future__ import annotations

from spine.agents.prompt_format import (
    Tag,
    assert_hostage_layout,
    get_block,
    parse_tags,
)
from spine.workflow.critic_review import _build_review_prompt


_SPEC_JSON = (
    '{"title": "T", "scope_inclusions": ["spine/cli"], '
    '"scope_exclusions": ["spine/billing"]}'
)
_PLAN_JSON = '{"feature_slices": [{"id": "a", "title": "A"}]}'


def test_plan_critic_prompt_includes_spec_block_in_hostage_layout():
    prompt = _build_review_prompt(
        reviewed_phase="plan",
        structured_payload=_PLAN_JSON,
        description="Implement CLI verbose flag",
        spec_payload=_SPEC_JSON,
    )

    # The spec must land in its own <specification> block and the plan in
    # <findings> — never the other way round.
    spec_block = get_block(prompt, Tag.SPECIFICATION)
    assert "scope_inclusions" in spec_block
    assert "scope_exclusions" in spec_block
    assert "spine/cli" in spec_block

    findings = get_block(prompt, Tag.FINDINGS)
    assert "feature_slices" in findings

    # Hostage layout: directive sits AFTER every closing tag.
    assert_hostage_layout(prompt)

    # Order matters for U-curve attention — the objective + spec should
    # arrive before the plan being reviewed.
    found = [name for name, _ in parse_tags(prompt)]
    assert found.index(Tag.OBJECTIVE.value) < found.index(Tag.SPECIFICATION.value)
    assert found.index(Tag.SPECIFICATION.value) < found.index(Tag.FINDINGS.value)


def test_plan_critic_prompt_omits_spec_when_absent():
    prompt = _build_review_prompt(
        reviewed_phase="plan",
        structured_payload=_PLAN_JSON,
        description="Implement CLI verbose flag",
        spec_payload=None,
    )
    tags = {name for name, _ in parse_tags(prompt)}
    assert Tag.SPECIFICATION.value not in tags
    assert Tag.OBJECTIVE.value in tags
    assert Tag.FINDINGS.value in tags
    assert_hostage_layout(prompt)


def test_specify_critic_prompt_does_not_inject_spec_block():
    """SPECIFY review treats the spec as the structured payload itself —
    there's no separate <specification> block to inject."""
    prompt = _build_review_prompt(
        reviewed_phase="specify",
        structured_payload=_SPEC_JSON,
        description="Implement CLI verbose flag",
        spec_payload=None,
    )
    tags = {name for name, _ in parse_tags(prompt)}
    assert Tag.SPECIFICATION.value not in tags
    # The spec's JSON still appears, but inside <findings> (the payload under
    # review), not in a separate <specification> block.
    assert "scope_inclusions" in get_block(prompt, Tag.FINDINGS)
