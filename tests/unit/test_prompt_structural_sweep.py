"""Structural per-site sweep over every spine prompt constant.

Guards against future regressions where a new prompt site is added
without applying the project's XML-tagging convention (see
:mod:`spine.agents.prompt_format`). For every long system-prompt
constant, this file asserts:

1. The prompt parses cleanly via ``parse_tags`` (no malformed tags).
2. At least one tag from the canonical :class:`Tag` enum is present.
3. Specific role-prompt vs data-prompt expectations hold (system prompts
   typically have ``<role>`` and ``<constraints>``; user templates
   typically have ``<objective>`` + a hostage-layout directive tail).

The runtime-built user templates (manager context, synthesizer prompts,
supervisor / worker user messages) are exercised in their own dedicated
tests — those files already cover the per-call shape. This file pins
the constants that DON'T have dedicated tests, so a future contributor
can't silently drop the convention.
"""

from __future__ import annotations

import pytest

from spine.agents.prompt_format import Tag, assert_has_tags, parse_tags


# ── System-prompt constants (lazily imported to keep failure messages
#    pointed at the right module when something is broken) ────────────


def _system_prompt_inventory() -> dict[str, str]:
    """Return ``{display_name: prompt_text}`` for every spine system-prompt
    constant under the XML convention.

    Imported lazily so a single broken module surfaces a focused error
    rather than failing import-time on the whole inventory.
    """
    from spine.agents.classification import _CLASSIFICATION_SYSTEM
    from spine.agents.decomposer import _FALLBACK_PROMPT, _PLAN_PROMPT
    from spine.agents.exploration_agents import (
        _RESEARCH_MANAGER_PLAN,
        _RESEARCH_MANAGER_SPECIFY,
    )
    from spine.agents.factory import (
        SPINE_FILESYSTEM_EXEC_PROMPT,
        SPINE_FILESYSTEM_PROMPT,
    )
    from spine.agents.plan_do import _PLAN_SYSTEM_PROMPT
    from spine.agents.profile import SPINE_BASE_PROMPT
    from spine.agents.researcher_supervisor import _SUPERVISOR_SYSTEM_PROMPT
    from spine.agents.subagents import SUBAGENT_PROMPTS

    inventory: dict[str, str] = {
        "SPINE_BASE_PROMPT": SPINE_BASE_PROMPT,
        "SPINE_FILESYSTEM_PROMPT": SPINE_FILESYSTEM_PROMPT,
        "SPINE_FILESYSTEM_EXEC_PROMPT": SPINE_FILESYSTEM_EXEC_PROMPT,
        "_RESEARCH_MANAGER_SPECIFY": _RESEARCH_MANAGER_SPECIFY,
        "_RESEARCH_MANAGER_PLAN": _RESEARCH_MANAGER_PLAN,
        "_SUPERVISOR_SYSTEM_PROMPT": _SUPERVISOR_SYSTEM_PROMPT,
        "_PLAN_SYSTEM_PROMPT": _PLAN_SYSTEM_PROMPT,
        "_CLASSIFICATION_SYSTEM": _CLASSIFICATION_SYSTEM,
        "decomposer._PLAN_PROMPT": _PLAN_PROMPT,
        "decomposer._FALLBACK_PROMPT": _FALLBACK_PROMPT,
    }
    for name, prompt in SUBAGENT_PROMPTS.items():
        inventory[f"SUBAGENT_PROMPTS[{name!r}]"] = prompt
    return inventory


@pytest.mark.parametrize(
    "name,prompt",
    list(_system_prompt_inventory().items()),
    ids=list(_system_prompt_inventory().keys()),
)
def test_system_prompt_parses_and_has_canonical_tags(name: str, prompt: str):
    """Every spine system prompt must parse cleanly and emit at least one
    tag from the canonical :class:`Tag` vocabulary.

    The XML-tagging convention is a contract; this test pins it. If a new
    prompt site is added without tags, this test surfaces the omission
    against a specific module rather than a generic failure elsewhere.
    """
    canonical = {t.value for t in Tag}

    parsed = parse_tags(prompt)
    assert parsed, (
        f"{name} has no XML tags — wrap data-shaped sections via "
        f"spine.agents.prompt_format.xml_block(Tag.X, ...)."
    )
    found = {t for t, _ in parsed}
    unknown = found - canonical
    assert not unknown, (
        f"{name} uses non-canonical tag(s) {unknown}. Add them to "
        f"spine.agents.prompt_format.Tag or rewrite the renderer."
    )
    overlap = found & canonical
    assert overlap, f"{name} parses tags but none are canonical: {found}"


@pytest.mark.parametrize(
    "name,prompt",
    [
        ("_RESEARCH_MANAGER_SPECIFY", "_RESEARCH_MANAGER_SPECIFY"),
        ("_RESEARCH_MANAGER_PLAN", "_RESEARCH_MANAGER_PLAN"),
        ("_SUPERVISOR_SYSTEM_PROMPT", "_SUPERVISOR_SYSTEM_PROMPT"),
        ("_PLAN_SYSTEM_PROMPT", "_PLAN_SYSTEM_PROMPT"),
        ("_CLASSIFICATION_SYSTEM", "_CLASSIFICATION_SYSTEM"),
        ("decomposer._PLAN_PROMPT", "decomposer._PLAN_PROMPT"),
        ("decomposer._FALLBACK_PROMPT", "decomposer._FALLBACK_PROMPT"),
    ],
)
def test_role_governed_system_prompts_have_role_and_constraints(name: str, prompt: str):
    """Prompts that define an agent's role MUST carry both <role> and
    <constraints> blocks — those are the two universal elements of any
    role-governed prompt in this codebase. SPINE_BASE_PROMPT,
    filesystem prompts, and subagent prompts have their own per-site
    expectations in the inventory test above.
    """
    inv = _system_prompt_inventory()
    text = inv[prompt]
    assert_has_tags(text, Tag.ROLE, Tag.CONSTRAINTS)


def test_subagent_prompts_have_output_schema():
    """Every SUBAGENT_PROMPTS entry must declare its expected output
    structure inside an ``<output_schema>`` block so the downstream
    structured-output parser has documentation co-located with the prompt.
    """
    from spine.agents.subagents import SUBAGENT_PROMPTS

    for name, prompt in SUBAGENT_PROMPTS.items():
        assert_has_tags(prompt, Tag.OUTPUT_SCHEMA)


def test_filesystem_prompts_carry_tools_block():
    """The filesystem-tool prompts must describe the tool surface inside a
    ``<tools>`` block so subagents that splice them in get a consistent
    machine-parseable tool catalog.
    """
    from spine.agents.factory import (
        SPINE_FILESYSTEM_EXEC_PROMPT,
        SPINE_FILESYSTEM_PROMPT,
    )

    assert_has_tags(SPINE_FILESYSTEM_PROMPT, Tag.TOOLS)
    assert_has_tags(SPINE_FILESYSTEM_EXEC_PROMPT, Tag.TOOLS)
