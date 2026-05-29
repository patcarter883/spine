"""Tests for ``spine.agents.prompt_format`` — the helper module that owns
the canonical XML tag vocabulary, the runtime block builders, and the
structural assertions used across the prompt test suite.
"""

from __future__ import annotations

import pytest

from spine.agents.prompt_format import (
    Tag,
    assert_has_tags,
    assert_hostage_layout,
    assert_tag_order,
    get_block,
    hostage_layout,
    parse_tags,
    xml_block,
    xml_blocks,
)


# ── xml_block / xml_blocks ─────────────────────────────────────────────


def test_xml_block_wraps_content_with_tag_name():
    out = xml_block(Tag.FINDINGS, "summary text here")
    assert out == "<findings>\nsummary text here\n</findings>"


def test_xml_block_elides_empty_content():
    assert xml_block(Tag.FINDINGS, "") == ""
    assert xml_block(Tag.FINDINGS, "   \n  ") == ""
    assert xml_block(Tag.FINDINGS, None) == ""  # type: ignore[arg-type]


def test_xml_block_strips_surrounding_whitespace():
    assert xml_block(Tag.OBJECTIVE, "  goal  \n") == "<objective>\ngoal\n</objective>"


def test_xml_blocks_stacks_blocks_separated_by_blank_lines():
    out = xml_blocks(
        (Tag.OBJECTIVE, "the goal"),
        (Tag.FINDINGS, "two findings"),
    )
    assert out == (
        "<objective>\nthe goal\n</objective>"
        "\n\n"
        "<findings>\ntwo findings\n</findings>"
    )


def test_xml_blocks_elides_empty_pairs_preserving_order():
    out = xml_blocks(
        (Tag.OBJECTIVE, "first"),
        (Tag.SPECIFICATION, ""),  # elide
        (Tag.PRIOR_RESEARCH, "third"),
    )
    assert "specification" not in out
    assert out.index("<objective>") < out.index("<prior_research>")


def test_xml_blocks_returns_empty_string_when_all_pairs_empty():
    assert xml_blocks((Tag.OBJECTIVE, ""), (Tag.FINDINGS, "")) == ""


def test_xml_blocks_ignores_malformed_pair():
    # Non-tuple, wrong-length tuple — must not raise, just skip.
    out = xml_blocks((Tag.OBJECTIVE, "ok"), "not a pair")  # type: ignore[arg-type]
    assert out == "<objective>\nok\n</objective>"


# ── hostage_layout ─────────────────────────────────────────────────────


def test_hostage_layout_appends_directive_after_blocks():
    blocks = xml_blocks((Tag.OBJECTIVE, "do X"))
    out = hostage_layout(blocks, "Analyse the objective and emit your verdict.")
    assert out.endswith("Analyse the objective and emit your verdict.")
    assert out.startswith("<objective>")
    # Sanity: closing tag precedes the directive.
    assert out.index("</objective>") < out.index("Analyse")


def test_hostage_layout_works_with_empty_blocks():
    """If a prompt has no data sections (rare but valid), the directive
    alone is a legitimate output."""
    out = hostage_layout("", "Do the thing.")
    assert out == "Do the thing."


def test_hostage_layout_raises_on_empty_directive():
    with pytest.raises(ValueError, match="non-empty directive"):
        hostage_layout(xml_blocks((Tag.OBJECTIVE, "x")), "")
    with pytest.raises(ValueError):
        hostage_layout("", "   \n  ")


# ── parse_tags ─────────────────────────────────────────────────────────


def test_parse_tags_extracts_in_order():
    text = (
        "<objective>\ngoal\n</objective>\n\n"
        "<findings>\nbody\n</findings>\n\n"
        "Directive at the bottom."
    )
    out = parse_tags(text)
    assert out == [("objective", "goal"), ("findings", "body")]


def test_parse_tags_handles_nested_lines_via_dotall():
    """Inner content with newlines must be captured fully."""
    text = "<findings>\nline 1\nline 2\nline 3\n</findings>"
    out = parse_tags(text)
    assert out == [("findings", "line 1\nline 2\nline 3")]


def test_parse_tags_ignores_html_like_noise_inside_code_blocks():
    """A code snippet inside a tag block might contain ``<Foo>`` — the
    parser must ignore those because they don't match the snake_case
    identifier rule.
    """
    text = (
        "<retrieved_code>\n```python\nclass Foo(<X>):\n  pass\n```\n</retrieved_code>"
    )
    out = parse_tags(text)
    assert len(out) == 1
    assert out[0][0] == "retrieved_code"


def test_parse_tags_returns_empty_for_no_tags():
    assert parse_tags("just some prose") == []
    assert parse_tags("") == []


# ── assert_hostage_layout ──────────────────────────────────────────────


def test_assert_hostage_layout_passes_when_directive_follows_tags():
    text = "<objective>\nx\n</objective>\n\nDo the work."
    assert_hostage_layout(text)  # no raise


def test_assert_hostage_layout_fails_when_text_ends_with_closing_tag():
    text = "<objective>\nx\n</objective>"
    with pytest.raises(AssertionError, match="hostage layout"):
        assert_hostage_layout(text)


def test_assert_hostage_layout_fails_when_no_tags_present():
    with pytest.raises(AssertionError, match="no closing tag"):
        assert_hostage_layout("just prose")


def test_assert_hostage_layout_fails_on_empty_text():
    with pytest.raises(AssertionError, match="empty"):
        assert_hostage_layout("")


# ── assert_has_tags / assert_tag_order / get_block ─────────────────────


def test_assert_has_tags_passes_when_all_present():
    text = xml_blocks(
        (Tag.OBJECTIVE, "x"),
        (Tag.FINDINGS, "y"),
        (Tag.HISTORY, "z"),
    )
    assert_has_tags(text, Tag.OBJECTIVE, Tag.FINDINGS)


def test_assert_has_tags_fails_when_missing():
    text = xml_blocks((Tag.OBJECTIVE, "x"))
    with pytest.raises(AssertionError, match="findings"):
        assert_has_tags(text, Tag.OBJECTIVE, Tag.FINDINGS)


def test_assert_tag_order_passes_when_monotonic():
    text = xml_blocks(
        (Tag.OBJECTIVE, "1"),
        (Tag.PRIOR_RESEARCH, "2"),
        (Tag.FINDINGS, "3"),
    )
    assert_tag_order(text, Tag.OBJECTIVE, Tag.PRIOR_RESEARCH, Tag.FINDINGS)
    # Skipping intermediate tags is allowed.
    assert_tag_order(text, Tag.OBJECTIVE, Tag.FINDINGS)


def test_assert_tag_order_fails_when_out_of_order():
    text = xml_blocks(
        (Tag.FINDINGS, "first"),
        (Tag.OBJECTIVE, "second"),
    )
    with pytest.raises(AssertionError, match="out of order"):
        assert_tag_order(text, Tag.OBJECTIVE, Tag.FINDINGS)


def test_get_block_returns_first_occurrence_body():
    text = xml_blocks(
        (Tag.OBJECTIVE, "the goal"),
        (Tag.FINDINGS, "the findings"),
    )
    assert get_block(text, Tag.OBJECTIVE) == "the goal"
    assert get_block(text, Tag.FINDINGS) == "the findings"


def test_get_block_returns_empty_when_tag_absent():
    text = xml_blocks((Tag.OBJECTIVE, "x"))
    assert get_block(text, Tag.FINDINGS) == ""


def test_tag_enum_values_are_lowercase_snake_case():
    """Every Tag value must match the ``parse_tags`` regex so a prompt
    built via xml_block(Tag.X, ...) is always discoverable by tests.
    """
    import re

    pattern = re.compile(r"^[a-z][a-z0-9_]*$")
    for t in Tag:
        assert pattern.match(t.value), f"{t.name}={t.value!r} fails regex"
