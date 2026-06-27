"""Semantic XML tagging + hostage-layout helpers for LLM prompts.

The motivation is two-fold:

1. **Attention bleed.** Smaller and Mixture-of-Experts models (Qwen, DeepSeek,
   Llama-3 class) lose precision when prompts splice raw data into instruction
   prose — a comment inside a code snippet can read like a directive. Wrapping
   every dynamic data block in an explicit XML tag (e.g. ``<findings>...
   </findings>``) tells the model "this region is data, not instruction" via
   the self-attention mechanism, which these models are heavily fine-tuned to
   recognise.

2. **U-curve mis-attention.** Small-model attention is high at the top and the
   bottom of a prompt, low in the middle. Today many spine prompts put the
   instruction at the TOP — exactly the wrong place. The "hostage" layout
   moves the instruction to the absolute tail, after every closing tag, so the
   last thing the model sees is what it must do next.

This module exposes three thin runtime helpers (``xml_block`` / ``xml_blocks``
/ ``hostage_layout``) used everywhere a prompt is built, and four test helpers
(``parse_tags`` / ``assert_hostage_layout`` / ``assert_has_tags`` /
``assert_tag_order``) used to verify structure without pinning on transient
wording.

The module is pure-stdlib so tests can import it without dragging in langchain.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Iterable


class Tag(str, Enum):
    """Canonical tag vocabulary.

    Every spine prompt MUST use a member of this enum — no ad-hoc string
    literals — so adding a new tag is a deliberate vocabulary extension and
    tests can keep an authoritative list of "things the model might see".

    Data / context blocks (used in user prompts):
        OBJECTIVE: the global goal / topic / work description.
        SPECIFICATION: ``specification.md`` content (PLAN-phase only).
        PRIOR_RESEARCH: SPECIFY findings injected into PLAN.
        RETRIEVED_CODE: recall hits with code snippets.
        FINDINGS: accumulated ``ResearchFindings`` for the current phase.
        LATEST_FINDING: most recent ``StructuredFinding`` (supervisor turn).
        HISTORY: running list of prior cycles / findings.
        TOPICS_ALREADY_EXPLORED: manager's prior-round roll-up.
        COVERED_GROUND: compact digest of sibling/prior-round findings
            handed to each explore branch so researchers build on already-
            mapped ground instead of re-fetching it.
        CRITIC_FEEDBACK: last critic verdict + suggestions (rework path).
        SCRATCHPAD: working-memory accumulator.
        ERRORS: tool / system error dossier.
        ONBOARDING_DOCS: references to the project's on-disk onboarding
            documentation (paths the agent may read on demand).
        DIRECTIVE: supervisor's next-step instruction (when shown to a worker
            as a data block — distinct from the plain-text final-instruction
            tail that lives outside any tag, see :func:`hostage_layout`).
        REFERENCE_SYMBOLS: implement-phase — existing definitions the slice's
            code calls/extends/mimics, with their current source inlined so the
            implementer never surveys to find them.
        EDIT_PLAN: implement-phase — the planner's ordered targeted edits, each
            with the current source of the symbol it changes inlined.

    Role / system blocks (used in system prompts):
        ROLE: agent identity / mission paragraph.
        TOOLS: tool catalog.
        WORKFLOW: step-by-step process.
        CONSTRAINTS: hard limits / rules.
        OUTPUT_SCHEMA: required structured-output shape.
        EXAMPLES: few-shot examples when present.
    """

    # Data / context
    OBJECTIVE = "objective"
    SPECIFICATION = "specification"
    PRIOR_RESEARCH = "prior_research"
    RETRIEVED_CODE = "retrieved_code"
    FINDINGS = "findings"
    LATEST_FINDING = "latest_finding"
    HISTORY = "history"
    TOPICS_ALREADY_EXPLORED = "topics_already_explored"
    COVERED_GROUND = "covered_ground"
    CRITIC_FEEDBACK = "critic_feedback"
    SCRATCHPAD = "scratchpad"
    ERRORS = "errors"
    DIRECTIVE = "directive"
    ONBOARDING_DOCS = "onboarding_documentation"
    # Distilled lessons from prior runs' critic feedback for this phase —
    # cross-run "experience" injected so the agent avoids repeat defects.
    LEARNED_EXPERIENCE = "learned_experience"
    # Implement-phase slice payload: existing definitions the slice's code
    # extends (with their source inlined), and the planner's targeted edits.
    REFERENCE_SYMBOLS = "reference_symbols"
    EDIT_PLAN = "edit_plan"

    # Role / system
    ROLE = "role"
    TOOLS = "tools"
    WORKFLOW = "workflow"
    CONSTRAINTS = "constraints"
    OUTPUT_SCHEMA = "output_schema"
    EXAMPLES = "examples"


# Type alias for one (tag, content) pair when feeding xml_blocks.
TagPair = tuple[Tag, str]


# ── Runtime helpers ─────────────────────────────────────────────────────


def xml_block(tag: Tag, content: str) -> str:
    """Wrap ``content`` in ``<tag>...</tag>``.

    Empty / whitespace-only ``content`` returns ``""`` so optional sections
    elide cleanly when passed through :func:`xml_blocks`. Content is stripped
    of surrounding whitespace so the rendered block has predictable boundaries.
    """
    if not isinstance(content, str):
        return ""
    body = content.strip()
    if not body:
        return ""
    return f"<{tag.value}>\n{body}\n</{tag.value}>"


def xml_blocks(*pairs: TagPair) -> str:
    """Render a sequence of (tag, content) pairs as stacked XML blocks.

    Empty content elides. Blocks are separated by a blank line so the rendered
    prompt is visually parseable in trace logs. Pair order is preserved —
    callers should pass pairs in the order they want sections to appear.
    """
    rendered: list[str] = []
    for pair in pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            continue
        tag, content = pair
        block = xml_block(tag, content)
        if block:
            rendered.append(block)
    return "\n\n".join(rendered)


def hostage_layout(blocks: str, directive: str) -> str:
    """Compose a "hostage prompt": data blocks first, instruction last.

    The hostage layout places the directive AFTER every closing XML tag so the
    model's high-attention tail lands on the instruction, not on whatever data
    happened to be the last block. ``blocks`` should be the output of
    :func:`xml_blocks` (or any pre-built tagged content); ``directive`` is the
    plain-text instruction that tells the model what to do this turn.

    Raises:
        ValueError: when ``directive`` is empty — a hostage prompt with no
            instruction is a programming bug, not a runtime case.
    """
    directive_text = (directive or "").strip()
    if not directive_text:
        raise ValueError(
            "hostage_layout requires a non-empty directive — the model needs "
            "a final instruction to act on."
        )
    blocks_text = (blocks or "").strip()
    if not blocks_text:
        return directive_text
    return f"{blocks_text}\n\n{directive_text}"


# ── Test helpers ────────────────────────────────────────────────────────


# Matches ``<tag>...</tag>`` where ``tag`` is a snake_case identifier. The
# inner group captures the body non-greedily. DOTALL so newlines inside the
# block are part of the capture. Anchored to lowercase letters + underscore
# + digits — matches the Tag enum's value pattern and ignores stray ``<...>``
# in code samples (those start with capitals or punctuation).
_TAG_RE = re.compile(r"<([a-z][a-z0-9_]*)>(.*?)</\1>", re.DOTALL)


def parse_tags(text: str) -> list[tuple[str, str]]:
    """Extract ``(tag_name, inner_text)`` pairs from a prompt in order.

    Cheap regex parser — does NOT validate nesting or attribute syntax (spine
    prompts use no nesting and no attributes by convention). Inner text is
    returned exactly as found (with leading/trailing whitespace stripped) so
    tests can substring-search inside a known block.

    Tags whose names are not in :class:`Tag` are still returned — tests can
    decide whether to reject them.
    """
    if not text:
        return []
    return [(name, body.strip()) for name, body in _TAG_RE.findall(text)]


def _tag_value(tag: Tag | str) -> str:
    return tag.value if isinstance(tag, Tag) else tag


def assert_hostage_layout(text: str) -> None:
    """Assert the prompt's final non-whitespace content sits OUTSIDE any tag.

    The directive must follow every closing ``</tag>``. If the last ``</...>``
    in the prompt has only whitespace after it, the prompt has no directive
    tail and the assertion fails.

    Raises:
        AssertionError: when the prompt ends with a closing tag (no tail),
            when no tags are present at all (this helper expects at least one
            data block), or when the directive tail is empty.
    """
    if not text or not text.strip():
        raise AssertionError("prompt is empty")
    # Find the position immediately after the last closing tag.
    last_close = text.rfind("</")
    if last_close == -1:
        raise AssertionError(
            "prompt has no closing tag — call hostage_layout / xml_blocks "
            "to wrap dynamic data before the final directive."
        )
    end_of_tag = text.find(">", last_close)
    if end_of_tag == -1:
        raise AssertionError(
            f"prompt has a malformed closing tag near offset {last_close}"
        )
    tail = text[end_of_tag + 1:].strip()
    if not tail:
        raise AssertionError(
            "prompt ends with a closing tag — the directive must sit after "
            "every </tag> (hostage layout)."
        )


def assert_has_tags(text: str, *required: Tag | str) -> None:
    """Assert that every tag in ``required`` appears at least once in ``text``.

    Order does not matter. Use :func:`assert_tag_order` if order matters.
    """
    present = {name for name, _ in parse_tags(text)}
    missing = [
        _tag_value(t) for t in required if _tag_value(t) not in present
    ]
    if missing:
        raise AssertionError(
            f"prompt missing required tag(s): {missing}. "
            f"Present: {sorted(present)}"
        )


def assert_tag_order(text: str, *order: Tag | str) -> None:
    """Assert that the named tags appear in ``order`` in the prompt.

    Other tags between the named ones are allowed; the assertion is that the
    relative positions of ``order`` are monotonic. Missing tags fail.
    """
    expected = [_tag_value(t) for t in order]
    found = [name for name, _ in parse_tags(text)]
    indices: list[int] = []
    cursor = 0
    for want in expected:
        try:
            i = found.index(want, cursor)
        except ValueError:
            raise AssertionError(
                f"prompt missing tag {want!r} (or out of order). "
                f"Found order: {found}"
            )
        indices.append(i)
        cursor = i + 1
    # If we got here, the positions are monotonic by construction.


def get_block(text: str, tag: Tag | str) -> str:
    """Return the body of the FIRST occurrence of ``tag``, or ``""``.

    Convenience for tests that want to scope a substring check to a single
    block (e.g. assert "spine/config.py" appears in the <prior_research>
    body, not anywhere in the prompt).
    """
    want = _tag_value(tag)
    for name, body in parse_tags(text):
        if name == want:
            return body
    return ""


__all__ = [
    "Tag",
    "TagPair",
    "xml_block",
    "xml_blocks",
    "hostage_layout",
    "parse_tags",
    "assert_hostage_layout",
    "assert_has_tags",
    "assert_tag_order",
    "get_block",
]
