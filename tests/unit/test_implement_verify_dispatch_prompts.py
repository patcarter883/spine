"""Implement / Verify slice-dispatch user messages must use the
project's XML-tagged hostage layout.

Both ``_slice_implementer_node`` and ``_run_slice_verifier_node`` build
a one-shot user message that splices the slice JSON + the
``active_slice_directive`` into a prompt sent to the subagent. Prior
to this change, the message was a raw f-string with the directive at
the HEAD and the slice JSON trailing — the inverted layout that
defeats the U-curve attention pattern (see
``spine.agents.prompt_format`` and trace ``019e721d`` audit notes).

This file pins the new structural contract: data blocks first
(``<objective>``, ``<findings>``, optional ``<directive>``); plain-text
instruction last. The same contract applies to both phases.
"""

from __future__ import annotations

from typing import Any

import pytest

from spine.agents.prompt_format import (
    Tag,
    assert_has_tags,
    assert_hostage_layout,
    assert_tag_order,
    get_block,
    parse_tags,
)


def _base_state() -> dict[str, Any]:
    return {
        "work_id": "wid-test",
        "work_type": "task",
        "workspace_root": "/tmp/spine-test",
        "description": "test description",
        "feedback": [],
        "last_critic_review": None,
        "artifacts": {},
        "messages": [],
        "phase_status": "",
        "read_cache": {},
    }


def _capture_prompt(monkeypatch) -> dict[str, Any]:
    """Stub the heavy dependencies of slice-dispatch nodes so we can
    capture the prompt string passed to the agent's ainvoke without
    spinning up real models.
    """
    captured: dict[str, Any] = {}

    class _FakeTool:
        def __init__(self, name: str) -> None:
            self.name = name

    monkeypatch.setattr(
        "spine.agents.subagents.build_subagent_spec",
        lambda **kw: {
            "system_prompt": "subagent-role",
            "tools": [_FakeTool("read_file"), _FakeTool("ls")],
            "model": object(),
            "response_format": None,
        },
    )
    monkeypatch.setattr(
        "spine.agents.factory.build_phase_agent",
        lambda **kw: object(),
    )

    async def _fake_ainvoke(agent, input_dict, *args, **kwargs):
        captured["prompt"] = input_dict["messages"][0]["content"]
        # Return something non-None so the caller can attempt to walk
        # outputs without crashing. Actual finding extraction will fail
        # later, which the test ignores via try/except.
        return {"messages": []}

    # ``ainvoke_with_retry`` is imported as a name at the top of each
    # subgraph module, so we must patch every import site (not just the
    # source). Otherwise the consumer module keeps its bound reference
    # to the real function and the fake never fires.
    for path in (
        "spine.agents.retry.ainvoke_with_retry",
        "spine.workflow.subgraphs.implement_subgraph.ainvoke_with_retry",
        "spine.workflow.subgraphs.verify_subgraph.ainvoke_with_retry",
    ):
        try:
            monkeypatch.setattr(path, _fake_ainvoke)
        except AttributeError:
            # Module may not have the name imported at top level. The
            # source-level patch above will still apply for any local
            # ``from ... import`` that resolves at call time.
            pass
    return captured


# ── IMPLEMENT slice-dispatcher ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_implement_dispatch_prompt_has_hostage_layout(monkeypatch):
    """The implement node's user message must parse with the canonical
    tag set and end with a plain-text instruction tail.
    """
    captured = _capture_prompt(monkeypatch)

    from spine.workflow.subgraphs.implement_subgraph import _slice_implementer_node

    state = {
        **_base_state(),
        "phase": "implement",
        "active_slice": {
            "id": "slice-1",
            "title": "test slice",
            "target_files": ["spine/x.py"],
            "acceptance_criteria": ["X happens"],
        },
        "pending_slices": [],
        "completed_slices": [],
        "failed_slices": [],
    }
    try:
        await _slice_implementer_node(state, config=None)
    except Exception:
        pass  # the fake agent fails on result-extraction; we only need the prompt

    prompt = captured.get("prompt")
    assert prompt, "expected the implementer to build a user-message prompt"

    # Structural invariants from spine.agents.prompt_format.
    assert_hostage_layout(prompt)
    assert_has_tags(prompt, Tag.OBJECTIVE, Tag.FINDINGS)
    assert_tag_order(prompt, Tag.OBJECTIVE, Tag.FINDINGS)

    # Per-block semantic checks: slice JSON lands in <findings>; the
    # objective references the slice id.
    objective = get_block(prompt, Tag.OBJECTIVE)
    assert "slice-1" in objective

    findings = get_block(prompt, Tag.FINDINGS)
    assert '"id": "slice-1"' in findings
    assert "acceptance_criteria" in findings


# ── VERIFY slice-dispatcher ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_verify_dispatch_prompt_has_hostage_layout(monkeypatch):
    """The verify node's user message must parse with the canonical tag
    set and end with a plain-text instruction tail."""
    captured = _capture_prompt(monkeypatch)

    from spine.workflow.subgraphs.verify_subgraph import _run_slice_verifier_node

    state = {
        **_base_state(),
        "phase": "verify",
        "slice": {
            "id": "slice-1",
            "title": "test slice",
            "target_files": ["spine/x.py"],
            "acceptance_criteria": ["X is verified"],
        },
        "verification_results": [],
    }
    try:
        await _run_slice_verifier_node(state, config=None)
    except Exception:
        pass

    prompt = captured.get("prompt")
    assert prompt, "expected the verifier to build a user-message prompt"

    assert_hostage_layout(prompt)
    assert_has_tags(prompt, Tag.OBJECTIVE, Tag.FINDINGS)
    assert_tag_order(prompt, Tag.OBJECTIVE, Tag.FINDINGS)

    objective = get_block(prompt, Tag.OBJECTIVE)
    assert "slice-1" in objective

    findings = get_block(prompt, Tag.FINDINGS)
    assert '"id": "slice-1"' in findings
    assert "acceptance_criteria" in findings


# ── Directive-block splicing (both phases) ──────────────────────────────


@pytest.mark.asyncio
async def test_implement_prompt_includes_directive_when_present(monkeypatch):
    """When ``active_slice_directive`` is on state and non-empty, the
    rendered <directive> block from ``format_directive_for_prompt``
    splices into the prompt between the data blocks and the tail."""
    captured = _capture_prompt(monkeypatch)

    from spine.workflow.subgraphs.implement_subgraph import _slice_implementer_node

    # Seed a directive in the shape SubagentDirective.model_dump() returns.
    state = {
        **_base_state(),
        "phase": "implement",
        "active_slice": {
            "id": "slice-1",
            "title": "test slice",
            "target_files": ["spine/x.py"],
            "acceptance_criteria": ["X happens"],
        },
        "active_slice_directive": {
            "approach": "Edit X.py to add the foo() helper",
            "target_files": ["spine/x.py"],
            "tool_calls_to_make": ["read_edit_lint spine/x.py"],
            "acceptance": ["X happens"],
            "notes": "",
        },
        "pending_slices": [],
        "completed_slices": [],
        "failed_slices": [],
    }
    try:
        await _slice_implementer_node(state, config=None)
    except Exception:
        pass

    prompt = captured.get("prompt")
    tag_names = [name for name, _ in parse_tags(prompt or "")]
    assert Tag.DIRECTIVE.value in tag_names, (
        f"directive block missing — got tags {tag_names}"
    )
    directive_body = get_block(prompt, Tag.DIRECTIVE)
    assert "foo() helper" in directive_body
    # Hostage layout still holds even with the directive spliced in.
    assert_hostage_layout(prompt)
