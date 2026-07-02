"""Tests for verify-gap feedback injection into the IMPLEMENT editors.

Run 019f20a5: gap_plan diagnosed a verify failure perfectly ("use
config.get('providers', ...) instead of attribute access") but `gap_plan_path`
was never read by anything — the no-tool synthesis editor regenerated
byte-identical failing code three cycles in a row. These tests cover the two
halves of the fix:

* `_gap_fixes_body` — loads gap_plan.json, extracts THIS slice's remediation
  (including for split sub-slices via `_parent_slice_id`), fails open to ''.
* `build_synthesis_prompt` — renders the gaps as a <critic_feedback> block
  with a rework tail; lint-placement `feedback` keeps tail priority.
"""

from __future__ import annotations

import json
from pathlib import Path

from spine.agents.synthesis_implementer import build_synthesis_prompt
from spine.workflow.subgraphs.implement_subgraph import _gap_fixes_body

GAP_PLAN = {
    "summary": "fix dict access",
    "remediation_items": [
        {
            "slice_id": "ui-provider-config",
            "priority": "critical",
            "failures": [
                "api.get_embedding_provider accesses config.embedding_provider directly",
            ],
            "root_cause": "attribute access on a dict",
            "fixes": [
                {
                    "file_path": "spine/ui_api/api.py",
                    "issue_description": "direct attribute access",
                    "suggested_fix": "return config.get('providers', {}).get('embedding', None)",
                    "acceptance_criteria": [
                        "get_embedding_provider returns a string or None",
                    ],
                }
            ],
        },
        {
            "slice_id": "other-slice",
            "priority": "low",
            "failures": ["unrelated"],
            "root_cause": "n/a",
            "fixes": [
                {
                    "file_path": "x.py",
                    "issue_description": "other",
                    "suggested_fix": "other",
                    "acceptance_criteria": ["other"],
                }
            ],
        },
    ],
}


def _state(tmp_path: Path, gap_dir: str | None = "gaps") -> dict:
    if gap_dir:
        d = tmp_path / gap_dir
        d.mkdir(parents=True, exist_ok=True)
        (d / "gap_plan.json").write_text(json.dumps(GAP_PLAN), encoding="utf-8")
    return {
        "workspace_root": str(tmp_path),
        "gap_plan_path": gap_dir,
    }


def test_no_gap_plan_path_returns_empty(tmp_path: Path) -> None:
    state = _state(tmp_path, gap_dir=None)
    state["gap_plan_path"] = None
    assert _gap_fixes_body(state, {"id": "ui-provider-config"}) == ""


def test_missing_file_fails_open(tmp_path: Path) -> None:
    state = {"workspace_root": str(tmp_path), "gap_plan_path": "nope"}
    assert _gap_fixes_body(state, {"id": "ui-provider-config"}) == ""


def test_malformed_json_fails_open(tmp_path: Path) -> None:
    d = tmp_path / "gaps"
    d.mkdir()
    (d / "gap_plan.json").write_text("{not json", encoding="utf-8")
    state = {"workspace_root": str(tmp_path), "gap_plan_path": "gaps"}
    assert _gap_fixes_body(state, {"id": "ui-provider-config"}) == ""


def test_renders_only_this_slices_fixes(tmp_path: Path) -> None:
    body = _gap_fixes_body(_state(tmp_path), {"id": "ui-provider-config"})
    assert "FAILED verification" in body
    assert "config.get('providers', {}).get('embedding', None)" in body
    assert "attribute access on a dict" in body
    assert "Must satisfy: get_embedding_provider returns a string or None" in body
    assert "other-slice" not in body
    assert "unrelated" not in body


def test_sub_slice_matches_parent_remediation(tmp_path: Path) -> None:
    sub = {"id": "ui-provider-config::1-api.py", "_parent_slice_id": "ui-provider-config"}
    body = _gap_fixes_body(_state(tmp_path), sub)
    assert "config.get('providers', {}).get('embedding', None)" in body


def test_unmatched_slice_returns_empty(tmp_path: Path) -> None:
    assert _gap_fixes_body(_state(tmp_path), {"id": "render-config-view"}) == ""


def test_degenerate_gap_plan_is_capped(tmp_path: Path) -> None:
    huge = {
        "remediation_items": [
            {
                "slice_id": "s",
                "failures": [f"failure {i}: " + "x" * 200 for i in range(200)],
                "root_cause": "r",
                "fixes": [],
            }
        ]
    }
    d = tmp_path / "gaps"
    d.mkdir()
    (d / "gap_plan.json").write_text(json.dumps(huge), encoding="utf-8")
    state = {"workspace_root": str(tmp_path), "gap_plan_path": "gaps"}
    body = _gap_fixes_body(state, {"id": "s"})
    assert len(body) < 4200
    assert "truncated" in body


# ── build_synthesis_prompt ────────────────────────────────────────────────────


def test_prompt_without_gaps_has_default_tail() -> None:
    p = build_synthesis_prompt(slice_json="{}", refs_body="", plan_body="")
    assert "critic_feedback" not in p
    assert "VERIFICATION REWORK" not in p
    assert "do not survey" in p


def test_prompt_with_gaps_renders_block_and_rework_tail() -> None:
    p = build_synthesis_prompt(
        slice_json="{}",
        refs_body="",
        plan_body="",
        gaps_body="Required fix in api.py: use dict access",
    )
    assert "<critic_feedback>" in p
    assert "use dict access" in p
    assert "VERIFICATION REWORK" in p
    assert "resolve exactly those failures" in p


def test_lint_feedback_keeps_tail_priority_over_gaps() -> None:
    p = build_synthesis_prompt(
        slice_json="{}",
        refs_body="",
        plan_body="",
        gaps_body="verify gap",
        feedback="E999 SyntaxError",
    )
    # Both blocks render, but the tail addresses the placement failure.
    assert "<critic_feedback>" in p
    assert "E999 SyntaxError" in p
    assert "FAILED to place" in p
    assert "VERIFICATION REWORK" not in p
