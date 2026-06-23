"""Tests for the survey-trap fixes: path auto-resolve, ambiguity-tolerant
reads, one-shot target reads, source-inlined prompt blocks, compact directive,
and the decomposer's one-entry-per-method constraint.

Covers the model-independent IMPLEMENT read-spiral diagnosed in trace 019ef2ae
(api.py read 89×): the editor surveyed because the survey was mandatory but the
survey tools were crippled, and the prompt tripled every datum.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spine.agents.tools.read_edit_lint import ReadEditLintTool


def _decode(out: str) -> dict | None:
    try:
        return json.loads(out)
    except ValueError:
        return None  # a rendered "[read: ...]" body, not a JSON status


# ── Fix B: path auto-resolution to the slice's target_files ──────────────


def test_wrong_path_autoresolves_to_unique_target(tmp_path: Path) -> None:
    (tmp_path / "spine" / "ui_api").mkdir(parents=True)
    real = tmp_path / "spine" / "ui_api" / "api.py"
    real.write_text("class UIApi:\n    def existing(self):\n        return 1\n")
    tool = ReadEditLintTool(
        workspace_root=str(tmp_path),
        target_files=["spine/ui_api/api.py"],
    )
    # The editor guesses the wrong sibling path (spine/ui/api.py).
    out = tool._run(file_path="spine/ui/api.py")  # bare = whole-file read
    assert "[read:" in out
    assert "auto-corrected to the slice target spine/ui_api/api.py" in out
    assert "class UIApi" in out


def test_wrong_path_not_resolved_when_target_missing(tmp_path: Path) -> None:
    tool = ReadEditLintTool(
        workspace_root=str(tmp_path),
        target_files=["spine/ui_api/api.py"],  # does not exist on disk
    )
    out = _decode(tool._run(file_path="spine/ui/api.py", read_symbol="UIApi"))
    assert out is not None and out["status"] == "not_found"


# ── Fix C: one orienting whole-file read of a target, then refused ───────


def test_target_file_allows_one_whole_read_then_refuses(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    f = tmp_path / "pkg" / "mod.py"
    f.write_text("def a():\n    return 1\n\n\ndef b():\n    return 2\n")
    tool = ReadEditLintTool(
        workspace_root=str(tmp_path),
        target_files=["pkg/mod.py"],
    )
    first = tool._run(file_path="pkg/mod.py")
    assert "[read:" in first and "def a" in first
    second = _decode(tool._run(file_path="pkg/mod.py"))
    assert second is not None and second["status"] == "read_disabled"


def test_non_target_whole_read_is_disabled(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "other.py").write_text("def a():\n    return 1\n")
    tool = ReadEditLintTool(
        workspace_root=str(tmp_path),
        target_files=["pkg/mod.py"],
    )
    out = _decode(tool._run(file_path="pkg/other.py"))
    assert out is not None and out["status"] == "read_disabled"


# ── Fix D: ambiguous anchored reads return the first region, not a refusal


def test_ambiguous_read_around_returns_first_region(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    f = tmp_path / "pkg" / "mod.py"
    f.write_text(
        "x = 1  # marker\n"
        "y = 2\n"
        "z = 3  # marker\n"
    )
    tool = ReadEditLintTool(workspace_root=str(tmp_path), target_files=["pkg/mod.py"])
    out = tool._run(file_path="pkg/mod.py", read_around="# marker")
    assert "[read:" in out
    assert "Showing the FIRST" in out


# ── Source-inlined prompt blocks ─────────────────────────────────────────


def test_edit_plan_body_inlines_source(monkeypatch: pytest.MonkeyPatch) -> None:
    from spine.workflow.subgraphs import implement_subgraph as impl

    monkeypatch.setattr(
        "spine.agents.tools.codebase_query.get_symbol_source",
        lambda db, root, name: f"def {name.split('.')[-1]}(self):\n    return 1",
    )
    slice_ = {
        "edit_plan": [
            {"file": "a.py", "symbol": "C.m1", "action": "replace", "intent": "do x"},
            {"file": "a.py", "symbol": "C.m2", "action": "insert_after", "intent": "add y"},
        ]
    }
    body = impl._edit_plan_body(slice_, db_path="db", workspace_root=".")
    assert "C.m1" in body and "do x" in body
    assert "C.m2" in body and "add y" in body
    assert body.count("```python") == 2  # source inlined per entry
    assert "edit this with ast_edit replace" in body  # replace gets the strong label


def test_reference_symbols_body_degrades_without_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spine.workflow.subgraphs import implement_subgraph as impl

    monkeypatch.setattr(
        "spine.agents.tools.codebase_query.get_symbol_source",
        lambda db, root, name: None,
    )
    body = impl._reference_symbols_body(
        {"reference_symbols": ["X.y"]}, db_path="db", workspace_root="."
    )
    assert "X.y" in body and "read_symbol it" in body


def test_large_symbol_source_is_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    from spine.workflow.subgraphs import implement_subgraph as impl

    big = "\n".join(f"    line_{i} = {i}" for i in range(500))
    monkeypatch.setattr(
        "spine.agents.tools.codebase_query.get_symbol_source",
        lambda db, root, name: f"class Big:\n{big}",
    )
    out = impl._inline_symbol_source("db", ".", "Big")
    assert "truncated" in out
    assert out.count("\n") < 500


# ── Compact directive drops the redundant fields ─────────────────────────


def test_compact_directive_drops_targets_and_acceptance() -> None:
    from spine.agents.plan_do import format_directive_for_prompt

    directive = {
        "approach": "do the thing",
        "target_files": ["a.py"],
        "tool_calls_to_make": ["read_symbol: X"],
        "acceptance": ["it works"],
        "notes": "careful",
    }
    full = format_directive_for_prompt(directive)
    compact = format_directive_for_prompt(directive, compact=True)
    assert "Target files" in full and "Acceptance" in full
    assert "Target files" not in compact and "Acceptance" not in compact
    assert "do the thing" in compact and "careful" in compact


# ── Decomposer no longer instructs bundling new methods ──────────────────


def test_enrich_prompt_requires_one_entry_per_method() -> None:
    from spine.agents.decomposer import _ENRICH_PROMPT

    assert "ONE entry per change site" in _ENRICH_PROMPT
    assert "umbrella" in _ENRICH_PROMPT
    # the old foot-gun instruction is gone
    assert "use the last existing\nmethod of that class as symbol, set action" not in _ENRICH_PROMPT
