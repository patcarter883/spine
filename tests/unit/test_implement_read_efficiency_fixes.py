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


# ── Anti-clobber: inline the live target file so serialized same-file slices
#    build on the prior slice's edits instead of regenerating (trace 019f2005) ──


def test_target_files_body_inlines_current_content(tmp_path: Path) -> None:
    from spine.workflow.subgraphs import implement_subgraph as impl

    # config_view.py AFTER an earlier slice already added an embedding section.
    (tmp_path / "ui").mkdir()
    (tmp_path / "ui" / "config_view.py").write_text(
        "def render_llm_providers():\n    ...\n\n"
        "def render_embedding_providers():\n    ...\n",
        encoding="utf-8",
    )
    body = impl._target_files_body(["ui/config_view.py"], str(tmp_path))
    # The next same-file slice sees the prior slice's work and is told to keep it.
    assert "render_embedding_providers" in body
    assert "PRESERVE it" in body
    assert "```python" in body


def test_target_files_body_flags_missing_file_as_create(tmp_path: Path) -> None:
    from spine.workflow.subgraphs import implement_subgraph as impl

    body = impl._target_files_body(["ui/new_page.py"], str(tmp_path))
    assert "does not exist yet" in body
    assert "```python" not in body  # nothing to inline for a to-be-created file


def test_target_files_body_outlines_oversized_file(tmp_path: Path) -> None:
    from spine.workflow.subgraphs import implement_subgraph as impl

    # Larger than the inline cap → outline of existing anchors, not the body.
    big = "\n\n".join(f"def f{i}():\n    return {i}" for i in range(600))
    (tmp_path / "big.py").write_text(big, encoding="utf-8")
    body = impl._target_files_body(["big.py"], str(tmp_path))
    assert "too large to inline" in body
    assert "do NOT recreate" in body
    assert "def f0" in body and "def f599" in body  # anchors listed
    assert "return 0" not in body  # bodies NOT dumped (token guard)


def test_target_files_body_empty_without_targets() -> None:
    from spine.workflow.subgraphs import implement_subgraph as impl

    assert impl._target_files_body([], "/tmp") == ""


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


# ── plan_slice_implementer: deterministic directive for edit_plan slices ──


@pytest.mark.asyncio
async def test_edit_plan_slice_skips_llm_planner_and_forbids_exploration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langgraph.types import Command, Send

    from spine.workflow.subgraphs import implement_subgraph as impl

    async def _boom(**kwargs):
        raise AssertionError("run_plan_node must NOT run for an edit_plan slice")

    monkeypatch.setattr(impl, "run_plan_node", _boom)

    state = {
        "phase": "implement",
        "work_id": "test-work",
        "work_type": "feature",
        "workspace_root": "/tmp/test",
        "plan_path": ".spine/artifacts/test-work/plan",
        "active_slice": {
            "id": "backend",
            "title": "backend",
            "target_files": ["spine/ui_api/api.py"],
            "acceptance_criteria": ["it persists"],
            "edit_plan": [
                {"file": "spine/ui_api/api.py", "symbol": "UIApi.x", "action": "replace", "intent": "y"},
            ],
        },
    }
    out = await impl._plan_slice_implementer_node(state, None)
    assert isinstance(out, Command) and isinstance(out.goto, Send)
    approach = out.goto.arg["active_slice_directive"]["approach"].lower()
    # Deterministic edit-first directive that explicitly FORBIDS exploration
    # (the negated "do NOT explore" mention is the fix, not a foot-gun).
    assert approach.startswith("apply the edits in <edit_plan>")
    assert "do not explore" in approach


# ── Decomposer length-escalation (reasoning-heavy local models) ──────────


@pytest.mark.asyncio
async def test_structured_escalation_doubles_cap_on_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spine.agents import decomposer as dc

    class _StubLength(Exception):
        pass

    monkeypatch.setattr(dc, "_LengthFinishReasonError", _StubLength)
    calls: list[int] = []
    # _bind_capped returns the cap itself so the fake invoke can branch on it.
    monkeypatch.setattr(dc, "_bind_capped", lambda base, schema, cap: cap)

    async def fake_invoke(model, messages, *, label):
        calls.append(model)
        if model < 30000:
            raise _StubLength()
        return "ok"

    monkeypatch.setattr(dc, "ainvoke_structured_with_retry", fake_invoke)
    out = await dc._ainvoke_structured_escalating(
        object(), object(), [], label="t", base_cap=16384, window=65536, max_escalations=1
    )
    assert out == "ok"
    assert calls == [16384, 32768]  # doubled once, then succeeded


@pytest.mark.asyncio
async def test_structured_escalation_respects_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spine.agents import decomposer as dc

    class _StubLength(Exception):
        pass

    monkeypatch.setattr(dc, "_LengthFinishReasonError", _StubLength)
    calls: list[int] = []
    monkeypatch.setattr(dc, "_bind_capped", lambda base, schema, cap: cap)

    async def always_length(model, messages, *, label):
        calls.append(model)
        raise _StubLength()

    monkeypatch.setattr(dc, "ainvoke_structured_with_retry", always_length)
    # window leaves no room to double (32768 + 2048 >= 33000) → no escalation.
    with pytest.raises(_StubLength):
        await dc._ainvoke_structured_escalating(
            object(), object(), [], label="t", base_cap=16384, window=33000, max_escalations=3
        )
    assert calls == [16384]


# ── enrich gets reference-symbol signatures + module paths ───────────────


def test_reference_signatures_block_groups_by_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spine.agents import decomposer as dc

    sigs = {
        "UIApi.set_phase_provider": ("spine/ui_api/api.py", "def set_phase_provider(self, phase, config): ..."),
        "UIApi.get_providers": ("spine/ui_api/api.py", "def get_providers(self): ..."),
        "SpineConfig": ("spine/config.py", "class SpineConfig: ..."),
    }
    monkeypatch.setattr(
        "spine.agents.tools.codebase_query.get_symbol_signature",
        lambda db, root, name: sigs.get(name),
    )
    block = dc._build_reference_signatures_block(
        "db", ".", ["UIApi.set_phase_provider", "UIApi.get_providers", "SpineConfig"]
    )
    # correct module path is surfaced, grouped by file, with the real signatures
    assert "# spine/ui_api/api.py" in block
    assert "# spine/config.py" in block
    assert "set_phase_provider" in block and "get_providers" in block
    assert "do NOT invent a module path" in block


def test_reference_signatures_block_empty_when_no_refs() -> None:
    from spine.agents import decomposer as dc

    assert dc._build_reference_signatures_block("db", ".", []) == ""


def test_reference_symbols_body_inlines_constructor_of_referenced_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slice referencing Class.method must also see Class.__init__.

    Regression (run 86a3ab17): the test slice referenced
    ArtifactStore.save_artifact / artifact_exists, so only those methods'
    source was inlined — the editor then invented constructor kwargs
    (root_dir / base_dir / root; the real parameter is base_path) across
    three verify cycles.
    """
    from spine.workflow.subgraphs import implement_subgraph as impl

    sources = {
        "Store.save": "def save(self, name):\n    ...\n",
        "Store.__init__": "def __init__(self, base_path: str = '.x'):\n    ...\n",
    }
    monkeypatch.setattr(
        "spine.agents.tools.codebase_query.get_symbol_source",
        lambda db, root, name: sources.get(name),
    )
    body = impl._reference_symbols_body(
        {"reference_symbols": ["Store.save"]}, db_path="db", workspace_root="."
    )
    assert "Store.__init__" in body
    assert "base_path" in body
    assert "EXACTLY these" in body


def test_constructor_not_duplicated_when_class_itself_referenced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from spine.workflow.subgraphs import implement_subgraph as impl

    sources = {
        "Store": "class Store:\n    def __init__(self, base_path):\n        ...\n",
        "Store.__init__": "def __init__(self, base_path):\n    ...\n",
    }
    monkeypatch.setattr(
        "spine.agents.tools.codebase_query.get_symbol_source",
        lambda db, root, name: sources.get(name),
    )
    body = impl._reference_symbols_body(
        {"reference_symbols": ["Store"]}, db_path="db", workspace_root="."
    )
    # The whole class (constructor included) is already inlined — no extra block.
    assert body.count("__init__") == 1
