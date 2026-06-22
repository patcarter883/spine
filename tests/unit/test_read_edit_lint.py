"""Unit tests for :mod:`spine.agents.tools.read_edit_lint`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spine.agents.tools.read_edit_lint import ReadEditLintTool


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    return tmp_path


def _tool(workspace: Path) -> ReadEditLintTool:
    return ReadEditLintTool(workspace_root=str(workspace))


def _decode(out: str) -> dict:
    return json.loads(out)


def test_full_replace_creates_file_with_valid_python(workspace: Path) -> None:
    out = _decode(
        _tool(workspace)._run(
            file_path="src/hello.py",
            full_replace="def hello():\n    return 'hi'\n",
        )
    )
    assert out["status"] == "ok"
    assert out["created"] is True
    assert (workspace / "src" / "hello.py").read_text() == "def hello():\n    return 'hi'\n"


def test_full_replace_rejects_invalid_python_without_writing(workspace: Path) -> None:
    target = workspace / "src" / "broken.py"
    out = _decode(
        _tool(workspace)._run(
            file_path="src/broken.py",
            full_replace="def hello(:\n    return 'hi'\n",  # syntax error
        )
    )
    assert out["status"] == "syntax_error"
    assert "SyntaxError" in out["detail"]
    assert out["wrote"] is False
    assert not target.exists()


def test_edit_with_indent_break_does_not_change_mtime(workspace: Path) -> None:
    target = workspace / "src" / "indent.py"
    target.write_text("def hello():\n    return 'hi'\n")
    original_mtime = target.stat().st_mtime

    # Make a syntactically-broken edit (un-indented body).
    out = _decode(
        _tool(workspace)._run(
            file_path="src/indent.py",
            old_str="    return 'hi'",
            new_str="return 'hi'",  # missing indent
        )
    )
    assert out["status"] == "syntax_error"
    assert target.read_text() == "def hello():\n    return 'hi'\n"
    # mtime preserved (atomic write would have changed it)
    assert target.stat().st_mtime == original_mtime


def test_no_match_returns_clear_error(workspace: Path) -> None:
    target = workspace / "src" / "miss.py"
    target.write_text("x = 1\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/miss.py",
            old_str="not in file",
            new_str="replacement",
        )
    )
    assert out["status"] == "no_match"
    assert "not found" in out["detail"]


def test_ambiguous_match_returns_clear_error(workspace: Path) -> None:
    target = workspace / "src" / "dupe.py"
    target.write_text("x = 1\nx = 1\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/dupe.py",
            old_str="x = 1\n",
            new_str="x = 2\n",
        )
    )
    assert out["status"] == "ambiguous_match"
    assert "2 locations" in out["detail"]


def test_exclusive_args_validated(workspace: Path) -> None:
    out = _decode(
        _tool(workspace)._run(
            file_path="src/x.py",
            old_str="a",
            new_str="b",
            full_replace="full",
        )
    )
    assert out["status"] == "input_error"


def test_unknown_extension_skips_lint(workspace: Path) -> None:
    out = _decode(
        _tool(workspace)._run(
            file_path="src/notes.md",
            full_replace="this is `not python` and that is fine",
        )
    )
    assert out["status"] == "ok"


def test_typescript_syntax_error_is_caught(workspace: Path) -> None:
    target = workspace / "src" / "broken.ts"
    out = _decode(
        _tool(workspace)._run(
            file_path="src/broken.ts",
            full_replace="function broken(: string) { return ; }\n",
        )
    )
    assert out["status"] == "syntax_error"
    assert not target.exists()


def test_leading_slash_resolves_workspace_relative(workspace: Path) -> None:
    out = _decode(
        _tool(workspace)._run(
            file_path="/src/leading_slash.py",
            full_replace="def f(): pass\n",
        )
    )
    assert out["status"] == "ok"
    assert (workspace / "src" / "leading_slash.py").exists()


# ── Batch edits (all-or-nothing) ────────────────────────────────────


def test_batch_edits_apply_in_order(workspace: Path) -> None:
    target = workspace / "src" / "batch.py"
    target.write_text("a = 1\nb = 2\nc = 3\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/batch.py",
            edits=[
                {"old_str": "a = 1", "new_str": "a = 10"},
                {"old_str": "c = 3", "new_str": "c = 30"},
            ],
        )
    )
    assert out["status"] == "ok"
    assert target.read_text() == "a = 10\nb = 2\nc = 30\n"


def test_batch_edits_are_all_or_nothing(workspace: Path) -> None:
    target = workspace / "src" / "atomic.py"
    target.write_text("a = 1\nb = 2\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/atomic.py",
            edits=[
                {"old_str": "a = 1", "new_str": "a = 10"},
                {"old_str": "not present", "new_str": "x"},
            ],
        )
    )
    assert out["status"] == "no_match"
    assert out["edit_index"] == 1
    # First edit must NOT have landed — the whole batch is rolled back.
    assert target.read_text() == "a = 1\nb = 2\n"


def test_batch_edit_sees_earlier_results(workspace: Path) -> None:
    # The second edit targets text produced by the first.
    target = workspace / "src" / "chain.py"
    target.write_text("x = 1\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/chain.py",
            edits=[
                {"old_str": "x = 1", "new_str": "x = 2"},
                {"old_str": "x = 2", "new_str": "x = 3"},
            ],
        )
    )
    assert out["status"] == "ok"
    assert target.read_text() == "x = 3\n"


def test_batch_rejected_when_result_has_syntax_error(workspace: Path) -> None:
    target = workspace / "src" / "syn.py"
    target.write_text("def f():\n    return 1\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/syn.py",
            edits=[{"old_str": "    return 1", "new_str": "return 1"}],  # bad indent
        )
    )
    assert out["status"] == "syntax_error"
    assert target.read_text() == "def f():\n    return 1\n"


# ── Line-range mode ─────────────────────────────────────────────────


def test_line_range_replaces_lines(workspace: Path) -> None:
    target = workspace / "src" / "range.py"
    target.write_text("a = 1\nb = 2\nc = 3\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/range.py",
            start_line=2,
            end_line=2,
            replacement="b = 20",
        )
    )
    assert out["status"] == "ok"
    assert target.read_text() == "a = 1\nb = 20\nc = 3\n"


def test_line_range_deletes_when_replacement_empty(workspace: Path) -> None:
    target = workspace / "src" / "del.py"
    target.write_text("a = 1\nb = 2\nc = 3\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/del.py",
            start_line=2,
            end_line=2,
            replacement="",
        )
    )
    assert out["status"] == "ok"
    assert target.read_text() == "a = 1\nc = 3\n"


def test_line_range_out_of_bounds(workspace: Path) -> None:
    target = workspace / "src" / "oob.py"
    target.write_text("a = 1\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/oob.py",
            start_line=1,
            end_line=5,
            replacement="x = 1",
        )
    )
    assert out["status"] == "range_error"
    assert target.read_text() == "a = 1\n"


def test_line_range_expected_guard_rejects_stale(workspace: Path) -> None:
    target = workspace / "src" / "stale.py"
    target.write_text("a = 1\nb = 2\nc = 3\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/stale.py",
            start_line=2,
            end_line=2,
            replacement="b = 20",
            expected="something else",  # does not match current line 2
        )
    )
    assert out["status"] == "stale"
    assert target.read_text() == "a = 1\nb = 2\nc = 3\n"


def test_line_range_expected_guard_allows_match(workspace: Path) -> None:
    target = workspace / "src" / "fresh.py"
    target.write_text("a = 1\nb = 2\nc = 3\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/fresh.py",
            start_line=2,
            end_line=2,
            replacement="b = 20",
            expected="b = 2",
        )
    )
    assert out["status"] == "ok"
    assert target.read_text() == "a = 1\nb = 20\nc = 3\n"


def test_multiple_modes_rejected(workspace: Path) -> None:
    target = workspace / "src" / "multi.py"
    target.write_text("a = 1\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/multi.py",
            old_str="a = 1",
            new_str="a = 2",
            start_line=1,
            end_line=1,
            replacement="a = 3",
        )
    )
    assert out["status"] == "input_error"


# ── Read mode (trace 019eb502: read_file removed from the implementer) ──


def test_bare_whole_file_read_is_disabled(workspace: Path) -> None:
    (workspace / "src" / "r.py").write_text("a = 1\nb = 2\nc = 3\n")
    out = _decode(_tool(workspace)._run(file_path="src/r.py"))
    assert out["status"] == "read_disabled"
    assert "read_symbol" in out["detail"]


def test_arbitrary_line_range_read_is_disabled(workspace: Path) -> None:
    (workspace / "src" / "r2.py").write_text("".join(f"x{i} = {i}\n" for i in range(1, 11)))
    out = _decode(_tool(workspace)._run(file_path="src/r2.py", start_line=4, end_line=6))
    assert out["status"] == "read_disabled"


def test_read_symbol_returns_definition_source(workspace: Path) -> None:
    (workspace / "src" / "rs.py").write_text(
        "import os\n\n\ndef alpha():\n    return 1\n\n\ndef beta():\n    return 2\n"
    )
    out = _tool(workspace)._run(file_path="src/rs.py", read_symbol="beta")
    assert "def beta():" in out and "return 2" in out
    assert "def alpha" not in out  # only the requested symbol


def test_read_symbol_unknown_lists_available(workspace: Path) -> None:
    (workspace / "src" / "rs2.py").write_text("def alpha():\n    return 1\n")
    out = _decode(_tool(workspace)._run(file_path="src/rs2.py", read_symbol="zeta"))
    assert out["status"] == "no_match"
    assert "alpha" in out.get("available_symbols", [])


def test_read_symbol_tolerates_module_qualified_name(workspace: Path) -> None:
    (workspace / "src" / "rs3.py").write_text("class UIApi:\n    def m(self):\n        return 1\n")
    out = _tool(workspace)._run(file_path="src/rs3.py", read_symbol="pkg.mod.rs3.UIApi")
    assert "class UIApi:" in out


def test_read_around_returns_region_with_context(workspace: Path) -> None:
    (workspace / "src" / "ra.py").write_text(
        "".join(f"line{i} = {i}\n" for i in range(1, 21))
    )
    out = _tool(workspace)._run(file_path="src/ra.py", read_around="line10 = 10")
    assert "10| line10 = 10" in out
    assert "line1 = 1" not in out  # not the whole file
    assert "line20" not in out


def test_read_missing_file_reports_not_found(workspace: Path) -> None:
    out = _decode(_tool(workspace)._run(file_path="src/nope.py", read_symbol="x"))
    assert out["status"] == "not_found"


# ── already_applied (trace 019eb502: re-sent edits read as failures) ──


def test_already_applied_when_new_str_present(workspace: Path) -> None:
    target = workspace / "src" / "aa.py"
    target.write_text("value = 2\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/aa.py", old_str="value = 1", new_str="value = 2"
        )
    )
    assert out["status"] == "already_applied"
    assert "do NOT retry" in out["detail"]
    assert target.read_text() == "value = 2\n"


def test_plain_no_match_still_reported_when_new_str_absent(workspace: Path) -> None:
    target = workspace / "src" / "nm.py"
    target.write_text("value = 3\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/nm.py", old_str="value = 1", new_str="value = 2"
        )
    )
    assert out["status"] == "no_match"


def test_batch_already_applied_names_the_edit(workspace: Path) -> None:
    target = workspace / "src" / "ba.py"
    target.write_text("a = 10\nb = 2\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/ba.py",
            edits=[
                {"old_str": "a = 1\n", "new_str": "a = 10\n"},  # already applied
                {"old_str": "b = 2", "new_str": "b = 20"},
            ],
        )
    )
    assert out["status"] == "already_applied"
    assert out["edit_index"] == 0
    assert "WITHOUT edits[0]" in out["detail"]
    # All-or-nothing: the second edit must NOT have been applied.
    assert target.read_text() == "a = 10\nb = 2\n"


# ── ruff report on successful Python writes ──


def test_ok_result_carries_ruff_field(workspace: Path) -> None:
    import shutil

    if shutil.which("ruff") is None:
        pytest.skip("ruff not on PATH")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/lint_me.py",
            full_replace="import os\n\n\nvalue = 1\n",  # F401: unused import
        )
    )
    assert out["status"] == "ok"
    assert "ruff" in out
    assert "F401" in out["ruff"]


def test_clean_write_reports_ruff_clean(workspace: Path) -> None:
    import shutil

    if shutil.which("ruff") is None:
        pytest.skip("ruff not on PATH")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/clean.py",
            full_replace="value = 1\n",
        )
    )
    assert out["status"] == "ok"
    assert out.get("ruff") == "clean"


def test_ruff_unavailable_fails_open(workspace: Path, monkeypatch) -> None:
    import spine.agents.tools.read_edit_lint as rel

    def _raise(*a, **kw):
        raise FileNotFoundError("no ruff")

    monkeypatch.setattr(rel.subprocess, "run", _raise)
    out = _decode(
        _tool(workspace)._run(file_path="src/no_ruff.py", full_replace="v = 1\n")
    )
    assert out["status"] == "ok"
    assert "ruff" not in out


# ── patch mode (whitespace-tolerant) ────────────────────────────────


def test_patch_exact_match_replaces(workspace: Path) -> None:
    (workspace / "src" / "p.py").write_text("def f():\n    return 1\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/p.py",
            patch=[{"search": "    return 1", "replace": "    return 2"}],
        )
    )
    assert out["status"] == "ok"
    assert (workspace / "src" / "p.py").read_text() == "def f():\n    return 2\n"


def test_patch_tolerates_indentation_drift(workspace: Path) -> None:
    # File body is indented 4 spaces; model supplies the snippet at 2 spaces.
    (workspace / "src" / "q.py").write_text("def f():\n    x = 1\n    return x\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/q.py",
            patch=[{"search": "  x = 1", "replace": "  x = 42"}],  # wrong indent
        )
    )
    assert out["status"] == "ok"
    # Replacement re-indented to the matched block (4 spaces), stays valid.
    assert (workspace / "src" / "q.py").read_text() == "def f():\n    x = 42\n    return x\n"


def test_patch_no_match_reports_index(workspace: Path) -> None:
    (workspace / "src" / "r.py").write_text("a = 1\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/r.py",
            patch=[{"search": "nonexistent line", "replace": "z = 9"}],
        )
    )
    assert out["status"] == "no_match"
    assert out["edit_index"] == 0


def test_patch_rejects_result_failing_syntax_check(workspace: Path) -> None:
    target = workspace / "src" / "s.py"
    target.write_text("def f():\n    return 1\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/s.py",
            patch=[{"search": "    return 1", "replace": "    return ("}],
        )
    )
    assert out["status"] == "syntax_error"
    assert target.read_text() == "def f():\n    return 1\n"  # untouched


# ── ast_edit mode (symbol-anchored) ─────────────────────────────────


def test_ast_edit_replaces_function_by_name(workspace: Path) -> None:
    (workspace / "src" / "m.py").write_text(
        "import os\n\n\ndef greet(name):\n    return 'hi ' + name\n\n\nX = 1\n"
    )
    out = _decode(
        _tool(workspace)._run(
            file_path="src/m.py",
            ast_edit={
                "symbol": "greet",
                "action": "replace",
                "code": "def greet(name):\n    return f'hello {name}'",
            },
        )
    )
    assert out["status"] == "ok"
    text = (workspace / "src" / "m.py").read_text()
    assert "return f'hello {name}'" in text
    assert "X = 1" in text  # surrounding code preserved


def test_ast_edit_insert_after_symbol(workspace: Path) -> None:
    (workspace / "src" / "n.py").write_text("def a():\n    return 1\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/n.py",
            ast_edit={
                "symbol": "a",
                "action": "insert_after",
                "code": "def b():\n    return 2",
            },
        )
    )
    assert out["status"] == "ok"
    text = (workspace / "src" / "n.py").read_text()
    assert "def a():" in text and "def b():" in text
    assert text.index("def a()") < text.index("def b()")


def test_ast_edit_unknown_symbol_lists_available(workspace: Path) -> None:
    (workspace / "src" / "o.py").write_text("def alpha():\n    return 1\n")
    out = _decode(
        _tool(workspace)._run(
            file_path="src/o.py",
            ast_edit={"symbol": "beta", "action": "replace", "code": "def beta():\n    pass"},
        )
    )
    assert out["status"] == "no_match"
    assert "alpha" in out.get("available_symbols", [])


def test_ast_edit_replace_method_by_qualified_name(workspace: Path) -> None:
    (workspace / "src" / "c.py").write_text(
        "class C:\n    def m(self):\n        return 1\n\n    def k(self):\n        return 2\n"
    )
    out = _decode(
        _tool(workspace)._run(
            file_path="src/c.py",
            ast_edit={
                "symbol": "C.m",
                "action": "replace",
                "code": "    def m(self):\n        return 99",
            },
        )
    )
    assert out["status"] == "ok"
    text = (workspace / "src" / "c.py").read_text()
    assert "return 99" in text and "return 2" in text


def test_ast_edit_creation_anchor_guard_suggests_last_sibling(workspace: Path) -> None:
    # A weak model trying to CREATE a method via action='replace' anchored to the
    # not-yet-existing symbol gets a grounded recovery anchor (the last existing
    # method of the same class), not a bare symbol list — the misstep that began
    # the GLM destructive-recovery spiral (GLM_QWEN_BENCH_ANALYSIS.md, PR-A).
    (workspace / "src" / "svc.py").write_text(
        "class Svc:\n"
        "    def alpha(self):\n"
        "        return 1\n"
        "\n"
        "    def beta(self):\n"
        "        return 2\n"
    )
    out = _decode(
        _tool(workspace)._run(
            file_path="src/svc.py",
            ast_edit={
                "symbol": "Svc.gamma",
                "action": "replace",
                "code": "    def gamma(self):\n        return 3\n",
            },
        )
    )
    assert out["status"] == "no_match"
    # PR-B reference contract: defect + resolvable target + concrete next action.
    assert out["target"] == "Svc.beta"
    assert "insert_after" in out["next_action"]
    assert "creating" in out["detail"].lower()


def test_ast_edit_no_match_without_creation_intent_offers_no_target(workspace: Path) -> None:
    # A genuine wrong/typo anchor (code does NOT define the requested symbol)
    # must NOT trigger the creation guard — no target, just the symbol menu and
    # a generic next_action.
    (workspace / "src" / "svc2.py").write_text(
        "class Svc:\n    def alpha(self):\n        return 1\n"
    )
    out = _decode(
        _tool(workspace)._run(
            file_path="src/svc2.py",
            ast_edit={
                "symbol": "Svc.typo",
                "action": "replace",
                "code": "    def something_else(self):\n        return 9\n",
            },
        )
    )
    assert out["status"] == "no_match"
    assert "target" not in out
    assert out["next_action"]
    assert "Svc.alpha" in out.get("available_symbols", [])


def test_ast_edit_ambiguous_match_carries_target_and_next_action(workspace: Path) -> None:
    (workspace / "src" / "amb.py").write_text(
        "def dup():\n    return 1\n\n\ndef dup():\n    return 2\n"
    )
    out = _decode(
        _tool(workspace)._run(
            file_path="src/amb.py",
            ast_edit={
                "symbol": "dup",
                "action": "replace",
                "code": "def dup():\n    return 3\n",
            },
        )
    )
    assert out["status"] == "ambiguous_match"
    assert out["target"].startswith("src/amb.py:")
    assert "retry ast_edit" in out["next_action"]


def test_ast_edit_conflict_error_carries_next_action(workspace: Path) -> None:
    (workspace / "src" / "conf.py").write_text(
        "class C:\n    def a(self):\n        return 1\n"
    )
    out = _decode(
        _tool(workspace)._run(
            file_path="src/conf.py",
            ast_edit={
                "symbol": "C.a",
                "action": "insert_after",
                "code": "    def a(self):\n        return 2\n",
            },
        )
    )
    assert out["status"] == "conflict_error"
    assert "a" in out["target"]
    assert "replace" in out["next_action"]
