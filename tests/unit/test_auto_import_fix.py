"""Deterministic F821 auto-import repair on write (run 019f253c).

The synthesis editor's schema only speaks whole def/class constructs, so it
cannot express "add import yaml" — and the advisory ruff report never blocks a
write. A missing module import therefore plateaued every gap cycle (totals
12/12/12). The write path now auto-inserts ``import <name>`` for F821 names
that resolve to importable modules; typo'd variables stay for the model.
"""

from __future__ import annotations

from spine.agents.tools.read_edit_lint import ReadEditLintTool


def _tool(tmp_path) -> ReadEditLintTool:
    return ReadEditLintTool(workspace_root=str(tmp_path), target_files=["mod.py"])


def test_missing_module_import_is_auto_added(tmp_path):
    code = (
        '"""Module."""\n'
        "\n"
        "import os\n"
        "\n"
        "\n"
        "def load(path):\n"
        "    if os.path.exists(path):\n"
        "        with open(path) as fh:\n"
        "            return yaml.safe_load(fh)\n"
        "    return None\n"
    )
    out = _tool(tmp_path)._run("mod.py", full_replace=code)
    text = (tmp_path / "mod.py").read_text()
    assert "import yaml" in text
    # Inserted after the existing import block, not mid-function.
    assert text.index("import os") < text.index("import yaml") < text.index("def load")
    assert "auto_imports_added" in out
    assert "F821" not in text
    # The file must still be valid Python.
    compile(text, "mod.py", "exec")


def test_typoed_name_is_not_auto_imported(tmp_path):
    code = "def f():\n    return not_a_real_module_xyz.value\n"
    out = _tool(tmp_path)._run("mod.py", full_replace=code)
    text = (tmp_path / "mod.py").read_text()
    assert "import not_a_real_module_xyz" not in text
    assert "F821" in out  # advisory report still surfaces the real problem


def test_docstring_only_file_inserts_after_docstring(tmp_path):
    code = '"""Doc."""\n\n\ndef f():\n    return json.dumps({})\n'
    _tool(tmp_path)._run("mod.py", full_replace=code)
    text = (tmp_path / "mod.py").read_text()
    lines = text.splitlines()
    assert lines[0] == '"""Doc."""'
    assert "import json" in text
    assert text.index('"""Doc."""') < text.index("import json") < text.index("def f")
    compile(text, "mod.py", "exec")


def test_clean_write_untouched(tmp_path):
    code = "import json\n\n\ndef f():\n    return json.dumps({})\n"
    out = _tool(tmp_path)._run("mod.py", full_replace=code)
    assert "auto_imports_added" not in out
    assert (tmp_path / "mod.py").read_text() == code


def test_syntax_error_feedback_includes_offending_region(tmp_path):
    """A bare 'invalid syntax (line N, offset M)' is useless to a no-tool
    retry editor (run 019f25b8: the same broken insert re-failed identically
    three cycles running). The error now carries the offending source lines."""
    (tmp_path / "mod.py").write_text("def ok():\n    return 1\n")
    tool = ReadEditLintTool(workspace_root=str(tmp_path), target_files=["mod.py"])
    out = tool._run(
        "mod.py",
        full_replace="def ok():\n    return 1\n\ndef broken(:\n    return 2\n",
    )
    assert "syntax_error" in out
    assert "Offending region" in out
    assert "def broken(:" in out
