"""Deterministic E402/F811 late-import hoist on write (runs ce6f887d, 717fda0e).

Appending code to an existing file lands the appended block's imports
mid-file (E402) and duplicates imports the head already has (F811) — and the
synthesis editor regenerates the same shape every gap cycle, so the lint gaps
plateau until the run parks. The write path now hoists late module-level
imports into the head block and drops exact duplicates.
"""

from __future__ import annotations

import json

from spine.agents.tools.read_edit_lint import ReadEditLintTool


def _tool(tmp_path) -> ReadEditLintTool:
    return ReadEditLintTool(workspace_root=str(tmp_path), target_files=["mod.py"])


# The appended-tests shape from run 717fda0e: existing tests, then the new
# block re-imports the module under test plus pytest, mid-file.
APPENDED = (
    "import pytest\n"
    "from spine_fake import format_bytes\n"
    "\n"
    "\n"
    "def test_old():\n"
    "    assert format_bytes is not None\n"
    "\n"
    "\n"
    "from spine_fake import format_bytes\n"
    "import pytest\n"
    "import textwrap\n"
    "\n"
    "\n"
    "def test_new():\n"
    "    assert pytest is not None and textwrap is not None\n"
)


def test_late_imports_hoisted_and_deduped(tmp_path):
    out = _tool(tmp_path)._run("mod.py", full_replace=APPENDED)
    text = (tmp_path / "mod.py").read_text()
    lines = text.splitlines()
    # All imports now sit before the first def.
    first_def = next(i for i, ln in enumerate(lines) if ln.startswith("def "))
    import_lines = [
        i for i, ln in enumerate(lines) if ln.startswith(("import ", "from "))
    ]
    assert all(i < first_def for i in import_lines)
    # Duplicates collapsed to one occurrence each.
    assert text.count("import pytest") == 1
    assert text.count("from spine_fake import format_bytes") == 1
    # The genuinely new import survived the move.
    assert "import textwrap" in text
    assert "imports_hoisted" in out
    compile(text, "mod.py", "exec")
    payload = json.loads(out)
    assert "E402" not in str(payload.get("ruff", ""))
    assert "F811" not in str(payload.get("ruff", ""))


def test_conditional_imports_are_not_touched(tmp_path):
    code = (
        "import os\n"
        "\n"
        "try:\n"
        "    import fast_json as json_impl\n"
        "except ImportError:\n"
        "    import json as json_impl\n"
        "\n"
        "\n"
        "def f():\n"
        "    return json_impl.dumps(os.getcwd())\n"
    )
    out = _tool(tmp_path)._run("mod.py", full_replace=code)
    text = (tmp_path / "mod.py").read_text()
    assert "imports_hoisted" not in out
    assert text == code


def test_clean_file_untouched(tmp_path):
    code = "import json\n\n\ndef f():\n    return json.dumps({})\n"
    out = _tool(tmp_path)._run("mod.py", full_replace=code)
    assert "imports_hoisted" not in out
    assert (tmp_path / "mod.py").read_text() == code
