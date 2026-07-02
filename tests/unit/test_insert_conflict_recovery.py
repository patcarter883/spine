"""Duplicate-insert idempotence recovery (run 019f2194).

Rework editors bundle every definition a slice needs into ONE ast_edit insert
— including definitions an earlier cycle already landed. The re-definition
guard used to reject the whole batch with conflict_error, leaving the slice
partial forever. Now the batch is split deterministically: chunks re-defining
an existing symbol REPLACE it in place, the rest insert at the anchor.
Ambiguous cases keep the hard error so nothing is silently mangled.
"""

from __future__ import annotations

from spine.agents.tools.read_edit_lint import ReadEditLintTool

SRC = '''\
class UIApi:
    """API facade."""

    def get_embedding_provider(self):
        return "old"

    def unrelated(self):
        return 1
'''


def _tool(tmp_path, src: str = SRC) -> ReadEditLintTool:
    (tmp_path / "api.py").write_text(src)
    return ReadEditLintTool(workspace_root=str(tmp_path), target_files=["api.py"])


def test_mixed_bundle_replaces_existing_and_inserts_new(tmp_path):
    tool = _tool(tmp_path)
    code = (
        '    def get_embedding_provider(self):\n'
        '        return "new"\n'
        "\n"
        "    def set_embedding_provider(self, config):\n"
        "        return config\n"
    )
    out = tool._run(
        "api.py",
        ast_edit={"symbol": "UIApi.unrelated", "action": "insert_after", "code": code},
    )
    assert "conflict_error" not in out
    text = (tmp_path / "api.py").read_text()
    assert text.count("def get_embedding_provider") == 1
    assert 'return "new"' in text
    assert 'return "old"' not in text
    assert "def set_embedding_provider" in text
    # The new method landed after the anchor, not inside another def.
    assert text.index("def unrelated") < text.index("def set_embedding_provider")


def test_all_conflicting_bundle_replaces_in_place(tmp_path):
    tool = _tool(tmp_path)
    code = '    def get_embedding_provider(self):\n        return "v2"\n'
    out = tool._run(
        "api.py",
        ast_edit={"symbol": "UIApi.unrelated", "action": "insert_after", "code": code},
    )
    assert "conflict_error" not in out
    text = (tmp_path / "api.py").read_text()
    assert text.count("def get_embedding_provider") == 1
    assert 'return "v2"' in text
    assert "def unrelated" in text


def test_shared_method_name_resolves_to_anchor_scope(tmp_path):
    src = (
        "class A:\n"
        "    def render(self):\n"
        '        return "a"\n'
        "\n"
        "\n"
        "class B:\n"
        "    def render(self):\n"
        '        return "b"\n'
        "\n"
        "    def other(self):\n"
        "        return 2\n"
    )
    tool = _tool(tmp_path, src)
    code = '    def render(self):\n        return "b2"\n'
    out = tool._run(
        "api.py",
        ast_edit={"symbol": "B.other", "action": "insert_after", "code": code},
    )
    assert "conflict_error" not in out
    text = (tmp_path / "api.py").read_text()
    assert 'return "a"' in text  # A.render untouched
    assert 'return "b2"' in text  # B.render replaced
    assert 'return "b"\n' not in text.replace('return "b2"', "")


def test_unresolvable_ambiguity_keeps_hard_error(tmp_path):
    src = (
        "class A:\n"
        "    def render(self):\n"
        '        return "a"\n'
        "\n"
        "\n"
        "class B:\n"
        "    def render(self):\n"
        '        return "b"\n'
        "\n"
        "\n"
        "def standalone():\n"
        "    return 3\n"
    )
    tool = _tool(tmp_path, src)
    # Module-level anchor gives no scope to disambiguate two Cls.render defs.
    code = 'def render(self):\n    return "x"\n'
    out = tool._run(
        "api.py",
        ast_edit={"symbol": "standalone", "action": "insert_after", "code": code},
    )
    assert "conflict_error" in out
    text = (tmp_path / "api.py").read_text()
    assert 'return "a"' in text and 'return "b"' in text  # nothing mangled


def test_non_conflicting_insert_unchanged(tmp_path):
    tool = _tool(tmp_path)
    code = "    def brand_new(self):\n        return 42\n"
    out = tool._run(
        "api.py",
        ast_edit={"symbol": "UIApi.unrelated", "action": "insert_after", "code": code},
    )
    assert "conflict_error" not in out
    text = (tmp_path / "api.py").read_text()
    assert "def brand_new" in text
    assert 'return "old"' in text
