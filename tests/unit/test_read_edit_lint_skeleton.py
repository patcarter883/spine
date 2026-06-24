"""God-class anchored reads fold to a signature skeleton (trace 019ef809).

A `read_symbol` on a large class must not dump the whole class body — that is
what blew the implement phase to 1.7M tokens. It returns a header + per-method
signatures instead, while a `Class.method` read still returns the full body.
"""

from __future__ import annotations

from spine.agents.tools.read_edit_lint import ReadEditLintTool


def _big_class_source(n_methods: int = 40) -> str:
    lines = ["class Big:", '    """A god-class."""', "", "    attr = 1", ""]
    for i in range(n_methods):
        lines += [
            f"    def method_{i}(self, x: int) -> int:",
            f'        """Method {i}."""',
            f"        total = x + {i}",
            "        return total",
            "",
        ]
    return "\n".join(lines) + "\n"


def test_godclass_read_returns_skeleton(tmp_path):
    src = _big_class_source()
    f = tmp_path / "big.py"
    f.write_text(src)
    tool = ReadEditLintTool(workspace_root=str(tmp_path), target_files=["big.py"])

    out = tool._run("big.py", read_symbol="Big")

    assert out.startswith("[read:")
    assert "SKELETON" in out
    # Signatures are present...
    assert "def method_0(self, x: int) -> int:" in out
    assert "def method_39(self, x: int) -> int:" in out
    # ...but the bodies are folded away.
    assert "total = x + 0" not in out
    assert "return total" not in out


def test_method_read_returns_full_body(tmp_path):
    src = _big_class_source()
    f = tmp_path / "big.py"
    f.write_text(src)
    tool = ReadEditLintTool(workspace_root=str(tmp_path), target_files=["big.py"])

    out = tool._run("big.py", read_symbol="Big.method_3")

    assert out.startswith("[read:")
    assert "SKELETON" not in out
    assert "total = x + 3" in out  # full body, not just the signature


def test_small_class_shown_in_full(tmp_path):
    src = _big_class_source(n_methods=2)  # well under the skeleton threshold
    f = tmp_path / "small.py"
    f.write_text(src)
    tool = ReadEditLintTool(workspace_root=str(tmp_path), target_files=["small.py"])

    out = tool._run("small.py", read_symbol="Big")

    assert "SKELETON" not in out
    assert "return total" in out  # bodies shown for a small class
