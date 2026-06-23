"""Anti-spiral behaviours of ReadEditLintTool (read-dedup, path hints, pressure).

These cover the four fixes added after trace 019ef1e5, where a slice editor
made 332 anchored reads / 8 edits and exhausted the token budget:

1. a repeat anchored read of an unchanged file returns ``already_read`` instead
   of re-injecting the body;
2. a successful edit invalidates that file's read cache (next read is fresh);
3. ``not_found`` carries a ``did_you_mean`` list grounded on the slice's
   ``target_files`` and the real workspace tree;
4. sustained reading with no edits appends an edit-pressure nudge.
"""
from __future__ import annotations

import json

import pytest

from spine.agents.tools.read_edit_lint import (
    _READ_PRESSURE_SOFT,
    ReadEditLintTool,
)

_SRC = (
    "class Foo:\n"
    "    def bar(self):\n"
    "        return 1\n\n"
    "    def baz(self):\n"
    "        return 2\n"
)


@pytest.fixture
def ws(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "mod.py").write_text(_SRC, encoding="utf-8")
    return tmp_path


def _status(out: str) -> str:
    return json.loads(out)["status"]


def test_repeat_read_returns_already_read(ws):
    tool = ReadEditLintTool(workspace_root=str(ws))
    first = tool._run("pkg/mod.py", read_symbol="Foo.bar")
    assert first.startswith("[read:")
    second = tool._run("pkg/mod.py", read_symbol="Foo.bar")
    assert _status(second) == "already_read"
    # different anchor in the same file is NOT a cache hit
    other = tool._run("pkg/mod.py", read_symbol="Foo.baz")
    assert other.startswith("[read:")


def test_edit_invalidates_read_cache(ws):
    tool = ReadEditLintTool(workspace_root=str(ws))
    assert tool._run("pkg/mod.py", read_symbol="Foo.bar").startswith("[read:")
    assert _status(tool._run("pkg/mod.py", read_symbol="Foo.bar")) == "already_read"
    edited = tool._run(
        "pkg/mod.py",
        full_replace=_SRC.replace("return 1", "return 99"),
    )
    assert _status(edited) == "ok"
    # after the edit the same symbol re-reads fresh, not from cache
    assert tool._run("pkg/mod.py", read_symbol="Foo.bar").startswith("[read:")


def test_not_found_suggests_target_file(ws):
    tool = ReadEditLintTool(
        workspace_root=str(ws), target_files=["pkg/mod.py"]
    )
    out = tool._run("pkg/wrong_mod.py", read_symbol="Foo.bar")
    payload = json.loads(out)
    assert payload["status"] == "not_found"
    # same basename → pkg/mod.py is suggested
    out2 = tool._run("other/mod.py", read_symbol="Foo.bar")
    sugg = json.loads(out2).get("did_you_mean", [])
    assert "pkg/mod.py" in sugg
    # target_files are ranked first
    assert sugg[0] == "pkg/mod.py"


def test_edit_not_found_also_suggests(ws):
    tool = ReadEditLintTool(workspace_root=str(ws), target_files=["pkg/mod.py"])
    out = tool._run("other/mod.py", old_str="return 1", new_str="return 9")
    payload = json.loads(out)
    assert payload["status"] == "not_found"
    assert "pkg/mod.py" in payload.get("did_you_mean", [])


def test_edit_pressure_nudges_after_threshold(ws):
    tool = ReadEditLintTool(workspace_root=str(ws))
    outs = [
        tool._run("pkg/mod.py", read_symbol="Foo.bar")
        for _ in range(_READ_PRESSURE_SOFT)
    ]
    # the call landing exactly on the soft threshold carries the nudge
    assert "reads since your last edit" in outs[_READ_PRESSURE_SOFT - 1]
    # ...and earlier reads do not
    assert "reads since your last edit" not in outs[0]


def test_reference_only_file_rejects_edits(ws):
    (ws / "pkg" / "ref.yaml").write_text("ref: 1\n", encoding="utf-8")
    tool = ReadEditLintTool(
        workspace_root=str(ws), reference_only_files=["pkg/ref.yaml"]
    )
    # edit is blocked
    out = tool._run("pkg/ref.yaml", full_replace="ref: 2\n")
    assert _status(out) == "reference_only"
    assert (ws / "pkg" / "ref.yaml").read_text() == "ref: 1\n"  # untouched
    # reading it is still allowed
    assert tool._run("pkg/ref.yaml", read_around="ref").startswith("[read:")


def test_spine_runtime_path_is_reference_only(ws):
    (ws / ".spine").mkdir()
    (ws / ".spine" / "config.reference.yaml").write_text("a: 1\n", encoding="utf-8")
    tool = ReadEditLintTool(workspace_root=str(ws))  # no explicit list
    out = tool._run(".spine/config.reference.yaml", full_replace="a: 2\n")
    assert _status(out) == "reference_only"


def test_pressure_resets_after_edit(ws):
    tool = ReadEditLintTool(workspace_root=str(ws))
    for _ in range(_READ_PRESSURE_SOFT - 1):
        tool._run("pkg/mod.py", read_symbol="Foo.bar")
    tool._run("pkg/mod.py", full_replace=_SRC.replace("return 1", "return 7"))
    # counter reset → the next read is read #1, no nudge
    nxt = tool._run("pkg/mod.py", read_symbol="Foo.bar")
    assert "reads since your last edit" not in nxt
