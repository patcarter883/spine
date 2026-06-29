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
    _FILE_READ_CAP,
    _READ_PRESSURE_SOFT,
    _READ_PRESSURE_WALL,
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
    # Target that does NOT exist on disk: a unique-basename match cannot be
    # auto-resolved (that needs the target to exist — see
    # test_autoresolve_redirects_to_existing_target), so the miss returns
    # not_found WITH a did_you_mean suggestion.
    tool = ReadEditLintTool(
        workspace_root=str(ws), target_files=["pkg/ghost.py"]
    )
    out = tool._run("pkg/wrong_mod.py", read_symbol="Foo.bar")
    payload = json.loads(out)
    assert payload["status"] == "not_found"
    # same basename as the (missing) target → pkg/ghost.py is suggested
    out2 = tool._run("other/ghost.py", read_symbol="Foo.bar")
    sugg = json.loads(out2).get("did_you_mean", [])
    assert "pkg/ghost.py" in sugg
    # target_files are ranked first
    assert sugg[0] == "pkg/ghost.py"


def test_edit_not_found_also_suggests(ws):
    tool = ReadEditLintTool(workspace_root=str(ws), target_files=["pkg/ghost.py"])
    out = tool._run("other/ghost.py", old_str="return 1", new_str="return 9")
    payload = json.loads(out)
    assert payload["status"] == "not_found"
    assert "pkg/ghost.py" in payload.get("did_you_mean", [])


def test_autoresolve_redirects_to_existing_target(ws):
    # A wrong path whose basename uniquely matches an EXISTING target_file is
    # silently redirected and read, rather than bounced as not_found
    # (trace 019ef2ae — stops the editor inventing path variants).
    tool = ReadEditLintTool(workspace_root=str(ws), target_files=["pkg/mod.py"])
    out = tool._run("other/mod.py", read_symbol="Foo.bar")
    # Redirected & read: an auto-correct note + the real source, not not_found.
    assert "auto-corrected to the slice target pkg/mod.py" in out
    assert "[read: pkg/mod.py" in out
    assert "def bar" in out


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


# ── Per-file read budget (anchor-varied re-reads) ───────────────────────
# Distinct anchors into the same file slip past the exact-anchor cache; the
# per-file budget caps them. Five distinct snippets in _SRC.
_ANCHORS = ["class Foo", "def bar", "return 1", "def baz", "return 2"]


def test_file_read_cap_withholds_body_after_cap(ws):
    tool = ReadEditLintTool(workspace_root=str(ws))
    outs = [tool._run("pkg/mod.py", read_around=a) for a in _ANCHORS]
    # the first _FILE_READ_CAP distinct-anchor reads return real bodies
    for o in outs[:_FILE_READ_CAP]:
        assert o.startswith("[read:")
    # the read landing ON the cap flags itself as the final full read
    assert "final full read" in outs[_FILE_READ_CAP - 1]
    # reads beyond the cap are refused without a body
    assert _status(outs[_FILE_READ_CAP]) == "read_capped"


def test_file_read_cap_resets_after_edit(ws):
    tool = ReadEditLintTool(workspace_root=str(ws))
    for a in _ANCHORS[:_FILE_READ_CAP]:
        tool._run("pkg/mod.py", read_around=a)
    assert _status(tool._run("pkg/mod.py", read_around="return 2")) == "read_capped"
    # a successful edit clears the budget → reads flow again
    tool._run("pkg/mod.py", full_replace=_SRC.replace("return 1", "return 8"))
    assert tool._run("pkg/mod.py", read_around="return 2").startswith("[read:")


def test_global_read_wall_refuses_breadth_spiral(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    n_files = (_READ_PRESSURE_WALL // _FILE_READ_CAP) + 2
    for i in range(n_files):
        (pkg / f"m{i}.py").write_text(_SRC, encoding="utf-8")
    tool = ReadEditLintTool(workspace_root=str(tmp_path))
    outs = []
    for i in range(n_files):
        for a in _ANCHORS[:_FILE_READ_CAP]:
            outs.append(tool._run(f"pkg/m{i}.py", read_around=a))
    # once reads pile up past the wall, a fresh file read is still refused
    assert any("reading is now disabled" in o for o in outs)


def test_not_found_escalates_after_repeated_guesses(ws):
    tool = ReadEditLintTool(workspace_root=str(ws), target_files=["pkg/mod.py"])
    first = json.loads(tool._run("a/x.py", read_symbol="Foo.bar"))
    assert first["status"] == "not_found"
    assert "STOP guessing" not in first["detail"]
    # second miss with no intervening edit → hard directive to use target_files
    second = json.loads(tool._run("b/y.py", read_symbol="Foo.bar"))
    assert "STOP guessing" in second["detail"]
    assert "pkg/mod.py" in second["did_you_mean"]
