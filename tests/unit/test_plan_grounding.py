"""Deterministic grounding of plan target files (fix for the config.reference
.yaml mis-scope, trace 019ef1e5)."""
from __future__ import annotations

import pytest

from spine.agents.plan_grounding import (
    build_workspace_index,
    classify_target,
    ground_slice_targets,
)


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "spine").mkdir()
    (tmp_path / "spine" / "config.py").write_text("x = 1\n")
    (tmp_path / ".spine").mkdir()
    (tmp_path / ".spine" / "config.reference.yaml").write_text("ref: true\n")
    return tmp_path


def test_classify(ws):
    idx = build_workspace_index(str(ws))
    # existing path → writable
    assert classify_target("spine/config.py", str(ws), idx)[0] == "writable"
    # genuinely new file (nowhere) → writable
    assert classify_target("tests/unit/test_new.py", str(ws), idx)[0] == "writable"
    # mis-pathed: root config.reference.yaml exists only under .spine/
    kind, real = classify_target("config.reference.yaml", str(ws), idx)
    assert kind == "reference"
    assert real[0] == ".spine/config.reference.yaml"
    # a .spine path itself → reference
    assert classify_target(".spine/config.reference.yaml", str(ws), idx)[0] == "reference"


def test_ground_demotes_mispathed_and_keeps_new(ws):
    slices = [{
        "id": "s1",
        "title": "t",
        "target_files": [
            "spine/config.py", "config.reference.yaml", "tests/unit/test_new.py",
        ],
        "execution_requirements": "Create config.reference.yaml and edit config.py.",
    }]
    out = ground_slice_targets(slices, str(ws))[0]
    assert out["target_files"] == ["spine/config.py", "tests/unit/test_new.py"]
    assert "config.reference.yaml" in out["reference_only_files"]
    # the clarifying note names the real reference path
    assert ".spine/config.reference.yaml" in out["execution_requirements"]
    assert "do NOT create or modify" in out["execution_requirements"]


def test_ground_never_empties_a_slice(ws):
    # a slice whose ONLY target would demote keeps its originals (don't strand it)
    slices = [{
        "id": "s1", "title": "t",
        "target_files": ["config.reference.yaml"],
        "execution_requirements": "do thing",
    }]
    out = ground_slice_targets(slices, str(ws))[0]
    assert out["target_files"] == ["config.reference.yaml"]
    assert not out.get("reference_only_files")


def test_ground_noop_when_all_grounded(ws):
    slices = [{
        "id": "s1", "title": "t",
        "target_files": ["spine/config.py"],
        "execution_requirements": "edit it",
        "reference_only_files": [],
    }]
    out = ground_slice_targets(slices, str(ws))[0]
    assert out["target_files"] == ["spine/config.py"]
    assert out["execution_requirements"] == "edit it"  # untouched
