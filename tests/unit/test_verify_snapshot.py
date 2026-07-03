"""Best-state snapshot/restore roundtrip (run 019f2579)."""

from __future__ import annotations

from spine.workflow.verify_snapshot import (
    load_best_findings,
    restore_best,
    snapshot_best,
)


def test_snapshot_restore_roundtrip(tmp_path):
    (tmp_path / "src").mkdir()
    f = tmp_path / "src" / "mod.py"
    f.write_text("BEST = 1\n")
    findings = [{"slice_name": "s", "verdict": "NOT_VERIFIED", "gaps": ["g"]}]

    assert snapshot_best(str(tmp_path), "w1", ["src/mod.py"], findings, total=9)

    f.write_text("REGRESSED = 2\n")
    assert restore_best(str(tmp_path), "w1")
    assert f.read_text() == "BEST = 1\n"
    assert load_best_findings(str(tmp_path), "w1") == findings


def test_missing_snapshot_fails_open(tmp_path):
    assert restore_best(str(tmp_path), "nope") is False
    assert load_best_findings(str(tmp_path), "nope") is None


def test_snapshot_skips_missing_files(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    assert snapshot_best(str(tmp_path), "w1", ["a.py", "ghost.py"], [], total=3)
    (tmp_path / "a.py").write_text("x = 2\n")
    assert restore_best(str(tmp_path), "w1")
    assert (tmp_path / "a.py").read_text() == "x = 1\n"
