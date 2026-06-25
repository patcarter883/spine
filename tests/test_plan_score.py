"""Tests for best-of-N plan scoring (spine.workflow.plan_score)."""

from __future__ import annotations

from spine.workflow.plan_score import score_plan

_SPEC = "Modify api.py and add a handler in routes.py. Update config.py."


def _slice(sid, files, deps=()):
    return {
        "id": sid,
        "title": sid.upper(),
        "target_files": list(files),
        "dependencies": list(deps),
    }


def test_clean_plan_scores_top_and_is_schedulable():
    plan = {
        "feature_slices": [
            _slice("s1", ["api.py"]),
            _slice("s2", ["routes.py"], ["s1"]),
            _slice("s3", ["config.py"]),
        ]
    }
    s = score_plan(plan, spec_body=_SPEC, work_id="t")
    assert s.schedulable is True
    assert s.total == 1.0


def test_dependency_cycle_is_unschedulable_zero():
    plan = {
        "feature_slices": [
            _slice("s1", ["a.py"], ["s2"]),
            _slice("s2", ["b.py"], ["s1"]),
        ]
    }
    s = score_plan(plan, spec_body=_SPEC, work_id="t")
    assert s.schedulable is False
    assert s.total == 0.0


def test_no_slices_is_zero():
    s = score_plan({"feature_slices": []}, work_id="t")
    assert s.total == 0.0
    assert s.schedulable is False


def test_empty_target_files_penalised_but_schedulable():
    plan = {"feature_slices": [_slice("s1", []), _slice("s2", [])]}
    s = score_plan(plan, spec_body=_SPEC, work_id="t")
    assert s.schedulable is True
    assert s.target_files_score == 0.0
    assert s.total < 0.5


def test_duplicate_target_files_explosion_penalised():
    plan = {"feature_slices": [_slice("s1", ["api.py"] * 250)]}
    s = score_plan(plan, spec_body=_SPEC, work_id="t")
    assert s.dup_penalty_score < 0.05  # 1/250


def test_clean_plan_outranks_every_pathology():
    """The selector's core contract: the clean plan wins."""
    clean = {
        "feature_slices": [
            _slice("s1", ["api.py"]),
            _slice("s2", ["routes.py"], ["s1"]),
            _slice("s3", ["config.py"]),
        ]
    }
    empty_tf = {"feature_slices": [_slice("s1", []), _slice("s2", [])]}
    dup = {"feature_slices": [_slice("s1", ["api.py"] * 250)]}
    cycle = {
        "feature_slices": [
            _slice("s1", ["a.py"], ["s2"]),
            _slice("s2", ["b.py"], ["s1"]),
        ]
    }
    clean_score = score_plan(clean, spec_body=_SPEC, work_id="t").total
    for bad in (empty_tf, dup, cycle):
        assert clean_score > score_plan(bad, spec_body=_SPEC, work_id="t").total


def test_coverage_neutral_when_spec_names_no_files():
    plan = {"feature_slices": [_slice("s1", ["whatever.py"])]}
    s = score_plan(plan, spec_body="A prose spec with no file paths at all.", work_id="t")
    assert s.coverage_score == 1.0
