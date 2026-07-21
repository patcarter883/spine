"""Final-mile targeted-edit mode (run 019f25f5).

At 3/25 failing criteria, two wholesale re-syntheses scored WORSE (6, 9) and
the budget expired — regeneration variance outweighs a 3-criterion delta.
When a slice is down to a handful of failing criteria with a passing
majority, the editor is constrained to the smallest edit set that fixes
exactly those, and placement prefers the smallest clean candidate.
"""

from __future__ import annotations

from spine.agents.synthesis_implementer import (
    SynthesizedEdit,
    SynthesizedSlice,
    build_synthesis_prompt,
    place_best_candidate,
)
from spine.workflow.subgraphs.implement_subgraph import _final_mile_fails


def _state(fails: int, passing: int, slice_id: str = "s1") -> dict:
    checklist = [
        {"criterion": f"fail-{i}", "passed": False} for i in range(fails)
    ] + [{"criterion": f"pass-{i}", "passed": True} for i in range(passing)]
    return {
        "verification_findings": [
            {"slice_name": slice_id, "verdict": "NOT_VERIFIED", "checklist": checklist}
        ]
    }


def test_qualifies_with_few_fails_and_passing_majority():
    fails = _final_mile_fails(_state(3, 22), {"id": "s1"})
    assert fails == ["fail-0", "fail-1", "fail-2"]


def test_too_many_fails_disqualifies():
    assert _final_mile_fails(_state(6, 22), {"id": "s1"}) == []


def test_no_passing_majority_disqualifies():
    assert _final_mile_fails(_state(3, 3), {"id": "s1"}) == []


def test_no_findings_or_all_passing_disqualifies():
    assert _final_mile_fails({}, {"id": "s1"}) == []
    assert _final_mile_fails(_state(0, 25), {"id": "s1"}) == []


def test_other_slice_findings_ignored():
    assert _final_mile_fails(_state(2, 20, slice_id="other"), {"id": "s1"}) == []


def test_final_mile_prompt_tail_lists_exact_criteria():
    p = build_synthesis_prompt(
        slice_json="{}",
        refs_body="",
        plan_body="",
        gaps_body="gap detail",
        final_mile_fails=["save persists to yaml", "st.rerun called after save"],
    )
    assert "FINAL MILE" in p
    assert "SMALLEST possible edit set" in p
    assert "- save persists to yaml" in p
    assert "- st.rerun called after save" in p
    assert "VERIFICATION REWORK" not in p  # final-mile tail replaces it


def test_lint_feedback_leads_but_final_mile_constraint_survives():
    # Original contract dropped FINAL MILE entirely on a placement retry —
    # run 019f82b1 showed those retries regenerating wholesale and
    # regressing near-passing slices (two ratchet restores). The error-fix
    # instruction still LEADS the tail; the minimal-edit constraint now
    # rides along instead of vanishing.
    p = build_synthesis_prompt(
        slice_json="{}",
        refs_body="",
        plan_body="",
        final_mile_fails=["c1"],
        feedback="E999 boom",
    )
    assert "FAILED to place" in p
    assert "FINAL MILE still applies" in p
    assert p.index("FAILED to place") < p.index("FINAL MILE still applies")


def test_prefer_minimal_picks_smallest_clean_candidate(tmp_path):
    (tmp_path / "m.py").write_text(
        "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
    )
    small = SynthesizedSlice(
        edits=[SynthesizedEdit(file="m.py", symbol="a", action="replace",
                               code="def a():\n    return 10\n")]
    )
    big = SynthesizedSlice(
        edits=[
            SynthesizedEdit(file="m.py", symbol="a", action="replace",
                            code="def a():\n    return 99\n"),
            SynthesizedEdit(file="m.py", symbol="b", action="replace",
                            code="def b():\n    return 99\n"),
        ]
    )
    winner, placement = place_best_candidate(
        [big, small],
        workspace_root=str(tmp_path),
        target_files=["m.py"],
        prefer_minimal=True,
    )
    assert winner is small
    assert placement.clean
    text = (tmp_path / "m.py").read_text()
    assert "return 10" in text and "return 2" in text  # b untouched


def test_default_scoring_still_prefers_most_applied(tmp_path):
    (tmp_path / "m.py").write_text(
        "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
    )
    small = SynthesizedSlice(
        edits=[SynthesizedEdit(file="m.py", symbol="a", action="replace",
                               code="def a():\n    return 10\n")]
    )
    big = SynthesizedSlice(
        edits=[
            SynthesizedEdit(file="m.py", symbol="a", action="replace",
                            code="def a():\n    return 99\n"),
            SynthesizedEdit(file="m.py", symbol="b", action="replace",
                            code="def b():\n    return 99\n"),
        ]
    )
    winner, _ = place_best_candidate(
        [big, small],
        workspace_root=str(tmp_path),
        target_files=["m.py"],
    )
    assert winner is big
