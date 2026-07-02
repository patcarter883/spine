"""Progress-based verify gap-cycle budget (run 019f2194).

The fixed 2-gap-cycle cap cut off a run whose total gap count was converging
18→12→8. The first _VERIFY_MIN_CYCLES cycles stay unconditional (pre-existing
behavior); beyond that a cycle is granted only while the total strictly
decreases, up to _VERIFY_MAX_CYCLES. Plateau or increase stops immediately, so
a stuck loop costs no more than before.
"""

from __future__ import annotations

from spine.workflow.compose import (
    _VERIFY_MAX_CYCLES,
    _VERIFY_MIN_CYCLES,
    _total_gap_count,
    _verify_result_mapper,
)


def _findings(*gap_counts: int) -> list[dict]:
    return [
        {"slice_name": f"s{i}", "verdict": "NOT_VERIFIED", "gaps": [f"g{j}" for j in range(n)]}
        for i, n in enumerate(gap_counts)
    ]


def _run_mapper(attempts: int, totals: list[int], gaps_now: list[dict]) -> dict:
    subgraph_result = {
        "phase_status": "needs_review",
        "verification_findings": gaps_now,
    }
    parent = {
        "work_id": "w",
        "verify_attempts": attempts,
        "verify_gap_totals": totals,
        "retry_count": {},
    }
    return _verify_result_mapper(subgraph_result, parent)


def test_total_gap_count():
    assert _total_gap_count(_findings(3, 2)) == 5
    # VERIFIED slices don't count; a failing slice with no itemized gaps
    # counts as one so it can't fake convergence to zero.
    assert _total_gap_count(
        [{"verdict": "VERIFIED", "gaps": []}, {"verdict": "NOT_VERIFIED", "gaps": []}]
    ) == 1
    assert _total_gap_count([]) is None
    assert _total_gap_count(None) is None


def test_floor_cycles_granted_unconditionally():
    for attempts in range(_VERIFY_MIN_CYCLES):
        out = _run_mapper(attempts, totals=[], gaps_now=_findings(5))
        assert out["status"] == "needs_gap_fix"
        assert out["verify_attempts"] == attempts + 1


def test_extra_cycle_granted_while_strictly_decreasing():
    out = _run_mapper(_VERIFY_MIN_CYCLES, totals=[18, 12], gaps_now=_findings(4, 4))
    assert out["status"] == "needs_gap_fix"
    assert out["verify_gap_totals"] == [18, 12, 8]


def test_plateau_stops_after_floor():
    out = _run_mapper(_VERIFY_MIN_CYCLES, totals=[12, 8], gaps_now=_findings(4, 4))
    assert out["status"] == "needs_review"
    assert out["needs_review_phase"] == "verify"


def test_increase_stops_after_floor():
    out = _run_mapper(_VERIFY_MIN_CYCLES, totals=[12, 8], gaps_now=_findings(9))
    assert out["status"] == "needs_review"


def test_hard_ceiling_stops_even_while_decreasing():
    totals = list(range(20, 20 - _VERIFY_MAX_CYCLES, -1))
    out = _run_mapper(_VERIFY_MAX_CYCLES, totals=totals, gaps_now=_findings(1))
    assert out["status"] == "needs_review"


def test_uncountable_findings_stop_after_floor():
    out = _run_mapper(_VERIFY_MIN_CYCLES, totals=[9, 5], gaps_now=[])
    assert out["status"] == "needs_review"
