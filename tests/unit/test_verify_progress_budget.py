"""Progress-based verify gap-cycle budget (run 019f2194).

The fixed 2-gap-cycle cap cut off a run whose total gap count was converging
18→12→8. The first _VERIFY_MIN_CYCLES cycles stay unconditional (pre-existing
behavior); beyond that a cycle is granted while the run keeps setting new BEST
totals, with a patience of one non-improving cycle (verifier noise — run
019f25b8), up to _VERIFY_MAX_CYCLES. Two consecutive cycles without a new
best stop the run. Totals count FAILED CHECKLIST CRITERIA (stable) rather
than free-text gap entries (itemization noise).
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


def test_total_gap_count_prefers_checklist_fails():
    # gap entries are free-text itemization noise (run 019f25b8: 13 gap
    # entries vs 17 checklist fails on the same state) — the checklist wins.
    finding = {
        "verdict": "NOT_VERIFIED",
        "gaps": ["a", "b"],  # 2 entries...
        "checklist": [
            {"criterion": "c1", "passed": False},
            {"criterion": "c2", "passed": False},
            {"criterion": "c3", "passed": False},
            {"criterion": "c4", "passed": True},
        ],  # ...but 3 real fails
    }
    assert _total_gap_count([finding]) == 3


def test_one_stall_cycle_is_tolerated_then_stops():
    # Patience: one non-improving cycle continues (judge noise), two stop.
    out = _run_mapper(2, totals=[28], gaps_now=_findings(15, 15))  # 30 > 28
    assert out["status"] == "needs_gap_fix"
    out = _run_mapper(3, totals=[28, 30], gaps_now=_findings(15, 14))  # 29, still no best
    assert out["status"] == "needs_review"


def test_floor_cycles_granted_unconditionally():
    for attempts in range(_VERIFY_MIN_CYCLES):
        out = _run_mapper(attempts, totals=[], gaps_now=_findings(5))
        assert out["status"] == "needs_gap_fix"
        assert out["verify_attempts"] == attempts + 1


def test_extra_cycle_granted_while_improving():
    out = _run_mapper(_VERIFY_MIN_CYCLES, totals=[18, 12], gaps_now=_findings(4, 4))
    assert out["status"] == "needs_gap_fix"
    assert out["verify_gap_totals"] == [18, 12, 8]


def test_two_cycle_plateau_stops_after_floor():
    out = _run_mapper(3, totals=[12, 8, 8], gaps_now=_findings(4, 4))
    assert out["status"] == "needs_review"
    assert out["needs_review_phase"] == "verify"


def test_two_nonimproving_cycles_stop_after_floor():
    out = _run_mapper(3, totals=[12, 8, 9], gaps_now=_findings(9))
    assert out["status"] == "needs_review"


def test_hard_ceiling_stops_even_while_decreasing():
    totals = list(range(20, 20 - _VERIFY_MAX_CYCLES, -1))
    out = _run_mapper(_VERIFY_MAX_CYCLES, totals=totals, gaps_now=_findings(1))
    assert out["status"] == "needs_review"


def test_uncountable_findings_stop_after_floor():
    out = _run_mapper(_VERIFY_MIN_CYCLES, totals=[9, 5], gaps_now=[])
    assert out["status"] == "needs_review"


# ── Best-state ratchet (run 019f2579: 43→22→11→9 then a regression to 23) ──


class _SnapSpy:
    def __init__(self):
        self.snapshots: list[int] = []
        self.restores = 0

    def install(self, monkeypatch, best_findings=None):
        import spine.workflow.verify_snapshot as vs

        monkeypatch.setattr(
            vs, "snapshot_best",
            lambda ws, wid, files, findings, total: self.snapshots.append(total) or True,
        )
        monkeypatch.setattr(
            vs, "restore_best",
            lambda ws, wid: (setattr(self, "restores", self.restores + 1) or True),
        )
        monkeypatch.setattr(
            vs, "load_best_findings", lambda ws, wid: best_findings
        )


def _run_ratchet(attempts, totals, gaps_now, best=None, retries=0, monkeypatch=None, spy=None):
    subgraph_result = {
        "phase_status": "needs_review",
        "verification_findings": gaps_now,
    }
    parent = {
        "work_id": "w",
        "workspace_root": "/tmp/x",
        "verify_attempts": attempts,
        "verify_gap_totals": totals,
        "verify_best": best,
        "verify_regression_retries": retries,
        "retry_count": {},
        "execution_waves": [[{"id": "s0", "target_files": ["a.py"]}]],
    }
    return _verify_result_mapper(subgraph_result, parent)


def test_new_best_is_snapshotted(monkeypatch):
    spy = _SnapSpy()
    spy.install(monkeypatch)
    out = _run_ratchet(3, [43, 22], _findings(11), best={"total": 22}, monkeypatch=monkeypatch, spy=spy)
    assert spy.snapshots == [11]
    assert out["verify_best"] == {"total": 11}
    assert out["status"] == "needs_gap_fix"


def test_regression_restores_and_patience_grants_retry(monkeypatch):
    best_findings = _findings(4, 5)
    spy = _SnapSpy()
    spy.install(monkeypatch, best_findings=best_findings)
    out = _run_ratchet(5, [43, 22, 11, 9], _findings(23), best={"total": 9})
    assert spy.restores == 1
    assert spy.snapshots == []
    # Findings handed downstream describe the RESTORED state, not the regression.
    assert out["verification_findings"] == best_findings
    # First non-improving cycle → patience grants the next one, retried from
    # the restored best state.
    assert out["status"] == "needs_gap_fix"
    assert "RESTORED" in out["feedback"][-1]["reason"]


def test_second_regression_stops_but_still_restores(monkeypatch):
    best_findings = _findings(9)
    spy = _SnapSpy()
    spy.install(monkeypatch, best_findings=best_findings)
    out = _run_ratchet(6, [43, 22, 11, 9, 23], _findings(25), best={"total": 9}, retries=1)
    assert spy.restores == 1
    # Two consecutive non-improving cycles exhaust the patience → stop, with
    # the workspace restored to (and findings describing) the best state.
    assert out["status"] == "needs_review"
    assert out["verification_findings"] == best_findings


def test_plateau_neither_snapshots_nor_restores(monkeypatch):
    spy = _SnapSpy()
    spy.install(monkeypatch)
    out = _run_ratchet(3, [12, 12], _findings(12), best={"total": 12})
    assert spy.snapshots == [] and spy.restores == 0
    # Third identical total = two trailing stall cycles → stop.
    assert out["status"] == "needs_review"
