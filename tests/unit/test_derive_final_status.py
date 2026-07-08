"""Tests for dispatcher._derive_final_status.

This guards the shared final-status derivation used by submit_work,
resume_interrupted_work, and _run_workflow_graph_inner. Key invariants:
reviewed-plan work types (reviewed_task / critical_reviewed_task) that pass
critic_plan must end at "awaiting_approval" so they surface for human review —
regardless of which dispatcher entry point ran the graph — and the graph's
terminal status is authoritative: stale needs_review feedback from verify
cycles the gap-fix loop subsequently FIXED must not flip a converged run.

Regressions for: critical_reviewed_task a8511aac marked "completed" after a
restart (the restart path lacked the awaiting_approval conversion); task
06c2d55c converged on verify cycle 4 yet finalized needs_review off stale
cycle-1..3 feedback, rolling back a fully verified patch.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the spine package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.work.dispatcher import _derive_final_status


def test_reviewed_plan_completed_becomes_awaiting_approval():
    # Graph that ends normally leaves status "running"; for a reviewed-plan
    # type that passes critic, it must become "awaiting_approval".
    for status in ("running", "completed"):
        result = {"status": status, "current_phase": "critic"}
        out = _derive_final_status(
            result, stalled=False, work_type="critical_reviewed_task"
        )
        assert out == "awaiting_approval", status


def test_reviewed_task_completed_becomes_awaiting_approval():
    result = {"status": "completed"}
    out = _derive_final_status(result, stalled=False, work_type="reviewed_task")
    assert out == "awaiting_approval"


def test_graph_status_needs_review_is_preserved():
    # When the critic exhausts retries, its result mapper sets
    # status="needs_review" in graph state; that must be preserved, not
    # converted to awaiting_approval.
    result = {"status": "needs_review"}
    out = _derive_final_status(
        result, stalled=False, work_type="critical_reviewed_task"
    )
    assert out == "needs_review"


def test_recovered_run_with_stale_feedback_completes():
    # Regression (work 06c2d55c): verify cycles 1-3 failed and left
    # needs_review entries in the append-only feedback list, then the
    # gap-fix loop CONVERGED — cycle 4 verified all criteria and the graph
    # ended normally (status "running"). The old feedback scan flipped the
    # run to needs_review, so the sandbox rolled back a fully verified
    # patch. Stale feedback must not override the graph's terminal status.
    result = {
        "status": "running",
        "verification_passed": True,
        "feedback": [
            {"status": "needs_review", "tier": "verify", "reason": "cycle 1"},
            {"status": "needs_review", "tier": "verify", "reason": "cycle 2"},
            {"status": "needs_review", "tier": "verify", "reason": "cycle 3"},
        ],
    }
    out = _derive_final_status(result, stalled=False, work_type="task")
    assert out == "completed"


def test_regular_task_stays_completed():
    result = {"status": "running"}
    out = _derive_final_status(result, stalled=False, work_type="task")
    assert out == "completed"


def test_stalled_takes_precedence():
    result = {"status": "running"}
    out = _derive_final_status(
        result, stalled=True, work_type="critical_reviewed_task"
    )
    assert out == "stalled"
