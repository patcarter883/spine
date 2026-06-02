"""Tests for dispatcher._derive_final_status.

This guards the shared final-status derivation used by submit_work,
resume_interrupted_work, and _run_workflow_graph_inner. The key invariant:
reviewed-plan work types (reviewed_task / critical_reviewed_task) that pass
critic_plan must end at "awaiting_approval" so they surface for human review —
regardless of which dispatcher entry point ran the graph.

Regression for: critical_reviewed_task a8511aac marked "completed" after a
restart (the restart path lacked the awaiting_approval conversion).
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
            result, stalled=False, feedback=[], work_type="critical_reviewed_task"
        )
        assert out == "awaiting_approval", status


def test_reviewed_task_completed_becomes_awaiting_approval():
    result = {"status": "completed"}
    out = _derive_final_status(
        result, stalled=False, feedback=[], work_type="reviewed_task"
    )
    assert out == "awaiting_approval"


def test_needs_review_feedback_is_not_overridden():
    # When the critic exhausts retries it routes to END with needs_review
    # feedback; that must be preserved, not converted to awaiting_approval.
    result = {"status": "completed"}
    feedback = [{"status": "needs_review", "tier": "critic_plan"}]
    out = _derive_final_status(
        result, stalled=False, feedback=feedback, work_type="critical_reviewed_task"
    )
    assert out == "needs_review"


def test_regular_task_stays_completed():
    result = {"status": "running"}
    out = _derive_final_status(
        result, stalled=False, feedback=[], work_type="task"
    )
    assert out == "completed"


def test_stalled_takes_precedence():
    result = {"status": "running"}
    out = _derive_final_status(
        result, stalled=True, feedback=[], work_type="critical_reviewed_task"
    )
    assert out == "stalled"
