"""Tests for the adversarial review stage (critical work types).

The adversarial stage runs after ``critic_plan`` for ``critical_task`` and
``critical_reviewed_task``. It red-teams the approved plan, loops
autonomously-fixable findings back to PLAN on its OWN retry budget, and
escalates human-judgement findings. It replaces the old ``critic_specify``
gate as the defining feature of a critical task.
"""

from __future__ import annotations

import pytest

from spine.models.enums import PhaseName

try:
    from spine.workflow.compose import (
        WORKFLOW_SEQUENCES,
        build_workflow_graph,
        adversarial_router,
        _adversarial_result_mapper,
        _plan_state_mapper,
        get_restart_phases,
    )

    _AVAILABLE = True
except Exception:  # pragma: no cover - dep guard
    _AVAILABLE = False

pytestmark = pytest.mark.skipif(not _AVAILABLE, reason="workflow compose deps not available")

ADV = f"{PhaseName.ADVERSARIAL.value}_plan"
CRITIC_PLAN = f"{PhaseName.CRITIC.value}_plan"


# ── Sequence + graph structure ───────────────────────────────────────────────


@pytest.mark.parametrize("work_type", ["critical_task", "critical_reviewed_task"])
def test_adversarial_follows_critic_plan_and_no_critic_specify(work_type):
    names = [n for n, _ in WORKFLOW_SEQUENCES[work_type]]
    assert ADV in names
    assert names.index(ADV) == names.index(CRITIC_PLAN) + 1
    assert "critic_specify" not in names


def test_critical_task_reaches_implement_after_adversarial():
    names = [n for n, _ in WORKFLOW_SEQUENCES["critical_task"]]
    assert names[-2:] == [PhaseName.IMPLEMENT.value, PhaseName.VERIFY.value]


def test_critical_reviewed_task_ends_at_adversarial():
    names = [n for n, _ in WORKFLOW_SEQUENCES["critical_reviewed_task"]]
    assert names[-1] == ADV


def test_adversarial_node_present_and_excluded_from_restart():
    for wt in ("critical_task", "critical_reviewed_task"):
        nodes = set(build_workflow_graph(wt).get_graph().nodes)
        assert ADV in nodes
        assert ADV not in get_restart_phases(wt)


def test_critical_task_adversarial_edges_route_through_implement_gate():
    g = build_workflow_graph("critical_task").get_graph()
    edges = {(e.source, e.target) for e in g.edges}
    out = {t for s, t in edges if s == ADV}
    # passed → implement gate; needs_revision → plan; needs_review → flag (autonomous)
    assert "gate_adversarial_plan_to_implement" in out
    assert PhaseName.PLAN.value in out
    assert "flag_needs_review" in out
    # The implement gate chains through the plan-artifact prerequisite gate.
    assert ("gate_adversarial_plan_to_implement", "prereq_gate_implement") in edges
    assert ("prereq_gate_implement", PhaseName.IMPLEMENT.value) in edges


def test_critical_reviewed_task_adversarial_escalates_to_human_review():
    g = build_workflow_graph("critical_reviewed_task").get_graph()
    out = {e.target for e in g.edges if e.source == ADV}
    assert "human_review" in out  # reviewed type pauses for a human
    assert PhaseName.PLAN.value in out  # needs_revision loopback
    assert "__end__" in out  # passed → awaiting approval


# ── Result mapper + router logic ─────────────────────────────────────────────


def _map(agent_status, *, prior=0, maxadv=2, blocker=None):
    sub = {
        "agent_result": {
            "status": agent_status,
            "tier": "adversarial",
            "reason": "r",
            "suggestions": ["s"],
            "blocker_category": blocker,
        },
        "phase_status": agent_status,
    }
    parent = {
        "adversarial_retry_count": prior,
        "max_adversarial_retries": maxadv,
        # Critic accounting that MUST stay untouched.
        "retry_count": {"plan": 5},
        "last_critic_review": {"phase": "plan", "status": "passed"},
    }
    base = _adversarial_result_mapper(sub, parent)
    route = adversarial_router({**parent, **base})
    return base, route


def test_passed_proceeds_and_sets_completion_flag():
    base, route = _map("passed")
    assert route == "passed"
    assert base["adversarial_plan_completed"] is True


def test_needs_revision_within_budget_loops_and_counts():
    base, route = _map("needs_revision", prior=0, maxadv=2)
    assert route == "needs_revision"
    assert base["adversarial_retry_count"] == 1
    assert base["status"] == "running"


def test_needs_revision_at_budget_escalates():
    base, route = _map("needs_revision", prior=2, maxadv=2)
    assert route == "needs_review"
    assert base["needs_review_kind"] == "adversarial_exhausted"
    assert base["needs_review_phase"] == PhaseName.PLAN.value


def test_needs_review_escalates_as_flagged():
    base, route = _map("needs_review")
    assert route == "needs_review"
    assert base["needs_review_kind"] == "adversarial_flagged"


def test_spec_contradiction_targets_specify():
    base, route = _map("needs_review", blocker="spec_contradiction")
    assert route == "needs_review"
    assert base["needs_review_kind"] == "spec_amendment"
    assert base["needs_review_phase"] == PhaseName.SPECIFY.value


def test_error_fails():
    base, route = _map("error")
    assert route == "failed"
    assert base["status"] == "failed"


def test_result_mapper_never_touches_critic_accounting():
    """Isolation guarantee: the adversarial budget and the critic budget are
    independent — the mapper must not write retry_count or last_critic_review."""
    for status in ("passed", "needs_revision", "needs_review"):
        base, _ = _map(status, prior=0, maxadv=2)
        assert "retry_count" not in base
        assert "last_critic_review" not in base
        assert "last_adversarial_review" in base


# ── Plan rework feedback (Option A) ──────────────────────────────────────────


def test_plan_mapper_uses_adversarial_review_on_adversarial_loopback():
    """An adversarial-driven loopback drives rework off the adversarial round
    and surfaces the adversarial verdict, not the critic's stale PASS."""
    parent = {
        "work_id": "wk-adv",
        "workspace_root": ".",
        "current_phase": PhaseName.ADVERSARIAL.value,
        "retry_count": {"plan": 0},
        "adversarial_retry_count": 1,
        "last_critic_review": {"phase": "plan", "status": "passed", "reason": "ok"},
        "last_adversarial_review": {
            "phase": "plan",
            "status": "needs_revision",
            "tier": "adversarial",
            "reason": "unhandled failure mode",
            "suggestions": ["add a rollback slice"],
            "attempt": 1,
        },
    }
    mapped = _plan_state_mapper(parent, None)
    assert mapped["retry_count"] == 1  # adversarial round → rework mode
    assert mapped["last_critic_review"]["tier"] == "adversarial"
    assert mapped["last_critic_review"]["status"] == "needs_revision"


def test_plan_mapper_uses_critic_review_on_critic_loopback():
    parent = {
        "work_id": "wk-crit",
        "workspace_root": ".",
        "current_phase": PhaseName.CRITIC.value,
        "retry_count": {"plan": 2},
        "adversarial_retry_count": 1,
        "last_critic_review": {"phase": "plan", "status": "needs_revision", "tier": "agent"},
        "last_adversarial_review": {"phase": "plan", "status": "needs_revision", "tier": "adversarial"},
    }
    mapped = _plan_state_mapper(parent, None)
    assert mapped["retry_count"] == 2  # critic round drives rework
    assert mapped["last_critic_review"]["tier"] == "agent"
