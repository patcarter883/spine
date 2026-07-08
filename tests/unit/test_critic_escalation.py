"""Tests for early critic escalation — stagnation, spec contradiction, budget.

Covers the _critic_result_mapper convergence wiring and the critic_router edge
decision so the two stay in agreement. Regression target: trace 019ed383,
where three identical plan rejections burned the whole retry budget before a
forced human escalation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.models.enums import PhaseName, ReviewStatus
from spine.workflow.critic_review import critic_router
from spine.workflow.compose import _critic_result_mapper

PLAN = PhaseName.PLAN.value
SPECIFY = PhaseName.SPECIFY.value

_ASKS = [
    "Add get_embedding_providers and get_reranker_providers methods",
    "Split the large config-ui-extensions slice into two",
    "Resolve phase_max_retries vs max_critic_retries schema mismatch",
]


def _agent_result(status, *, suggestions=None, reason="r", blocker=None):
    out = {
        "status": status,
        "tier": "agent",
        "reason": reason,
        "suggestions": suggestions or [],
    }
    if blocker is not None:
        out["blocker_category"] = blocker
    return out


def _subgraph_result(status, **kw):
    return {"agent_result": _agent_result(status, **kw), "phase_status": status}


def _merge(parent, base):
    """Emulate the LangGraph reducers the parent graph would apply."""
    merged = {**parent, **base}
    rc = {**parent.get("retry_count", {}), **base.get("retry_count", {})}
    merged["retry_count"] = rc
    return merged


def _run(parent, subgraph_result, reviewed_phase=PLAN):
    base = _critic_result_mapper(reviewed_phase)(subgraph_result, parent)
    state = _merge(parent, base)
    route = critic_router(state)
    return base, state, route


class TestFirstRevisionRound:
    def test_revision_with_budget_reworks(self):
        parent = {"max_retries": 3, "retry_count": {}}
        base, state, route = _run(
            parent, _subgraph_result(ReviewStatus.NEEDS_REVISION.value, suggestions=_ASKS)
        )
        assert route == "needs_revision"
        assert base["status"] == "running"
        assert base["last_critic_review"]["stagnation_streak"] == 0
        assert base["last_critic_review"]["attempt"] == 1
        assert state["retry_count"][PLAN] == 1
        assert "needs_review_phase" not in base


class TestStagnationEscalation:
    def test_second_consecutive_repeat_escalates_early(self):
        # Prior round already a repeat (streak 1); the same asks recur again.
        parent = {
            "max_retries": 5,  # budget remains — escalation is by stagnation only
            "retry_count": {PLAN: 1},
            "last_critic_review": {
                "phase": PLAN,
                "status": ReviewStatus.NEEDS_REVISION.value,
                "suggestions": _ASKS,
                "stagnation_streak": 1,
            },
        }
        base, state, route = _run(
            parent, _subgraph_result(ReviewStatus.NEEDS_REVISION.value, suggestions=_ASKS)
        )
        assert route == "needs_review"
        assert base["status"] == "needs_review"
        assert base["needs_review_phase"] == PLAN
        assert base["needs_review_kind"] == "stagnation"
        assert base["last_critic_review"]["stagnation_streak"] == 2
        assert base["last_critic_review"]["unaddressed_points"] == _ASKS

    def test_first_repeat_still_reworks_with_delta(self):
        # One repeat is a warning round: rework continues but the unaddressed
        # delta is recorded for the rework prompt.
        parent = {
            "max_retries": 5,
            "retry_count": {PLAN: 1},
            "last_critic_review": {
                "phase": PLAN,
                "status": ReviewStatus.NEEDS_REVISION.value,
                "suggestions": _ASKS,
                "stagnation_streak": 0,
            },
        }
        base, state, route = _run(
            parent, _subgraph_result(ReviewStatus.NEEDS_REVISION.value, suggestions=_ASKS)
        )
        assert route == "needs_revision"
        assert base["last_critic_review"]["stagnation_streak"] == 1
        assert base["last_critic_review"]["unaddressed_points"] == _ASKS

    def test_progress_resets_streak(self):
        parent = {
            "max_retries": 5,
            "retry_count": {PLAN: 1},
            "last_critic_review": {
                "phase": PLAN,
                "status": ReviewStatus.NEEDS_REVISION.value,
                "suggestions": _ASKS,
                "stagnation_streak": 1,
            },
        }
        base, state, route = _run(
            parent,
            _subgraph_result(
                ReviewStatus.NEEDS_REVISION.value,
                suggestions=["Add unit tests for the new UIApi methods"],
            ),
        )
        assert route == "needs_revision"
        assert base["last_critic_review"]["stagnation_streak"] == 0
        assert base["last_critic_review"]["unaddressed_points"] == []


class TestChurnEscalation:
    """trace 019f01c2: the critic shifted goalposts every round (fresh asks),
    so the repeat-based stagnation streak never moved and the loop burned all
    five retries. Churn detection escalates after two consecutive shifts."""

    def test_second_consecutive_shift_escalates_early(self):
        # Prior round already a goalpost shift (churn 1); a wholly different
        # verdict shifts again — escalate despite remaining budget.
        parent = {
            "max_retries": 5,
            "retry_count": {PLAN: 2},
            "last_critic_review": {
                "phase": PLAN,
                "status": ReviewStatus.NEEDS_REVISION.value,
                "suggestions": ["Consolidate the embeddings and rerankers sections"],
                "churn_streak": 1,
            },
        }
        base, state, route = _run(
            parent,
            _subgraph_result(
                ReviewStatus.NEEDS_REVISION.value,
                suggestions=["Split phase-provider-config-ui into smaller slices"],
            ),
        )
        assert route == "needs_review"
        assert base["status"] == "needs_review"
        assert base["needs_review_phase"] == PLAN
        assert base["needs_review_kind"] == "non_convergence"
        assert base["last_critic_review"]["churn_streak"] == 2

    def test_first_shift_still_reworks(self):
        # A single goalpost shift is a warning round — rework continues.
        parent = {
            "max_retries": 5,
            "retry_count": {PLAN: 1},
            "last_critic_review": {
                "phase": PLAN,
                "status": ReviewStatus.NEEDS_REVISION.value,
                "suggestions": _ASKS,
                "churn_streak": 0,
            },
        }
        base, state, route = _run(
            parent,
            _subgraph_result(
                ReviewStatus.NEEDS_REVISION.value,
                suggestions=["Consolidate the embeddings and rerankers sections"],
            ),
        )
        assert route == "needs_revision"
        assert base["status"] == "running"
        assert base["last_critic_review"]["churn_streak"] == 1
        assert base["last_critic_review"]["escalation_kind"] is None

    def test_repeat_does_not_count_as_churn(self):
        # A repeat verdict drives the stagnation streak, not churn.
        parent = {
            "max_retries": 5,
            "retry_count": {PLAN: 1},
            "last_critic_review": {
                "phase": PLAN,
                "status": ReviewStatus.NEEDS_REVISION.value,
                "suggestions": _ASKS,
                "churn_streak": 1,
                "stagnation_streak": 0,
            },
        }
        base, state, route = _run(
            parent, _subgraph_result(ReviewStatus.NEEDS_REVISION.value, suggestions=_ASKS)
        )
        assert base["last_critic_review"]["churn_streak"] == 0
        assert base["last_critic_review"]["stagnation_streak"] == 1


class TestSpecContradiction:
    def test_spec_contradiction_routes_to_spec_amendment(self):
        parent = {"max_retries": 3, "retry_count": {}}
        base, state, route = _run(
            parent,
            _subgraph_result(
                ReviewStatus.NEEDS_REVIEW.value,
                suggestions=["Amend the spec to permit a phase_max_retries field"],
                blocker="spec_contradiction",
            ),
        )
        assert route == "needs_review"
        assert base["status"] == "needs_review"
        # Reworking targets SPECIFY, not PLAN — the plan can't fix a spec gap.
        assert base["needs_review_phase"] == SPECIFY
        assert base["needs_review_kind"] == "spec_amendment"

    def test_spec_contradiction_escalates_even_on_revision_status(self):
        # Defensive: blocker flag wins even if the critic returned NEEDS_REVISION.
        parent = {"max_retries": 3, "retry_count": {}}
        base, _, route = _run(
            parent,
            _subgraph_result(
                ReviewStatus.NEEDS_REVISION.value, blocker="spec_contradiction"
            ),
        )
        assert route == "needs_review"
        assert base["needs_review_phase"] == SPECIFY
        assert base["needs_review_kind"] == "spec_amendment"


class TestRetriesExhausted:
    def test_exhausted_budget_escalates_and_sets_phase(self):
        # new_attempt == max_retries; verdict differs so stagnation isn't the cause.
        parent = {"max_retries": 3, "retry_count": {PLAN: 2}}
        base, state, route = _run(
            parent,
            _subgraph_result(
                ReviewStatus.NEEDS_REVISION.value, suggestions=["something new"]
            ),
        )
        assert route == "needs_review"
        assert base["status"] == "needs_review"
        # Regression: exhausted-retry escalations now set needs_review_phase.
        assert base["needs_review_phase"] == PLAN
        assert base["needs_review_kind"] == "retries_exhausted"
        assert state["retry_count"][PLAN] == 3


class TestPassed:
    def test_passed_proceeds(self):
        parent = {"max_retries": 3, "retry_count": {PLAN: 1}}
        base, state, route = _run(parent, _subgraph_result(ReviewStatus.PASSED.value))
        assert route == "passed"
        assert base["status"] == "running"
        assert "needs_review_phase" not in base
        assert base["last_critic_review"]["escalate"] is False


class TestReworkDelta:
    """rec 3: the rework prompt hoists still-unaddressed asks to the top."""

    def test_unaddressed_points_rendered_first(self):
        from spine.workflow.subgraphs.exploration_subgraph import (
            _render_rework_feedback,
        )

        lcr = {
            "phase": PLAN,
            "status": "needs_revision",
            "tier": "agent",
            "attempt": 2,
            "reason": "critical gaps remain",
            "suggestions": ["Add get_embedding_providers() to ui-api slice"],
            "unaddressed_points": [
                "Add get_embedding_providers and get_reranker_providers methods",
                "Split the large config-ui-extensions slice into two",
            ],
        }
        out = _render_rework_feedback(lcr, [])
        assert "STILL NOT ADDRESSED" in out
        # Hoisted above the general verdict line.
        assert out.index("STILL NOT ADDRESSED") < out.index("attempt 2")
        for point in lcr["unaddressed_points"]:
            assert point in out

    def test_no_delta_block_on_first_round(self):
        from spine.workflow.subgraphs.exploration_subgraph import (
            _render_rework_feedback,
        )

        lcr = {
            "phase": PLAN,
            "status": "needs_revision",
            "tier": "agent",
            "attempt": 1,
            "reason": "first round",
            "suggestions": ["do x"],
        }
        out = _render_rework_feedback(lcr, [])
        assert "STILL NOT ADDRESSED" not in out
        assert "do x" in out


class TestGuardVerdictAttemptAccounting:
    """A truncated-critic guard verdict must not consume a retry attempt.

    Regression: trace 019f404a (work 06c2d55c resume) — the critic's own
    response truncated at its token cap, the guard verdict routed the plan
    into a ~6-minute regeneration AND pushed retry_count to 5-of-5, so the
    next real critic round escalated. Attempt accounting is now verdict-
    source-aware like the streaks: the first guard round is free, but
    consecutive guard rounds count so a critic that always truncates still
    exhausts the budget instead of looping forever.
    """

    def _guard_result(self):
        return {
            "agent_result": {
                "status": ReviewStatus.NEEDS_REVISION.value,
                "tier": "agent",
                "verdict_source": "guard",
                "reason": "Critic response was truncated at the token limit",
                "suggestions": [],
            },
            "phase_status": ReviewStatus.NEEDS_REVISION.value,
        }

    def test_first_guard_round_is_free(self):
        # At attempt 4-of-5 a guard verdict must not reach the threshold.
        parent = {
            "max_retries": 5,
            "retry_count": {PLAN: 4},
            "last_critic_review": {
                "phase": PLAN,
                "status": ReviewStatus.NEEDS_REVISION.value,
                "verdict_source": "agent",
                "suggestions": _ASKS,
            },
        }
        base, state, route = _run(parent, self._guard_result())
        assert route == "needs_revision"
        assert base["status"] == "running"
        assert base["last_critic_review"]["attempt"] == 4
        assert state["retry_count"][PLAN] == 4
        assert "needs_review_phase" not in base

    def test_consecutive_guard_rounds_still_count(self):
        # Two guard rounds in a row: the second consumes an attempt, so a
        # critic that always truncates cannot loop indefinitely.
        parent = {
            "max_retries": 5,
            "retry_count": {PLAN: 4},
            "last_critic_review": {
                "phase": PLAN,
                "status": ReviewStatus.NEEDS_REVISION.value,
                "verdict_source": "guard",
                "suggestions": [],
            },
        }
        base, state, route = _run(parent, self._guard_result())
        assert base["last_critic_review"]["attempt"] == 5
        assert base["status"] == "needs_review"
        assert base["needs_review_kind"] == "retries_exhausted"
        assert route == "needs_review"


class TestSymbolNonexistenceDemotion:
    """An agent-tier spec_contradiction alleging a symbol doesn't exist,
    while the deterministic gate PASSED the round, is stale-feedback
    re-litigation, not ground truth (run a24d4fca: 'self._base is not
    defined' parked the run one round after the gate began resolving that
    exact attribute). It must demote to a normal revision round.
    """

    def _contradiction(self, reason):
        return {
            "agent_result": {
                "status": ReviewStatus.NEEDS_REVIEW.value,
                "tier": "agent",
                "reason": reason,
                "suggestions": [],
                "blocker_category": "spec_contradiction",
            },
            "phase_status": ReviewStatus.NEEDS_REVIEW.value,
            # gate passed this round
            "reference_gate_result": None,
        }

    def test_nonexistence_claim_with_clean_gate_demotes(self):
        parent = {"max_retries": 5, "retry_count": {}}
        base, state, route = _run(
            parent,
            self._contradiction(
                "The plan references `self._base`, but this attribute is "
                "not defined in the codebase and scope exclusions prevent "
                "adding it."
            ),
        )
        assert route == "needs_revision"
        assert base["status"] == "running"
        assert base["last_critic_review"]["blocker_category"] is None
        assert base["last_critic_review"]["escalate"] is False
        assert "DEMOTED" in base["feedback"][0]["reason"]

    def test_contradiction_without_nonexistence_claim_still_escalates(self):
        parent = {"max_retries": 5, "retry_count": {}}
        base, state, route = _run(
            parent,
            self._contradiction(
                "The spec's scope_exclusions forbid backend changes, but the "
                "requirement demands a new API endpoint."
            ),
        )
        assert route == "needs_review"
        assert base["status"] == "needs_review"
        assert base["needs_review_kind"] == "spec_amendment"

    def test_nonexistence_claim_with_gate_findings_still_escalates(self):
        # When the gate ITSELF flagged symbols this round, the claim has
        # ground truth behind it — escalation stands.
        parent = {"max_retries": 5, "retry_count": {}}
        sub = self._contradiction(
            "Symbol `Ghost.method` does not exist in the codebase and the "
            "spec forbids creating it."
        )
        sub["reference_gate_result"] = {"dangling_leafs": ["method"]}
        base, state, route = _run(parent, sub)
        assert base["status"] == "needs_review"
        assert base["needs_review_kind"] == "spec_amendment"
