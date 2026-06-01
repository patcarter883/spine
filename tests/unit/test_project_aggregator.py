"""Tests for the deterministic project coverage aggregator.

The aggregator must compute per-requirement coverage by normalized-EXACT string
match (never fuzzy) and treat absent checkpoint state as unverified. These tests
monkeypatch CheckpointStore so no real DB is touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.config import SpineConfig
from spine.models.types import ProjectSpec, RequirementRef, Roadmap, RoadmapPhase
import spine.persistence.checkpoint as checkpoint_mod


def _state(requirements, passed):
    """Build a fake WorkflowState channel-values dict."""
    return {
        "specification_json": json.dumps({"requirements": requirements}),
        "verification_passed": passed,
    }


@pytest.fixture
def fake_checkpoint(monkeypatch):
    """Patch CheckpointStore to return canned states from a dict keyed by work_id."""
    states: dict[str, dict | None] = {}

    class _FakeStore:
        def __init__(self, db_path=".spine/spine.db"):
            pass

        async def get_state(self, work_id):
            return states.get(work_id)

        async def close(self):
            pass

    monkeypatch.setattr(checkpoint_mod, "CheckpointStore", _FakeStore)
    return states


def _run(spec):
    import asyncio

    from spine.project.aggregator import aggregate_project_coverage

    return asyncio.run(aggregate_project_coverage(spec, SpineConfig()))


def _project(requirements, members, phases=None):
    return ProjectSpec(
        id="p",
        title="P",
        requirements=requirements,
        member_work_ids=members,
        roadmap=Roadmap(phases=phases or []),
        created_at="t",
        updated_at="t",
    )


def test_all_covering_members_verified_is_satisfied(fake_checkpoint):
    fake_checkpoint["w1"] = _state(["Build login"], passed=True)
    spec = _project([RequirementRef(id="R-001", text="Build login")], ["w1"])
    cov = _run(spec)
    assert cov["requirements"][0]["status"] == "satisfied"
    assert cov["summary"] == {"satisfied": 1, "partial": 0, "unsatisfied": 0}


def test_subset_verified_is_partial(fake_checkpoint):
    fake_checkpoint["w1"] = _state(["Build login"], passed=True)
    fake_checkpoint["w2"] = _state(["Build login"], passed=False)
    spec = _project([RequirementRef(id="R-001", text="Build login")], ["w1", "w2"])
    cov = _run(spec)
    assert cov["requirements"][0]["status"] == "partial"


def test_covering_but_none_verified_is_unsatisfied(fake_checkpoint):
    fake_checkpoint["w1"] = _state(["Build login"], passed=False)
    spec = _project([RequirementRef(id="R-001", text="Build login")], ["w1"])
    cov = _run(spec)
    assert cov["requirements"][0]["status"] == "unsatisfied"


def test_unclaimed_requirement_is_unsatisfied(fake_checkpoint):
    fake_checkpoint["w1"] = _state(["Something else"], passed=True)
    spec = _project([RequirementRef(id="R-001", text="Build login")], ["w1"])
    cov = _run(spec)
    assert cov["requirements"][0]["status"] == "unsatisfied"
    assert cov["requirements"][0]["covering"] == []


def test_missing_checkpoint_state_treated_as_unverified(fake_checkpoint):
    # w1 has no state (purged/never-run) → contributes no spec, not verified.
    spec = _project([RequirementRef(id="R-001", text="Build login")], ["w1"])
    cov = _run(spec)
    assert cov["requirements"][0]["status"] == "unsatisfied"
    assert cov["members_with_state"] == 0
    assert cov["verified_members"] == 0


def test_normalized_match_hits_but_near_miss_does_not(fake_checkpoint):
    # Whitespace/case differences match; a different wording does NOT.
    fake_checkpoint["w1"] = _state(["  build LOGIN  "], passed=True)
    fake_checkpoint["w2"] = _state(["Log user actions"], passed=True)
    spec = _project(
        [
            RequirementRef(id="R-001", text="Build login"),
            RequirementRef(id="R-002", text="Log user activity"),  # != "Log user actions"
        ],
        ["w1", "w2"],
    )
    cov = _run(spec)
    by_id = {r["id"]: r for r in cov["requirements"]}
    assert by_id["R-001"]["status"] == "satisfied"  # normalized exact match
    assert by_id["R-002"]["status"] == "unsatisfied"  # NO fuzzy match


def test_phase_rollup_derivation(fake_checkpoint):
    fake_checkpoint["w1"] = _state(["A", "B"], passed=True)
    spec = _project(
        [
            RequirementRef(id="R-001", text="A"),
            RequirementRef(id="R-002", text="B"),
            RequirementRef(id="R-003", text="C"),  # unclaimed → unsatisfied
        ],
        ["w1"],
        phases=[
            RoadmapPhase(id="M-001", title="done", requirement_ids=["R-001", "R-002"]),
            RoadmapPhase(id="M-002", title="mixed", requirement_ids=["R-002", "R-003"]),
            RoadmapPhase(id="M-003", title="empty", requirement_ids=["R-003"]),
        ],
    )
    cov = _run(spec)
    by_id = {p["id"]: p for p in cov["phases"]}
    assert by_id["M-001"]["status"] == "complete"
    assert by_id["M-002"]["status"] == "in_progress"
    assert by_id["M-003"]["status"] == "pending"
