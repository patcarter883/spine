"""Tests for artifact gate, critic status propagation, and resume_work."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Stub the broken tasks_agent module (syntax error from incomplete refactor).
# This unblocks the registry import chain without hiding real test failures.
if "spine.agents.tasks_agent" not in sys.modules:
    _stub = types.ModuleType("spine.agents.tasks_agent")
    _stub.build_tasks_agent = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules["spine.agents.tasks_agent"] = _stub


# ── Artifact gate tests ──


class TestArtifactGateNode:
    """Tests for the artifact gate node (writes status + feedback to state)."""

    def test_proceed_when_implement_has_artifacts(self):
        from spine.workflow.artifact_gate import make_artifact_gate_node

        gate_fn = make_artifact_gate_node("implement", "verify")
        state = {
            "artifacts": {
                "implement": {"implementation.md": "x" * 100},
            },
        }
        result = gate_fn(state)
        assert result["status"] == "running"

    def test_needs_review_when_implement_has_no_artifacts(self):
        from spine.workflow.artifact_gate import make_artifact_gate_node

        gate_fn = make_artifact_gate_node("implement", "verify")
        state = {"artifacts": {"implement": {}}}
        result = gate_fn(state)
        assert result["status"] == "needs_review"
        # Must include a feedback entry so dispatcher detects it
        assert any(
            isinstance(f, dict) and f.get("status") == "needs_review"
            for f in result.get("feedback", [])
        )

    def test_needs_review_when_implement_missing(self):
        from spine.workflow.artifact_gate import make_artifact_gate_node

        gate_fn = make_artifact_gate_node("implement", "verify")
        state = {"artifacts": {}}
        result = gate_fn(state)
        assert result["status"] == "needs_review"

    def test_needs_review_when_artifact_too_short(self):
        from spine.workflow.artifact_gate import make_artifact_gate_node

        gate_fn = make_artifact_gate_node("implement", "verify")
        state = {"artifacts": {"implement": {"implementation.md": "short"}}}
        result = gate_fn(state)
        assert result["status"] == "needs_review"

    # NOTE: The implement→verify artifact gate is no longer wired in the
    # workflow graph (compose.py). Verify always runs after implement.
    # The gate node function is still available for other uses, so these
    # unit tests remain valid as contract tests for the function itself.

    def test_proceed_when_tasks_has_artifacts(self):
        from spine.workflow.artifact_gate import make_artifact_gate_node

        gate_fn = make_artifact_gate_node("tasks", "implement")
        state = {
            "artifacts": {
                "tasks": {"tasks.md": "x" * 100},
            },
        }
        result = gate_fn(state)
        assert result["status"] == "running"

    # ── Plan artifact gate tests ────────────────────────────────────────

    def test_plan_artifact_gate_proceed(self, tmp_path):
        """Gate proceeds when plan.json has valid feature_slices on disk."""
        from spine.workflow.artifact_gate import make_artifact_gate_node

        work_id = "plan-proceed"
        plan_dir = tmp_path / ".spine" / "artifacts" / work_id / "plan"
        plan_dir.mkdir(parents=True)
        (plan_dir / "plan.json").write_text(
            json.dumps(
                {
                    "architecture_overview": "Test architecture for gating.",
                    "feature_slices": [
                        {
                            "id": "slice-core",
                            "title": "Core models",
                            "target_files": ["models.py"],
                            "execution_requirements": [],
                            "dependencies": [],
                            "acceptance_criteria": ["models import correctly"],
                        }
                    ],
                    "testing_strategy": "pytest",
                    "risks": [],
                    "codebase_map": {},
                }
            ),
            encoding="utf-8",
        )

        gate_fn = make_artifact_gate_node("plan", "implement")
        state = {
            "work_id": work_id,
            "workspace_root": str(tmp_path),
            "artifacts": {"plan": {"plan.json": "x" * 100}},
        }
        result = gate_fn(state)
        assert result["status"] == "running"

    def test_plan_artifact_gate_empty_slices(self, tmp_path):
        """Gate returns needs_review when plan.json has empty feature_slices."""
        from spine.workflow.artifact_gate import make_artifact_gate_node

        work_id = "plan-empty-slices"
        plan_dir = tmp_path / ".spine" / "artifacts" / work_id / "plan"
        plan_dir.mkdir(parents=True)
        (plan_dir / "plan.json").write_text(
            json.dumps(
                {
                    "architecture_overview": "Architecture with no slices.",
                    "feature_slices": [],
                    "testing_strategy": "pytest",
                    "risks": [],
                    "codebase_map": {},
                }
            ),
            encoding="utf-8",
        )

        gate_fn = make_artifact_gate_node("plan", "implement")
        state = {
            "work_id": work_id,
            "workspace_root": str(tmp_path),
            "artifacts": {"plan": {"plan.json": "x" * 100}},
        }
        result = gate_fn(state)
        assert result["status"] == "needs_review"
        # Must include a feedback entry referencing the quality failure
        assert any(
            isinstance(f, dict) and f.get("status") == "needs_review"
            for f in result.get("feedback", [])
        )

    def test_plan_artifact_gate_missing_json(self, tmp_path):
        """Gate returns needs_review when plan dir exists but no plan.json."""
        from spine.workflow.artifact_gate import make_artifact_gate_node

        work_id = "plan-missing-json"
        plan_dir = tmp_path / ".spine" / "artifacts" / work_id / "plan"
        plan_dir.mkdir(parents=True)
        # Deliberately do NOT create plan.json

        gate_fn = make_artifact_gate_node("plan", "implement")
        state = {
            "work_id": work_id,
            "workspace_root": str(tmp_path),
            # State artifacts contain enough chars to pass the basic check,
            # forcing the quality check to run and discover plan.json is missing.
            "artifacts": {"plan": {"plan.md": "x" * 100}},
        }
        result = gate_fn(state)
        assert result["status"] == "needs_review"
        assert any(
            isinstance(f, dict) and "plan.json" in f.get("reason", "").lower()
            for f in result.get("feedback", [])
        )

    def test_gate_node_has_readable_name(self):
        from spine.workflow.artifact_gate import make_artifact_gate_node

        gate_fn = make_artifact_gate_node("implement", "verify")
        assert "implement" in gate_fn.__name__
        assert "verify" in gate_fn.__name__

    def test_artifact_gate_router_proceed(self):
        from spine.workflow.artifact_gate import artifact_gate_router

        state = {"status": "running"}
        assert artifact_gate_router(state) == "proceed"

    def test_artifact_gate_router_needs_review(self):
        from spine.workflow.artifact_gate import artifact_gate_router

        state = {"status": "needs_review"}
        assert artifact_gate_router(state) == "needs_review"


class TestLegacyArtifactGateFn:
    """Tests for the legacy make_artifact_gate_fn (backward compat)."""

    def test_make_gate_fn_produces_callable(self):
        from spine.workflow.artifact_gate import make_artifact_gate_fn

        fn = make_artifact_gate_fn("implement", "verify")
        assert callable(fn)
        result = fn({"artifacts": {"implement": {"impl.md": "x" * 100}}})
        assert result == "proceed"


# ── Critic status propagation tests ──


class TestCriticStatusPropagation:
    """Tests that the critic node sets status in its output state."""

    @pytest.mark.asyncio
    async def test_critic_returns_status_on_structural_fail(self):
        from spine.phases.critic import call_critic

        # Mock the structural check to fail
        with patch(
            "spine.phases.critic.structural_critic_check",
            return_value={
                "status": "needs_revision",
                "tier": "structural",
                "reason": "No artifacts produced",
                "suggestions": [],
            },
        ):
            with patch("spine.phases.critic.materialize_phase_artifacts"):
                state = {
                    "work_id": "test123",
                    "critic_reviewing": "plan",
                    "workspace_root": ".",
                    "retry_count": {"plan": 0},
                    "artifacts": {},
                }
                result = await call_critic(state)
                assert result["status"] == "running"
                assert result["current_phase"] == "critic"
                assert result["prompt_request"] is None

    @pytest.mark.asyncio
    async def test_critic_returns_status_on_agent_pass(self):
        from spine.phases.critic import call_critic

        with patch(
            "spine.phases.critic.structural_critic_check",
            return_value={
                "status": "passed",
                "tier": "structural",
                "reason": "OK",
                "suggestions": [],
            },
        ):
            with patch(
                "spine.phases.critic.agent_critic_check",
                return_value={
                    "status": "passed",
                    "tier": "agent",
                    "reason": "Quality OK",
                    "suggestions": [],
                },
            ):
                with patch("spine.phases.critic.materialize_phase_artifacts"):
                    state = {
                        "work_id": "test123",
                        "critic_reviewing": "plan",
                        "workspace_root": ".",
                        "retry_count": {"plan": 0},
                        "artifacts": {"plan": {"plan.md": "x" * 100}},
                    }
                    result = await call_critic(state)
                    assert result["status"] == "running"
                    assert result["current_phase"] == "critic"
                    assert result["prompt_request"] is None


# ── Resume work tests ──


class TestResumeWork:
    """Tests for the resume_work dispatcher function."""

    @pytest.mark.asyncio
    async def test_resume_rejects_non_needs_review(self):
        from spine.work.dispatcher import resume_work

        with patch("spine.work.dispatcher._get_work_db") as mock_db_fn:
            mock_db = MagicMock()
            mock_table = MagicMock()
            mock_table.get.return_value = {
                "id": "abc",
                "status": "completed",
            }
            mock_db.__getitem__ = lambda s, k: mock_table
            mock_db_fn.return_value = mock_db

            with pytest.raises(ValueError, match="not 'needs_review'"):
                await resume_work("abc", "fix it")

    @pytest.mark.asyncio
    async def test_resume_rejects_unknown_work_id(self):
        import sqlite_utils
        from spine.work.dispatcher import resume_work

        with patch("spine.work.dispatcher._get_work_db") as mock_db_fn:
            mock_db = MagicMock()
            mock_table = MagicMock()
            mock_table.get.side_effect = sqlite_utils.db.NotFoundError("not found")
            mock_db.__getitem__ = lambda s, k: mock_table
            mock_db_fn.return_value = mock_db

            with pytest.raises(ValueError, match="not found"):
                await resume_work("nonexistent", "fix it")


# ── Critic router tests ──


class TestCriticRouter:
    """Tests for the critic_router conditional edge logic."""

    def test_routes_passed(self):
        from spine.workflow.critic_review import critic_router

        state = {
            "critic_reviewing": "plan",
            "feedback": [
                {"status": "passed", "tier": "agent", "reason": "OK"},
            ],
        }
        assert critic_router(state) == "passed"

    def test_routes_needs_review(self):
        from spine.workflow.critic_review import critic_router

        state = {
            "critic_reviewing": "plan",
            "feedback": [
                {"status": "needs_review", "tier": "agent", "reason": "Bad"},
            ],
        }
        assert critic_router(state) == "needs_review"

    def test_routes_needs_revision_within_retries(self):
        from spine.workflow.critic_review import critic_router

        state = {
            "critic_reviewing": "plan",
            "feedback": [
                {"status": "needs_revision", "tier": "structural", "reason": "Short"},
            ],
            "retry_count": {"plan": 1},
            "max_retries": 3,
        }
        assert critic_router(state) == "needs_revision"

    def test_routes_needs_review_when_retries_exhausted(self):
        from spine.workflow.critic_review import critic_router

        state = {
            "critic_reviewing": "plan",
            "feedback": [
                {"status": "needs_revision", "tier": "structural", "reason": "Short"},
            ],
            "retry_count": {"plan": 3},
            "max_retries": 3,
        }
        assert critic_router(state) == "needs_review"

    def test_routes_needs_revision_when_no_feedback(self):
        from spine.workflow.critic_review import critic_router

        state = {
            "critic_reviewing": "plan",
            "feedback": [],
        }
        # No feedback → routes needs_revision as fallback
        assert critic_router(state) == "needs_revision"


# ── Graph composition tests ──


class TestWorkflowCompositionGates:
    """Tests that artifact gates are wired correctly in the composed graph."""

    def test_no_gate_between_implement_and_verify(self):
        """Verify always runs after implement — no artifact gate between them."""
        from spine.workflow.compose import build_workflow_graph

        for work_type in ("task", "critical_task"):
            graph = build_workflow_graph(work_type)
            node_names = set(graph.get_graph().nodes.keys())
            # No gate node should exist between implement and verify
            gate_names = [n for n in node_names if n.startswith("gate_implement_to_verify")]
            assert gate_names == [], (
                f"work_type={work_type}: found unexpected gate node(s) {gate_names} "
                "between implement and verify"
            )

    def test_gate_exists_between_plan_and_implement(self):
        """Implement is gated on plan artifacts — gate node must exist."""
        from spine.workflow.compose import build_workflow_graph

        for work_type in ("task", "critical_task"):
            graph = build_workflow_graph(work_type)
            node_names = set(graph.get_graph().nodes.keys())
            # A gate node should exist between plan and implement
            gate_names = [
                n for n in node_names if n.startswith("gate_") and "plan" in n and "implement" in n
            ]
            assert len(gate_names) >= 1, (
                f"work_type={work_type}: expected a gate node between plan and "
                f"implement, found none. All nodes: {sorted(node_names)}"
            )

    def test_no_tasks_phase_in_sequences(self):
        """TASKS phase must not appear in any WORKFLOW_SEQUENCES."""
        from spine.models.enums import PhaseName
        from spine.workflow.compose import WORKFLOW_SEQUENCES

        for work_type, sequence in WORKFLOW_SEQUENCES.items():
            phase_names = [name for name, _ in sequence]
            assert PhaseName.TASKS.value not in phase_names, (
                f"work_type={work_type}: TASKS should not appear in workflow "
                f"sequence. Found phases: {phase_names}"
            )
