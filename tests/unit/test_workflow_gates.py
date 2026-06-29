"""Tests for artifact gate, critic status propagation, and resume_work."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


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

    @pytest.mark.asyncio
    async def test_resume_rework_starts_from_needs_review_phase(self, monkeypatch):
        """When action='rework', resume should start from the needs_review_phase."""
        from spine.work.dispatcher import resume_work

        # work_type='task' is code-producing, so resume_work enters a mandatory
        # WorktreeSandbox (lazy `from spine.git import WorktreeSandbox`). Without
        # mocking it the sandbox runs real git (worktree add + `git checkout
        # <main>` / reset --hard) against THIS repo — flipping HEAD to main and
        # clobbering the working tree. Mock it so the test stays hermetic.
        mock_sandbox = MagicMock()
        mock_sandbox.enter.return_value = SimpleNamespace(workspace_root="/tmp/sbx")
        with patch("spine.work.dispatcher._get_work_db") as mock_db_fn, patch(
            "spine.git.WorktreeSandbox", return_value=mock_sandbox
        ):
            mock_db = MagicMock()
            mock_table = MagicMock()
            mock_table.get.return_value = {
                "id": "abc",
                "status": "needs_review",
                "work_type": "task",
                "description": "test task",
                "result": "",
            }
            mock_db.__getitem__ = lambda s, k: mock_table
            mock_db_fn.return_value = mock_db

            saved_state = {
                "needs_review_phase": "plan",
                "artifacts": {"plan": {"plan.md": "content"}},
                "feedback": [],
                "retry_count": {},
            }

            async def mock_astream(*args, **kwargs):
                yield {
                    "type": "updates",
                    "ns": (),
                    "data": {
                        "some_node": {
                            "current_phase": "plan",
                            "status": "running",
                            "artifacts": {},
                        }
                    },
                }

            mock_graph = MagicMock()
            mock_graph.astream = mock_astream

            import sys as _sys

            mock_compose = MagicMock()
            mock_compose.WORKFLOW_SEQUENCES = {
                "task": [("specify", None), ("plan", None), ("implement", None)]
            }
            mock_compose.build_workflow_graph.return_value = mock_graph

            mock_cp_mod = MagicMock()
            mock_cp = MagicMock()
            mock_cp_mod.CheckpointStore = MagicMock(return_value=mock_cp)
            mock_cp.get_checkpointer = AsyncMock()
            mock_cp.get_state = AsyncMock(return_value=saved_state)
            mock_cp.delete_state = AsyncMock(return_value=True)

            # Use monkeypatch.setitem so the ORIGINAL modules are restored at
            # teardown. The previous manual ``sys.modules.pop(...)`` deleted the
            # real cached modules, polluting any later test that had imported
            # them (e.g. test_project_aggregator imports spine.persistence.
            # checkpoint at module load) — an order-dependent failure.
            monkeypatch.setitem(_sys.modules, "spine.workflow.compose", mock_compose)
            monkeypatch.setitem(
                _sys.modules, "spine.persistence.checkpoint", mock_cp_mod
            )

            with patch("spine.work.dispatcher.ArtifactStore", MagicMock()):
                await resume_work("abc", "fix it", action="rework")
                mock_cp.delete_state.assert_called_once_with("abc")


# ── Critic router tests ──


class TestCriticRouter:
    """Tests for the critic_router conditional edge logic."""

    def test_routes_passed(self):
        from spine.workflow.critic_review import critic_router

        state = {
            "last_critic_review": {
                "phase": "plan",
                "status": "passed",
                "tier": "agent",
                "reason": "OK",
            },
        }
        assert critic_router(state) == "passed"

    def test_routes_needs_review(self):
        from spine.workflow.critic_review import critic_router

        state = {
            "last_critic_review": {
                "phase": "plan",
                "status": "needs_review",
                "tier": "agent",
                "reason": "Bad",
            },
        }
        assert critic_router(state) == "needs_review"

    def test_routes_needs_revision_within_retries(self):
        from spine.workflow.critic_review import critic_router

        state = {
            "last_critic_review": {
                "phase": "plan",
                "status": "needs_revision",
                "tier": "structural",
                "reason": "Short",
            },
            "retry_count": {"plan": 1},
            "max_retries": 3,
        }
        assert critic_router(state) == "needs_revision"

    def test_routes_needs_review_when_retries_exhausted(self):
        from spine.workflow.critic_review import critic_router

        state = {
            "last_critic_review": {
                "phase": "plan",
                "status": "needs_revision",
                "tier": "structural",
                "reason": "Short",
            },
            "retry_count": {"plan": 3},
            "max_retries": 3,
        }
        assert critic_router(state) == "needs_review"

    def test_routes_needs_revision_when_record_missing(self):
        from spine.workflow.critic_review import critic_router

        state = {"critic_reviewing": "plan"}
        # Missing last_critic_review → routes needs_revision as fallback
        assert critic_router(state) == "needs_revision"

    def test_ignores_stale_feedback_entry(self):
        from spine.workflow.critic_review import critic_router

        # feedback[-1] says PASSED (stale carryover from a prior tier), but
        # last_critic_review is authoritative and says NEEDS_REVISION.
        state = {
            "last_critic_review": {
                "phase": "plan",
                "status": "needs_revision",
                "tier": "agent",
                "reason": "Plan slices missing acceptance criteria",
            },
            "feedback": [
                {"status": "passed", "tier": "structural", "reason": "OK"},
            ],
            "retry_count": {"plan": 1},
            "max_retries": 3,
        }
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


class TestReviewedTaskHumanApprovalGate:
    """Reviewed work types MUST terminate before implement.

    The human-review gate for reviewed_task / critical_reviewed_task is the
    graph reaching END after critic_plan: the dispatcher then relabels
    "completed" → "awaiting_approval" and the user approves via
    approve_and_spawn (which spawns fresh ``task`` items for execution).

    If the sequence ever includes IMPLEMENT or VERIFY again, the human gate
    is bypassed — implementation runs before any human looks at the plan.
    This was a real regression once; these tests exist so it can't recur.
    """

    def test_reviewed_task_sequence_stops_at_critic_plan(self):
        from spine.models.enums import PhaseName
        from spine.workflow.compose import WORKFLOW_SEQUENCES

        sequence = WORKFLOW_SEQUENCES["reviewed_task"]
        phase_names = [name for name, _ in sequence]
        assert phase_names[-1] == f"{PhaseName.CRITIC.value}_plan", (
            f"reviewed_task must end at critic_plan, got: {phase_names}"
        )
        assert PhaseName.IMPLEMENT.value not in phase_names
        assert PhaseName.VERIFY.value not in phase_names

    def test_critical_reviewed_task_sequence_stops_at_adversarial_plan(self):
        from spine.models.enums import PhaseName
        from spine.workflow.compose import WORKFLOW_SEQUENCES

        sequence = WORKFLOW_SEQUENCES["critical_reviewed_task"]
        phase_names = [name for name, _ in sequence]
        # The adversarial review is the final, human-review gate for critical
        # reviewed tasks — it runs after critic_plan and before approval.
        assert phase_names[-1] == f"{PhaseName.ADVERSARIAL.value}_plan", (
            f"critical_reviewed_task must end at adversarial_plan, got: {phase_names}"
        )
        assert f"{PhaseName.CRITIC.value}_plan" in phase_names
        assert f"{PhaseName.CRITIC.value}_specify" not in phase_names
        assert PhaseName.IMPLEMENT.value not in phase_names
        assert PhaseName.VERIFY.value not in phase_names

    def test_reviewed_task_graph_has_no_implement_node(self):
        """Compiled graph for reviewed types must not contain implement/verify nodes."""
        from spine.models.enums import PhaseName
        from spine.workflow.compose import build_workflow_graph

        for work_type in ("reviewed_task", "critical_reviewed_task"):
            graph = build_workflow_graph(work_type)
            nodes = set(graph.get_graph().nodes.keys())
            assert PhaseName.IMPLEMENT.value not in nodes, (
                f"{work_type}: implement node leaked into compiled graph — "
                f"the human-review gate would be bypassed. Nodes: {sorted(nodes)}"
            )
            assert PhaseName.VERIFY.value not in nodes, (
                f"{work_type}: verify node leaked into compiled graph. "
                f"Nodes: {sorted(nodes)}"
            )
