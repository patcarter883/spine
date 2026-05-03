"""Tests for persistence.py - human handoff protocol."""

import os
import sys
import tempfile

import importlib.util
spec = importlib.util.spec_from_file_location("persistence", "spine/core/persistence.py")
persistence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(persistence)

ResumeMarker = persistence.ResumeMarker
ResumeAction = persistence.ResumeAction
create_resume_marker = persistence.create_resume_marker
Checkpoint = persistence.Checkpoint
RecoveryStrategy = persistence.RecoveryStrategy
ExecutionPlan = persistence.ExecutionPlan
Context = persistence.Context
ContinuityManager = persistence.ContinuityManager


class TestResumeMarker:
    def test_init_defaults(self):
        marker = ResumeMarker()
        assert marker.resume_version == "1.0"
        assert marker.work_item_id == ""
        assert marker.checkpoint_ref == ""
        assert marker.saved_at != ""
        assert marker.reason == ""
        assert marker.next_action == ""

    def test_to_dict(self):
        marker = ResumeMarker(
            work_item_id="test_123",
            checkpoint_ref=".spine/checkpoints/test.json",
            reason="timeout",
        )
        result = marker.to_dict()
        assert result["work_item_id"] == "test_123"
        assert result["checkpoint_ref"] == ".spine/checkpoints/test.json"
        assert result["reason"] == "timeout"
        assert "saved_at" in result

    def test_from_dict(self):
        data = {
            "resume_version": "1.0",
            "work_item_id": "feat_auth",
            "checkpoint_ref": ".spine/state/check.json",
            "saved_at": "2024-01-15T14:22:33Z",
            "reason": "timeout",
            "next_action": "resume_execution",
            "human_instructions": {"message": "test"},
        }
        marker = ResumeMarker.from_dict(data)
        assert marker.work_item_id == "feat_auth"
        assert marker.checkpoint_ref == ".spine/state/check.json"
        assert marker.reason == "timeout"

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, ".spine", "resume.json")
            marker = ResumeMarker(
                work_item_id="feat_20240115",
                checkpoint_ref=".spine/state/checkpoints/exec.json",
                reason="timeout",
            )
            marker.save(path)

            assert os.path.exists(path)
            loaded = ResumeMarker.load(path)
            assert loaded.work_item_id == "feat_20240115"
            assert loaded.checkpoint_ref == ".spine/state/checkpoints/exec.json"

    def test_create_handoff(self):
        marker = ResumeMarker()
        result = marker.create_handoff(
            work_item_id="feat_auth_20240115",
            checkpoint_ref=".spine/state/checkpoints/exec_20240115_1422.json",
            reason="timeout",
            phase="EXECUTION",
            active_subphases=["BACKEND"],
            pending_gates=["reviewer", "test_engineer"],
        )

        assert result.work_item_id == "feat_auth_20240115"
        assert result.checkpoint_ref == ".spine/state/checkpoints/exec_20240115_1422.json"
        assert result.reason == "timeout"
        assert result.next_action == "resume_execution"
        assert result.human_instructions["message"] == "Work paused due to timeout. Ready to resume EXECUTION phase."
        assert "swarm_state" in result.human_instructions
        assert result.human_instructions["swarm_state"]["active_subphases"] == ["BACKEND"]
        assert result.human_instructions["swarm_state"]["pending_gates"] == ["reviewer", "test_engineer"]
        assert len(result.human_instructions["options"]) == 4
        assert result.human_instructions["options"][0]["action"] == ResumeAction.RESUME.value

    def test_load_nonexistent_returns_none(self):
        marker = ResumeMarker.load("/nonexistent/path/resume.json")
        assert marker is None


class TestCreateResumeMarker:
    def test_factory_function(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, ".spine", "resume.json")
            marker = create_resume_marker(
                work_item_id="feat_test",
                checkpoint_ref=".spine/check.json",
                reason="manual_pause",
                phase="PLANNING",
                active_subphases=["ANALYZE", "TECH_RESEARCH"],
                pending_gates=[],
                path=path,
            )

            assert marker.work_item_id == "feat_test"
            loaded = ResumeMarker.load(path)
            assert loaded.reason == "manual_pause"


class TestResumeAction:
    def test_action_values(self):
        assert ResumeAction.RESUME.value == "resume"
        assert ResumeAction.INSPECT.value == "inspect"
        assert ResumeAction.ADJUST.value == "adjust"
        assert ResumeAction.CANCEL.value == "cancel"


class TestRecoveryStrategy:
    def test_resume_builds_execution_plan(self):
        checkpoint = Checkpoint(
            checkpoint_id="test_ckpt",
            work_item_id="feat_test",
            phase_name="EXECUTION",
            phase_progress=1.0,
            state={
                "completed_tasks": ["task_1"],
                "failed_tasks": [],
            },
            dag={
                "execution_plan": ["task_1", "task_2", "task_3"],
                "dependencies": {
                    "task_3": ["task_1", "task_2"]
                },
                "results": {
                    "task_1": {"status": "success"}
                }
            },
            swarm_state={
                "file_reservations": {"worker-a": ["src/test.py"]},
                "pending_gates": ["reviewer"]
            }
        )

        strategy = RecoveryStrategy()
        plan = strategy.resume(checkpoint)

        assert "task_2" in plan.tasks
        assert "task_3" in plan.tasks
        assert plan.verification_needed is False
        assert plan.file_reservations == {"worker-a": ["src/test.py"]}
        assert plan.pending_gates == ["reviewer"]

    def test_excludes_completed_tasks(self):
        checkpoint = Checkpoint(
            dag={
                "execution_plan": ["a", "b", "c"],
                "dependencies": {},
                "results": {
                    "a": {"status": "success"},
                    "b": {"status": "success"}
                }
            },
            state={"completed_tasks": []}
        )

        strategy = RecoveryStrategy()
        plan = strategy.resume(checkpoint)

        assert "c" in plan.tasks
        assert "a" not in plan.tasks
        assert "b" not in plan.tasks

    def test_needs_verification_with_failed_tasks(self):
        checkpoint = Checkpoint(
            phase_progress=1.0,
            state={"failed_tasks": ["task_1"]}
        )

        strategy = RecoveryStrategy()
        plan = strategy.resume(checkpoint)

        assert plan.verification_needed is True

    def test_needs_verification_with_incomplete_progress(self):
        checkpoint = Checkpoint(
            phase_progress=0.5,
            state={}
        )

        strategy = RecoveryStrategy()
        plan = strategy.resume(checkpoint)

        assert plan.verification_needed is True

    def test_in_flight_tasks_identified(self):
        checkpoint = Checkpoint(
            dag={
                "execution_plan": ["task_1", "task_2"],
                "results": {
                    "task_1": {"status": "running"},
                    "task_2": {"status": "success"}
                }
            }
        )

        strategy = RecoveryStrategy()
        plan = strategy.resume(checkpoint)

        assert len(plan.in_flight_recovery) == 1
        assert plan.in_flight_recovery[0]["task_id"] == "task_1"
