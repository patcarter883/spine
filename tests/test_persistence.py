"""Tests for persistence.py - human handoff protocol."""

import os
import tempfile

from spine.core.persistence import ResumeMarker, ResumeAction, create_resume_marker


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
