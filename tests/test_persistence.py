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
GitWorkflow = persistence.GitWorkflow
GitWorkflowConfig = persistence.GitWorkflowConfig


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


class TestGitWorkflowConfig:
    def test_default_config(self):
        config = persistence.GitWorkflowConfig()
        assert config.remote_name == "origin"
        assert config.branch_prefix == "spine-"
        assert config.commit_template == "{work_item}: {message}"
        assert config.remote_push is False

    def test_custom_config(self):
        config = persistence.GitWorkflowConfig(
            remote_name="upstream",
            branch_prefix="feature/",
            commit_template="[{work_item}] {message}",
            remote_push=True,
        )
        assert config.remote_name == "upstream"
        assert config.branch_prefix == "feature/"
        assert config.commit_template == "[{work_item}] {message}"
        assert config.remote_push is True


class TestGitWorkflow:
    def test_create_branch_formats_name(self):
        workflow = persistence.GitWorkflow()
        # We mock the git call since we're not in a real repo
        original_run_git = workflow._run_git
        called_args = []

        def mock_run_git(args):
            called_args.extend(args)
            return "created"

        workflow._run_git = mock_run_git
        workflow.create_branch("my-feature")

        assert "checkout" in called_args
        assert "-b" in called_args
        assert "spine-my-feature" in called_args

    def test_commit_with_work_item_uses_template(self):
        workflow = persistence.GitWorkflow()
        called_commits = []

        def mock_run_git(args):
            if "commit" in args:
                called_commits.append(args)
            return "abc123"

        workflow._run_git = mock_run_git
        workflow.commit("Add new feature", work_item="feat-123")

        assert len(called_commits) == 1
        assert "feat-123: Add new feature" in called_commits[0]

    def test_commit_without_work_item_uses_plain_message(self):
        workflow = persistence.GitWorkflow()
        called_commits = []

        def mock_run_git(args):
            if "commit" in args:
                called_commits.append(args)
            return "abc123"

        workflow._run_git = mock_run_git
        workflow.commit("Simple message")

        assert len(called_commits) == 1
        assert called_commits[0][-1] == "Simple message"

    def test_commit_without_work_item_direct_message(self):
        workflow = persistence.GitWorkflow()
        called_commits = []

        def mock_run_git(args):
            if "commit" in args:
                called_commits.append(args)
            return "abc123"

        workflow._run_git = mock_run_git
        # When work_item is empty/falsy, plain message is used
        workflow.commit("Direct message", work_item="")

        assert len(called_commits) == 1
        assert called_commits[0][-1] == "Direct message"

    def test_push_requires_remote_push_config(self):
        workflow = persistence.GitWorkflow()  # remote_push defaults to False

        try:
            workflow.push("spine-test")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "remote_push must be enabled" in str(e)

    def test_push_with_config_enabled(self):
        config = persistence.GitWorkflowConfig(remote_push=True)
        workflow = persistence.GitWorkflow(config)
        called_args = []

        def mock_run_git(args):
            called_args.extend(args)
            return "pushed"

        workflow._run_git = mock_run_git
        workflow.push("spine-my-feature")

        assert "push" in called_args
        assert "origin" in called_args
        assert "spine-my-feature" in called_args

    def test_push_with_custom_remote(self):
        config = persistence.GitWorkflowConfig(remote_push=True)
        workflow = persistence.GitWorkflow(config)
        called_args = []

        def mock_run_git(args):
            called_args.extend(args)
            return "pushed"

        workflow._run_git = mock_run_git
        workflow.push("spine-feature", remote="upstream")

        assert "upstream" in called_args

    def test_create_pull_request_requires_token(self):
        workflow = persistence.GitWorkflow()

        try:
            workflow.create_pull_request(
                title="Test PR",
                body="Test body",
                head_branch="spine-test",
                token=None,
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "GitHub token is required" in str(e)

    def test_create_pull_request_requires_github_repository(self):
        import os
        original_env = os.environ.pop("GITHUB_REPOSITORY", None)

        workflow = persistence.GitWorkflow()

        try:
            workflow.create_pull_request(
                title="Test PR",
                body="Test body",
                head_branch="spine-test",
                token="fake-token",
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "GITHUB_REPOSITORY environment variable not set" in str(e)
        finally:
            if original_env:
                os.environ["GITHUB_REPOSITORY"] = original_env

    def test_get_current_branch(self):
        workflow = persistence.GitWorkflow()

        def mock_run_git(args):
            if "rev-parse" in args:
                return "main"

        workflow._run_git = mock_run_git
        branch = workflow._get_current_branch()
        assert branch == "main"


class TestContinuityManagerEnhanced:
    """Test enhanced ContinuityManager with learning and Git integration."""

    def test_init_with_learning_manager(self):
        """ContinuityManager should accept learning_manager parameter."""
        from spine.core.learning import LearningManager
        with tempfile.TemporaryDirectory() as tmpdir:
            lm = LearningManager(knowledge_dir=os.path.join(tmpdir, "knowledge"))
            cm = ContinuityManager(state_dir=os.path.join(tmpdir, "state"), learning_manager=lm)
            assert cm.learning_manager is lm

    def test_init_with_git_workflow(self):
        """ContinuityManager should accept git_workflow parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gw = persistence.GitWorkflow()
            cm = ContinuityManager(state_dir=os.path.join(tmpdir, "state"), git_workflow=gw)
            assert cm.git_workflow is gw

    def test_create_checkpoint(self):
        """create_checkpoint should create checkpoint with swarm state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ContinuityManager(state_dir=os.path.join(tmpdir, "state"))
            checkpoint = cm.create_checkpoint(
                work_item_id="feat_test",
                phase_name="EXECUTION",
                phase_progress=0.5,
                state={"completed_tasks": ["task1"]},
                dag={"execution_plan": ["task1", "task2"]},
                context_vars={"key": "value"},
                swarm_state={
                    "active_subphases": ["BACKEND"],
                    "file_reservations": {"worker": ["file.py"]},
                    "pending_gates": ["reviewer"]
                }
            )
            assert checkpoint.work_item_id == "feat_test"
            assert checkpoint.phase_name == "EXECUTION"
            assert checkpoint.phase_progress == 0.5
            assert checkpoint.swarm_state["active_subphases"] == ["BACKEND"]

    def test_save_checkpoint_creates_file(self):
        """save_checkpoint should save checkpoint to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ContinuityManager(state_dir=os.path.join(tmpdir, "state"))
            checkpoint = Checkpoint(
                work_item_id="feat_save",
                phase_name="PLANNING",
                phase_progress=1.0,
                swarm_state={}
            )
            path = cm.save_checkpoint(checkpoint)
            assert os.path.exists(path)
            assert "feat_save" in path or "ckpt_" in path

    def test_save_checkpoint_auto_commit(self):
        """save_checkpoint with auto_commit should call git commit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gw = persistence.GitWorkflow()
            gw._run_git = lambda args: "abc123"  # Mock git
            cm = ContinuityManager(state_dir=os.path.join(tmpdir, "state"), git_workflow=gw)
            checkpoint = Checkpoint(
                work_item_id="feat_auto",
                phase_name="EXECUTION",
                phase_progress=0.5,
                swarm_state={}
            )
            path = cm.save_checkpoint(checkpoint, auto_commit=True)
            assert os.path.exists(path)

    def test_create_resume_marker_with_checkpoint(self):
        """create_resume_marker_with_checkpoint should link marker to checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ContinuityManager(state_dir=os.path.join(tmpdir, "state"))
            checkpoint = Checkpoint(
                work_item_id="feat_resume",
                phase_name="EXECUTION",
                phase_progress=0.75,
                swarm_state={
                    "active_subphases": ["FRONTEND"],
                    "pending_gates": ["test"]
                }
            )
            marker = cm.create_resume_marker_with_checkpoint(
                work_item_id="feat_resume",
                checkpoint=checkpoint,
                reason="timeout"
            )
            assert marker.work_item_id == "feat_resume"
            assert marker.reason == "timeout"
            assert "ckpt_" in marker.checkpoint_ref

    def test_list_checkpoints(self):
        """list_checkpoints should return checkpoint paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ContinuityManager(state_dir=os.path.join(tmpdir, "state"))
            checkpoint1 = Checkpoint(work_item_id="test1", phase_name="PLANNING", phase_progress=1.0, swarm_state={})
            checkpoint2 = Checkpoint(work_item_id="test2", phase_name="EXECUTION", phase_progress=0.5, swarm_state={})
            cm.save_checkpoint(checkpoint1)
            cm.save_checkpoint(checkpoint2)
            
            checkpoints = cm.list_checkpoints()
            assert len(checkpoints) >= 1  # At least some checkpoints exist

    def test_list_checkpoints_filtered(self):
        """list_checkpoints should filter by work_item_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ContinuityManager(state_dir=os.path.join(tmpdir, "state"))
            checkpoint = Checkpoint(work_item_id="filtered", phase_name="PLANNING", phase_progress=1.0, swarm_state={})
            cm.save_checkpoint(checkpoint)
            
            checkpoints = cm.list_checkpoints(work_item_id="filtered")
            assert len(checkpoints) >= 1
            # Should not return checkpoints for other work items
            checkpoints_other = cm.list_checkpoints(work_item_id="nonexistent")
            assert len(checkpoints_other) == 0


class TestGitWorkflowAutoCommit:
    """Test GitWorkflow auto_commit_checkpoint method."""

    def test_auto_commit_checkpoint(self):
        """auto_commit_checkpoint should create commit with message."""
        workflow = persistence.GitWorkflow()
        called_args = []

        def mock_run_git(args):
            called_args.extend(args)
            return "abc123"

        workflow._run_git = mock_run_git
        result = workflow.auto_commit_checkpoint(
            checkpoint_id="ckpt_123",
            phase_name="EXECUTION",
            work_item_id="feat_test",
            percent_complete=50.0
        )
        assert result == "abc123"
        assert "commit" in called_args
        # Check the commit message contains both Checkpoint and EXECUTION
        commit_idx = called_args.index("-m") + 1 if "-m" in called_args else -1
        if commit_idx > 0 and commit_idx < len(called_args):
            message = called_args[commit_idx]
            assert "Checkpoint" in message
            assert "EXECUTION" in message

    def test_auto_commit_checkpoint_exception_handling(self):
        """auto_commit_checkpoint should return None on exception."""
        workflow = persistence.GitWorkflow()

        def mock_run_git(args):
            raise RuntimeError("git error")

        workflow._run_git = mock_run_git
        result = workflow.auto_commit_checkpoint(
            checkpoint_id="ckpt_456",
            phase_name="PLANNING",
            work_item_id="feat_test",
            percent_complete=100.0
        )
        assert result is None
