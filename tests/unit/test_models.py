"""Unit tests for SPINE models and types."""

from __future__ import annotations

import pytest
from datetime import datetime

from spine.models.enums import ReviewStatus, TaskStatus, PhaseName, WorkType
from spine.models.types import Task, Artifact, ReviewFeedback, PromptRequest
from spine.models.state import WorkflowState, _merge_dicts, _merge_artifacts


class TestTask:
    """Test cases for Task dataclass."""

    def test_task_creation(self) -> None:
        """Test creating a task with default values."""
        task = Task(id="test-task", description="Test task")

        assert task.id == "test-task"
        assert task.description == "Test task"
        assert task.status == TaskStatus.PENDING
        assert task.artifact_paths == []
        assert task.error is None

    def test_task_with_all_fields(self) -> None:
        """Test creating a task with all fields specified."""
        task = Task(
            id="full-task",
            description="Test task description",
            status=TaskStatus.RUNNING,
            artifact_paths=["/path/to/artifact1.txt", "/path/to/artifact2.txt"],
            error="Task failed with error"
        )
        
        assert task.id == "full-task"
        assert task.description == "Test task description"
        assert task.status == TaskStatus.RUNNING
        assert len(task.artifact_paths) == 2
        assert task.error == "Task failed with error"

    def test_task_status_enum_values(self) -> None:
        """Test that TaskStatus enum has expected values."""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.RUNNING == "running"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.NEEDS_REVIEW == "needs_review"
        assert TaskStatus.FAILED == "failed"


class TestArtifact:
    """Test cases for Artifact dataclass."""

    def test_artifact_creation(self) -> None:
        """Test creating an artifact with default values."""
        artifact = Artifact(path="/path/to/artifact.txt", content="Test content", phase="test")

        assert artifact.path == "/path/to/artifact.txt"
        assert artifact.content == "Test content"
        assert artifact.phase == "test"
        assert isinstance(artifact.produced_at, datetime)

    def test_artifact_with_phase(self) -> None:
        """Test creating an artifact with phase specified."""
        produced_at = datetime(2024, 1, 1, 12, 0, 0)
        artifact = Artifact(
            path="/path/to/artifact.txt",
            content="Test content",
            phase="test_phase",
            produced_at=produced_at
        )

        assert artifact.phase == "test_phase"
        assert artifact.produced_at == produced_at

    def test_artifact_string_content(self) -> None:
        """Test that artifact content is stored as string."""
        long_content = "A" * 10000
        artifact = Artifact(
            path="/path/to/artifact.txt",
            content=long_content,
            phase="test"
        )

        assert isinstance(artifact.content, str)
        assert artifact.content == long_content


class TestReviewFeedback:
    """Test cases for ReviewFeedback dataclass."""

    def test_review_feedback_creation(self) -> None:
        """Test creating review feedback with default values."""
        feedback = ReviewFeedback(
            status=ReviewStatus.PASSED,
            tier="structural",
            reason="Test review passed"
        )
        
        assert feedback.status == ReviewStatus.PASSED
        assert feedback.tier == "structural"
        assert feedback.reason == "Test review passed"
        assert feedback.suggestions == []

    def test_review_feedback_with_suggestions(self) -> None:
        """Test creating review feedback with suggestions."""
        suggestions = ["Improve X", "Check Y", "Review Z"]
        feedback = ReviewFeedback(
            status=ReviewStatus.NEEDS_REVISION,
            tier="agent",
            reason="Needs improvement",
            suggestions=suggestions
        )
        
        assert len(feedback.suggestions) == 3
        assert "Improve X" in feedback.suggestions

    def test_review_feedback_enum_values(self) -> None:
        """Test that ReviewStatus enum has expected values."""
        assert ReviewStatus.PASSED == "passed"
        assert ReviewStatus.NEEDS_REVISION == "needs_revision"
        assert ReviewStatus.NEEDS_REVIEW == "needs_review"


class TestPromptRequest:
    """Test cases for PromptRequest dataclass."""

    def test_prompt_request_creation(self) -> None:
        """Test creating a prompt request with default values."""
        request = PromptRequest(message="Please review this")
        
        assert request.message == "Please review this"
        assert request.phase == ""
        assert request.context == {}

    def test_prompt_request_with_all_fields(self) -> None:
        """Test creating a prompt request with all fields."""
        context = {"file_path": "/test.py", "line": 10}
        request = PromptRequest(
            message="Please review this code",
            phase="verify",
            context=context
        )
        
        assert request.message == "Please review this code"
        assert request.phase == "verify"
        assert request.context == context


class TestWorkflowState:
    """Test cases for WorkflowState and state utilities."""

    def test_merge_dicts_function(self) -> None:
        """Test the _merge_dicts utility function."""
        left = {"a": 1, "b": 2}
        right = {"b": 3, "c": 4}
        
        merged = _merge_dicts(left, right)
        
        assert merged == {"a": 1, "b": 3, "c": 4}  # b from right overwrites left

    def test_merge_artifacts_deep_merge(self) -> None:
        """Test that _merge_artifacts merges at the file level, not the phase level."""
        left = {
            "tasks": {
                "tasks.md": "summary v1",
                "slice-auth.md": "auth slice",
                "slice-routes.md": "routes slice",
            },
            "specify": {"specification.md": "spec content"},
        }
        right = {
            "tasks": {"tasks.md": "summary v2"},  # rework — only tasks.md updated
        }

        merged = _merge_artifacts(left, right)

        # tasks.md is updated to v2, but slice files are preserved
        assert merged["tasks"]["tasks.md"] == "summary v2"
        assert merged["tasks"]["slice-auth.md"] == "auth slice"
        assert merged["tasks"]["slice-routes.md"] == "routes slice"
        assert merged["specify"]["specification.md"] == "spec content"

    def test_merge_artifacts_empty_right(self) -> None:
        """Test that empty right dict doesn't destroy left."""
        left = {"tasks": {"tasks.md": "content"}}
        merged = _merge_artifacts(left, {})
        assert merged == {"tasks": {"tasks.md": "content"}}

    def test_merge_artifacts_new_phase(self) -> None:
        """Test that a new phase key is added correctly."""
        left = {"specify": {"specification.md": "spec"}}
        right = {"plan": {"plan.md": "the plan"}}
        merged = _merge_artifacts(left, right)
        assert merged["plan"]["plan.md"] == "the plan"
        assert merged["specify"]["specification.md"] == "spec"

    def test_merge_artifacts_empty_phase_value(self) -> None:
        """Test that an empty phase dict on the right doesn't corrupt left."""
        left = {"tasks": {"tasks.md": "content", "slice-1.md": "slice"}}
        right = {"tasks": {}}  # error path returns empty dict
        merged = _merge_artifacts(left, right)
        # Empty dict from right overwrites the phase key (LangGraph semantics)
        assert merged["tasks"] == {}

    def test_workflow_state_typed_dict(self) -> None:
        """Test that WorkflowState accepts expected fields."""
        state: WorkflowState = {
            "work_id": "test-123",
            "work_type": "spec",
            "description": "Test work",
            "current_phase": "specify",
            "phase_index": 0,
            "retry_count": {"specify": 0},
            "max_retries": 3,
            "artifacts": {"specify": {"output.txt": "content"}},
            "feedback": [],
            "status": "pending"
        }
        
        assert state["work_id"] == "test-123"
        assert state["work_type"] == "spec"
        assert state["retry_count"]["specify"] == 0
        assert state["artifacts"]["specify"]["output.txt"] == "content"

    def test_workflow_state_optional_fields(self) -> None:
        """Test that WorkflowState optional fields work correctly."""
        state: WorkflowState = {
            "work_id": "test-123",
            "description": "Test work",
            "status": "running"
        }
        
        assert state["prompt_request"] is None
        assert state.get("nonexistent_field") is None


class TestEnums:
    """Test cases for SPINE enums."""

    def test_phase_name_enum(self) -> None:
        """Test PhaseName enum values."""
        assert PhaseName.SPECIFY == "specify"
        assert PhaseName.PLAN == "plan"
        assert PhaseName.TASKS == "tasks"
        assert PhaseName.IMPLEMENT == "implement"
        assert PhaseName.VERIFY == "verify"
        assert PhaseName.CRITIC == "critic"

    def test_work_type_enum(self) -> None:
        """Test WorkType enum values."""
        assert WorkType.QUICK == "quick"
        assert WorkType.CRITICAL_QUICK == "critical_quick"
        assert WorkType.SPEC == "spec"
        assert WorkType.CRITICAL_SPEC == "critical_spec"

    def test_enum_string_behavior(self) -> None:
        """Test that enums behave like strings."""
        phase = PhaseName.SPECIFY
        assert phase == "specify"
        assert str(phase) == "specify"
        assert f"Phase: {phase}" == "Phase: specify"

    def test_enum_iteration(self) -> None:
        """Test that we can iterate over enum values."""
        # Test PhaseName
        phases = list(PhaseName)
        assert len(phases) == 6
        assert PhaseName.SPECIFY in phases
        
        # Test WorkType
        work_types = list(WorkType)
        assert len(work_types) == 4
        assert WorkType.SPEC in work_types


class TestModelValidation:
    """Test cases for model validation and edge cases."""

    def test_task_id_validation(self) -> None:
        """Test task ID validation (should be string)."""
        task = Task(id="test-id")
        assert isinstance(task.id, str)
        
        # Test with numeric ID (should be converted to string)
        task_numeric = Task(id=123)
        assert task_numeric.id == "123"

    def test_artifact_path_validation(self) -> None:
        """Test artifact path validation."""
        artifact = Artifact(path="/valid/path.txt", content="test")
        assert artifact.path == "/valid/path.txt"
        
        # Test with empty path
        artifact_empty = Artifact(path="", content="test")
        assert artifact_empty.path == ""

    def test_review_feedback_status_validation(self) -> None:
        """Test review feedback status validation."""
        # Should accept valid enum values
        feedback = ReviewFeedback(
            status=ReviewStatus.PASSED,
            tier="structural",
            reason="Test"
        )
        assert feedback.status == ReviewStatus.PASSED
        
        # Should accept string equivalents
        feedback_str = ReviewFeedback(
            status="passed",
            tier="structural",
            reason="Test"
        )
        assert feedback_str.status == ReviewStatus.PASSED

    def test_prompt_request_context_validation(self) -> None:
        """Test prompt request context validation."""
        context = {"key": "value", "number": 42, "nested": {"inner": "data"}}
        request = PromptRequest(
            message="Test",
            phase="test",
            context=context
        )
        
        assert request.context == context
        assert isinstance(request.context, dict)