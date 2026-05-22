"""Unit tests for SPINE exceptions and error handling."""

from __future__ import annotations


from spine.exceptions import (
    SpineError,
    WorkflowError,
    CriticError,
    MaxRetriesExceeded,
    PromptRequestError,
    AgentUnavailableError,
    ConfigurationError,
)


class TestSpineError:
    """Test cases for the base SpineError exception."""

    def test_spine_error_creation(self) -> None:
        """Test creating a basic SpineError."""
        error = SpineError("Test error message")

        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_spine_error_without_message(self) -> None:
        """Test creating SpineError without a message."""
        error = SpineError()

        assert str(error) == ""
        assert isinstance(error, Exception)


class TestWorkflowError:
    """Test cases for WorkflowError exception."""

    def test_workflow_error_inheritance(self) -> None:
        """Test that WorkflowError inherits from SpineError."""
        error = WorkflowError("Workflow failed")

        assert isinstance(error, SpineError)
        assert isinstance(error, Exception)
        assert str(error) == "Workflow failed"

    def test_workflow_error_with_context(self) -> None:
        """Test WorkflowError with contextual information."""
        error = WorkflowError("Phase transition failed")

        assert "Phase transition failed" in str(error)


class TestCriticError:
    """Test cases for CriticError exception."""

    def test_critic_error_inheritance(self) -> None:
        """Test that CriticError inherits from SpineError."""
        error = CriticError("Critic review failed")

        assert isinstance(error, SpineError)
        assert isinstance(error, Exception)
        assert str(error) == "Critic review failed"


class TestMaxRetriesExceeded:
    """Test cases for MaxRetriesExceeded exception."""

    def test_max_retries_exceeded_inheritance(self) -> None:
        """Test that MaxRetriesExceeded inherits from WorkflowError."""
        error = MaxRetriesExceeded("test_phase", 3)

        assert isinstance(error, WorkflowError)
        assert isinstance(error, SpineError)
        assert isinstance(error, Exception)

    def test_max_retries_exceeded_message_format(self) -> None:
        """Test the error message format for MaxRetriesExceeded."""
        error = MaxRetriesExceeded("analyze", 5)

        expected_message = "Phase 'analyze' exceeded max retries (5)"
        assert str(error) == expected_message

    def test_max_retries_exceeded_attributes(self) -> None:
        """Test that phase and retries are stored correctly."""
        phase = "review"
        retries = 2
        error = MaxRetriesExceeded(phase, retries)

        assert error.phase == phase
        assert error.retries == retries

    def test_max_retries_exceeded_different_values(self) -> None:
        """Test MaxRetriesExceeded with different phase and retry values."""
        error = MaxRetriesExceeded("finalize", 10)

        assert error.phase == "finalize"
        assert error.retries == 10
        assert "finalize" in str(error)
        assert "10" in str(error)


class TestPromptRequestError:
    """Test cases for PromptRequestError exception."""

    def test_prompt_request_error_inheritance(self) -> None:
        """Test that PromptRequestError inherits from SpineError."""
        error = PromptRequestError("Invalid prompt request")

        assert isinstance(error, SpineError)
        assert isinstance(error, Exception)
        assert str(error) == "Invalid prompt request"


class TestAgentUnavailableError:
    """Test cases for AgentUnavailableError exception."""

    def test_agent_unavailable_error_inheritance(self) -> None:
        """Test that AgentUnavailableError inherits from SpineError."""
        error = AgentUnavailableError("No agents available for task")

        assert isinstance(error, SpineError)
        assert isinstance(error, Exception)
        assert str(error) == "No agents available for task"


class TestConfigurationError:
    """Test cases for ConfigurationError exception."""

    def test_configuration_error_inheritance(self) -> None:
        """Test that ConfigurationError inherits from SpineError."""
        error = ConfigurationError("Invalid configuration")

        assert isinstance(error, SpineError)
        assert isinstance(error, Exception)
        assert str(error) == "Invalid configuration"

    def test_configuration_error_with_details(self) -> None:
        """Test ConfigurationError with detailed error message."""
        error = ConfigurationError("Missing required API key in configuration")

        assert "Missing required API key" in str(error)
        assert "configuration" in str(error)
