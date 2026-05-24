"""SPINE exceptions — custom exception hierarchy."""

from __future__ import annotations


class SpineError(Exception):
    """Base exception for all SPINE errors."""


class WorkflowError(SpineError):
    """Error in workflow execution or phase transition."""


class CriticError(SpineError):
    """Error in critic review execution."""


class MaxRetriesExceeded(WorkflowError):
    """A phase exceeded its configured critic retry limit."""

    def __init__(self, phase: str, retries: int) -> None:
        self.phase = phase
        self.retries = retries
        super().__init__(f"Phase '{phase}' exceeded max retries ({retries})")


class PromptRequestError(SpineError):
    """Error related to human prompt request handling."""


class AgentUnavailableError(SpineError):
    """No agent provider is available for phase execution."""


class ConfigurationError(SpineError):
    """Invalid or missing configuration."""


class TransientAPIError(SpineError):
    """A transient (retryable) LLM API error — 5xx, 429, or provider error.

    Used internally by ``invoke_with_retry()`` to signal that an error
    was classified as transient. Not raised to callers by default
    (retries exhaust first), but useful for logging and testing.
    """

    def __init__(self, original: Exception) -> None:
        self.original = original
        super().__init__(f"Transient API error: {type(original).__name__}: {original}")


class CriticalContractFailure(SpineError):
    """A phase precondition or invariant was violated.

    Raised when a required artifact is missing, a phase flag is incorrect,
    or any other critical workflow contract is broken. This signals that
    the workflow cannot progress safely without human intervention.
    """

    def __init__(self, phase: str, reason: str) -> None:
        self.phase = phase
        self.reason = reason
        super().__init__(f"Critical contract failure in '{phase}': {reason}")
