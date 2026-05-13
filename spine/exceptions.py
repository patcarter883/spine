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
