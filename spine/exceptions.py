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

    ``carryover`` optionally holds subgraph state worth seeding into the
    wrapper's fresh-thread structural retry — e.g. exploration findings
    that preceded a failed synthesis, so the retry re-runs only the
    synthesis instead of repeating the whole research loop (trace
    019eb940: a plan-synthesis contract failure re-ran ~15 minutes of
    exploration whose findings were intact).
    """

    def __init__(
        self,
        phase: str,
        reason: str,
        carryover: dict | None = None,
    ) -> None:
        self.phase = phase
        self.reason = reason
        self.carryover = carryover or {}
        super().__init__(f"Critical contract failure in '{phase}': {reason}")


class GitOrchestratorError(SpineError):
    """Base exception for the transactional git-sandbox orchestrator."""


class SandboxPreparationError(GitOrchestratorError):
    """Failed to create the sandbox worktree or branch.

    Raised when the working tree is dirty or the underlying ``git
    worktree``/``git checkout`` command exits non-zero.
    """


class ValidationError(GitOrchestratorError):
    """A validation gate in the pipeline failed.

    Carries the offending gate's name, the command that was run, and the
    captured combined output so callers can surface actionable detail.
    """

    def __init__(self, gate_name: str, command: str, output: str) -> None:
        self.gate_name = gate_name
        self.command = command
        self.output = output
        super().__init__(
            f"Validation gate '{gate_name}' failed (command: {command}):\n{output}"
        )


class MergeError(GitOrchestratorError):
    """A fast-forward merge of the verified patch branch failed.

    Typically indicates the main branch advanced and the patch can no
    longer be fast-forwarded (a conflict).
    """
