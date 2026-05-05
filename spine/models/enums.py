"""SPINE data model enums."""

from enum import Enum


class PhaseName(str, Enum):
    """Phase names for the SPINE state machine."""
    INIT = "INIT"
    PLANNING = "PLANNING"
    EXECUTION = "EXECUTION"
    VERIFICATION = "VERIFICATION"
    REWORK = "REWORK"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"


class StateStatus(str, Enum):
    """Status values for states."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class SubPhaseStatus(str, Enum):
    """Status values for sub-phases during wave execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    REWORKING = "reworking"
    CANCELLED = "cancelled"


class ErrorState(str, Enum):
    """Error states for error handling paths."""
    INIT = "INIT"
    TRANSIENT = "TRANSIENT"
    FATAL = "FATAL"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    TIMEOUT = "TIMEOUT"


class PhaseStateStatus(str, Enum):
    """Status values for phase state tracking."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    ERROR = "error"


__all__ = ["PhaseName", "StateStatus", "SubPhaseStatus"]