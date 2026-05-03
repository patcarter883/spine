"""SPINE constants and enums."""

from enum import Enum


class PhaseName(str, Enum):
    """Phase names for the SPINE state machine."""
    INIT = "INIT"
    PLANNING = "PLANNING"
    EXECUTION = "EXECUTION"
    VERIFICATION = "VERIFICATION"
    REWORK = "REWORK"
    BLOCKED = "BLOCKED"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"
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