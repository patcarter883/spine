"""SPINE enums — all enumeration types for the workflow engine."""

from __future__ import annotations

from enum import Enum


class PhaseName(str, Enum):
    """Workflow phase names, used as node identifiers in the LangGraph StateGraph."""

    SPECIFY = "specify"
    PLAN = "plan"
    TASKS = "tasks"
    IMPLEMENT = "implement"
    VERIFY = "verify"
    CRITIC = "critic"


class WorkType(str, Enum):
    """Work type determines the phase composition of the workflow."""

    QUICK = "quick"
    CRITICAL_QUICK = "critical_quick"
    SPEC = "spec"
    CRITICAL_SPEC = "critical_spec"


class ReviewStatus(str, Enum):
    """Outcome of a critic review."""

    PASSED = "passed"
    NEEDS_REVISION = "needs_revision"
    NEEDS_REVIEW = "needs_review"


class TaskStatus(str, Enum):
    """Status of a work item or sub-task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"
    STALLED = "stalled"
