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
    GAP_PLAN = "gap_plan"


class WorkType(str, Enum):
    """Work type determines the phase composition of the workflow.

    - TASK: specify -> plan -> critic_plan -> implement -> verify
    - CRITICAL_TASK: specify -> critic_specify -> plan -> critic_plan -> implement -> verify
    - REVIEWED_TASK: same as TASK but pauses for approval after critic_plan
    - CRITICAL_REVIEWED_TASK: same as CRITICAL_TASK but pauses for approval after critic_plan
    """

    TASK = "task"
    CRITICAL_TASK = "critical_task"
    REVIEWED_TASK = "reviewed_task"
    CRITICAL_REVIEWED_TASK = "critical_reviewed_task"


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
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
