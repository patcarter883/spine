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
    ADVERSARIAL = "adversarial"
    GAP_PLAN = "gap_plan"
    PROJECT_VERIFY = "project_verify"
    PROJECT_REVIEW = "project_review"


class WorkType(str, Enum):
    """Work type determines the phase composition of the workflow.

    - TASK: specify -> plan -> critic_plan -> implement -> verify
    - CRITICAL_TASK: specify -> plan -> critic_plan -> adversarial_plan ->
      implement -> verify. The adversarial stage red-teams the approved plan;
      autonomously-fixable findings loop back to plan, human-judgement findings
      escalate.
    - REVIEWED_TASK: specify -> plan -> critic_plan, then stops for human
      approval. On approval, the SAME work item is re-keyed to TASK and
      continued from implement (reusing the approved spec + plan).
    - CRITICAL_REVIEWED_TASK: specify -> plan -> critic_plan -> adversarial_plan,
      then stops for human approval. On approval, the SAME work item is
      re-keyed to CRITICAL_TASK and continued from implement.
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
    # Terminal status for a user-requested Stop Work. Because it is a member of
    # this enum it is included in RalphLoopWorker._TERMINAL_STATUSES, so the
    # queue loop preserves it instead of remapping a cancelled run to
    # "completed", and it is excluded from the UI's running/stalled active set.
    CANCELLED = "cancelled"
