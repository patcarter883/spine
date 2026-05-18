"""SPINE types — data models for tasks, artifacts, reviews, and prompt requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from spine.models.enums import ReviewStatus, TaskStatus, WorkType


# ── Work Unit Models (for plan decomposition) ──


class WorkUnit(BaseModel):
    """A single unit of execution spawned from a planning work item."""

    title: str
    description: str
    priority: str = "medium"  # "low" | "medium" | "high" | "critical"
    is_critical: bool = False


class PlanDecomposition(BaseModel):
    """Output from the plan resolver - a decomposition of a plan into work units."""

    units: list[WorkUnit]

    @field_validator("units")
    @classmethod
    def units_non_empty(cls, v: list[WorkUnit]) -> list[WorkUnit]:
        if not v:
            raise ValueError("Plan decomposition must contain at least one work unit")
        return v


class WorkSpawnSpec(BaseModel):
    """Specification for spawning work items from an approved plan."""

    title: str
    description: str
    work_type: WorkType = WorkType.QUICK
    plan_id: str
    priority: str = "medium"  # "low" | "medium" | "high" | "critical"


# ── Legacy Task Models ──


@dataclass
class Task:
    """A unit of work within a workflow phase."""

    id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    artifact_paths: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class Artifact:
    """An output artifact produced by a workflow phase."""

    path: str
    content: str
    phase: str
    produced_at: datetime = field(default_factory=datetime.now)


@dataclass
class ReviewFeedback:
    """Feedback from a critic review, either structural or agent-based."""

    status: ReviewStatus
    tier: str  # "structural" or "agent"
    reason: str
    suggestions: list[str] = field(default_factory=list)


@dataclass
class PromptRequest:
    """A request from a phase for human input."""

    message: str
    phase: str
    context: dict = field(default_factory=dict)
