"""SPINE types — data models for tasks, artifacts, reviews, and prompt requests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from spine.models.enums import ReviewStatus, TaskStatus, WorkType
from typing import Literal


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
    work_type: WorkType = WorkType.TASK
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


# ── Slice Planning Models ──


@dataclass
class FeatureSlice:
    """A single, self-contained implementation slice within a structured plan.

    Each slice declares its target files, dependencies on other slices, and
    acceptance criteria so the orchestrator can topologically sort and execute
    slices in the correct order.
    """

    id: str
    title: str
    target_files: list[str] = field(default_factory=list)
    execution_requirements: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    complexity: str = "small"  # "small" | "medium" | "large"

    def to_dict(self) -> dict[str, Any]:
        """Serialize this slice to a plain dict suitable for JSON encoding."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureSlice:
        """Deserialize a FeatureSlice from a plain dict.

        Unknown keys are silently ignored so forward-compatible payloads
        don't break older consumers.
        """
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


@dataclass
class StructuredPlan:
    """Machine-readable plan output composed of ordered feature slices.

    Replaces the prose-based ``plan.md`` with a structured declaration
    that the orchestrator can parse, validate, and topologically sort.
    """

    architecture_overview: str = ""
    technology_choices: list[str] = field(default_factory=list)
    feature_slices: list[FeatureSlice] = field(default_factory=list)
    testing_strategy: str = ""
    risks: list[str] = field(default_factory=list)
    codebase_map: dict[str, Any] = field(default_factory=dict)


# ── Specification and Gap Planning Models ──


class Specification(BaseModel):
    """Structured specification output from SPECIFY phase."""

    title: str = Field(description="Specification title")
    summary: str = Field(description="Executive summary (2-3 sentences)")
    objectives: list[str] = Field(description="High-level goals", default_factory=list)
    requirements: list[str] = Field(description="Functional requirements", default_factory=list)
    constraints: list[str] = Field(
        description="Non-functional constraints", default_factory=list
    )
    scope_inclusions: list[str] = Field(
        description="Scope inclusions", default_factory=list
    )
    scope_exclusions: list[str] = Field(
        description="Scope exclusions", default_factory=list
    )
    known_risks: list[str] = Field(description="Known risks", default_factory=list)


class FixInstruction(BaseModel):
    """Structured fix instruction for one gap."""

    slice_id: str = Field(description="ID of the slice containing this gap")
    file_path: str = Field(description="File path to modify")
    change_type: Literal["add", "modify", "delete"] = Field(
        description="Type of change to make"
    )
    specific_change: str = Field(
        description="Precise description of what to change"
    )
    acceptance_criteria: list[str] = Field(
        description="Acceptance criteria for the fix", default_factory=list
    )
    estimated_complexity: Literal["small", "medium", "large"] = Field(
        default="small", description="Estimated complexity"
    )


class GapPlan(BaseModel):
    """Structured gap plan output."""

    verification_summary: str = Field(description="Summary of verification failures")
    gaps_identified: int = Field(description="Number of gaps found")
    fix_instructions: list[FixInstruction] = Field(
        description="Structured fix instructions", default_factory=list
    )
    re_verify_slices: list[str] = Field(
        description="Slice IDs that need re-verification", default_factory=list
    )


class CriticReview(BaseModel):
    """Structured critic output."""

    status: Literal["PASSED", "NEEDS_REVISION", "NEEDS_REVIEW"] = Field(
        description="Review status"
    )
    tier: Literal["structural", "agent"] = Field(description="Review tier")
    reason: str = Field(description="Reason for the review decision")
    suggestions: list[str] = Field(default_factory=list, description="Suggestions for improvement")
    score: int | None = Field(
        default=None, description="Optional 1-10 quality score"
    )
