"""SPINE types — data models for tasks, artifacts, reviews, and prompt requests."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
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

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the full plan to a JSON string.

        Feature slices are serialized via their ``to_dict()`` method so
        the roundtrip through ``from_json`` is lossless.
        """
        data = {
            "architecture_overview": self.architecture_overview,
            "technology_choices": self.technology_choices,
            "feature_slices": [s.to_dict() for s in self.feature_slices],
            "testing_strategy": self.testing_strategy,
            "risks": self.risks,
            "codebase_map": self.codebase_map,
        }
        return json.dumps(data, indent=indent)

    @classmethod
    def from_json(cls, raw: str) -> StructuredPlan:
        """Deserialize a StructuredPlan from a JSON string.

        Raises ``json.JSONDecodeError`` for malformed input and ``TypeError``
        if the top-level value is not a JSON object.
        """
        data = json.loads(raw)
        slices = [FeatureSlice.from_dict(s) for s in data.get("feature_slices", [])]
        return cls(
            architecture_overview=data.get("architecture_overview", ""),
            technology_choices=data.get("technology_choices", []),
            feature_slices=slices,
            testing_strategy=data.get("testing_strategy", ""),
            risks=data.get("risks", []),
            codebase_map=data.get("codebase_map", {}),
        )
