"""SPINE types — data models for tasks, artifacts, reviews, and prompt requests."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from spine.models.enums import ReviewStatus, TaskStatus
from typing import Literal


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
    # Qualified names of existing symbols the implementer must read to write
    # this slice (the code it calls/extends/mimics). Lets the implementer
    # read_symbol them directly instead of surveying files. Empty on older plans.
    reference_symbols: list[str] = field(default_factory=list)
    # Optional planner-provided targeted edits (file/symbol/mode/intent dicts)
    # that let a lightweight implementer apply changes without re-discovering
    # where to work. Empty on older plans — the implementer falls back to the
    # read-then-edit flow.
    edit_plan: list[dict[str, Any]] = field(default_factory=list)

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
    hard_boundaries: list[str] = Field(
        description=(
            "No-touch file path globs (workspace-root-relative, fnmatch syntax, "
            "e.g. 'spine/billing/**'). Files written by IMPLEMENT that match any "
            "of these are a hard scope violation and are enforced deterministically "
            "by the scope-boundary gate — distinct from the prose scope_exclusions, "
            "which remain advisory input to the critic."
        ),
        default_factory=list,
    )
    known_risks: list[str] = Field(description="Known risks", default_factory=list)


# ── Project / Milestone Layer Models ──
# A persistent project-level envelope spanning many work items. A project is an
# EXPLICIT list of top-level work_ids (membership stored project-side); it does
# NOT rely on the dead ``spawned_work_ids`` field, and ``project_id`` is
# orthogonal to ``plan_id`` (both are back-references, not a hierarchy).


class RequirementRef(BaseModel):
    """A project-level requirement with a stable, caller-assigned ID.

    ``id`` is assigned once at creation (e.g. "R-001") and is IMMUTABLE for the
    project's lifetime: the coverage aggregator keys per-requirement status off
    it, so renumbering would silently invalidate all historical coverage. Treat
    any ID change as a breaking operation.
    """

    id: str = Field(description="Stable requirement ID, e.g. 'R-001'. Immutable.")
    text: str = Field(description="The requirement statement.")
    rationale: str = Field(default="", description="Why this requirement exists.")


class RoadmapPhase(BaseModel):
    """One phase/milestone of a project roadmap. Status is DERIVED by the
    aggregator from member verification state — never stored here."""

    id: str = Field(description="Stable phase ID, e.g. 'M-001'. Immutable.")
    title: str = Field(description="Phase title.")
    description: str = Field(default="", description="What the phase delivers.")
    requirement_ids: list[str] = Field(
        default_factory=list,
        description="RequirementRef.id values this phase is responsible for.",
    )
    member_work_ids: list[str] = Field(
        default_factory=list,
        description="Subset of the project's member work_ids assigned to this phase.",
    )


class Roadmap(BaseModel):
    """An ordered list of roadmap phases."""

    phases: list[RoadmapPhase] = Field(default_factory=list)


class ProjectSpec(BaseModel):
    """Persistent project-level specification + roadmap spanning many work items.

    Membership (``member_work_ids``) is the source of truth for which work items
    belong to the project. Coverage is computed read-only by
    ``spine.project.aggregator``: it reflects members that have RUN AND PASSED
    verification; members whose checkpoint state is absent (purged on approve or
    never run) are treated as unverified, not as failures.
    """

    id: str = Field(description="Unique project slug; the on-disk directory key.")
    title: str = Field(description="Project title.")
    summary: str = Field(default="", description="Executive summary.")
    objectives: list[str] = Field(default_factory=list)
    requirements: list[RequirementRef] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    hard_boundaries: list[str] = Field(default_factory=list)
    roadmap: Roadmap = Field(default_factory=Roadmap)
    member_work_ids: list[str] = Field(default_factory=list)
    created_at: str = Field(description="ISO timestamp of project creation.")
    updated_at: str = Field(description="ISO timestamp of last modification.")


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
    cited_exclusions: list[str] = Field(
        default_factory=list,
        description=(
            "When (and only when) you flag a scope-creep / out-of-scope "
            "VIOLATION, copy here — verbatim, character-for-character — the "
            "exact scope_exclusions bullet(s) from the <specification> that the "
            "work overlaps. Leave empty otherwise. NEVER cite a scope_inclusions "
            "item here: inclusions are IN scope by definition. A scope-exclusion "
            "violation asserted without a verbatim citation that matches a real "
            "scope_exclusions bullet is treated as unsupported and overturned "
            "automatically, so do not flag one you cannot quote."
        ),
    )
    score: int | None = Field(
        default=None, description="Optional 1-10 quality score"
    )
    blocker_category: Literal["spec_contradiction"] | None = Field(
        default=None,
        description=(
            "Set to 'spec_contradiction' ONLY when the sole blocking issue is "
            "that the specification excludes or omits something the requirement "
            "needs, so the author cannot fix it by reworking this phase — it "
            "requires amending the spec. Reworking the plan cannot resolve a "
            "spec gap, so this is escalated to a spec-amendment review instead "
            "of consuming retry attempts. Leave null for ordinary defects."
        ),
    )
