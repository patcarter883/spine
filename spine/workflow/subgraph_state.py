"""Per-phase subgraph state schemas for the SPINE orchestrator.

Each subgraph has its own TypedDict so DA agent message history
and phase-internal state don't leak into the parent graph's state.
"""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict


class BaseSubgraphState(TypedDict, total=False):
    """Fields shared by all phase subgraphs."""

    phase: str
    work_id: str
    work_type: str
    description: str  # Only used by SPECIFY (always) and TASKS (quick workflows).
                      # Other phases work from prior artifacts on disk.
    workspace_root: str
    retry_count: int
    feedback: list
    messages: list[Any]
    artifacts_output: dict  # {filename: content} — what this phase produced
    phase_status: str  # "success" | "needs_review" | "error"


class SpecifySubgraphState(BaseSubgraphState, total=False):
    """SPECIFY phase — produces specification.md."""


class PlanSubgraphState(BaseSubgraphState, total=False):
    """PLAN phase — reads spec, produces plan.md."""

    spec_path: str


class TasksSubgraphState(BaseSubgraphState, total=False):
    """TASKS phase — reads plan, produces tasks.md + slice-*.md."""

    plan_path: str
    spec_path: str  # Only for spec/critical_spec workflows


class ImplementSubgraphState(BaseSubgraphState, total=False):
    """IMPLEMENT phase — reads tasks, writes code."""

    tasks_path: str


class VerifySubgraphState(BaseSubgraphState, total=False):
    """VERIFY phase — confirms implementation."""

    tasks_path: str
    spec_path: str | None  # Only for spec/critical_spec workflows
    plan_path: str | None


class CriticSubgraphState(BaseSubgraphState, total=False):
    """CRITIC phase — reviews a preceding phase's output."""

    reviewed_phase: str
    reviewed_phase_path: str
