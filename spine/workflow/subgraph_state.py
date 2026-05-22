"""Per-phase subgraph state schemas for the SPINE orchestrator.

Each subgraph has its own TypedDict so DA agent message history
and phase-internal state don't leak into the parent graph's state.
"""

from __future__ import annotations

from operator import add as _op_add
from typing import Annotated, Any

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
    """PLAN phase — reads spec (if available), produces plan.md + plan.json."""

    spec_path: str  # None for quick workflows (no SPECIFY phase)
    has_spec: bool  # True when a specification artifact exists
    plan_json: str  # Raw plan.json content (set by run_agent, read by save_artifacts)
    execution_waves: list  # Computed after agent completes (for IMPLEMENT)


class TasksSubgraphState(BaseSubgraphState, total=False):
    """TASKS phase — reads plan, produces tasks.md + slice-*.md."""

    plan_path: str
    spec_path: str  # Only for spec/critical_spec workflows


class ImplementSubgraphState(BaseSubgraphState, total=False):
    """IMPLEMENT phase — reads plan artifacts, dispatches slice-implementers."""

    plan_path: str
    gap_plan_path: str | None  # Set when re-running for a gap fix
    execution_waves: list  # Execution waves from PLAN phase (for wave dispatch)


class VerifySubgraphState(BaseSubgraphState, total=False):
    """VERIFY phase — confirms implementation."""

    tasks_path: str
    spec_path: str | None  # Only for spec/critical_spec workflows
    plan_path: str | None


class CriticSubgraphState(BaseSubgraphState, total=False):
    """CRITIC phase — reviews a preceding phase's output."""

    reviewed_phase: str
    reviewed_phase_path: str


class GapPlanSubgraphState(BaseSubgraphState, total=False):
    """GAP_PLAN phase — reads verify feedback, produces gap_plan.md.

    Does NOT re-explore the codebase — uses the existing codebase map
    and plan artifacts from the original planning phase.
    """

    verify_path: str
    plan_path: str


class ExplorationSubgraphState(BaseSubgraphState, total=False):
    """Multi-node exploration → synthesis subgraph (SPECIFY, PLAN).

    Accumulates findings across parallel explore rounds via
    ``operator.add``, then routes to synthesis when research
    is sufficient.
    """

    # Exploration loop control
    research_round: int  # Current round number (0-based)
    max_rounds: int  # Safety valve — max exploration rounds (default 3)
    manager_decision: str  # "explore" | "done" — set by research_manager

    # Accumulated research (operator.add reducer merges per-round findings)
    topics: Annotated[list[str], _op_add]  # Areas being explored this round
    findings: Annotated[list[dict], _op_add]  # ResearchFindings dicts from explore nodes

    # Synthesis output
    agent_response: str  # Final spec/plan text from synthesizer

    # PLAN-specific fields
    spec_path: str  # Path to specification.md (for PLAN explore agents)
    has_spec: bool  # True when a specification artifact exists
    plan_json: str  # Raw plan.json content (only used in PLAN phase)
    execution_waves: list  # Computed execution waves (only used in PLAN phase)
