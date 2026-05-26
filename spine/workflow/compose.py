"""SPINE workflow composer — builds a LangGraph StateGraph from a WorkType.

The composer reads the WorkType, determines the phase sequence, and wires
the graph with conditional edges for critic review rework loops.

Each critic instance gets a unique node name (e.g. ``critic_specify``,
``critic_plan``) so the same critic function can appear multiple times in
a workflow graph — each reviewing a different preceding phase.

Artifact gates are wired as **nodes** (not just conditional edge functions)
so they can write ``status = "needs_review"`` and feedback entries to state
when they fail. This ensures the dispatcher detects the human-review condition
instead of silently marking the work as completed.

Currently, only the plan→implement transition is gated.  Verify always runs
after implement — if implement produced nothing, verify detects and reports
that; there is no reason for a human review gate between those two phases.

Phase sequences by WorkType:
    task:              SPECIFY → PLAN → CRITIC_PLAN → IMPLEMENT → VERIFY
    critical_task:     SPECIFY → CRITIC_SPECIFY → PLAN → CRITIC_PLAN → IMPLEMENT → VERIFY
    reviewed_task:     SPECIFY → PLAN → CRITIC_PLAN → IMPLEMENT → VERIFY
    critical_reviewed: SPECIFY → CRITIC_SPECIFY → PLAN → CRITIC_PLAN → IMPLEMENT → VERIFY

Note: reviewed_task and critical_reviewed_task share identical phase sequences
with their non-reviewed counterparts. The difference is handled at the stream
level via ``interrupt_after=["critic_plan"]`` in submit_work().
"""

from typing import Any, Callable, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import interrupt

from spine.models.enums import PhaseName, ReviewStatus, WorkType
from spine.models.state import WorkflowState
from spine.workflow.phase_progress import mark_phase_started
from spine.workflow.registry import get_registry
from spine.workflow.critic_review import critic_router
from spine.workflow.artifact_gate import (
    make_artifact_gate_node,
    artifact_gate_router,
)
from spine.agents.artifacts import artifact_path
from spine.workflow.subgraph_wrapper import (
    make_subgraph_node,
    make_success_result_mapper,
)
from spine.workflow.subgraphs.verify_subgraph import build_verify_subgraph
from spine.workflow.subgraphs.implement_subgraph import build_implement_subgraph
from spine.workflow.subgraphs.tasks_subgraph import build_tasks_subgraph
from spine.workflow.subgraphs.specify_subgraph import build_specify_subgraph
from spine.workflow.subgraphs.plan_subgraph import build_plan_subgraph
from spine.workflow.subgraphs.critic_subgraph import build_critic_subgraph
from spine.workflow.subgraphs.exploration_subgraph import build_exploration_subgraph
from spine.workflow.subgraphs.gap_plan_subgraph import build_gap_plan_subgraph
from spine.workflow.artifact_gate import (
    make_prerequisite_gate_node,
    _check_spec_prerequisite,
    _check_plan_prerequisite,
    _check_implement_prerequisite,
    _check_verify_prerequisite,
)

# ── Subgraph builder registry ──────────────────────────────────────────
# Used by the per-phase checkpointer to recompile subgraphs at runtime
# with phase-specific SQLite databases instead of sharing the parent's.

_SUBGRAPH_BUILDER_REGISTRY: dict[str, Callable] = {}


def register_subgraph_builder(phase: str, builder: Callable) -> None:
    """Register a subgraph builder so per-phase checkpointers can recompile."""
    _SUBGRAPH_BUILDER_REGISTRY[phase] = builder


def get_subgraph_builder(phase: str) -> Callable | None:
    """Get the registered builder for a phase, or None."""
    return _SUBGRAPH_BUILDER_REGISTRY.get(phase)


# Register all phase builders at import time.
register_subgraph_builder(PhaseName.VERIFY.value, build_verify_subgraph)
register_subgraph_builder(PhaseName.IMPLEMENT.value, build_implement_subgraph)
register_subgraph_builder(PhaseName.TASKS.value, build_tasks_subgraph)
register_subgraph_builder(PhaseName.SPECIFY.value, build_specify_subgraph)
register_subgraph_builder(PhaseName.PLAN.value, build_plan_subgraph)
# Critic is parameterized by reviewed_phase — register keyed variants.
register_subgraph_builder(f"{PhaseName.CRITIC.value}_tasks", build_critic_subgraph)
register_subgraph_builder(f"{PhaseName.CRITIC.value}_plan", build_critic_subgraph)
register_subgraph_builder(f"{PhaseName.CRITIC.value}_specify", build_critic_subgraph)
register_subgraph_builder(PhaseName.GAP_PLAN.value, build_gap_plan_subgraph)


# Feature flags for per-phase subgraph migration.
# During rollout, phases can be enabled independently.
_SUBGRAPH_ENABLED: dict[str, bool] = {
    PhaseName.VERIFY.value: True,
    PhaseName.IMPLEMENT.value: True,
    PhaseName.TASKS.value: True,
    PhaseName.SPECIFY.value: True,
    PhaseName.PLAN.value: True,
    PhaseName.CRITIC.value: True,
    PhaseName.GAP_PLAN.value: True,
}

# Feature flags for exploration subgraph rollout.
# When True, SPECIFY/PLAN use the multi-node research_manager → explore →
# synthesize subgraph instead of the linear run_agent → save_artifacts
# subgraph.  Default: SPECIFY enabled, PLAN pending implementation.
_USE_EXPLORATION_SUBGRAPH: dict[str, bool] = {
    PhaseName.SPECIFY.value: True,
    PhaseName.PLAN.value: True,
}


def _base_state_mapper(parent_state: WorkflowState, config) -> dict:
    """Base fields shared by all phase state mappers."""
    return {
        "work_id": parent_state.get("work_id", "unknown"),
        "work_type": parent_state.get("work_type", ""),
        "description": parent_state.get("description", ""),
        "workspace_root": parent_state.get("workspace_root", "."),
        "feedback": parent_state.get("feedback", []),
        "messages": [],
        "artifacts_output": {},
        "phase_status": "",
    }


def _specify_state_mapper(parent_state: WorkflowState, config) -> dict:
    return {
        **_base_state_mapper(parent_state, config),
        "phase": PhaseName.SPECIFY.value,
        "retry_count": parent_state.get("retry_count", {}).get(PhaseName.SPECIFY.value, 0),
        "scratchpad": parent_state.get("scratchpad", ""),
        "task_category": parent_state.get("task_category"),
    }


def _plan_state_mapper(parent_state: WorkflowState, config) -> dict:
    work_id = parent_state.get("work_id", "")
    # All work types now run specify, so has_spec is always True
    return {
        **_base_state_mapper(parent_state, config),
        "phase": PhaseName.PLAN.value,
        "retry_count": parent_state.get("retry_count", {}).get(PhaseName.PLAN.value, 0),
        "spec_path": artifact_path(work_id, PhaseName.SPECIFY.value),
        "has_spec": True,
        "scratchpad": parent_state.get("scratchpad", ""),
    }


def _tasks_state_mapper(parent_state: WorkflowState, config) -> dict:
    work_id = parent_state.get("work_id", "")
    # All work types now run specify, so has_spec is always True
    return {
        **_base_state_mapper(parent_state, config),
        "phase": PhaseName.TASKS.value,
        "retry_count": parent_state.get("retry_count", {}).get(PhaseName.TASKS.value, 0),
        "plan_path": artifact_path(work_id, PhaseName.PLAN.value),
        "spec_path": artifact_path(work_id, PhaseName.SPECIFY.value),
    }


def _implement_state_mapper(parent_state: WorkflowState, config) -> dict:
    work_id = parent_state.get("work_id", "")
    verify_attempts = parent_state.get("verify_attempts", 0)
    return {
        **_base_state_mapper(parent_state, config),
        "phase": PhaseName.IMPLEMENT.value,
        "retry_count": parent_state.get("retry_count", {}).get(PhaseName.IMPLEMENT.value, 0),
        "plan_path": artifact_path(work_id, PhaseName.PLAN.value),
        "gap_plan_path": artifact_path(work_id, PhaseName.GAP_PLAN.value) if verify_attempts > 0 else None,
        "execution_waves": parent_state.get("execution_waves", []),
    }


def _verify_state_mapper(parent_state: WorkflowState, config) -> dict:
    """Map parent WorkflowState to VerifySubgraphState."""
    work_id = parent_state.get("work_id", "")
    # All work types now run specify, so has_spec is always True
    return {
        **_base_state_mapper(parent_state, config),
        "phase": PhaseName.VERIFY.value,
        "retry_count": parent_state.get("retry_count", {}).get(PhaseName.VERIFY.value, 0),
        "plan_path": artifact_path(work_id, PhaseName.PLAN.value),
        "spec_path": artifact_path(work_id, PhaseName.SPECIFY.value),
        "execution_waves": parent_state.get("execution_waves", []),
    }


def _critic_state_mapper(reviewed_phase: str):
    """Create a state mapper for a critic subgraph reviewing a specific phase."""

    def mapper(parent_state: WorkflowState, config) -> dict:
        work_id = parent_state.get("work_id", "")
        return {
            **_base_state_mapper(parent_state, config),
            "phase": PhaseName.CRITIC.value,
            "retry_count": parent_state.get("retry_count", {}).get(reviewed_phase, 0),
            "reviewed_phase": reviewed_phase,
            "reviewed_phase_path": artifact_path(work_id, reviewed_phase),
            "artifacts": parent_state.get("artifacts", {}),
        }

    return mapper


def _verify_result_mapper(subgraph_result: dict, parent_state: WorkflowState) -> dict[str, Any]:
    """Map VerifySubgraphState output back to parent WorkflowState.

    When verification fails (phase_status="needs_review"), checks
    ``verify_attempts`` to decide between a gap-fix cycle and human review.
    Up to 2 gap-fix cycles are attempted before escalating to human review.
    """
    base = make_success_result_mapper(PhaseName.VERIFY.value)(subgraph_result, parent_state)
    phase_status = subgraph_result.get("phase_status", "")
    if phase_status == "needs_review":
        verify_attempts = parent_state.get("verify_attempts", 0)
        if verify_attempts < 2:
            base["status"] = "needs_gap_fix"
            base["verify_attempts"] = verify_attempts + 1
        else:
            base["status"] = "needs_review"
            base["needs_review_phase"] = PhaseName.VERIFY.value
        base["feedback"] = base.get("feedback", []) + [
            {
                "status": "needs_review",
                "tier": "verify",
                "reason": "Verification did not pass. See verification.md for details.",
                "suggestions": [],
            }
        ]
    elif phase_status == "error":
        base["status"] = "failed"
    # Set completion invariants
    base["verify_completed"] = phase_status == "success"
    base["verification_attempted"] = True
    base["verification_passed"] = phase_status == "success"
    vf = subgraph_result.get("verification_findings", [])
    if vf:
        base["verification_findings"] = vf
    return base


def _implement_result_mapper(subgraph_result: dict, parent_state: WorkflowState) -> dict[str, Any]:
    """Map ImplementSubgraphState output back to parent WorkflowState."""
    base = make_success_result_mapper(PhaseName.IMPLEMENT.value)(subgraph_result, parent_state)
    phase_status = subgraph_result.get("phase_status", "")
    if phase_status == "needs_review":
        base["status"] = "needs_review"
        base["needs_review_phase"] = PhaseName.IMPLEMENT.value
    elif phase_status == "error":
        base["status"] = "failed"
    # Set completion invariants
    base["implement_completed"] = phase_status == "success"
    base["slices_dispatched"] = subgraph_result.get("slices_dispatched", False)
    base["implementation_files_written"] = subgraph_result.get("implementation_files_written", False)
    return base


def _tasks_result_mapper(subgraph_result: dict, parent_state: WorkflowState) -> dict[str, Any]:
    """Map TasksSubgraphState output back to parent WorkflowState."""
    base = make_success_result_mapper(PhaseName.TASKS.value)(subgraph_result, parent_state)
    phase_status = subgraph_result.get("phase_status", "")
    if phase_status == "needs_review":
        base["status"] = "needs_review"
        base["needs_review_phase"] = PhaseName.TASKS.value
    elif phase_status == "error":
        base["status"] = "failed"
    # Set completion invariants
    base["tasks_completed"] = phase_status == "success"
    base["work_units_count"] = len(subgraph_result.get("work_units", []))
    return base


def _specify_result_mapper(subgraph_result: dict, parent_state: WorkflowState) -> dict[str, Any]:
    """Map SpecifySubgraphState output back to parent WorkflowState."""
    base = make_success_result_mapper(PhaseName.SPECIFY.value)(subgraph_result, parent_state)
    phase_status = subgraph_result.get("phase_status", "")
    if phase_status == "needs_review":
        base["status"] = "needs_review"
        base["needs_review_phase"] = PhaseName.SPECIFY.value
    elif phase_status == "error":
        base["status"] = "failed"
    # Forward early commitment results to parent state
    if subgraph_result.get("task_category"):
        base["task_category"] = subgraph_result["task_category"]
    if subgraph_result.get("retrieved_context"):
        base["retrieved_context"] = subgraph_result["retrieved_context"]
    if "classification_confidence" in subgraph_result:
        base["classification_confidence"] = subgraph_result["classification_confidence"]
    # Set completion invariants
    base["spec_completed"] = phase_status == "success"
    return base


def _plan_result_mapper(subgraph_result: dict, parent_state: WorkflowState) -> dict[str, Any]:
    """Map PlanSubgraphState output back to parent WorkflowState."""
    base = make_success_result_mapper(PhaseName.PLAN.value)(subgraph_result, parent_state)
    phase_status = subgraph_result.get("phase_status", "")
    if phase_status == "needs_review":
        base["status"] = "needs_review"
        base["needs_review_phase"] = PhaseName.PLAN.value
    elif phase_status == "error":
        base["status"] = "failed"
    # Forward execution waves so IMPLEMENT can use wave-based dispatch
    execution_waves = subgraph_result.get("execution_waves", [])
    if execution_waves:
        base["execution_waves"] = execution_waves
    # Set completion invariants
    base["plan_completed"] = phase_status == "success"
    base["feature_slices_count"] = len(subgraph_result.get("feature_slices", []))
    return base


def _critic_result_mapper(reviewed_phase: str):
    """Create a result mapper for a critic subgraph reviewing a specific phase.
    
    Two-tier review logic:
    1. Structural check: If it fails, the agent check still runs (for feedback).
    2. Agent check: Determines the final status (PASSED, NEEDS_REVISION, NEEDS_REVIEW).
    
    The effective feedback comes from the agent if available, otherwise from structural.
    """

    def mapper(subgraph_result: dict, parent_state: WorkflowState) -> dict[str, Any]:
        base: dict[str, Any] = {
            "current_phase": PhaseName.CRITIC.value,
            "status": "running",
            "prompt_request": None,
        }

        structural_result = subgraph_result.get("structural_result", {})
        agent_result = subgraph_result.get("agent_result", {})

        if agent_result:
            effective_result = agent_result
        elif structural_result:
            effective_result = structural_result
        else:
            effective_result = {
                "status": "passed",
                "tier": "structural",
                "reason": "No review performed",
                "suggestions": [],
            }

        base["feedback"] = [effective_result]

        phase_status = subgraph_result.get("phase_status", "")
        if phase_status == ReviewStatus.NEEDS_REVIEW.value:
            base["status"] = "needs_review"
            base["needs_review_phase"] = reviewed_phase
            retry_count = parent_state.get("retry_count", {})
            current = retry_count.get(reviewed_phase, 0)
            base["retry_count"] = {reviewed_phase: current + 1}
        elif phase_status == ReviewStatus.NEEDS_REVISION.value:
            retry_count = parent_state.get("retry_count", {})
            current = retry_count.get(reviewed_phase, 0)
            base["retry_count"] = {reviewed_phase: current + 1}
            base["status"] = "running"
        elif phase_status == "success":
            base["status"] = "running"
        elif phase_status == "error":
            base["status"] = "failed"

        if reviewed_phase == PhaseName.SPECIFY.value:
            base["critic_specify_completed"] = True
        elif reviewed_phase == PhaseName.PLAN.value:
            base["critic_plan_completed"] = True

        return base

    return mapper


def _gap_plan_state_mapper(parent_state: WorkflowState, config) -> dict:
    """Map parent WorkflowState to GapPlanSubgraphState."""
    work_id = parent_state.get("work_id", "")
    return {
        **_base_state_mapper(parent_state, config),
        "phase": PhaseName.GAP_PLAN.value,
        "retry_count": 0,
        "verify_path": artifact_path(work_id, PhaseName.VERIFY.value),
        "plan_path": artifact_path(work_id, PhaseName.PLAN.value),
    }


def _gap_plan_result_mapper(subgraph_result: dict, parent_state: WorkflowState) -> dict[str, Any]:
    """Map GapPlanSubgraphState output back to parent WorkflowState."""
    base = make_success_result_mapper(PhaseName.GAP_PLAN.value)(subgraph_result, parent_state)
    phase_status = subgraph_result.get("phase_status", "")
    if phase_status == "needs_review":
        base["status"] = "needs_review"
        base["needs_review_phase"] = PhaseName.GAP_PLAN.value
    elif phase_status == "error":
        base["status"] = "failed"
    # Set completion invariants
    base["gap_plan_completed"] = phase_status == "success"
    base["gaps_identified"] = len(subgraph_result.get("gaps", []))
    return base


# ── Phase sequences per work type ──
# Each tuple is (node_name, reviewed_phase_or_None).
# For critic nodes, reviewed_phase tells the critic which phase to review.
#
# All 4 work types share identical phase sequences. The difference between
# "reviewed" and non-reviewed types is handled at the stream level via
# interrupt_after=["critic_plan"] in submit_work().

WORKFLOW_SEQUENCES: dict[str, list[tuple[str, str | None]]] = {
    WorkType.TASK.value: [
        (PhaseName.SPECIFY.value, None),
        (PhaseName.PLAN.value, None),
        (f"{PhaseName.CRITIC.value}_plan", PhaseName.PLAN.value),
        (PhaseName.IMPLEMENT.value, None),
        (PhaseName.VERIFY.value, None),
    ],
    WorkType.CRITICAL_TASK.value: [
        (PhaseName.SPECIFY.value, None),
        (f"{PhaseName.CRITIC.value}_specify", PhaseName.SPECIFY.value),
        (PhaseName.PLAN.value, None),
        (f"{PhaseName.CRITIC.value}_plan", PhaseName.PLAN.value),
        (PhaseName.IMPLEMENT.value, None),
        (PhaseName.VERIFY.value, None),
    ],
    WorkType.REVIEWED_TASK.value: [
        (PhaseName.SPECIFY.value, None),
        (PhaseName.PLAN.value, None),
        (f"{PhaseName.CRITIC.value}_plan", PhaseName.PLAN.value),
        (PhaseName.IMPLEMENT.value, None),
        (PhaseName.VERIFY.value, None),
    ],
    WorkType.CRITICAL_REVIEWED_TASK.value: [
        (PhaseName.SPECIFY.value, None),
        (f"{PhaseName.CRITIC.value}_specify", PhaseName.SPECIFY.value),
        (PhaseName.PLAN.value, None),
        (f"{PhaseName.CRITIC.value}_plan", PhaseName.PLAN.value),
        (PhaseName.IMPLEMENT.value, None),
        (PhaseName.VERIFY.value, None),
    ],
}


def _human_review_interrupt(state: WorkflowState) -> dict:
    """Interrupt the graph for human review between phases.

    The workflow pauses here. A human (via UI or CLI) reviews the
    current state and calls ``Command(resume={...})`` to continue.

    Returns:
        A dict with the human decision and feedback.
    """
    needs_review_phase = state.get("needs_review_phase", "")
    feedback = state.get("feedback", [])
    phase_results = state.get("phase_results", {})

    last_fb = feedback[-1] if feedback else {}
    review_info = {
        "phase": needs_review_phase or state.get("current_phase", ""),
        "reason": last_fb.get("reason", "No reason provided"),
        "suggestions": last_fb.get("suggestions", []),
        "phase_results": phase_results,
    }

    # interrupt() pauses the graph. Human response comes back via Command(resume=...)
    human_decision = interrupt(review_info)

    return {
        "human_feedback": human_decision,
        "needs_review_phase": None,
    }


def _make_human_review_router(phase_seq: list[tuple[str, str | None]]):
    """Create a human review router that knows the phase sequence.

    Returns a router function for add_conditional_edges that routes:
    - "rework" → back to the needs_review_phase node
    - "approve" → to the phase after needs_review_phase in the sequence
    - "abort" → END
    """

    # Build a lookup: phase_name → index in phase_seq
    phase_index = {}
    for idx, (name, _) in enumerate(phase_seq):
        phase_index[name] = idx

    def router(state: WorkflowState) -> str:
        human_feedback = state.get("human_feedback", {})
        action = (
            human_feedback.get("action", "abort") if isinstance(human_feedback, dict) else "abort"
        )

        if action == "rework":
            needs_review = state.get("needs_review_phase")
            if needs_review and needs_review in phase_index:
                return needs_review
            # Fallback: if we can't find the phase, still try rework
            return needs_review or "abort"

        if action == "approve":
            needs_review = state.get("needs_review_phase")
            if needs_review and needs_review in phase_index:
                idx = phase_index[needs_review]
                if idx + 1 < len(phase_seq):
                    return phase_seq[idx + 1][0]
                return END
            # Fallback: advance from current_phase
            current = state.get("current_phase", "")
            if current in phase_index:
                idx = phase_index[current]
                if idx + 1 < len(phase_seq):
                    return phase_seq[idx + 1][0]
            return END

        return "abort"

    return router


def _phase_status_router(state: WorkflowState) -> str:
    """Generic post-phase router.

    Reads ``state["status"]`` after a phase subgraph completes and decides
    whether to proceed to the next node or halt for human review.  This is
    the missing guard that previously let needs_review propagate forward
    silently (Bug B): the result mapper set status=needs_review, but the
    bare ``graph.add_edge(phase, next)`` ignored it.

    Returns:
        ``"proceed"`` when status is ``"running"`` (the phase succeeded),
        ``"needs_review"`` when status is ``"needs_review"`` or anything
        unexpected.  Routing targets are bound at edge-construction time
        in :func:`build_workflow_graph`.
    """
    status = state.get("status", "running")
    if status == "needs_review":
        return "needs_review"
    if status == "failed":
        return "failed"
    return "proceed"


def _verify_router(state: WorkflowState) -> str:
    """Post-verify conditional edge router.

    Unlike the generic ``_phase_status_router``, this router can send a
    failed verification into a ``gap_plan → implement → verify`` loop
    instead of immediately flagging for human review.

    Returns:
        ``"passed"`` when verification succeeded,
        ``"needs_gap_fix"`` when verification failed with gap attempts remaining,
        ``"needs_review"`` when verification failed and gap attempts exhausted,
        ``"failed"`` for hard errors.
    """
    status = state.get("status", "running")
    if status == "needs_gap_fix":
        return "needs_gap_fix"
    if status == "needs_review":
        return "needs_review"
    if status == "failed":
        return "failed"
    return "passed"


def _gate_node_name(source_node: str, next_node: str) -> str:
    """Derive a unique node name for a gate between two phases."""
    return f"gate_{source_node}_to_{next_node}"


def build_workflow_graph(
    work_type: str,
    checkpointer: BaseCheckpointSaver | None = None,
    start_from_phase: str | None = None,
) -> Any:
    """Build a compiled LangGraph StateGraph for the given work type.

    The graph wires phase nodes with:
    - Sequential edges for non-critic phases
    - Conditional edges after critic nodes that route to rework or next phase
    - Artifact gate nodes that check prerequisites before verify/implement
    - Retry counting and needs_review escalation

    Critic nodes are named ``critic_{reviewed_phase}`` to allow multiple
    critic instances in a single workflow (e.g. critical_task has 2).

    Args:
        work_type: One of "task", "critical_task", "reviewed_task",
            "critical_reviewed_task".
        checkpointer: Optional BaseCheckpointSaver for persistence.
        start_from_phase: Optional phase name to start execution from.
            When set, the START edge routes directly to this phase instead
            of the first phase in the sequence. Used by restart_from_phase
            to resume a stalled job partway through the workflow.

    Returns:
        A compiled StateGraph ready for ``.invoke()`` or ``.stream()``.

    Raises:
        ValueError: If the work_type is not recognised.
    """
    if work_type not in WORKFLOW_SEQUENCES:
        raise ValueError(
            f"Unknown work type '{work_type}'. Must be one of: {list(WORKFLOW_SEQUENCES.keys())}"
        )

    phase_seq = WORKFLOW_SEQUENCES[work_type]
    registry = get_registry()

    # Validate start_from_phase if provided
    if start_from_phase is not None:
        valid_nodes = {name for name, _ in phase_seq}
        if start_from_phase not in valid_nodes:
            raise ValueError(
                f"Phase '{start_from_phase}' is not a valid node for work type "
                f"'{work_type}'. Valid nodes: {sorted(valid_nodes)}"
            )

    # ── Build the graph ──
    graph = StateGraph(WorkflowState)

    # Collect which edges need artifact gates.
    # We gate based on the *target* node: implement requires plan artifacts.
    # Verify always runs after implement — it's the phase that confirms
    # implementation meets requirements.  If implement produced nothing,
    # verify can detect and report that; there is no reason for a human
    # review gate between implement and verify.
    gate_edges: dict[tuple[str, str], str] = {}  # (src, dst) → required_phase
    _human_review_targets: dict[str, dict[str, str]] = {}  # node → {rework, approve}
    for i, (node_name, _reviewed_phase) in enumerate(phase_seq):
        if i < len(phase_seq) - 1:
            next_node_name = phase_seq[i + 1][0]
            if next_node_name == PhaseName.IMPLEMENT.value:
                gate_edges[(node_name, next_node_name)] = PhaseName.PLAN.value

    # ── Exploration subgraph override ──
    # When exploration mode is enabled for a phase, replace the standard
    # linear subgraph builder with the multi-node research loop.
    # This must happen before the phase node loop below so the builder
    # lookup finds the exploration subgraph instead of the standard one.
    if _USE_EXPLORATION_SUBGRAPH.get(PhaseName.SPECIFY.value, False):
        register_subgraph_builder(
            PhaseName.SPECIFY.value,
            lambda: build_exploration_subgraph(phase=PhaseName.SPECIFY.value),
        )
    if _USE_EXPLORATION_SUBGRAPH.get(PhaseName.PLAN.value, False):
        register_subgraph_builder(
            PhaseName.PLAN.value,
            lambda: build_exploration_subgraph(phase=PhaseName.PLAN.value),
        )

    # Add all phase/critic nodes
    for node_name, reviewed_phase in phase_seq:
        if node_name.startswith(PhaseName.CRITIC.value):
            # Critic node — use subgraph if enabled, else legacy
            _reviewed = reviewed_phase or "unknown"
            if _SUBGRAPH_ENABLED.get(PhaseName.CRITIC.value, False):
                critic_subgraph = build_critic_subgraph(_reviewed).compile()
                graph.add_node(
                    node_name,
                    make_subgraph_node(
                        critic_subgraph,
                        node_name,
                        _critic_state_mapper(_reviewed),
                        _critic_result_mapper(_reviewed),
                        use_per_phase_checkpointer=True,
                    ),
                )
            else:
                critic_def = registry.require(PhaseName.CRITIC.value)
                critic_fn = critic_def.call_fn or critic_def.subgraph_node_fn
                if critic_fn is None:
                    raise ValueError(
                        f"Critic phase '{node_name}' has no call_fn or subgraph_node_fn"
                    )
                graph.add_node(
                    node_name,
                    _make_critic_node(critic_fn, _reviewed),
                )
        elif _SUBGRAPH_ENABLED.get(node_name, False):
            # Subgraph node (new style) — look up builder, mapper, and result mapper
            # from the module-level registry and lookup tables.
            _STATE_MAPPERS = {
                PhaseName.VERIFY.value: _verify_state_mapper,
                PhaseName.IMPLEMENT.value: _implement_state_mapper,
                PhaseName.TASKS.value: _tasks_state_mapper,
                PhaseName.SPECIFY.value: _specify_state_mapper,
                PhaseName.PLAN.value: _plan_state_mapper,
                PhaseName.GAP_PLAN.value: _gap_plan_state_mapper,
            }
            _RESULT_MAPPERS = {
                PhaseName.VERIFY.value: _verify_result_mapper,
                PhaseName.IMPLEMENT.value: _implement_result_mapper,
                PhaseName.TASKS.value: _tasks_result_mapper,
                PhaseName.SPECIFY.value: _specify_result_mapper,
                PhaseName.PLAN.value: _plan_result_mapper,
                PhaseName.GAP_PLAN.value: _gap_plan_result_mapper,
            }

            builder_fn = _SUBGRAPH_BUILDER_REGISTRY.get(node_name)
            if builder_fn is not None:
                subgraph = builder_fn().compile()
                state_mapper = _STATE_MAPPERS[node_name]
                result_mapper = _RESULT_MAPPERS[node_name]

                def _build_node(
                    phase: str,
                    sub: Any,
                    sm: Any,
                    rm: Any,
                ) -> Any:
                    return make_subgraph_node(
                        sub,
                        phase,
                        sm,
                        rm,
                        use_per_phase_checkpointer=True,
                    )

                graph.add_node(
                    node_name,
                    _build_node(
                        node_name,
                        subgraph,
                        state_mapper,
                        result_mapper,
                    ),
                )
            else:
                # Fallback to legacy for phases not yet migrated
                phase_def = registry.require(node_name)
                if phase_def.subgraph_node_fn:
                    graph.add_node(
                        node_name, _make_legacy_node(node_name, phase_def.subgraph_node_fn)
                    )
                elif phase_def.call_fn:
                    graph.add_node(node_name, _make_legacy_node(node_name, phase_def.call_fn))
                else:
                    raise ValueError(f"Phase '{node_name}' has no call_fn or subgraph_node_fn")
        else:
            # Legacy phase node — wrap with phase-start tracking
            phase_def = registry.require(node_name)
            if phase_def.call_fn is None:
                raise ValueError(f"Phase '{node_name}' has no call_fn (legacy mode)")
            graph.add_node(node_name, _make_legacy_node(node_name, phase_def.call_fn))

    # Add human review interrupt node (once, not per gate)
    # Build the human_review conditional-edge map.  The router can return:
    #   - a phase node name (rework or approve → that node),
    #   - "abort" → END,
    #   - END (approve past last phase).
    # Every possible return value must appear in the map or LangGraph raises
    # KeyError at runtime.
    _hr_ends: dict[str, str] = {"abort": END, END: END}
    for name, _ in phase_seq:
        _hr_ends[name] = name
    _hr_ends[PhaseName.GAP_PLAN.value] = PhaseName.GAP_PLAN.value

    graph.add_node("human_review", _human_review_interrupt)
    graph.add_conditional_edges(
        "human_review",
        _make_human_review_router(phase_seq),
        _hr_ends,
    )

    # Add gap_plan node — not in WORKFLOW_SEQUENCES because it's a
    # conditional node reached via the verify_router, not a linear step.
    if _SUBGRAPH_ENABLED.get(PhaseName.GAP_PLAN.value, False):
        gap_subgraph = build_gap_plan_subgraph().compile()
        graph.add_node(
            PhaseName.GAP_PLAN.value,
            make_subgraph_node(
                gap_subgraph,
                PhaseName.GAP_PLAN.value,
                _gap_plan_state_mapper,
                _gap_plan_result_mapper,
                use_per_phase_checkpointer=True,
            ),
        )
    else:
        # Legacy gap_plan node — wrap with phase-start tracking
        gap_plan_def = registry.require(PhaseName.GAP_PLAN.value)
        if gap_plan_def.call_fn is None:
            raise ValueError(f"Phase '{PhaseName.GAP_PLAN.value}' has no call_fn (legacy mode)")
        graph.add_node(PhaseName.GAP_PLAN.value, _make_legacy_node(PhaseName.GAP_PLAN.value, gap_plan_def.call_fn))

    # ── Prerequisite Gate Nodes ─────────────────────────────────────────
    # These gates check phase completion invariants before allowing phases to run.
    # They are wired inline with the graph edges to block empty progression.

    # Gate: PLAN requires SPECIFY completed
    prereq_gate_plan = make_prerequisite_gate_node(_check_spec_prerequisite, PhaseName.PLAN.value)
    graph.add_node("prereq_gate_plan", prereq_gate_plan)

    # Gate: IMPLEMENT requires PLAN completed
    prereq_gate_implement = make_prerequisite_gate_node(_check_plan_prerequisite, PhaseName.IMPLEMENT.value)
    graph.add_node("prereq_gate_implement", prereq_gate_implement)

    # Gate: VERIFY requires IMPLEMENT completed
    prereq_gate_verify = make_prerequisite_gate_node(_check_implement_prerequisite, PhaseName.VERIFY.value)
    graph.add_node("prereq_gate_verify", prereq_gate_verify)

    # Gate: GAP_PLAN requires VERIFY attempted
    prereq_gate_gap_plan = make_prerequisite_gate_node(_check_verify_prerequisite, PhaseName.GAP_PLAN.value)
    graph.add_node("prereq_gate_gap_plan", prereq_gate_gap_plan)

    # Add artifact gate nodes and their outgoing conditional edges.
    # Gate nodes route to ``next_node`` on proceed or ``human_review`` on
    # needs_review.  Adding the conditional edges here (rather than in the
    # per-phase loop below) ensures the gate's outgoing edges exist
    # regardless of whether its source is a phase node or a critic node —
    # the prior placement skipped the gate-outgoing edges when the gate
    # source was a critic, leaving the gate as a dead-end node.
    for (src, dst), required_phase in gate_edges.items():
        gate_name = _gate_node_name(src, dst)
        graph.add_node(
            gate_name,
            make_artifact_gate_node(required_phase, dst),
        )
        # Route through prerequisite gate when target has one.
        # Without this, critic_plan → gate_* → IMPLEMENT bypasses
        # prereq_gate_implement, skipping the plan_completed invariant check.
        prereq_gate_map = {
            PhaseName.PLAN.value: "prereq_gate_plan",
            PhaseName.IMPLEMENT.value: "prereq_gate_implement",
            PhaseName.VERIFY.value: "prereq_gate_verify",
        }
        actual_dst = prereq_gate_map.get(dst, dst)
        graph.add_conditional_edges(
            gate_name,
            artifact_gate_router,
            {
                "proceed": actual_dst,
                "needs_review": "human_review",
            },
        )

    # ── Prerequisite Gate Router ───────────────────────────────────────
    def prereq_gate_router(state: WorkflowState) -> str:
        """Route based on prerequisite gate check result."""
        return state.get("status")  # "running" → proceed, "needs_review" → human_review

    # Wire the graph
    if start_from_phase:
        graph.add_edge(START, start_from_phase)
    else:
        graph.add_edge(START, phase_seq[0][0])

    for i, (node_name, reviewed_phase) in enumerate(phase_seq):
        is_last = i == len(phase_seq) - 1
        next_node = phase_seq[i + 1][0] if not is_last else None

        # Check if there's a gate between this node and the next
        edge_key = (node_name, next_node) if next_node else None
        has_gate = edge_key in gate_edges if edge_key else False

        # Determine the target for "proceed" (after phase succeeds)
        # This includes both artifact gates AND prerequisite gates
        def get_proceed_target(next_node_name: str | None) -> str:
            """Get the target node, potentially routing through prerequisite gate."""
            if next_node_name is None:
                return END
            # Map target phases to their prerequisite gates
            prereq_gate = {
                PhaseName.PLAN.value: "prereq_gate_plan",
                PhaseName.IMPLEMENT.value: "prereq_gate_implement",
                PhaseName.VERIFY.value: "prereq_gate_verify",
            }
            return prereq_gate.get(next_node_name, next_node_name)

        if node_name.startswith(PhaseName.CRITIC.value):
            # Critic node → conditional edge
            pre_critic = phase_seq[i - 1][0] if i > 0 else phase_seq[0][0]

            # Determine where "passed" routes to
            if has_gate and next_node:
                gate_name = _gate_node_name(node_name, next_node)
                critic_proceed_target: str = gate_name
            elif not is_last and next_node:
                critic_proceed_target = get_proceed_target(next_node)
            else:
                critic_proceed_target = END

            graph.add_conditional_edges(
                node_name,
                critic_router,
                {
                    "passed": critic_proceed_target,
                    "needs_revision": pre_critic,  # rework loop
                    "needs_review": "human_review",  # interrupt for human
                    "failed": END,
                },
            )
        elif has_gate and next_node:
            # Route to the artifact gate node; its outgoing conditional edges
            # were registered in the gate-node loop above.
            gate_name = _gate_node_name(node_name, next_node)
            graph.add_edge(node_name, gate_name)
        elif node_name == PhaseName.VERIFY.value:
            # Verify uses its own router for gap-fix loop support.
            # On passed → END. On needs_gap_fix → prereq_gate_gap_plan → gap_plan → implement → verify.
            # On needs_review (gap attempts exhausted) → human_review.
            graph.add_conditional_edges(
                node_name,
                _verify_router,
                {
                    "passed": END,
                    "needs_gap_fix": "prereq_gate_gap_plan",
                    "needs_review": "human_review",
                    "failed": END,
                },
            )
        elif next_node is not None:
            # Status guard: if the phase produced needs_review, route
            # to human_review instead of charging into the next phase.
            # Without this, a failing specify silently feeds an empty
            # plan, which feeds an empty critic, and so on — burning
            # tokens and exceeding budgets.
            # Also route through prerequisite gate for the next phase.
            target = get_proceed_target(next_node)
            graph.add_conditional_edges(
                node_name,
                _phase_status_router,
                {
                    "proceed": target,
                    "needs_review": "human_review",
                    "failed": END,
                },
            )
        else:
            # Terminal phase: still guard so a needs_review on the
            # final phase routes to human_review for resume support.
            graph.add_conditional_edges(
                node_name,
                _phase_status_router,
                {
                    "proceed": END,
                    "needs_review": "human_review",
                    "failed": END,
                },
            )

    # Wire prerequisite gates to their target phases via conditional edges
    # (failure routes to human_review, success routes to target phase)
    graph.add_conditional_edges(
        "prereq_gate_plan",
        _phase_status_router,
        {
            "proceed": PhaseName.PLAN.value,
            "needs_review": "human_review",
            "failed": END,
        },
    )
    graph.add_conditional_edges(
        "prereq_gate_implement",
        _phase_status_router,
        {
            "proceed": PhaseName.IMPLEMENT.value,
            "needs_review": "human_review",
            "failed": END,
        },
    )
    graph.add_conditional_edges(
        "prereq_gate_verify",
        _phase_status_router,
        {
            "proceed": PhaseName.VERIFY.value,
            "needs_review": "human_review",
            "failed": END,
        },
    )
    graph.add_conditional_edges(
        "prereq_gate_gap_plan",
        _phase_status_router,
        {
            "proceed": PhaseName.GAP_PLAN.value,
            "needs_review": "human_review",
            "failed": END,
        },
    )

    # Wire gap_plan → implement (gap-fix loop: verify → gap_plan → implement → verify)
    graph.add_edge(PhaseName.GAP_PLAN.value, PhaseName.IMPLEMENT.value)

    # Compile with optional checkpointer
    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    return graph.compile(**compile_kwargs)


def _make_critic_node(
    critic_fn: Any,
    reviewed_phase: str,
) -> Any:
    """Create a critic node function that knows which phase it reviews.

    Wraps the generic ``call_critic`` so the reviewed phase is determined
    by the graph position, not by inspecting state artifacts.

    The wrapper is async because ``call_critic`` is async — LangGraph
    handles async node functions natively.

    Args:
        critic_fn: The base critic call function (async).
        reviewed_phase: The phase this critic instance reviews.

    Returns:
        An async node function with the correct reviewed_phase.
    """

    async def critic_node(state: WorkflowState, config: Optional[RunnableConfig] = None) -> dict:
        """Critic node that reviews a specific phase."""
        mark_phase_started(state, f"critic_{reviewed_phase}")
        # Inject which phase this critic reviews into state
        # so _get_reviewed_phase and critic_router can use it
        augmented_state = {**state, "critic_reviewing": reviewed_phase}
        result = await critic_fn(augmented_state, config)
        return result

    return critic_node


def _make_legacy_node(
    phase_name: str,
    call_fn: Any,
) -> Any:
    """Create a legacy node function with phase-start tracking.

    Wraps the generic phase call function so it marks the phase as
    started before executing. This ensures the UI shows the correct
    phase immediately.

    Args:
        phase_name: The phase identifier (e.g. "specify", "plan").
        call_fn: The async call function for this phase.

    Returns:
        An async node function that marks the phase started, then calls
        the original phase function.
    """

    async def legacy_node(state: WorkflowState, config: Optional[RunnableConfig] = None) -> dict:
        """Legacy node wrapper with phase-start tracking."""
        mark_phase_started(state, phase_name)
        result = await call_fn(state, config)
        return result

    return legacy_node


def get_restart_phases(work_type: str) -> list[str]:
    """Return the list of valid phase names for restart_from_phase.

    Filters out critic nodes since restarting into a critic doesn't
    make sense — the critic is always called after its reviewed phase.

    Args:
        work_type: One of the valid WorkType values.

    Returns:
        Sorted list of non-critic phase names from the workflow sequence.

    Raises:
        ValueError: If the work_type is not recognised.
    """
    if work_type not in WORKFLOW_SEQUENCES:
        raise ValueError(
            f"Unknown work type '{work_type}'. Must be one of: {list(WORKFLOW_SEQUENCES.keys())}"
        )
    return sorted(
        name
        for name, _ in WORKFLOW_SEQUENCES[work_type]
        if not name.startswith(PhaseName.CRITIC.value)
    )
