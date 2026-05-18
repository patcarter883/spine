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

Currently, only the tasks→implement transition is gated.  Verify always runs
after implement — if implement produced nothing, verify detects and reports
that; there is no reason for a human review gate between those two phases.

Phase sequences by WorkType:
    quick:           TASKS → IMPLEMENT → VERIFY
    critical_quick:  TASKS → CRITIC → IMPLEMENT → VERIFY
    spec:            SPECIFY → PLAN → CRITIC → TASKS → IMPLEMENT → VERIFY
    critical_spec:   SPECIFY → CRITIC_SPECIFY → PLAN → CRITIC_PLAN →
                     TASKS → CRITIC_TASKS → IMPLEMENT → VERIFY
"""

from typing import Any, Callable, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import interrupt

from spine.models.enums import PhaseName, ReviewStatus, WorkType
from spine.models.state import WorkflowState
from spine.workflow.registry import get_registry
from spine.workflow.critic_review import critic_router
from spine.workflow.artifact_gate import (
    make_artifact_gate_node,
    artifact_gate_router,
)
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


# Feature flags for per-phase subgraph migration.
# During rollout, phases can be enabled independently.
_SUBGRAPH_ENABLED: dict[str, bool] = {
    PhaseName.VERIFY.value: True,
    PhaseName.IMPLEMENT.value: True,
    PhaseName.TASKS.value: True,
    PhaseName.SPECIFY.value: True,
    PhaseName.PLAN.value: True,
    PhaseName.CRITIC.value: True,
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
    }


def _plan_state_mapper(parent_state: WorkflowState, config) -> dict:
    work_id = parent_state.get("work_id", "")
    return {
        **_base_state_mapper(parent_state, config),
        "phase": PhaseName.PLAN.value,
        "retry_count": parent_state.get("retry_count", {}).get(PhaseName.PLAN.value, 0),
        "spec_path": f".spine/artifacts/{work_id}/specify",
    }


def _tasks_state_mapper(parent_state: WorkflowState, config) -> dict:
    work_id = parent_state.get("work_id", "")
    work_type = parent_state.get("work_type", "")
    has_spec = "spec" in work_type
    return {
        **_base_state_mapper(parent_state, config),
        "phase": PhaseName.TASKS.value,
        "retry_count": parent_state.get("retry_count", {}).get(PhaseName.TASKS.value, 0),
        "plan_path": f".spine/artifacts/{work_id}/plan" if has_spec else None,
        "spec_path": f".spine/artifacts/{work_id}/specify" if has_spec else None,
    }


def _implement_state_mapper(parent_state: WorkflowState, config) -> dict:
    work_id = parent_state.get("work_id", "")
    return {
        **_base_state_mapper(parent_state, config),
        "phase": PhaseName.IMPLEMENT.value,
        "retry_count": parent_state.get("retry_count", {}).get(PhaseName.IMPLEMENT.value, 0),
        "tasks_path": f".spine/artifacts/{work_id}/tasks",
    }


def _verify_state_mapper(parent_state: WorkflowState, config) -> dict:
    """Map parent WorkflowState to VerifySubgraphState."""
    work_id = parent_state.get("work_id", "")
    work_type = parent_state.get("work_type", "")
    has_spec = "spec" in work_type
    return {
        **_base_state_mapper(parent_state, config),
        "phase": PhaseName.VERIFY.value,
        "retry_count": parent_state.get("retry_count", {}).get(PhaseName.VERIFY.value, 0),
        "tasks_path": f".spine/artifacts/{work_id}/tasks",
        "spec_path": f".spine/artifacts/{work_id}/specify" if has_spec else None,
        "plan_path": f".spine/artifacts/{work_id}/plan" if has_spec else None,
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
            "reviewed_phase_path": f".spine/artifacts/{work_id}/{reviewed_phase}",
        }
    return mapper


def _verify_result_mapper(subgraph_result: dict, parent_state: WorkflowState) -> dict[str, Any]:
    """Map VerifySubgraphState output back to parent WorkflowState."""
    base = make_success_result_mapper(PhaseName.VERIFY.value)(subgraph_result, parent_state)
    phase_status = subgraph_result.get("phase_status", "")
    if phase_status == "needs_review":
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
        base["status"] = "needs_review"
        base["needs_review_phase"] = PhaseName.VERIFY.value
    return base


def _implement_result_mapper(subgraph_result: dict, parent_state: WorkflowState) -> dict[str, Any]:
    """Map ImplementSubgraphState output back to parent WorkflowState."""
    base = make_success_result_mapper(PhaseName.IMPLEMENT.value)(subgraph_result, parent_state)
    phase_status = subgraph_result.get("phase_status", "")
    if phase_status == "needs_review":
        base["status"] = "needs_review"
        base["needs_review_phase"] = PhaseName.IMPLEMENT.value
    elif phase_status == "error":
        base["status"] = "needs_review"
        base["needs_review_phase"] = PhaseName.IMPLEMENT.value
    return base


def _tasks_result_mapper(subgraph_result: dict, parent_state: WorkflowState) -> dict[str, Any]:
    """Map TasksSubgraphState output back to parent WorkflowState."""
    base = make_success_result_mapper(PhaseName.TASKS.value)(subgraph_result, parent_state)
    phase_status = subgraph_result.get("phase_status", "")
    if phase_status == "needs_review":
        base["status"] = "needs_review"
        base["needs_review_phase"] = PhaseName.TASKS.value
    elif phase_status == "error":
        base["status"] = "needs_review"
        base["needs_review_phase"] = PhaseName.TASKS.value
    return base


def _specify_result_mapper(subgraph_result: dict, parent_state: WorkflowState) -> dict[str, Any]:
    """Map SpecifySubgraphState output back to parent WorkflowState."""
    base = make_success_result_mapper(PhaseName.SPECIFY.value)(subgraph_result, parent_state)
    phase_status = subgraph_result.get("phase_status", "")
    if phase_status == "needs_review":
        base["status"] = "needs_review"
        base["needs_review_phase"] = PhaseName.SPECIFY.value
    elif phase_status == "error":
        base["status"] = "needs_review"
        base["needs_review_phase"] = PhaseName.SPECIFY.value
    return base


def _plan_result_mapper(subgraph_result: dict, parent_state: WorkflowState) -> dict[str, Any]:
    """Map PlanSubgraphState output back to parent WorkflowState."""
    base = make_success_result_mapper(PhaseName.PLAN.value)(subgraph_result, parent_state)
    phase_status = subgraph_result.get("phase_status", "")
    if phase_status == "needs_review":
        base["status"] = "needs_review"
        base["needs_review_phase"] = PhaseName.PLAN.value
    elif phase_status == "error":
        base["status"] = "needs_review"
        base["needs_review_phase"] = PhaseName.PLAN.value
    return base


def _critic_result_mapper(reviewed_phase: str):
    """Create a result mapper for a critic subgraph reviewing a specific phase."""
    def mapper(subgraph_result: dict, parent_state: WorkflowState) -> dict[str, Any]:
        # Critic doesn't produce artifacts_output — it produces review results
        base: dict[str, Any] = {
            "current_phase": PhaseName.CRITIC.value,
            "status": "running",
            "prompt_request": None,
        }
        
        structural_result = subgraph_result.get("structural_result", {})
        agent_result = subgraph_result.get("agent_result", {})
        
        # Determine effective feedback from both tiers
        effective_result = agent_result if agent_result else structural_result
        if not effective_result:
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
        elif phase_status == ReviewStatus.NEEDS_REVISION.value:
            # Router will check retry count and route accordingly
            base["status"] = "running"
        
        return base
    return mapper


# ── Phase sequences per work type ──
# Each tuple is (node_name, reviewed_phase_or_None).
# For critic nodes, reviewed_phase tells the critic which phase to review.

WORKFLOW_SEQUENCES: dict[str, list[tuple[str, str | None]]] = {
    WorkType.QUICK.value: [
        (PhaseName.TASKS.value, None),
        (PhaseName.IMPLEMENT.value, None),
        (PhaseName.VERIFY.value, None),
    ],
    WorkType.CRITICAL_QUICK.value: [
        (PhaseName.TASKS.value, None),
        (f"{PhaseName.CRITIC.value}_tasks", PhaseName.TASKS.value),
        (PhaseName.IMPLEMENT.value, None),
        (PhaseName.VERIFY.value, None),
    ],
    WorkType.SPEC.value: [
        (PhaseName.SPECIFY.value, None),
        (PhaseName.PLAN.value, None),
        (f"{PhaseName.CRITIC.value}_plan", PhaseName.PLAN.value),
        (PhaseName.TASKS.value, None),
        (PhaseName.IMPLEMENT.value, None),
        (PhaseName.VERIFY.value, None),
    ],
    WorkType.CRITICAL_SPEC.value: [
        (PhaseName.SPECIFY.value, None),
        (f"{PhaseName.CRITIC.value}_specify", PhaseName.SPECIFY.value),
        (PhaseName.PLAN.value, None),
        (f"{PhaseName.CRITIC.value}_plan", PhaseName.PLAN.value),
        (PhaseName.TASKS.value, None),
        (f"{PhaseName.CRITIC.value}_tasks", PhaseName.TASKS.value),
        (PhaseName.IMPLEMENT.value, None),
        (PhaseName.VERIFY.value, None),
    ],
    WorkType.PLAN.value: [
        (PhaseName.SPECIFY.value, None),
        (PhaseName.PLAN.value, None),
        (f"{PhaseName.CRITIC.value}_plan", PhaseName.PLAN.value),
    ],
    WorkType.PLAN_SPEC.value: [
        (PhaseName.SPECIFY.value, None),
        (f"{PhaseName.CRITIC.value}_specify", PhaseName.SPECIFY.value),
        (PhaseName.PLAN.value, None),
        (f"{PhaseName.CRITIC.value}_plan", PhaseName.PLAN.value),
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
            human_feedback.get("action", "abort")
            if isinstance(human_feedback, dict)
            else "abort"
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
    return "proceed"


def _gate_node_name(source_node: str, next_node: str) -> str:
    """Derive a unique node name for a gate between two phases."""
    return f"gate_{source_node}_to_{next_node}"


def build_workflow_graph(
    work_type: str,
    checkpointer: BaseCheckpointSaver | None = None,
) -> Any:
    """Build a compiled LangGraph StateGraph for the given work type.

    The graph wires phase nodes with:
    - Sequential edges for non-critic phases
    - Conditional edges after critic nodes that route to rework or next phase
    - Artifact gate nodes that check prerequisites before verify/implement
    - Retry counting and needs_review escalation

    Critic nodes are named ``critic_{reviewed_phase}`` to allow multiple
    critic instances in a single workflow (e.g. critical_spec has 3).

    Args:
        work_type: One of "quick", "critical_quick", "spec", "critical_spec".
        checkpointer: Optional BaseCheckpointSaver for persistence.

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

    # ── Build the graph ──
    graph = StateGraph(WorkflowState)

    # Collect which edges need artifact gates.
    # We gate based on the *target* node: implement requires tasks artifacts.
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
                gate_edges[(node_name, next_node_name)] = PhaseName.TASKS.value

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
            }
            _RESULT_MAPPERS = {
                PhaseName.VERIFY.value: _verify_result_mapper,
                PhaseName.IMPLEMENT.value: _implement_result_mapper,
                PhaseName.TASKS.value: _tasks_result_mapper,
                PhaseName.SPECIFY.value: _specify_result_mapper,
                PhaseName.PLAN.value: _plan_result_mapper,
            }

            builder_fn = _SUBGRAPH_BUILDER_REGISTRY.get(node_name)
            if builder_fn is not None:
                subgraph = builder_fn().compile()
                state_mapper = _STATE_MAPPERS[node_name]
                result_mapper = _RESULT_MAPPERS[node_name]
                graph.add_node(
                    node_name,
                    make_subgraph_node(
                        subgraph,
                        node_name,
                        state_mapper,
                        result_mapper,
                        use_per_phase_checkpointer=True,
                    ),
                )
            else:
                # Fallback to legacy for phases not yet migrated
                phase_def = registry.require(node_name)
                if phase_def.subgraph_node_fn:
                    graph.add_node(node_name, phase_def.subgraph_node_fn)
                elif phase_def.call_fn:
                    graph.add_node(node_name, phase_def.call_fn)
                else:
                    raise ValueError(
                        f"Phase '{node_name}' has no call_fn or subgraph_node_fn"
                    )
        else:
            # Legacy phase node
            phase_def = registry.require(node_name)
            if phase_def.call_fn is None:
                raise ValueError(f"Phase '{node_name}' has no call_fn (legacy mode)")
            graph.add_node(node_name, phase_def.call_fn)

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

    graph.add_node("human_review", _human_review_interrupt)
    graph.add_conditional_edges(
        "human_review",
        _make_human_review_router(phase_seq),
        _hr_ends,
    )

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
        graph.add_conditional_edges(
            gate_name,
            artifact_gate_router,
            {
                "proceed": dst,
                "needs_review": "human_review",
            },
        )

    # Wire the graph
    graph.add_edge(START, phase_seq[0][0])

    for i, (node_name, reviewed_phase) in enumerate(phase_seq):
            is_last = i == len(phase_seq) - 1
            next_node = phase_seq[i + 1][0] if not is_last else None

            # Check if there's a gate between this node and the next
            edge_key = (node_name, next_node) if next_node else None
            has_gate = edge_key in gate_edges if edge_key else False

            if node_name.startswith(PhaseName.CRITIC.value):
                # Critic node → conditional edge
                pre_critic = phase_seq[i - 1][0] if i > 0 else phase_seq[0][0]

                # Determine where "passed" routes to
                if has_gate and next_node:
                    gate_name = _gate_node_name(node_name, next_node)
                    critic_proceed_target: str = gate_name
                elif not is_last and next_node:
                    critic_proceed_target = next_node
                else:
                    critic_proceed_target = END

                graph.add_conditional_edges(
                    node_name,
                    critic_router,
                    {
                        "passed": critic_proceed_target,
                        "needs_revision": pre_critic,  # rework loop
                        "needs_review": "human_review",  # interrupt for human
                    },
                )
            elif has_gate and next_node:
                # Route to the gate node; its outgoing conditional edges
                # were registered in the gate-node loop above.
                gate_name = _gate_node_name(node_name, next_node)
                graph.add_edge(node_name, gate_name)
            elif next_node is not None:
                # Status guard: if the phase produced needs_review, route
                # to human_review instead of charging into the next phase.
                # Without this, a failing specify silently feeds an empty
                # plan, which feeds an empty critic, and so on — burning
                # tokens and exceeding budgets.
                graph.add_conditional_edges(
                    node_name,
                    _phase_status_router,
                    {
                        "proceed": next_node,
                        "needs_review": "human_review",
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
                    },
                )

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
        # Inject which phase this critic reviews into state
        # so _get_reviewed_phase and critic_router can use it
        augmented_state = {**state, "critic_reviewing": reviewed_phase}
        result = await critic_fn(augmented_state, config)
        return result

    return critic_node
