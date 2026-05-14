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

Phase sequences by WorkType:
    quick:           TASKS → IMPLEMENT → VERIFY
    critical_quick:  TASKS → CRITIC → IMPLEMENT → VERIFY
    spec:            SPECIFY → PLAN → CRITIC → TASKS → IMPLEMENT → VERIFY
    critical_spec:   SPECIFY → CRITIC_SPECIFY → PLAN → CRITIC_PLAN →
                     TASKS → CRITIC_TASKS → IMPLEMENT → VERIFY
"""

from __future__ import annotations

from typing import Any

from typing import Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver

from spine.models.enums import PhaseName, WorkType
from spine.models.state import WorkflowState
from spine.workflow.registry import get_registry
from spine.workflow.critic_review import critic_router
from spine.workflow.artifact_gate import (
    make_artifact_gate_node,
    artifact_gate_router,
)


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
}


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
    # We gate based on the *target* node: verify requires implement artifacts,
    # implement requires tasks artifacts.  The source node can be any phase
    # (including a critic node) — we look at what comes just before the target.
    gate_edges: dict[tuple[str, str], str] = {}  # (src, dst) → required_phase
    for i, (node_name, _reviewed_phase) in enumerate(phase_seq):
        if i < len(phase_seq) - 1:
            next_node_name = phase_seq[i + 1][0]
            if next_node_name == PhaseName.VERIFY.value:
                gate_edges[(node_name, next_node_name)] = PhaseName.IMPLEMENT.value
            elif next_node_name == PhaseName.IMPLEMENT.value:
                gate_edges[(node_name, next_node_name)] = PhaseName.TASKS.value

    # Add all phase/critic nodes
    for node_name, reviewed_phase in phase_seq:
        if node_name.startswith(PhaseName.CRITIC.value):
            # Critic node — use call_critic, store reviewed_phase in a closure
            critic_def = registry.require(PhaseName.CRITIC.value)
            _reviewed = reviewed_phase or "unknown"
            graph.add_node(
                node_name,
                _make_critic_node(critic_def.call_fn, _reviewed),
            )
        else:
            # Regular phase node
            phase_def = registry.require(node_name)
            graph.add_node(node_name, phase_def.call_fn)

    # Add artifact gate nodes
    for (src, dst), required_phase in gate_edges.items():
        gate_name = _gate_node_name(src, dst)
        graph.add_node(
            gate_name,
            make_artifact_gate_node(required_phase, dst),
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
                    "needs_review": END,  # stop for human
                },
            )
        elif has_gate and next_node:
            # Route to the gate node, which then conditionally routes
            gate_name = _gate_node_name(node_name, next_node)
            graph.add_edge(node_name, gate_name)

            # Gate node → conditional edge (proceed or END)
            graph.add_conditional_edges(
                gate_name,
                artifact_gate_router,
                {
                    "proceed": next_node,
                    "needs_review": END,
                },
            )
        elif next_node is not None:
            graph.add_edge(node_name, next_node)
        else:
            graph.add_edge(node_name, END)

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

    Args:
        critic_fn: The base critic call function.
        reviewed_phase: The phase this critic instance reviews.

    Returns:
        A node function with the correct reviewed_phase.
    """

    def critic_node(state: WorkflowState, config: Optional[RunnableConfig] = None) -> dict:
        """Critic node that reviews a specific phase."""
        # Inject which phase this critic reviews into state
        # so _get_reviewed_phase and critic_router can use it
        augmented_state = {**state, "critic_reviewing": reviewed_phase}
        result = critic_fn(augmented_state, config)
        return result

    return critic_node
