"""SPINE workflow composer — builds a LangGraph StateGraph from a WorkType.

The composer reads the WorkType, determines the phase sequence, and wires
the graph with conditional edges for critic review rework loops.

Each critic instance gets a unique node name (e.g. ``critic_specify``,
``critic_plan``) so the same critic function can appear multiple times in
a workflow graph — each reviewing a different preceding phase.

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


def build_workflow_graph(
    work_type: str,
    checkpointer: BaseCheckpointSaver | None = None,
) -> Any:
    """Build a compiled LangGraph StateGraph for the given work type.

    The graph wires phase nodes with:
    - Sequential edges for non-critic phases
    - Conditional edges after critic nodes that route to rework or next phase
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

    # Wire the graph
    graph.add_edge(START, phase_seq[0][0])

    for i, (node_name, reviewed_phase) in enumerate(phase_seq):
        if i < len(phase_seq) - 1:
            next_node = phase_seq[i + 1][0]
        else:
            next_node = None  # last node → END

        if node_name.startswith(PhaseName.CRITIC.value):
            # Critic node → conditional edge
            pre_critic = phase_seq[i - 1][0] if i > 0 else phase_seq[0][0]
            after_critic = next_node or END

            graph.add_conditional_edges(
                node_name,
                critic_router,
                {
                    "passed": after_critic if after_critic != END else END,
                    "needs_revision": pre_critic,  # rework loop
                    "needs_review": END,  # stop for human
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
