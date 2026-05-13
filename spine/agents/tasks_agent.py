"""SPINE tasks agent — Deep Agent for the TASKS (decomposition) phase.

Uses the shared :func:`build_phase_agent` factory.  RLM guidance is provided
via the ``rlm-pattern`` skill (progressive disclosure) and decomposition
guidance via the ``feature-slice-decomposition`` skill.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.factory import build_phase_agent
from spine.agents.artifacts import build_artifact_prompt


def build_tasks_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the TASKS phase.

    Creates a deep agent configured for decomposing plans into feature
    slices with dependency tracking.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        A compiled Deep Agent ready for invocation.
    """
    workspace_root = state.get("workspace_root", ".")

    system_prompt = (
        "You are a task decomposition specialist. Given a plan, "
        "break it into smaller, executable feature slices.\n\n"
        f"Your workspace root is: {workspace_root}\n\n"
        "For each feature slice, specify:\n"
        "1. Name and description\n"
        "2. Files to create or modify\n"
        "3. Dependencies (which slices must complete first)\n"
        "4. Acceptance criteria\n"
        "5. Estimated complexity (small/medium/large)\n\n"
        "Group slices by dependency waves — slices with no dependencies "
        "can run in parallel. Use a DAG structure to show ordering.\n\n"
        "Output the slices in a structured markdown format with clear "
        "dependency annotations.\n\n"
        "Prior artifacts from earlier phases are available on disk — "
        "use `read_file` and `grep` to inspect them when needed.\n\n"
        + build_artifact_prompt(
            state.get("artifacts", {}), PhaseName.TASKS.value
        )
    )

    agent = build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.TASKS,
        system_prompt=system_prompt,
    )

    return agent
