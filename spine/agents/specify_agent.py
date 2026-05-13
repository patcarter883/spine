"""SPINE specify agent — Deep Agent for the SPECIFY phase."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.models.state import WorkflowState
from spine.agents.helpers import resolve_model, debug_enabled


def build_specify_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the SPECIFY phase.

    Creates a deep agent configured for specification generation. Uses
    subagents for research and documentation tasks.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        A compiled Deep Agent ready for invocation.
    """
    from deepagents import create_deep_agent

    from spine.agents.backend import build_backend

    model = resolve_model(config, session_id=state.get("work_id"))
    workspace_root = state.get("workspace_root", ".")
    backend = build_backend(workspace_root)

    agent = create_deep_agent(
        name="spine-specify",
        model=model,
        backend=backend,
        debug=debug_enabled(),
        system_prompt=(
            "You are a technical specification writer. Given a work description, "
            "produce a detailed specification document.\n\n"
            f"Your workspace root is: {workspace_root}\n\n"
            "The specification should include:\n"
            "1. Overview — summary of what needs to be built\n"
            "2. Requirements — functional and non-functional requirements\n"
            "3. Architecture — high-level design decisions\n"
            "4. Interfaces — API endpoints, data models, contracts\n"
            "5. Success criteria — measurable outcomes\n\n"
            "Be specific and technical. Avoid vague language."
        ),
    )

    return agent
