"""SPINE plan agent — Deep Agent for the PLAN phase."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.models.state import WorkflowState
from spine.agents.helpers import resolve_model, debug_enabled


def build_plan_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the PLAN phase.

    Creates a deep agent configured for technical architecture planning.
    Uses a LocalShellBackend so the agent can inspect existing project
    files when planning.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        A compiled Deep Agent ready for invocation.
    """
    from deepagents import create_deep_agent

    from spine.agents.backend import build_backend

    model = resolve_model(config)
    workspace_root = state.get("workspace_root", ".")
    backend = build_backend(workspace_root)

    agent = create_deep_agent(
        name="spine-plan",
        model=model,
        backend=backend,
        debug=debug_enabled(),
        system_prompt=(
            "You are a technical architect. Given a specification, "
            "create a detailed technical plan document.\n\n"
            f"Your workspace root is: {workspace_root}\n\n"
            "The plan should include:\n"
            "1. Architecture overview (components, data flow, interfaces)\n"
            "2. Technology choices and rationale\n"
            "3. Module/file structure\n"
            "4. API designs and data models\n"
            "5. Implementation order and dependencies\n"
            "6. Testing strategy\n\n"
            "Be specific about file paths, class names, and interfaces. "
            "The plan must be actionable — another developer should be able "
            "to implement directly from this document."
        ),
    )

    return agent
