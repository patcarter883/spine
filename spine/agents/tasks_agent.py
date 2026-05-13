"""SPINE tasks agent — Deep Agent for the TASKS (decomposition) phase."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.models.state import WorkflowState


def build_tasks_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the TASKS phase.

    Creates a deep agent configured for decomposing plans into feature slices
    with dependency tracking. Uses a LocalShellBackend so the agent can
    inspect existing project files when planning slices.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        A compiled Deep Agent ready for invocation.
    """
    from deepagents import create_deep_agent

    from spine.agents.backend import build_backend

    model = _resolve_model(config)
    workspace_root = state.get("workspace_root", ".")
    backend = build_backend(workspace_root)

    agent = create_deep_agent(
        name="spine-tasks",
        model=model,
        backend=backend,
        system_prompt=(
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
            "dependency annotations."
        ),
    )

    return agent


def _resolve_model(config: RunnableConfig | None) -> str:
    """Resolve the model identifier from config or SpineConfig."""
    if config and config.get("configurable", {}).get("model"):
        return config["configurable"]["model"]
    from spine.config import SpineConfig

    return SpineConfig.load().resolve_model()
