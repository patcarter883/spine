"""SPINE specify agent — Deep Agent for the SPECIFY phase."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.models.state import WorkflowState


def build_specify_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the SPECIFY phase.

    Creates a deep agent configured for specification generation with
    filesystem tools and research capabilities. Uses a LocalShellBackend
    so the agent can inspect existing project files when writing specs.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config (may contain provider settings).

    Returns:
        A compiled Deep Agent ready for invocation.
    """
    from deepagents import create_deep_agent

    from spine.agents.backend import build_backend

    model = _resolve_model(config)
    workspace_root = state.get("workspace_root", ".")
    backend = build_backend(workspace_root)

    agent = create_deep_agent(
        name="spine-specify",
        model=model,
        backend=backend,
        system_prompt=(
            "You are a specification writer. Given a work description, "
            "create a detailed, actionable specification document.\n\n"
            f"Your workspace root is: {workspace_root}\n\n"
            "The specification should include:\n"
            "1. Overview and objectives\n"
            "2. Requirements (functional and non-functional)\n"
            "3. Constraints and assumptions\n"
            "4. Success criteria\n"
            "5. Dependencies and risks\n\n"
            "Write clear, specific, and testable requirements. "
            "Avoid vague language. Structure the document with clear headers."
        ),
    )

    return agent


def _resolve_model(config: RunnableConfig | None) -> str:
    """Resolve the model identifier from config or SpineConfig."""
    if config and config.get("configurable", {}).get("model"):
        return config["configurable"]["model"]
    from spine.config import SpineConfig

    return SpineConfig.load().resolve_model()
