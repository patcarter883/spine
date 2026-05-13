"""SPINE implement agent — Deep Agent for the IMPLEMENT phase."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.models.state import WorkflowState


def build_implement_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the IMPLEMENT phase.

    Creates a deep agent configured for code generation. Uses subagents
    for parallel implementation of independent feature slices.

    The agent is given a LocalShellBackend rooted at the project workspace
    so that file writes land on disk and shell commands run in the right
    directory.

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
        name="spine-implement",
        model=model,
        backend=backend,
        system_prompt=(
            "You are an implementation engineer. Given feature slices, "
            "generate production-quality code to implement each one.\n\n"
            "Guidelines:\n"
            "1. Write clean, well-documented code\n"
            "2. Follow the project's coding conventions\n"
            "3. Include type hints and docstrings\n"
            "4. Handle errors gracefully\n"
            "5. Write code that is testable\n"
            "6. For independent slices, use the task tool to delegate "
            "to subagents for parallel execution\n\n"
            f"Your workspace root is: {workspace_root}\n"
            "All file paths should be relative to this root directory. "
            "Use the write_file tool to create files on disk.\n\n"
            "After implementing all slices, provide a summary of what "
            "was created and any decisions made during implementation."
        ),
    )

    return agent


def _resolve_model(config: RunnableConfig | None) -> str:
    """Resolve the model identifier from config or SpineConfig."""
    if config and config.get("configurable", {}).get("model"):
        return config["configurable"]["model"]
    from spine.config import SpineConfig

    return SpineConfig.load().resolve_model()
