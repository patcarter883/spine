"""SPINE implement agent — Deep Agent for the IMPLEMENT phase.

Uses the shared :func:`build_phase_agent` factory with summarization
middleware enabled (IMPLEMENT can be long-running with many slices).
RLM guidance via the ``rlm-pattern`` skill, prior artifacts on disk.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.factory import build_phase_agent
from spine.agents.artifacts import build_artifact_prompt


def build_implement_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the IMPLEMENT phase.

    Creates a deep agent configured for code generation with summarization
    middleware for long-running slice implementation.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        A compiled Deep Agent ready for invocation.
    """
    workspace_root = state.get("workspace_root", ".")

    system_prompt = (
        "You are an implementation engineer. Given feature slices, "
        "generate production-quality code to implement each one.\n\n"
        "Guidelines:\n"
        "1. Write clean, well-documented code\n"
        "2. Follow the project's existing coding conventions and patterns\n"
        "3. Include appropriate type annotations and docstrings\n"
        "4. Handle errors gracefully\n"
        "5. Write code that is testable\n"
        "6. For independent slices, use the task tool to delegate "
        "to subagents for parallel execution\n\n"
        f"Your workspace root is: {workspace_root}\n"
        "All file paths should be relative to this root directory. "
        "Use the write_file tool to create files on disk.\n\n"
        "After implementing all slices, provide a summary of what "
        "was created and any decisions made during implementation.\n\n"
        "Prior artifacts (specification, plan, feature slices) are on disk — "
        "use `read_file` and `grep` to inspect them when needed. "
        "Do NOT load everything into context at once.\n\n"
        + build_artifact_prompt(
            state.get("artifacts", {}), PhaseName.IMPLEMENT.value
        )
    )

    agent = build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.IMPLEMENT,
        system_prompt=system_prompt,
        add_summarization=True,  # IMPLEMENT can be long-running
    )

    return agent
