"""SPINE critic agent — Deep Agent for the CRITIC phase."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.models.state import WorkflowState
from spine.agents.helpers import resolve_model, debug_enabled


def build_critic_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the CRITIC phase.

    Creates a deep agent configured for quality review of phase outputs.
    Uses a LocalShellBackend so the critic can inspect actual files on
    disk when reviewing implementation artifacts.

    The critic evaluates completeness, correctness, and quality.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config (may contain provider settings).

    Returns:
        A compiled Deep Agent ready for invocation.
    """
    from deepagents import create_deep_agent

    from spine.agents.backend import build_backend

    model = resolve_model(config)
    workspace_root = state.get("workspace_root", ".")
    backend = build_backend(workspace_root)

    agent = create_deep_agent(
        name="spine-critic",
        model=model,
        backend=backend,
        debug=debug_enabled(),
        system_prompt=(
            "You are a quality reviewer. Review the output of a workflow "
            "phase and determine if it meets quality standards.\n\n"
            f"Your workspace root is: {workspace_root}\n\n"
            "You have filesystem and shell tools available. Use them to:\n"
            "- Inspect actual files when reviewing implementation\n"
            "- Run linters or tests to check code quality\n"
            "- Verify that referenced files actually exist\n\n"
            "Evaluate:\n"
            "1. Completeness — all required elements are present\n"
            "2. Correctness — the content is technically accurate\n"
            "3. Clarity — the document is well-structured and understandable\n"
            "4. Actionability — the output can be used by the next phase\n\n"
            "Respond with one of:\n"
            "- PASSED — the phase output meets quality standards\n"
            "- NEEDS_REVISION — the output needs improvement (specify what)\n"
            "- NEEDS_REVIEW — the output requires human judgment\n\n"
            "Always explain your reasoning and provide specific suggestions "
            "for improvement when recommending revision."
        ),
    )

    return agent
