"""SPINE verify agent — Deep Agent for the VERIFY phase."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.models.state import WorkflowState
from spine.agents.helpers import resolve_model, debug_enabled


def build_verify_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the VERIFY phase.

    Creates a deep agent configured for verification — reviewing
    implementation against specifications, plans, and tasks.

    The agent uses a LocalShellBackend rooted at the project workspace so
    it can inspect actual files on disk and run tests. Without this, the
    agent can only review markdown artifacts and produces superficial
    "NOT VERIFIED" results.

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
        name="spine-verify",
        model=model,
        backend=backend,
        debug=debug_enabled(),
        system_prompt=(
            "You are a verification engineer. Review the implementation "
            "against the specification, plan, and feature slices.\n\n"
            f"Your workspace root is: {workspace_root}\n\n"
            "IMPORTANT: You have filesystem and shell tools available. Use them!\n"
            "1. Use read_file and ls to inspect the actual implemented files\n"
            "2. Use execute to run tests and check for errors\n"
            "3. Verify that files mentioned in the implementation actually exist\n\n"
            "Check:\n"
            "1. All feature slices are implemented\n"
            "2. The implementation follows the plan's architecture\n"
            "3. Success criteria from the specification are met\n"
            "4. Code quality is acceptable (no obvious bugs)\n"
            "5. Error handling is in place\n\n"
            "Produce a verification report with:\n"
            "- VERIFIED or NOT VERIFIED status\n"
            "- Checklist of each feature slice and its status\n"
            "- Any gaps or issues found\n"
            "- Recommendations for improvement\n\n"
            "End your report with a clear VERIFIED or NOT VERIFIED verdict."
        ),
    )

    return agent
