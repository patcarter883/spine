"""SPINE verify agent — Deep Agent for the VERIFY phase.

Uses the shared :func:`build_phase_agent` factory with summarization
middleware enabled (VERIFY can be long-running across many slices).
RLM guidance via the ``rlm-pattern`` skill, verification guidance via
the ``code-review`` skill, prior artifacts on disk.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.factory import build_phase_agent
from spine.agents.artifacts import build_artifact_prompt
from spine.agents.subagents import build_phase_subagents


def _build_subagents(
    phase: PhaseName,
    state: WorkflowState,
    config: RunnableConfig | None,
) -> list[Any] | None:
    """Resolve subagent specs for the VERIFY phase."""
    return build_phase_subagents(phase, state, config)


def build_verify_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the VERIFY phase.

    Creates a deep agent configured for verification with summarization
    middleware for long-running multi-slice verification.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        A compiled Deep Agent ready for invocation.
    """
    workspace_root = state.get("workspace_root", ".")

    system_prompt = (
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
        "End your report with a clear VERIFIED or NOT VERIFIED verdict.\n\n"
        "Prior artifacts (specification, plan, feature slices) are on disk — "
        "use `read_file` and `grep` to inspect them when needed. "
        "Do NOT load everything into context at once.\n\n"
        + build_artifact_prompt(
            state.get("artifacts", {}), PhaseName.VERIFY.value
        )
    )

    agent = build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.VERIFY,
        system_prompt=system_prompt,
        add_summarization=True,  # VERIFY can be long-running
        subagents=_build_subagents(PhaseName.VERIFY, state, config),
    )

    return agent
