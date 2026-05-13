"""SPINE plan agent — Deep Agent for the PLAN phase.

Uses the shared :func:`build_phase_agent` factory with context engineering.
Prior artifacts (specification) are referenced by path on disk, not inlined.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.factory import build_phase_agent
from spine.agents.artifacts import build_artifact_prompt


def build_plan_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the PLAN phase.

    Creates a deep agent configured for technical architecture planning.
    Prior artifacts are referenced on disk rather than inlined.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        A compiled Deep Agent ready for invocation.
    """
    workspace_root = state.get("workspace_root", ".")

    system_prompt = (
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
        "to implement directly from this document.\n\n"
        "Prior artifacts from earlier phases are available on disk — "
        "use `read_file` and `grep` to inspect them when needed.\n\n"
        + build_artifact_prompt(
            state.get("artifacts", {}), PhaseName.PLAN.value
        )
    )

    agent = build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.PLAN,
        system_prompt=system_prompt,
    )

    return agent
