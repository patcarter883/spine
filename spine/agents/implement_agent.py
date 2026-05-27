"""SPINE implement agent — Deep Agent for the IMPLEMENT phase.

The IMPLEMENT phase is now dispatched via the Send API subgraph
(spine/workflow/subgraphs/implement_subgraph.py).  This module is
kept for the phase registry contract (``build_agent_fn``) and as a
fallback when subgraphs are disabled via the ``_SUBGRAPH_ENABLED``
feature flag.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.agents.artifacts import (
    build_current_phase_write_prompt,
)
from spine.agents.factory import build_phase_agent
from spine.agents.implement_tools import build_implement_orchestrator_tools
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState


def build_implement_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the IMPLEMENT phase.

    With the Send API subgraph enabled (default), this agent is not called
    — the subgraph handles all dispatch via ``Send("run_slice_implementer", ...)``.

    Kept for the phase registry contract and for when the
    ``_SUBGRAPH_ENABLED`` feature flag is turned off.
    """
    work_id = state.get("work_id", "")
    workspace_root = state.get("workspace_root", ".")

    execution_waves = state.get("execution_waves", [])
    total_slices = sum(len(wave) for wave in execution_waves)

    system_prompt = (
        "You are the IMPLEMENT phase synthesiser. Slice-implementer subagents "
        "have already been dispatched in parallel by the implement subgraph "
        "router; their results are in your context. Use `read_slice_files` "
        "to load slice definitions and the codebase map for cross-reference, "
        "then call `write_implementation_report` to record the outcome.\n\n"
        f"Total slices to implement: {total_slices}\n\n"
    )

    system_prompt += build_current_phase_write_prompt(
        work_id, PhaseName.IMPLEMENT.value, expected_files=["implementation.md"]
    )

    orchestrator_tools = build_implement_orchestrator_tools(
        workspace_root=workspace_root,
        work_id=work_id,
    )

    agent = build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.IMPLEMENT,
        system_prompt=system_prompt,
        extra_tools=orchestrator_tools,
        skip_filesystem_middleware=True,
    )

    return agent