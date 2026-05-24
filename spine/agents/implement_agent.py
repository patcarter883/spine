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
from spine.agents.subagents import build_phase_subagents
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState


def _build_subagents(
    phase: PhaseName,
    state: WorkflowState,
    config: RunnableConfig | None,
) -> list[Any] | None:
    """Resolve subagent specs for the IMPLEMENT phase."""
    return build_phase_subagents(phase, state, config)


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
        "You are the IMPLEMENT phase orchestrator. Your job is to dispatch "
        "slice-implementer subagents per feature slice and synthesize their "
        "results. Use `read_slice_files` to load slice definitions and the "
        "codebase map, then dispatch subagents via `task` inside `eval`.\n\n"
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
        subagents=_build_subagents(PhaseName.IMPLEMENT, state, config),
        extra_tools=orchestrator_tools,
        skip_filesystem_middleware=True,
    )

    return agent