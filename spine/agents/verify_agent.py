"""SPINE verify agent — Deep Agent for the VERIFY phase.

The VERIFY phase is now dispatched via the Send API subgraph
(spine/workflow/subgraphs/verify_subgraph.py).  This module is
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
from spine.agents.verify_tools import build_verify_orchestrator_tools
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState


def build_verify_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the VERIFY phase.

    With the Send API subgraph enabled (default), this agent is not called
    — the subgraph handles all dispatch via ``Send("run_slice_verifier", ...)``.

    Kept for the phase registry contract and for when the
    ``_SUBGRAPH_ENABLED`` feature flag is turned off.
    """
    work_id = state.get("work_id", "")
    workspace_root = state.get("workspace_root", ".")

    custom_tools = build_verify_orchestrator_tools(workspace_root, work_id)

    system_prompt = (
        "You are the VERIFY phase synthesiser. Slice-verifier subagents "
        "have already been dispatched in parallel by the verify subgraph "
        "router; their verdicts are in your context. Use `read_verify_context` "
        "to load structured slice definitions and implementation results, then "
        "call `write_verification_report` to record the combined verdict.\n\n"
    )
    system_prompt += build_current_phase_write_prompt(
        work_id, PhaseName.VERIFY.value, expected_files=["verification.md"]
    )

    agent = build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.VERIFY,
        system_prompt=system_prompt,
        extra_tools=custom_tools,
        skip_filesystem_middleware=True,
    )

    return agent