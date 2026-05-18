"""SPINE specify agent — Deep Agent for the SPECIFY phase.

Uses the shared :func:`build_phase_agent` factory with context engineering:
artifacts on disk, memory, skills, and SpineContext.  The RLM pattern
guidance is provided via the ``rlm-pattern`` skill (progressive disclosure)
instead of hardcoded in the system prompt.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.factory import build_phase_agent
from spine.agents.artifacts import build_artifact_prompt, build_current_phase_write_prompt
from spine.agents.subagents import build_phase_subagents


def _build_subagents(
    phase: PhaseName,
    state: WorkflowState,
    config: RunnableConfig | None,
) -> list[Any] | None:
    """Resolve subagent specs for the SPECIFY phase."""
    return build_phase_subagents(phase, state, config)


def build_specify_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the SPECIFY phase.

    Creates a deep agent configured for specification generation. Context
    engineering features (memory, skills, artifact references) are handled
    by the shared factory.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        A compiled Deep Agent ready for invocation.
    """
    work_id = state.get("work_id", "")

    system_prompt = (
        "You are a technical specification writer. Given a work description, "
        "produce a detailed specification document.\n\n"
        "Your filesystem is rooted at the project workspace. "
        "Use relative paths (e.g. `src/main.py`, `.spine/artifacts/...`).\n\n"
        "The specification should include:\n"
        "1. Overview — summary of what needs to be built\n"
        "2. Requirements — functional and non-functional requirements\n"
        "3. Architecture — high-level design decisions\n"
        "4. Interfaces — API endpoints, data models, contracts\n"
        "5. Success criteria — measurable outcomes\n\n"
        "Be specific and technical. Avoid vague language.\n\n"
        "Prior artifacts from earlier phases are available on disk — "
        "use `read_file` and `grep` to inspect them when needed. "
        "Do NOT load everything into context at once.\n\n"
        "When the interpreter is available, seed it with context on your first turn:\n"
        "```python\n"
        + f'globalThis.context = {{"work_id": "{work_id}", "phase": "specify", "artifact_dir": ".spine/artifacts/{work_id}/specify"}};\\n'
        + "```\n\n"
        + build_current_phase_write_prompt(
            work_id, PhaseName.SPECIFY.value, expected_files=["specification.md"]
        )
        + build_artifact_prompt(
            state.get("artifacts", {}), PhaseName.SPECIFY.value, work_id=work_id
        )
    )

    agent = build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.SPECIFY,
        system_prompt=system_prompt,
        subagents=_build_subagents(PhaseName.SPECIFY, state, config),
        add_summarization=True,  # SPECIFY can accumulate 80+ tool results
    )

    return agent
