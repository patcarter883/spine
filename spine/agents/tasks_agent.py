"""SPINE tasks agent — Deep Agent for the TASKS (decomposition) phase.

Uses the shared :func:`build_phase_agent` factory.  Structured
explore→decompose workflow with conditional researcher subagent dispatch
for quick workflows (no prior spec/plan exist).

For quick / critical_quick workflows, the agent uses ``researcher``
subagents dispatched via the interpreter to explore the codebase in
parallel — keeping the main agent's context lean and preventing
exploration-exhaustion deadlock.
"""

from __future__ import annotations

from typing import Any

import logging

from langchain_core.runnables import RunnableConfig
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.factory import build_phase_agent
from spine.agents.artifacts import build_artifact_prompt
from spine.agents.subagents import build_phase_subagents

logger = logging.getLogger(__name__)


def _build_subagents(
    phase: PhaseName,
    state: WorkflowState,
    config: RunnableConfig | None,
) -> list[Any] | None:
    """Resolve subagent specs for the TASKS phase.

    Returns researcher subagents when the workflow type includes ``spec``
    or ``quick`` (both need codebase exploration), **unless** the quick
    workflow is trivial (short UI-only description).  Returns ``None`` for
    trivial quick workflows and for rework passes where prior context
    already exists.
    """
    work_type = state.get("work_type", "")
    if "quick" in work_type and "critical" not in work_type:
        description = state.get("description", "")
        # Skip researcher dispatch for trivial quick tasks to save tokens
        if len(description) < 150:
            logger.info(
                "[%s] TASKS: skipping researcher subagents for trivial quick task "
                "(description %d chars)",
                state.get("work_id", ""),
                len(description),
            )
            return None
    if "quick" in work_type or "spec" in work_type:
        return build_phase_subagents(phase, state, config)
    return None


def build_tasks_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the TASKS phase.

    Creates a deep agent configured for decomposing plans into feature
    slices with dependency tracking.  Researcher subagents are provisioned
    for quick / spec workflows where codebase exploration is needed.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        A compiled Deep Agent ready for invocation.
    """
    workspace_root = state.get("workspace_root", ".")
    work_id = state.get("work_id", "")
    work_type = state.get("work_type", "")
    is_quick = "quick" in work_type
    tasks_artifact_dir = f".spine/artifacts/{work_id}/tasks"

    # ── Base system prompt ────────────────────────────────────────────
    system_prompt = (
        "You are a task decomposition specialist. Given a work description, "
        "break it into smaller, executable feature slices.\n\n"
        f"Your workspace root is: {workspace_root}\n\n"
        "## Workflow (follow this order)\n\n"
    )

    if is_quick:
        system_prompt += (
            "### Phase 1: Explore (1-2 turns)\n"
            "This is a quick workflow — no prior spec or plan exists.\n"
            "Use `eval` + researcher subagents for parallel exploration:\n"
            "- Dispatch 2-3 researcher subagents via "
            "`Promise.all(tools.task(...))`\n"
            "- Each researcher investigates one relevant module\n"
            "- Synthesize results in eval code\n\n"
        )
    else:
        system_prompt += (
            "### Phase 1: Explore (1-2 turns)\n"
            "Read prior artifacts (spec, plan) from disk — batch read them.\n\n"
        )

    system_prompt += (
        "### Phase 2: Decompose (1-2 turns)\n"
        "Write feature slices to disk:\n"
        f"- Write `slice-<name>.md` files to `{tasks_artifact_dir}/`\n"
        f"- Write `tasks.md` summary to `{tasks_artifact_dir}/`\n"
        f"- Write `codebase-map.md` to `{tasks_artifact_dir}/` (see below)\n"
        "- Group by dependency waves (DAG structure)\n\n"
        "## codebase-map.md\n"
        f"Write `codebase-map.md` to `{tasks_artifact_dir}/` — a structured "
        "summary of your exploration findings:\n"
        "- File paths with brief descriptions (what each file does)\n"
        "- Key classes and functions (names, signatures, line ranges)\n"
        "- Import chains and dependencies between the relevant modules\n"
        "- Conventions discovered (naming, patterns, error handling)\n"
        "This map eliminates re-exploration by subsequent phases.\n\n"
        "## Rules\n"
        "- You MUST call write_file — conversation-only output is lost\n"
        "- Batch reads — read ≥3 files per turn\n"
        "- Spend at most 2-3 turns exploring, then start writing\n"
        "- Each slice: name, description, files to modify, dependencies, "
        "acceptance criteria, complexity\n\n"
    )

    system_prompt += (
        "Output the slices in a structured markdown format with clear "
        "dependency annotations.\n\n"
        "**REMINDER**: You must call `write_file` to write artifacts to disk. "
        "Conversation-only output is lost — the next phase has nothing to "
        "implement.\n\n"
        "Prior artifacts from earlier phases are available on disk — "
        "use `read_file` and `grep` to inspect them when needed.\n\n"
        + build_artifact_prompt(
            state.get("artifacts", {}), PhaseName.TASKS.value, work_id=work_id
        )
    )

    agent = build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.TASKS,
        system_prompt=system_prompt,
        add_summarization=True,  # TASKS can be long-running with researcher subagents
        subagents=_build_subagents(PhaseName.TASKS, state, config),
    )

    return agent
