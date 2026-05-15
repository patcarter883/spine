"""SPINE tasks agent — Deep Agent for the TASKS (decomposition) phase.

Uses the shared :func:`build_phase_agent` factory.  RLM guidance is provided
via the ``rlm-pattern`` skill (progressive disclosure) and decomposition
guidance via the ``feature-slice-decomposition`` skill.

For quick / critical_quick workflows, the agent uses ``researcher`` subagents
dispatched via the interpreter to explore the codebase in parallel — the same
pattern SPECIFY uses for research.  This keeps the main agent's context lean
and prevents exploration-exhaustion deadlock.
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
    """Resolve subagent specs for the TASKS phase.

    Returns researcher subagents when the workflow type includes ``spec``
    or ``quick`` (both need codebase exploration).  Returns ``None`` for
    rework passes where prior context already exists.
    """
    work_type = state.get("work_type", "")
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

    # ── Base system prompt ────────────────────────────────────────────
    system_prompt = (
        "You are a task decomposition specialist. Given a plan, "
        "break it into smaller, executable feature slices.\n\n"
        f"Your workspace root is: {workspace_root}\n\n"
        "## YOUR ROLE\n\n"
        "You are the DECOMPOSER — not a researcher, not a planner. Your job is:\n"
        "1. Gather enough context (via researcher subagents or direct reads)\n"
        "2. Write `slice-<name>.md` files to disk using `write_file`\n"
        "3. Write a `tasks.md` summary to disk using `write_file`\n\n"
        "**You MUST call `write_file` to produce artifacts.** Describing slices\n"
        "in conversation without writing them to disk means the next phase has\n"
        "nothing to implement. Your final output must be files on disk.\n\n"
        "## Phase output format\n\n"
        "For each feature slice, specify:\n"
        "1. Name and description\n"
        "2. Files to create or modify\n"
        "3. Dependencies (which slices must complete first)\n"
        "4. Acceptance criteria\n"
        "5. Estimated complexity (small/medium/large)\n\n"
        "Group slices by dependency waves — slices with no dependencies "
        "can run in parallel. Use a DAG structure to show ordering.\n\n"
    )

    # ── RLM strategy: conditional on work type ────────────────────────
    # Quick workflows have no prior spec/plan — the agent must explore the
    # codebase itself.  Researcher subagents dispatched via the interpreter
    # do this in parallel, keeping the main agent's context lean.
    if is_quick:
        system_prompt += (
            "## Codebase exploration strategy (quick workflow)\n\n"
            "This is a quick workflow with no prior specification or plan. "
            "You must explore the codebase yourself to understand what needs "
            "to change — but do NOT read files one-by-one into conversation. "
            "Use the interpreter (`eval` tool) and researcher subagents:\n\n"
            "**Step 1 — Identify modules**: Use `grep` or `glob` in `eval` "
            "to find the 3-5 files or modules most relevant to the task. "
            "Load file paths into interpreter variables.\n\n"
            "**Step 2 — Parallel research**: Dispatch `researcher` subagents "
            "via `tools.task()` from `eval` — one per module. Use "
            "`Promise.all()` for independent modules, `Promise.allSettled()` "
            "for error tolerance. Each researcher investigates one area and "
            "returns structured findings.\n\n"
            "**Step 3 — Write slices**: IMMEDIATELY after receiving researcher "
            "results, write `slice-<name>.md` files and a `tasks.md` summary "
            "using the `write_file` tool directly. Do NOT keep exploring.\n\n"
            "**Budget**: Spend at most 2-3 turns on exploration. After that, "
            "you MUST start writing. Incomplete slices with partial knowledge "
            "are better than no slices at all.\n\n"
        )
    else:
        system_prompt += (
            "**RLM strategy for decomposition:** Use `eval` to read prior "
            "artifacts and the codebase in bulk, extract relevant structure "
            "into interpreter variables, then write slice files via "
            "`write_file`. For large codebases, dispatch `researcher` "
            "subagents per module via `tools.task()` from eval — collect "
            "results in the interpreter, then synthesize slices.\n\n"
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
        subagents=_build_subagents(PhaseName.TASKS, state, config),
    )

    return agent
