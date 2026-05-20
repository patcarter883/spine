"""SPINE tasks agent — Deep Agent for the TASKS (decomposition) phase.

Uses the shared :func:`build_phase_agent` factory. Structured
explore→decompose workflow with conditional researcher subagent dispatch
for quick workflows (no prior spec/plan exist).

The TASKS phase is upstream of two restricted orchestrator phases
(IMPLEMENT, VERIFY) that depend on the artifacts produced here. The
codebase-map.md it produces must be detailed enough that downstream
subagents can locate exact code without re-exploring.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.agents.artifacts import build_artifact_prompt
from spine.agents.factory import build_phase_agent
from spine.agents.subagents import build_phase_subagents
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState

logger = logging.getLogger(__name__)


def _build_subagents(
    phase: PhaseName,
    state: WorkflowState,
    config: RunnableConfig | None,
) -> list[Any] | None:
    """Resolve subagent specs for the TASKS phase.

    Returns researcher subagents when the workflow type includes ``spec``
    or ``quick`` (both need codebase exploration), **unless** the quick
    workflow is trivial (short UI-only description). Returns ``None`` for
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

    Decomposes the work description into feature slices and produces the
    codebase-map.md that downstream IMPLEMENT and VERIFY phases depend on.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        A compiled Deep Agent ready for invocation.
    """
    work_id = state.get("work_id", "")
    work_type = state.get("work_type", "")
    is_quick = "quick" in work_type
    tasks_artifact_dir = f".spine/artifacts/{work_id}/tasks"

    system_prompt = _build_tasks_prompt(
        work_id=work_id,
        tasks_dir=tasks_artifact_dir,
        is_quick=is_quick,
    ) + build_artifact_prompt(
        state.get("artifacts", {}), PhaseName.TASKS.value, work_id=work_id
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


# ── Prompt builder ─────────────────────────────────────────────────────


def _build_tasks_prompt(*, work_id: str, tasks_dir: str, is_quick: bool) -> str:
    """Build the tasks decomposition system prompt."""
    if is_quick:
        explore_section = (
            "### Step 1 — Explore (1-2 turns)\n"
            "Quick workflow — no prior spec or plan exists. Dispatch 2-3 "
            "`researcher` subagents in parallel inside one `eval` call via "
            "`Promise.all(tools.task(...))`. Each researcher investigates "
            "ONE module relevant to the work description.\n\n"
            "Subagent_type MUST be `researcher` (only valid type at this phase).\n\n"
        )
    else:
        explore_section = (
            "### Step 1 — Read prior artifacts (1 turn)\n"
            "Batch-read the spec and plan artifacts in one response. Do not "
            "re-explore — the plan already contains the design.\n\n"
        )

    return (
        "You are a task decomposition specialist. Break the work description "
        "into executable feature slices and produce a codebase map that "
        "downstream IMPLEMENT and VERIFY orchestrators will rely on.\n\n"
        "Filesystem is rooted at the workspace; use relative paths.\n\n"
        "## Why codebase-map.md matters\n"
        "The IMPLEMENT and VERIFY phases are DISPATCH-ONLY orchestrators — "
        "they cannot edit files themselves. They give your codebase-map.md "
        "directly to subagents. If the map lacks file paths, line ranges, "
        "or modification targets, subagents will waste turns re-exploring.\n\n"
        "## Workspace grounding — CRITICAL\n"
        "You are working on the **specific project in your workspace**, not a "
        "generic software project. Every file path you write in any artifact "
        "MUST refer to a file that actually exists in the workspace (for files "
        "to modify) or a directory that exists (for new files).\n\n"
        "**Before writing any slice or codebase-map.md:**\n"
        "Use `glob` or `ls` to confirm that at least 3 file paths from your "
        "researchers' findings exist on disk. If no researcher-identified paths "
        "exist in the workspace, STOP — you are working on the wrong project or "
        "have the wrong workspace root. Re-read `./` with `ls` to orient yourself "
        "before writing anything.\n\n"
        "**Forbidden:** Do NOT invent generic paths like `src/main.py`, "
        "`api/routes.py`, `web/components/`, or any path you have not confirmed "
        "exists (for modifications) or whose parent directory you have not "
        "confirmed exists (for new files). An artifact gate will reject slices "
        "that reference non-existent workspace paths.\n\n"
        "## Workflow\n\n"
        f"{explore_section}"
        "### Step 1b — Verify workspace grounding (1 turn)\n"
        "After researchers complete, pick 3 file paths from their `file_map` "
        "and confirm they exist: `glob` them or `ls` their parent directory. "
        "If none exist, `ls ./` and re-orient before proceeding.\n\n"
        "### Step 2 — Decompose (1-2 turns)\n"
        f"Write to `{tasks_dir}/`:\n"
        "- One `slice-<name>.md` per feature slice (DAG-ordered by deps)\n"
        "- `tasks.md` — index of slices, dependency waves, file change matrix\n"
        "- `codebase-map.md` — see required sections below\n\n"
        "Each slice must include: name, description, files to modify/create, "
        "dependencies, acceptance criteria, complexity estimate.\n\n"
        "## codebase-map.md — Required Sections\n"
        "1. **Files** — path → 1-line description → line count\n"
        "2. **Key Functions** — `name(args) → ret  [L<start>-<end>]` + 1-line desc.\n"
        "   Example: `submit_work(description, work_type, config) → dict  "
        "[L367-420]  — creates work entry and starts graph`\n"
        "3. **Import Chains** — which modules import which\n"
        "4. **Conventions** — naming, patterns, error handling\n"
        "5. **Modification Targets** — for each file to be modified, a 3-5 "
        "line code snippet around the change site with its line range. "
        "This is what subagents will use to locate exact insertion points.\n\n"
        "Example modification target:\n"
        "```python\n"
        "# spine/work/dispatcher.py [L407-420]\n"
        "        if any(\n"
        '            isinstance(f, dict) and f.get("status") == "needs_review"\n'
        "            for f in feedback\n"
        "        ):\n"
        '            final_status = "needs_review"\n'
        "```\n\n"
        "## Rules\n"
        "- You MUST call `write_file` for every artifact. Conversation-only "
        "output is lost; the next phase has nothing to dispatch.\n"
        "- Batch reads — read ≥3 files per turn.\n"
        "- Spend at most 2-3 turns exploring; then write.\n"
        "- All file paths in slice files MUST be real workspace paths "
        "(confirmed to exist, or inside a confirmed-existing directory).\n\n"
        "## Eval Context Seed (first eval call)\n"
        "```js\n"
        f"globalThis.context = {{work_id: \"{work_id}\", "
        f"phase: \"tasks\", artifact_dir: \"{tasks_dir}\"}};\n"
        "```\n\n"
    )
