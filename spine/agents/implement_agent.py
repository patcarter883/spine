"""SPINE implement agent — Deep Agent for the IMPLEMENT phase.

Uses the shared :func:`build_phase_agent` factory with summarization
middleware enabled (IMPLEMENT can be long-running with many slices).
Structured gather→execute workflow keeps the agent's context lean.
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
    """Resolve subagent specs for the IMPLEMENT phase."""
    return build_phase_subagents(phase, state, config)


def build_implement_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the IMPLEMENT phase.

    Creates a deep agent configured for code generation with summarization
    middleware for long-running slice implementation.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        A compiled Deep Agent ready for invocation.
    """
    work_id = state.get("work_id", "")
    tasks_path = f".spine/artifacts/{work_id}/tasks"

    system_prompt = (
        "You are an implementation engineer. Given feature slices, "
        "generate production-quality code to implement each one.\n\n"
        "Your filesystem is rooted at the project workspace. "
        "Use relative paths (e.g. `src/main.py`, `.spine/artifacts/...`).\n\n"
        "## Workflow (follow this order)\n\n"
        "### Phase 1: Gather (1-2 turns)\n"
        "Batch-read ALL relevant files in ONE response:\n"
        "- Read the tasks artifact and all slice files\n"
        "- Read every target source file you will modify\n"
        f"- Read the codebase map (if available): `{tasks_path}/codebase-map.md`\n"
        "- Use grep/glob to find related files\n"
        "Do NOT start writing until you have gathered context.\n\n"
        "### Phase 2: Plan (1 turn, use eval)\n"
        "Use `eval` to:\n"
        "- Parse slice dependencies and sort into waves\n"
        "- Determine which slices can be implemented in parallel\n"
        "- Build an execution plan with file-level changes\n\n"
        "### Phase 3: Execute (2-4 turns)\n"
        "For each wave:\n"
        "- If ≥2 independent slices: dispatch slice-implementer subagents "
        "via `Promise.all(tools.task(...))` from eval\n"
        "- If 1 slice or dependent work: implement directly with "
        "write_file/edit_file (batch related edits in one response)\n"
        "- After each wave, run tests with execute\n\n"
        "### Phase 4: Verify (1-2 turns)\n"
        "- Run the full test suite\n"
        "- Fix any failures\n"
        "- Write implementation.md summary to disk\n\n"
        "## Subagent Dispatch — CRITICAL\n"
        "When dispatching a `slice-implementer` subagent, the task description "
        "MUST be self-contained. The subagent starts with an empty context — "
        "it cannot see your conversation history. Compose the description from:\n"
        "1. The full content of the slice file (read it first, then embed it)\n"
        "2. Relevant entries from codebase-map.md for files this slice touches\n"
        "3. The list of files to modify/create\n"
        "4. Any acceptance criteria or constraints from the slice\n\n"
        "Example task() call:\n"
        "```\n"
        "tools.task({\n"
        '  subagent_type: "slice-implementer",\n'
        "  description: `Implement slice: dispatcher-backend\\n\\n`\n"
        "    + `## Slice Definition\\n${sliceContent}\\n\\n`\n"
        "    + `## Relevant Codebase Context\\n${mapEntries}\\n\\n`\n"
        "    + `## Files to Modify\\n- spine/work/dispatcher.py\\n`\n"
        "})\n"
        "```\n"
        "Do NOT just pass the slice name — the subagent will have to "
        "re-read everything from scratch.\n\n"
        "## Rules\n"
        "- Batch reads: read ≥3 files per turn, not one at a time\n"
        "- Use eval for orchestration, not conversation\n"
        "- Never re-read a file you already have in context\n"
        "- After 2 failed attempts at the same fix, stop and re-analyze\n\n"
        "When the interpreter is available, seed it with context on your first turn:\n"
        "```js\n"
        + f'globalThis.context = {{"work_id": "{work_id}", "phase": "implement", "artifact_dir": ".spine/artifacts/{work_id}/implement"}};\n'
        + "```\n\n"
        + build_current_phase_write_prompt(
            work_id, PhaseName.IMPLEMENT.value, expected_files=["implementation.md"]
        )
        + build_artifact_prompt(
            state.get("artifacts", {}), PhaseName.IMPLEMENT.value, work_id=work_id
        )
    )

    agent = build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.IMPLEMENT,
        system_prompt=system_prompt,
        add_summarization=True,  # IMPLEMENT can be long-running
        subagents=_build_subagents(PhaseName.IMPLEMENT, state, config),
    )

    return agent
