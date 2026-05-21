"""SPINE implement agent — Deep Agent for the IMPLEMENT phase.

This phase is the **orchestrator**: its only job is to dispatch one
``slice-implementer`` subagent per feature slice and synthesize their
results into ``implementation.md``. It does NOT touch source code.

Design rationale (see trace 743e5acb and 019e4447 assessments):

- Letting a single agent implement N slices serially causes runaway
  context growth (trace 743e5acb grew prompts from 17K → 84K tokens
  over 160 LLM calls, same files re-read up to 29 times; trace 019e4447
  hit the 73K hard context limit after 181 LLM turns with 0 eval calls).
- One subagent per slice gives each subagent a fresh, small context.
  The orchestrator's context stays bounded by the slice count, not slice
  complexity.
- The orchestrator is dispatch-only by **tool design**, not just by
  instruction. Generic filesystem tools (ls, read_file, glob, grep,
  write_file) are replaced entirely with two purpose-built tools:

  * ``read_slice_files`` — loads all slice definitions and the codebase
    map in a single call. Eliminates multi-turn exploration; the
    orchestrator has no need for ``ls``/``glob``/``grep``/``read_file``.
  * ``write_implementation_report`` — the only write surface. Accepts
    a structured result dict and writes to the fixed implementation.md
    path. Cannot write source files.

  With these tools the model's only valid actions are:
  (1) call ``read_slice_files`` once,
  (2) call ``eval`` to dispatch subagents in parallel,
  (3) call ``write_implementation_report`` to synthesize.
  There is literally nothing else to do.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.agents.artifacts import (
    build_artifact_prompt,
    build_current_phase_write_prompt,
    list_slice_files,
)
from spine.agents.factory import build_phase_agent
from spine.agents.implement_tools import build_implement_orchestrator_tools
from spine.agents.subagents import build_phase_subagents
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState

# ── Tool allowlist (legacy — kept for reference) ────────────────────────
# Previously the orchestrator used FilesystemMiddleware with an allowlist
# that kept only read-only tools + write_file. This still allowed a weak
# model to fall back to 100+ sequential read_file calls (trace 019e4447:
# 181 LLM turns, 106 read_file calls, context overflow at 73K tokens).
#
# The new approach (below) replaces FilesystemMiddleware entirely with
# two purpose-built tools. Kept here for historical reference only.
_IMPLEMENT_ORCHESTRATOR_TOOLS_LEGACY: list[str] = [
    "ls",
    "read_file",
    "glob",
    "grep",
    "write_file",
]


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

    Always builds a dispatch-only orchestrator. The orchestrator's only
    tools are read-only filesystem access plus ``write_file`` (reserved
    for ``implementation.md``). All code generation is delegated to
    ``slice-implementer`` subagents via the ``task`` tool, dispatched in
    parallel from ``eval``.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        A compiled Deep Agent ready for invocation.
    """
    work_id = state.get("work_id", "")
    workspace_root = state.get("workspace_root", ".")
    tasks_dir = f".spine/artifacts/{work_id}/tasks"

    # ── Discover slice files at build time ────────────────────────────
    # Inject the inventory directly into the prompt so the orchestrator
    # does not need to spend a turn discovering slices via ls/glob.
    slice_files = list_slice_files(workspace_root, work_id)
    slice_count = len(slice_files)

    if slice_count == 0:
        slice_inventory = (
            "⚠ No slice-*.md files found in tasks/ directory. "
            "Use `ls` + `glob` to locate slice files before proceeding."
        )
    else:
        slice_inventory = (
            f"{slice_count} slice file(s) found in `{tasks_dir}/`:\n"
            + "\n".join(f"  - `{tasks_dir}/{name}`" for name in slice_files)
        )

    system_prompt = _build_orchestrator_prompt() + build_current_phase_write_prompt(
        work_id, PhaseName.IMPLEMENT.value, expected_files=["implementation.md"]
    ) + build_artifact_prompt(
        state.get("artifacts", {}), PhaseName.IMPLEMENT.value, work_id=work_id
    )

    # ── Build custom orchestrator tools ───────────────────────────────
    # Replace generic filesystem tools with two purpose-built tools that
    # enforce dispatch-only behaviour at the tool level. FilesystemMiddleware
    # is skipped entirely — the orchestrator cannot call ls/glob/read_file/
    # write_file/grep/edit_file/execute even if it tries.
    orchestrator_tools = build_implement_orchestrator_tools(
        workspace_root=workspace_root,
        work_id=work_id,
    )

    agent = build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.IMPLEMENT,
        system_prompt=system_prompt,
        add_summarization=True,
        subagents=_build_subagents(PhaseName.IMPLEMENT, state, config),
        extra_tools=orchestrator_tools,
        skip_filesystem_middleware=True,
    )

    return agent


# ── Prompt builder ─────────────────────────────────────────────────────


def _build_orchestrator_prompt() -> str:
    """Build the orchestrator system prompt.

    Kept under ~3KB so the bulk of the prompt is the SPINE base prompt
    and per-call AGENTS.md memory injection, not phase boilerplate.
    """
    return (
        "You are the IMPLEMENT phase orchestrator. You do NOT write source "
        "code yourself — you dispatch one `slice-implementer` subagent per "
        "feature slice and synthesize their results.\n\n"
        "## Your tool surface (complete list)\n"
        "- `read_slice_files` — loads all slice definitions + codebase map "
        "in ONE call. No arguments needed. Call this FIRST.\n"
        "- `write_implementation_report` — writes implementation.md. Call "
        "this LAST after all subagents complete.\n"
        "- `task` (via eval) — dispatches a slice-implementer subagent.\n"
        "- `eval` — JavaScript REPL for parallel subagent dispatch.\n\n"
        "You do NOT have `ls`, `read_file`, `glob`, `grep`, `write_file`, "
        "`edit_file`, or `execute`. These tools do not exist in your session. "
        "Do not attempt to call them. There is nothing to explore — "
        "`read_slice_files` gives you everything you need.\n\n"
        "## Workflow (3 steps, ~3 turns total)\n\n"
        "### Step 1 — Call read_slice_files (1 turn)\n"
        "Call `read_slice_files` with no arguments. It returns a JSON object:\n"
        "```\n"
        "{\n"
        '  "slices": {"slice-foo.md": "<full content>", ...},\n'
        '  "codebase_map": "<full content of codebase-map.md>",\n'
        '  "slice_count": N,\n'
        '  "tasks_dir": "<path>"\n'
        "}\n"
        "```\n"
        "Store the result in eval: `globalThis.slices = result;`\n\n"
        "### Step 2 — Dispatch subagents in parallel (1 eval turn)\n"
        "Refer to Step 2 guidelines preloaded in your workspace environment on first turn.\n\n"
        "Use a single `eval` call with `Promise.allSettled` to dispatch all "
        "slices in parallel. Each task description MUST be fully self-contained "
        "— the subagent has an empty context and cannot see your conversation. "
        "Embed the full slice content, relevant codebase map sections, files "
        "to modify, and acceptance criteria.\n\n"
        "Dispatch pattern:\n"
        "```js\n"
        "// Step 1 result is in globalThis.slices\n"
        "const data = globalThis.slices;\n"
        "const map = data.codebase_map || '';\n"
        "const dispatches = Object.entries(data.slices).map(([name, content]) =>\n"
        "  tools.task({\n"
        '    subagent_type: "slice-implementer",\n'
        "    description: `Implement slice: ${name}\\n\\n"
        "## Slice Definition\\n${content}\\n\\n"
        "## Codebase Map\\n${map}`,\n"
        "  })\n"
        ");\n"
        "const results = await Promise.allSettled(dispatches);\n"
        "globalThis.sliceResults = results;\n"
        "console.log(JSON.stringify(results.map(r => r.status)));\n"
        "```\n\n"
        "### Step 3 — Call write_implementation_report (1 turn)\n"
        "Parse globalThis.sliceResults and call `write_implementation_report` "
        "with:\n"
        "- `slice_results`: list of dicts, one per slice, each with "
        "`slice_name`, `status` (implemented|partial|blocked), "
        "`files_modified`, `files_created`, `test_results`, `issues`\n"
        "- `summary`: overall summary of what was implemented\n\n"
        "## Strict Rules\n"
        "- You MUST call `read_slice_files` FIRST. Do not skip it.\n"
        "- You MUST dispatch one `slice-implementer` subagent per slice. "
        "Do not attempt to implement slices yourself.\n"
        "- The ONLY valid `subagent_type` is `slice-implementer`.\n"
        "- Subagent dispatch MUST happen inside `eval` with "
        "`Promise.allSettled` for parallelism.\n"
        "- You MUST call `write_implementation_report` to complete the phase. "
        "Without it the phase has no artifact and fails.\n"
        "- Total turns: ~3. More than 5 turns without dispatching subagents "
        "means something has gone wrong — stop and write the report with "
        "whatever results you have.\n\n"
        "## Eval context seed\n"
        "Access session-specific context properties via `globalThis.context` "
        "preloaded in your workspace environment on first turn (e.g., "
        "use `globalThis.context.work_id` or `globalThis.context.tasks_dir` inside eval).\n\n"
    )
