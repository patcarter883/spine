"""SPINE implement agent — Deep Agent for the IMPLEMENT phase.

This phase is the **orchestrator**: its only job is to dispatch one
``slice-implementer`` subagent per feature slice and synthesize their
results into ``implementation.md``. It does NOT touch source code.

Design rationale (see trace 743e5acb assessment):

- Letting a single agent implement N slices serially causes runaway
  context growth (the trace 743e5acb implement phase grew prompts from
  17K → 84K tokens over 160 LLM calls, with the same files re-read up
  to 29 times).
- One subagent per slice gives each subagent a fresh, small context.
  The orchestrator's context stays bounded by the slice count, not the
  slice complexity.
- The orchestrator is dispatch-only by tool restriction
  (``allowed_tools``), so the FilesystemMiddleware physically does not
  expose ``edit_file`` or ``execute``. The model cannot accidentally
  start patching files even if instructions get muddled.
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
from spine.agents.subagents import build_phase_subagents
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState

# ── Tool allowlist ─────────────────────────────────────────────────────
# Orchestrator role: no edit_file, no execute. The orchestrator reads
# slice files, dispatches subagents, and writes implementation.md only.
# Source-code mutation belongs to slice-implementer subagents.
_IMPLEMENT_ORCHESTRATOR_TOOLS: list[str] = [
    "ls",
    "read_file",
    "glob",
    "grep",
    "write_file",  # for implementation.md only; orchestrator MUST NOT touch source
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

    system_prompt = _build_orchestrator_prompt(
        work_id=work_id,
        tasks_dir=tasks_dir,
        slice_inventory=slice_inventory,
        slice_count=slice_count,
    ) + build_current_phase_write_prompt(
        work_id, PhaseName.IMPLEMENT.value, expected_files=["implementation.md"]
    ) + build_artifact_prompt(
        state.get("artifacts", {}), PhaseName.IMPLEMENT.value, work_id=work_id
    )

    agent = build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.IMPLEMENT,
        system_prompt=system_prompt,
        add_summarization=True,
        subagents=_build_subagents(PhaseName.IMPLEMENT, state, config),
        allowed_tools=_IMPLEMENT_ORCHESTRATOR_TOOLS,
    )

    return agent


# ── Prompt builder ─────────────────────────────────────────────────────


def _build_orchestrator_prompt(
    *,
    work_id: str,
    tasks_dir: str,
    slice_inventory: str,
    slice_count: int,
) -> str:
    """Build the orchestrator system prompt.

    Kept under ~3KB so the bulk of the prompt is the SPINE base prompt
    and per-call AGENTS.md memory injection, not phase boilerplate.
    """
    impl_dir = f".spine/artifacts/{work_id}/implement"

    # Singular/plural language and parallel dispatch wording
    if slice_count == 1:
        dispatch_note = (
            "There is 1 slice. Dispatch a single `slice-implementer` for "
            "consistency — same context-management benefit, no orchestrator "
            "drift between work items."
        )
    elif slice_count >= 2:
        dispatch_note = (
            f"There are {slice_count} slices. Dispatch all of them in "
            "parallel via `Promise.allSettled(tools.task(...))` inside a "
            "single `eval` call. Do NOT dispatch sequentially — the whole "
            "point of subagent dispatch is parallel isolated contexts."
        )
    else:
        dispatch_note = (
            "Slice files not yet discovered. Use `glob` to list "
            f"`{tasks_dir}/slice-*.md`, then dispatch one subagent per slice."
        )

    return (
        "You are the IMPLEMENT phase orchestrator. You do NOT write source "
        "code yourself — you dispatch one `slice-implementer` subagent per "
        "feature slice and synthesize their results into a single report.\n\n"
        "Your tools are restricted: you have `ls`, `read_file`, `glob`, "
        "`grep`, `write_file`, `task`, and `eval`. You do NOT have "
        "`edit_file` or `execute`. This is intentional — source-code "
        "mutation belongs to subagents.\n\n"
        "## Expected tool errors (BY DESIGN — do not recover)\n"
        "If you attempt to call `edit_file` or `execute`, you will see a "
        "'tool not found' or 'unknown tool' error. This is **not a bug** — "
        "your toolset is deliberately filtered. Do not try alternative "
        "tools or attempt to work around the restriction. Dispatch a "
        "`slice-implementer` subagent instead — they have those tools.\n\n"
        "## Workflow (3 steps, ~5 turns total)\n\n"
        "### Step 1 — Read slice definitions and codebase map\n"
        f"In ONE turn, batch-read these files:\n"
        f"- `{tasks_dir}/codebase-map.md` (architecture, file paths, conventions)\n"
        f"- Each slice file listed below\n\n"
        f"{slice_inventory}\n\n"
        "### Step 2 — Dispatch slice-implementer subagents in parallel\n"
        f"{dispatch_note}\n\n"
        "Each `task` description MUST be fully self-contained — the "
        "subagent has an empty context and cannot see your conversation. "
        "Embed:\n"
        "1. The full text of the slice file\n"
        "2. The codebase-map.md sections relevant to this slice's files\n"
        "3. The list of files to modify/create\n"
        "4. Acceptance criteria from the slice\n\n"
        "Dispatch pattern (do this inside one `eval` call):\n"
        "```js\n"
        "const sliceFiles = [/* slice filenames */];\n"
        "const dispatches = sliceFiles.map(async (name) => {\n"
        "  const slice = await tools.read_file({file_path: "
        f"`{tasks_dir}/${{name}}`}});\n"
        "  return tools.task({\n"
        '    subagent_type: "slice-implementer",  // ONLY valid type\n'
        "    description: `Implement slice: ${name}\\n\\n`\n"
        "      + `## Slice Definition\\n${slice.content}\\n\\n`\n"
        "      + `## Codebase Map (relevant excerpts)\\n${mapExcerpts}\\n`,\n"
        "  });\n"
        "});\n"
        "const results = await Promise.allSettled(dispatches);\n"
        "globalThis.sliceResults = results;\n"
        "console.log(JSON.stringify(results.map(r => r.status)));\n"
        "```\n\n"
        "### Step 3 — Synthesize implementation.md\n"
        f"Write `{impl_dir}/implementation.md` with `write_file`. Include:\n"
        "- One section per slice with the subagent's reported status\n"
        "- Aggregated list of files modified / created across all slices\n"
        "- Any slice that returned `blocked` or `partial` (verbatim issues)\n"
        "- Aggregated test results if subagents ran tests\n\n"
        "## Strict Rules\n"
        "- You MUST NOT call `edit_file`, `write_file` on source files, or "
        "`execute`. Your toolset is filtered — these tools are not available.\n"
        "- You MUST dispatch one `slice-implementer` subagent per slice. "
        "Do not attempt to implement slices inline.\n"
        "- The ONLY valid `subagent_type` is `slice-implementer`. Do NOT "
        "request `general-purpose` — it does not exist.\n"
        "- Subagent dispatch MUST happen inside `eval` so multiple "
        "subagents run in parallel via `Promise.allSettled`. Sequential "
        "tool calls from conversation are not parallel.\n"
        "- `implementation.md` is REQUIRED — without it the phase fails.\n\n"
        "## Eval Context Seed (first eval call)\n"
        "```js\n"
        f"globalThis.context = {{work_id: \"{work_id}\", "
        f"phase: \"implement\", tasks_dir: \"{tasks_dir}\", "
        f"impl_dir: \"{impl_dir}\"}};\n"
        "```\n\n"
    )
