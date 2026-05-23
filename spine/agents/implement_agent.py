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
    artifact_path,
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
    tasks_dir = artifact_path(work_id, "tasks")

    # ── Determine dispatch mode: waves (preferred) or legacy ──────────
    # When execution_waves is present in state (produced by the PLAN
    # scheduler), slices come from plan.json and are dispatched per wave
    # (parallel within, sequential across). Otherwise fall back to
    # discovering slice-*.md files from the tasks/ directory.
    execution_waves = state.get("execution_waves")
    has_waves = bool(execution_waves)

    if has_waves:
        total_slices = sum(len(wave) for wave in execution_waves)
        wave_count = len(execution_waves)
        plan_dir = artifact_path(work_id, "plan")
        plan_json_path = f"{plan_dir}/plan.json"
        slice_inventory = (
            f"{total_slices} slice(s) in {wave_count} execution wave(s) "
            f"from `{plan_json_path}`.\n"
            "Slices within each wave are independent (parallel); "
            "waves run sequentially.\n"
        )
        for i, wave in enumerate(execution_waves):
            slice_ids = [s.get("id", f"slice-{j}") for j, s in enumerate(wave)]
            slice_inventory += f"  Wave {i + 1}: {', '.join(slice_ids)} ({len(wave)} slice(s))\n"
    else:
        # Legacy fallback: discover slice-*.md files from tasks/ dir
        slice_files = list_slice_files(workspace_root, work_id)
        slice_count = len(slice_files)
        if slice_count == 0:
            slice_inventory = (
                "⚠ No slice-*.md files found in tasks/ directory. "
                "Use `ls` + `glob` to locate slice files before proceeding."
            )
        else:
            slice_inventory = f"{slice_count} slice file(s) found in `{tasks_dir}/`:\n" + "\n".join(
                f"  - `{tasks_dir}/{name}`" for name in slice_files
            )

    system_prompt = _build_orchestrator_prompt(has_waves=has_waves)
    system_prompt += slice_inventory + "\n\n"
    system_prompt += build_current_phase_write_prompt(
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
        subagents=_build_subagents(PhaseName.IMPLEMENT, state, config),
        extra_tools=orchestrator_tools,
        skip_filesystem_middleware=True,
    )

    return agent


# ── Prompt builder ─────────────────────────────────────────────────────


def _build_orchestrator_prompt(*, has_waves: bool = False) -> str:
    """Build the orchestrator system prompt.

    Kept under ~3KB so the bulk of the prompt is the SPINE base prompt
    and per-call AGENTS.md memory injection, not phase boilerplate.

    Args:
        has_waves: When True, the orchestrator works from plan.json
            execution_waves (parallel within wave, sequential across
            waves). When False, falls back to the legacy flat-parallel
            dispatch from tasks/slice-*.md files.
    """
    if has_waves:
        return _build_wave_orchestrator_prompt()
    return _build_legacy_orchestrator_prompt()


def _build_wave_orchestrator_prompt() -> str:
    """Orchestrator prompt for wave-based dispatch (plan.json mode)."""
    return (
        "You are the IMPLEMENT phase orchestrator. Your job is to dispatch one "
        "`slice-implementer` subagent per feature slice and synthesize their "
        "results.\n\n"
        "## Your tool surface (complete list)\n"
        "- `read_slice_files` — loads slice definitions + codebase map "
        "in ONE call. No arguments needed. Call this FIRST.\n"
        "- `write_implementation_report` — writes implementation.md. Call "
        "this LAST after all subagents complete.\n"
        "- `eval` — JavaScript REPL for subagent dispatch. "
        "Call `tools.task()` INSIDE eval for parallel dispatch.\n\n"
        "Your tools are restricted to `read_slice_files`, `write_implementation_report`, "
        "`task`, and `eval`. This is enforced by tool design — you cannot call "
        "`ls`, `read_file`, `glob`, `grep`, `write_file`, `edit_file`, or `execute`. "
        "Use `read_slice_files` to load all slice definitions at once.\n\n"
        "## Dispatch mode: execution waves\n"
        "Slices come from plan.json and are organized into execution waves. "
        "Slices within a wave are independent and run in parallel. "
        "Waves run sequentially — wait for wave N to complete before starting wave N+1.\n\n"
        "## Workflow\n\n"
        "### Step 1 — Call read_slice_files (1 turn)\n"
        "Call `read_slice_files` with no arguments. It returns a JSON object:\n"
        "```\n"
        "{\n"
        '  "slices": {\n'
        '    "add-user-model": {\n'
        '      "id": "add-user-model",\n'
        '      "title": "Add user model",\n'
        '      "target_files": ["src/models/user.py"],\n'
        '      "execution_requirements": "Create the User model class...",\n'
        '      "dependencies": [],\n'
        '      "acceptance_criteria": ["User.save() persists to DB"],\n'
        '      "complexity": "medium"\n'
        "    },\n"
        "    ...\n"
        "  },\n"
        '  "codebase_map": "<full content of codebase_map field>",\n'
        '  "slice_count": N,\n'
        '  "plan_dir": "<path>"\n'
        "}\n"
        "```\n"
        "Slices are keyed by their `id` field. Each value is the full "
        "feature-slice dict. Store the result in eval: "
        "`globalThis.planData = result;`\n\n"
        "### Step 2 — Dispatch subagents per wave (1 eval turn)\n"
        "Do this inside ONE eval call with Promise.allSettled:\n"
        "1. Get the execution_waves array from globalThis.planData or globalThis.execution_waves\n"
        "2. For each wave, create a Promise.allSettled call with all slices in that wave\n"
        "3. Wait for each wave's Promise.allSettled to complete before starting the next\n"
        "4. Store results in globalThis.sliceResults\n\n"
        "Template code:\n"
        "```js\n"
        "const waves = globalThis.execution_waves || Object.values(globalThis.planData.slices);\n"
        "for (let i = 0; i < waves.length; i++) {\n"
        "  const wave = waves[i];\n"
        "  const promises = wave.map(slice => tools.task({\n"
        "    subagent_type: 'slice-implementer',\n"
        "    description: `Slice: ${slice.id}\n\n` +\n"
        "      `Title: ${slice.title}\n` +\n"
        "      `Files: ${slice.target_files.join(', ')}\n` +\n"
        "      `Requirements: ${slice.execution_requirements}\n` +\n"
        "      `Acceptance Criteria: ${slice.acceptance_criteria.join('; ')}`\n"
        "  }));\n"
        "  const results = await Promise.allSettled(promises);\n"
        "  globalThis.sliceResults = globalThis.sliceResults || [];\n"
        "  globalThis.sliceResults.push(...results);\n"
        "}\n"
        "```\n\n"
        "### Step 3 — Call write_implementation_report (1 turn)\n"
        "Parse globalThis.sliceResults and call `write_implementation_report` "
        "with:\n"
        "- `slice_results`: list of dicts, one per slice, each with "
        "`slice_name`, `status` (implemented|partial|blocked), "
        "`files_modified`, `files_created`, `test_results`, `issues`\n"
        "- `summary`: overall summary of what was implemented\n\n"
        "## Strict Rules\n"
        "- Call `read_slice_files` FIRST. It provides all data needed for dispatch.\n"
        "- Dispatch one `slice-implementer` subagent per slice via `eval`.\n"
        "- Use only `slice-implementer` as the subagent_type.\n"
        "- Dispatch slices within a wave in parallel using `Promise.allSettled`.\n"
        "- Wait for Promise.allSettled(results) from wave N before starting wave N+1.\n"
        "- Call `write_implementation_report` to complete the phase.\n"
        "- Target ~3 turns total. If you reach 5+ turns without dispatching subagents, "
        "write the report with whatever results you have.\n\n"
        "## Eval context seed\n"
        "Access session-specific context properties via `globalThis.context` "
        "preloaded in your workspace environment on first turn (e.g., "
        "use `globalThis.context.work_id`, `globalThis.context.plan_json`, "
        "or `globalThis.execution_waves` inside eval).\n\n"
    )


def _build_legacy_orchestrator_prompt() -> str:
    """Orchestrator prompt for flat-parallel dispatch (plan.json mode)."""
    return (
        "You are the IMPLEMENT phase orchestrator. Your job is to dispatch one "
        "`slice-implementer` subagent per feature slice and synthesize their "
        "results.\n\n"
        "## Your tool surface (complete list)\n"
        "- `read_slice_files` — loads all slice definitions + codebase map "
        "in ONE call. No arguments needed. Call this FIRST.\n"
        "- `write_implementation_report` — writes implementation.md. Call "
        "this LAST after all subagents complete.\n"
        "- `eval` — JavaScript REPL for subagent dispatch. "
        "Call `tools.task()` INSIDE eval for parallel dispatch.\n\n"
        "Your tools are restricted to `read_slice_files`, `write_implementation_report`, "
        "`task`, and `eval`. This is enforced by tool design — you cannot call "
        "`ls`, `read_file`, `glob`, `grep`, `write_file`, `edit_file`, or `execute`. "
        "Use `read_slice_files` to load all slice definitions at once.\n\n"
        "## Workflow (3 steps, ~3 turns total)\n\n"
        "### Step 1 — Call read_slice_files (1 turn)\n"
        "Call `read_slice_files` with no arguments. It returns a JSON object:\n"
        "```\n"
        "{\n"
        '  "slices": {\n'
        '    "add-user-model": {\n'
        '      "id": "add-user-model",\n'
        '      "title": "Add user model",\n'
        '      "target_files": ["src/models/user.py"],\n'
        '      "execution_requirements": "Create the User model class...",\n'
        '      "dependencies": [],\n'
        '      "acceptance_criteria": ["User.save() persists to DB"],\n'
        '      "complexity": "medium"\n'
        "    },\n"
        "    ...\n"
        "  },\n"
        '  "codebase_map": "<full content of codebase_map field>",\n'
        '  "slice_count": N,\n'
        '  "plan_dir": "<path>"\n'
        "}\n"
        "```\n"
        "Slices are keyed by their `id` field. Each value is the full "
        "feature-slice dict. Store the result in eval: "
        "`globalThis.planData = result;`\n\n"
        "### Step 2 — Dispatch subagents in parallel (1 eval turn)\n"
        "Do this inside ONE eval call with Promise.allSettled:\n"
        "1. Get the slices array from globalThis.planData.slices (values)\n"
        "2. Create a Promise.allSettled call with all slices\n"
        "3. Store results in globalThis.sliceResults\n\n"
        "Template code:\n"
        "```js\n"
        "const slices = Object.values(globalThis.planData.slices);\n"
        "const promises = slices.map(slice => tools.task({\n"
        "  subagent_type: 'slice-implementer',\n"
        "  description: `Slice: ${slice.id}\n\n` +\n"
        "    `Title: ${slice.title}\n` +\n"
        "    `Files: ${slice.target_files.join(', ')}\n` +\n"
        "    `Requirements: ${slice.execution_requirements}\n` +\n"
        "    `Acceptance Criteria: ${slice.acceptance_criteria.join('; ')}`\n"
        "}));\n"
        "const results = await Promise.allSettled(promises);\n"
        "globalThis.sliceResults = results;\n"
        "```\n\n"
        "### Step 3 — Call write_implementation_report (1 turn)\n"
        "Parse globalThis.sliceResults and call `write_implementation_report` "
        "with:\n"
        "- `slice_results`: list of dicts, one per slice, each with "
        "`slice_name`, `status` (implemented|partial|blocked), "
        "`files_modified`, `files_created`, `test_results`, `issues`\n"
        "- `summary`: overall summary of what was implemented\n\n"
        "## Strict Rules\n"
        "- Call `read_slice_files` FIRST. It provides all data needed for dispatch.\n"
        "- Dispatch one `slice-implementer` subagent per slice via `eval`.\n"
        "- Use only `slice-implementer` as the subagent_type.\n"
        "- Dispatch slices in parallel using `Promise.allSettled`.\n"
        "- Call `write_implementation_report` to complete the phase.\n"
        "- Target ~3 turns total. If you reach 5+ turns without dispatching subagents, "
        "write the report with whatever results you have.\n\n"
        "## Eval context seed\n"
        "Access session-specific context properties via `globalThis.context` "
        "preloaded in your workspace environment on first turn (e.g., "
        "use `globalThis.context.work_id` or `globalThis.context.plan_dir` inside eval).\n\n"
    )
