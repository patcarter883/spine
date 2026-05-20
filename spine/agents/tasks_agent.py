"""SPINE tasks agent — Deep Agent for the TASKS (decomposition) phase.

Researches the codebase via ``search_codebase`` (and optional researcher
subagents), then writes all decomposition artifacts atomically via
``write_tasks_artifacts``.

Tool surface (complete list):
- ``read_prior_artifacts`` — loads spec/plan artifacts (spec workflows only)
- ``search_codebase`` — multi-query targeted file search (replaces ls/glob/grep/read_file)
- ``write_tasks_artifacts`` — atomic write of slices + tasks.md + codebase-map.md
- ``task`` (via SubAgentMiddleware) — dispatches researcher subagents
- ``eval`` (via CodeInterpreterMiddleware) — parallel subagent dispatch

No generic filesystem tools (ls, read_file, glob, grep, write_file,
edit_file, execute). The agent cannot read files directly — it uses
``search_codebase`` for targeted lookup. Researcher subagents do deep
exploration. ``write_tasks_artifacts`` is the only write surface and
calling it ends the phase.

Design rationale (trace 019e4483 analysis):
- With generic fs tools: 87 read_file calls, same files re-read 38×,
  no researcher dispatch, continued reading 20+ min AFTER writing.
- ``search_codebase`` answers "what code is relevant?" in one call.
- ``write_tasks_artifacts`` makes partial-output impossible and provides
  a clear phase-completion signal the agent can't ignore.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.agents.artifacts import _artifact_path
from spine.agents.factory import build_phase_agent
from spine.agents.subagents import build_phase_subagents
from spine.agents.tasks_tools import build_tasks_agent_tools
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState

logger = logging.getLogger(__name__)


def _build_subagents(
    phase: PhaseName,
    state: WorkflowState,
    config: RunnableConfig | None,
) -> list[Any] | None:
    """Resolve subagent specs for the TASKS phase.

    Returns researcher subagents for workflows that need codebase
    exploration, None for trivial quick tasks.
    """
    work_type = state.get("work_type", "")
    if "quick" in work_type and "critical" not in work_type:
        description = state.get("description", "")
        if len(description) < 150:
            logger.info(
                "[%s] TASKS: skipping researcher subagents for trivial quick task "
                "(%d chars)",
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

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        A compiled Deep Agent ready for invocation.
    """
    work_id = state.get("work_id", "")
    workspace_root = state.get("workspace_root", ".")
    work_type = state.get("work_type", "")
    description = state.get("description", "")
    feedback_raw = state.get("feedback", [])
    feedback = [str(f) for f in feedback_raw] if feedback_raw else []
    is_quick = "quick" in work_type

    tasks_dir = f".spine/artifacts/{work_id}/tasks"

    # Prior phase dirs for read_prior_artifacts (spec workflows only)
    prior_phase_dirs = _resolve_prior_phase_dirs(state, work_id)

    agent_tools = build_tasks_agent_tools(
        workspace_root=workspace_root,
        work_id=work_id,
        prior_phase_dirs=prior_phase_dirs,
        description=description,
        work_type=work_type,
        feedback=feedback,
    )

    system_prompt = _build_tasks_prompt(
        work_id=work_id,
        tasks_dir=tasks_dir,
        is_quick=is_quick,
        has_prior_artifacts=bool(prior_phase_dirs),
        is_rework=bool(feedback),
    )

    agent = build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.TASKS,
        system_prompt=system_prompt,
        add_summarization=True,
        subagents=_build_subagents(PhaseName.TASKS, state, config),
        extra_tools=agent_tools,
        skip_filesystem_middleware=True,
    )

    return agent


def _resolve_prior_phase_dirs(
    state: WorkflowState,
    work_id: str,
) -> dict[str, str]:
    """Map phases with existing artifacts to their directories."""
    artifacts = state.get("artifacts", {}) or {}
    dirs: dict[str, str] = {}
    for phase, phase_artifacts in artifacts.items():
        if phase_artifacts and isinstance(phase_artifacts, dict):
            dirs[phase] = _artifact_path(work_id, phase)
    return dirs


def _build_tasks_prompt(
    *,
    work_id: str,
    tasks_dir: str,
    is_quick: bool,
    has_prior_artifacts: bool,
    is_rework: bool,
) -> str:
    rework_note = (
        "\n**This is a REWORK pass.** `read_prior_artifacts` will include "
        "prior feedback. Address all feedback points in your decomposition.\n"
        if is_rework
        else ""
    )

    if is_quick:
        step1 = (
            "### Step 1 — Dispatch researcher subagents in parallel (1 eval turn)\n"
            "This is a quick workflow — no spec/plan exists. Your FIRST action "
            "MUST be an `eval` call that dispatches 2-3 `researcher` subagents "
            "in parallel via `Promise.allSettled`. Each researcher investigates "
            "ONE module or area relevant to the work description.\n\n"
            "Do NOT call `search_codebase` or explore yourself before dispatching "
            "researchers — they do the exploration. You orchestrate and synthesize.\n\n"
            "Researcher dispatch pattern:\n"
            "```js\n"
            "// Your FIRST eval call must look like this:\n"
            "const desc = '<work description>';\n"
            "const results = await Promise.allSettled([\n"
            "  tools.task({subagent_type: 'researcher',\n"
            "    description: `Research area 1 for: ${desc}\\n"
            "Investigate: <specific module/file/concern>`}),\n"
            "  tools.task({subagent_type: 'researcher',\n"
            "    description: `Research area 2 for: ${desc}\\n"
            "Investigate: <specific module/file/concern>`}),\n"
            "]);\n"
            "globalThis.research = results.map(r => r.value || r.reason);\n"
            "console.log('Researchers done:', results.map(r => r.status));\n"
            "```\n\n"
            "### Step 1b — Targeted follow-up (0-1 turns, optional)\n"
            "If researcher results reference specific files you need line-level "
            "detail for (e.g. a change site), call `search_codebase` with the "
            "exact function names or symbols. Maximum 1-2 `search_codebase` calls.\n\n"
        )
    elif has_prior_artifacts:
        step1 = (
            "### Step 1 — Load prior artifacts (1 turn)\n"
            "Call `read_prior_artifacts` with no arguments. It returns the "
            "full specification and plan from prior phases.\n"
            "```js\n"
            "globalThis.ctx = JSON.parse(result);\n"
            "// ctx.artifacts.specify, ctx.artifacts.plan\n"
            "```\n\n"
            "### Step 1b — Search codebase for modification targets (1 turn)\n"
            "Call `search_codebase` with queries derived from the plan's "
            "module structure and API designs. Find the exact files and "
            "functions you'll need to modify.\n\n"
        )
    else:
        step1 = (
            "### Step 1 — Search codebase (1 turn)\n"
            "Call `search_codebase` with 3-5 queries relevant to the work. "
            "Find the exact files that need to change and the functions "
            "around the change sites.\n\n"
        )

    return (
        "You are the TASKS phase agent. Your job is to decompose the work "
        "into executable feature slices with a precise codebase map that "
        "downstream IMPLEMENT and VERIFY orchestrators depend on.\n\n"
        f"{rework_note}"
        "## Your tool surface (complete list)\n"
        "- `read_prior_artifacts` — loads spec/plan artifacts. No args.\n"
        "### Codebase exploration (use MCP tools FIRST)\n"
        "MCP codebase-index tools answer symbol-level questions in "
        "sub-milliseconds with minimal token usage:\n"
        "- `mcp_codebase-index_find_symbol` — locate symbol definition\n"
        "- `mcp_codebase-index_get_function_source` — get function source\n"
        "- `mcp_codebase-index_get_dependencies` — what a symbol calls\n"
        "- `mcp_codebase-index_get_dependents` — who calls a symbol\n"
        "- `mcp_codebase-index_get_change_impact` — what breaks if you change\n"
        "- `mcp_codebase-index_search_codebase` — regex search across all files\n"
        "- `mcp_codebase-index_list_files` — list files by glob pattern\n"
        "- `mcp_codebase-index_get_project_summary` — high-level overview\n"
        "### Fallback search\n"
        "- `search_codebase` — find files by keyword/topic queries with "
        "content previews. Use for content-level queries MCP doesn't cover.\n"
        "### Output\n"
        "- `write_tasks_artifacts` — writes all artifacts atomically. "
        "Call this ONCE. It is the ONLY write tool.\n"
        "- `task` (via eval) — dispatches a `researcher` subagent.\n"
        "- `eval` — JavaScript REPL for parallel dispatch and caching.\n\n"
        "You do NOT have `ls`, `read_file`, `glob`, `grep`, `write_file`, "
        "`edit_file`, or `execute`. Do not attempt to call them.\n\n"
        "## Workflow (~3 turns total)\n\n"
        f"{step1}"
        "### Step 2 — Call write_tasks_artifacts (1 turn)\n"
        "Synthesize everything into one `write_tasks_artifacts` call. "
        "Provide:\n"
        "- `slices`: list of slice objects, each with name, description, "
        "files_to_modify, files_to_create, dependencies, acceptance_criteria, "
        "complexity, modification_targets\n"
        "- `overview`: 2-4 sentence summary of the decomposition\n"
        "- `dependency_waves`: which slices run in parallel vs. sequentially\n"
        "- `codebase_map`: ALL five required sections (Files, Key Functions, "
        "Import Chains, Conventions, Modification Targets)\n\n"
        "**PHASE IS COMPLETE AFTER write_tasks_artifacts.** "
        "Do NOT make any further tool calls after it returns. "
        "Do NOT read more files, do NOT verify your output, do NOT run any checks. "
        "The tool writes all files atomically — they are correct by construction.\n\n"
        "## Critical rules\n"
        "- Dispatch researchers FIRST (quick workflows) or load artifacts FIRST "
        "(spec workflows). Do not invent file paths from memory.\n"
        "- Every `files_to_modify` path MUST appear in MCP or `search_codebase` "
        "results or researcher output. Do not invent paths like `src/main.py`.\n"
        "- `modification_targets` MUST include actual code snippets from "
        "MCP `get_function_source` results or researcher output — not placeholder text.\n"
        "- Call `write_tasks_artifacts` exactly ONCE with ALL slices together.\n"
        "- After `write_tasks_artifacts` returns, stop immediately.\n\n"
        "## Eval context seed\n"
        "```js\n"
        f'globalThis.context = {{work_id: "{work_id}", '
        f'phase: "tasks", tasks_dir: "{tasks_dir}"}};\n'
        "```\n\n"
    )
