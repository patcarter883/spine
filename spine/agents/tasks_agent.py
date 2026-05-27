"""SPINE tasks agent — Deep Agent for the TASKS (decomposition) phase.

.. deprecated::
    The TASKS phase agent is deprecated. Use the PLAN phase agent for
    decomposition instead. This module is retained for backward
    compatibility and will be removed in a future release.

Researches the codebase via ``search_codebase`` (and optional researcher
subagents), then writes all decomposition artifacts atomically via
``write_tasks_artifacts``.

Tool surface (complete list):
- ``read_prior_artifacts`` — loads spec/plan artifacts (spec workflows only)
- ``search_codebase`` — multi-query targeted file search (replaces ls/glob/grep/read_file)
- ``write_tasks_artifacts`` — atomic write of slices + tasks.md + codebase-map.md

No generic filesystem tools (ls, read_file, glob, grep, write_file,
edit_file, execute). The agent cannot read files directly — it uses
``search_codebase`` for targeted lookup. ``write_tasks_artifacts`` is
the only write surface and calling it ends the phase.

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

from spine.agents.artifacts import artifact_path
from spine.agents.factory import build_phase_agent
from spine.agents.tasks_tools import build_tasks_agent_tools
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState

logger = logging.getLogger(__name__)


def build_tasks_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the TASKS phase.

    .. deprecated::
        The TASKS phase agent is deprecated. Use the PLAN phase agent
        for decomposition instead.

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

    system_prompt = _build_tasks_prompt()

    agent = build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.TASKS,
        system_prompt=system_prompt,
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
            dirs[phase] = artifact_path(work_id, phase)
    return dirs


def _build_tasks_prompt() -> str:
    return (
        "You are the TASKS phase agent. Your job is to decompose the work "
        "into executable feature slices with a precise codebase map that "
        "downstream IMPLEMENT and VERIFY orchestrators depend on.\n\n"
        "## Your tool surface (complete list)\n"
        "- `read_prior_artifacts` — loads spec/plan artifacts. No args.\n"
        "### Codebase exploration (use MCP tools FIRST)\n"
        "MCP codebase-index tools answer symbol-level questions in "
        "sub-milliseconds with minimal token usage.\n"
        "Call with native kwargs (no tool_input wrapper):\n"
        "- `mcp_codebase-index_find_symbol` — locate symbol. "
        'Call: `{"name": "symbol_name"}`\n'
        "- `mcp_codebase-index_get_function_source` — get function source. "
        'Call: `{"name": "func_name"}`\n'
        "- `mcp_codebase-index_get_dependencies` — what a symbol calls. "
        'Call: `{"name": "symbol_name"}`\n'
        "- `mcp_codebase-index_get_dependents` — who calls a symbol. "
        'Call: `{"name": "symbol_name"}`\n'
        "- `mcp_codebase-index_get_change_impact` — what breaks if you change. "
        'Call: `{"name": "symbol_name"}`\n'
        "- `mcp_codebase-index_search_codebase` — regex search across all files. "
        'Call: `{"pattern": "regex", "max_results": 20}`\n'
        "- `mcp_codebase-index_list_files` — list files by glob. "
        'Call: `{"pattern": "*.py"}`\n'
        "- `mcp_codebase-index_get_project_summary` — high-level overview. No args.\n"
        "### Fallback search\n"
        "- `search_codebase` — find files by keyword/topic queries with "
        "content previews. Use for content-level queries MCP doesn't cover.\n"
        "### Output\n"
        "- `write_tasks_artifacts` — writes all artifacts atomically. "
        "Call this ONCE. It is the ONLY write tool.\n\n"
        "You do NOT have `ls`, `read_file`, `glob`, `grep`, `write_file`, "
        "`edit_file`, or `execute`. Do not attempt to call them.\n\n"
        "## Workflow (~3 turns total)\n\n"
        "Refer to Step 1 & Step 1b guidelines preloaded in your user context pre-prompt.\n\n"
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
    )
