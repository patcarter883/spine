"""SPINE plan agent — Deep Agent for the PLAN phase.

Reads the specification artifact and codebase structure, then writes
the technical plan via the ``write_plan`` tool.

Tool surface (complete list):
- ``read_prior_artifacts`` — loads specification + context in one call
- ``search_codebase`` — multi-query codebase file search
- ``write_plan`` — structured write to plan.md (only)
- ``eval`` (via CodeInterpreterMiddleware) — orchestration / store results

No generic filesystem tools (ls, read_file, glob, grep, write_file,
edit_file, execute). The plan agent has targeted read access via
``read_prior_artifacts`` + ``search_codebase``, and write access only
to plan.md. It cannot browse the filesystem arbitrarily.

Note: PLAN has no subagents — it is a single-agent phase. All codebase
exploration is done via ``search_codebase``.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.agents.artifacts import _artifact_path
from spine.agents.factory import build_phase_agent
from spine.agents.plan_tools import build_plan_agent_tools
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState


def build_plan_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the PLAN phase.

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

    plan_dir = f".spine/artifacts/{work_id}/plan"

    # Build prior artifact dir mapping from state
    prior_phase_dirs = _resolve_prior_phase_dirs(state, work_id)

    agent_tools = build_plan_agent_tools(
        workspace_root=workspace_root,
        work_id=work_id,
        description=description,
        work_type=work_type,
        prior_phase_dirs=prior_phase_dirs,
        feedback=feedback,
    )

    system_prompt = _build_plan_prompt(
        work_id=work_id,
        plan_dir=plan_dir,
        has_spec=bool(prior_phase_dirs.get("specify")),
        is_rework=bool(feedback),
    )

    agent = build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.PLAN,
        system_prompt=system_prompt,
        add_summarization=True,
        extra_tools=agent_tools,
        skip_filesystem_middleware=True,
    )

    return agent


def _resolve_prior_phase_dirs(
    state: WorkflowState,
    work_id: str,
) -> dict[str, str]:
    """Map phase names to their artifact directories for phases with artifacts."""
    artifacts = state.get("artifacts", {}) or {}
    dirs: dict[str, str] = {}
    for phase, phase_artifacts in artifacts.items():
        if phase_artifacts and isinstance(phase_artifacts, dict):
            dirs[phase] = _artifact_path(work_id, phase)
    return dirs


def _build_plan_prompt(
    *,
    work_id: str,
    plan_dir: str,
    has_spec: bool,
    is_rework: bool,
) -> str:
    rework_note = (
        "\n**This is a REWORK pass.** `read_prior_artifacts` will include "
        "the prior plan and the critic feedback. Revise to address all feedback.\n"
        if is_rework
        else ""
    )

    spec_note = (
        "The `read_prior_artifacts` result will include the full specification "
        "from the SPECIFY phase. Read it carefully — your plan must implement "
        "exactly what the spec describes."
        if has_spec
        else
        "No prior specification exists (quick workflow). Work directly from "
        "the description returned by `read_prior_artifacts`."
    )

    return (
        "You are the PLAN phase agent. Your job is to create a detailed "
        "technical plan from the specification, grounded in the actual "
        "codebase structure.\n\n"
        f"{rework_note}"
        "## Your tool surface (complete list)\n"
        "- `read_prior_artifacts` — loads spec and all prior artifacts. "
        "No arguments. Call this FIRST.\n"
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
        "- `mcp_codebase-index_get_change_impact` — what breaks if you change a symbol. "
        'Call: `{"name": "symbol_name"}`\n'
        "- `mcp_codebase-index_get_call_chain` — path between two symbols. "
        'Call: `{"from_name": "A", "to_name": "B"}`\n'
        "- `mcp_codebase-index_search_codebase` — regex search across all files. "
        'Call: `{"pattern": "regex", "max_results": 20}`\n'
        "- `mcp_codebase-index_list_files` — list files by glob. "
        'Call: `{"pattern": "*.py"}`\n'
        "- `mcp_codebase-index_get_project_summary` — high-level overview. No args.\n"
        "- `mcp_codebase-index_get_functions` / `get_classes` — list symbols\n"
        "### Fallback search\n"
        "- `search_codebase` — find files by keyword/topic queries with "
        "content previews. Use this for content-level queries the MCP "
        "tools don't cover (e.g. finding specific error messages in code).\n"
        "### Output\n"
        "- `write_plan` — writes plan.md. Call this LAST. "
        "This is the ONLY write tool.\n"
        "- `eval` — JavaScript REPL for storing intermediate results.\n\n"
        "You do NOT have `ls`, `read_file`, `glob`, `grep`, `write_file`, "
        "`edit_file`, or `execute`. Do not attempt to call them. "
        "Use MCP tools and `search_codebase` for all file discovery.\n\n"
        "## Workflow (3 steps, ~4 turns)\n\n"
        "### Step 1 — Call read_prior_artifacts (1 turn)\n"
        f"{spec_note}\n\n"
        "```js\n"
        "globalThis.ctx = JSON.parse(result);\n"
        "// ctx.description, ctx.artifacts.specify, ctx.feedback\n"
        "```\n\n"
        "### Step 2 — Explore codebase (1-2 turns)\n"
        "**MCP tools first:** Start with `mcp_codebase-index_get_project_summary` "
        "for orientation. Then call `mcp_codebase-index_find_symbol` to locate "
        "key symbols from the spec (e.g. WorkflowState, PhaseName, build_phase_agent). "
        "Use `mcp_codebase-index_get_dependencies` to understand how they relate. "
        "Batch multiple MCP calls in a single turn — they're all sub-millisecond.\n\n"
        "**Fallback only if needed:** If MCP tools don't cover your query, call "
        "`search_codebase` with queries derived from the specification.\n\n"
        "Example MCP exploration:\n"
        "```\n"
        "// Turn 1: get_project_summary (files: 274, functions: 979, classes: 203)\n"
        "// Turn 2: find_symbol('WorkflowState') → spine/models/state.py:45\n"
        "//          find_symbol('build_phase_agent') → spine/agents/factory.py:174\n"
        "//          get_dependencies('build_workflow_graph') → [build_workflow_graph, ...]\n"
        "```\n\n"
        "Store results in eval:\n"
        "```js\n"
        "globalThis.codebase = JSON.parse(searchResult);\n"
        "```\n\n"
        "### Step 3 — Call write_plan (1 turn)\n"
        "Synthesize spec + codebase research into the six required sections "
        "and call `write_plan`:\n"
        "- `architecture_overview`: components, data flow, interfaces\n"
        "- `technology_choices`: libraries/tools with rationale\n"
        "- `module_structure`: file/module layout (all paths must be real "
        "workspace paths from search results)\n"
        "- `api_designs`: function signatures, data models, schemas\n"
        "- `implementation_order`: phases/waves with dependencies\n"
        "- `testing_strategy`: test file paths, what to add/modify\n\n"
        "## Strict rules\n"
        "- Call `read_prior_artifacts` first — always.\n"
        "- Every file path in the plan MUST come from MCP or `search_codebase` "
        "results or be a new file inside a directory confirmed to exist. "
        "Do not invent paths like `src/main.py` without verification.\n"
        "- Call `write_plan` exactly once, with all required fields.\n"
        "- Total turns: ~4. If you have not called `write_plan` by turn 5, "
        "write it with what you have.\n\n"
        "## Eval context seed\n"
        "```js\n"
        f'globalThis.context = {{work_id: "{work_id}", '
        f'phase: "plan", plan_dir: "{plan_dir}"}};\n'
        "```\n\n"
    )
