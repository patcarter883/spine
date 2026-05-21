"""SPINE plan agent — Deep Agent for the PLAN phase.

Reads the specification artifact and codebase structure, then writes
the technical plan via the ``write_plan`` tool.

Tool surface (complete list):
- ``read_prior_artifacts`` — loads specification + context in one call
- ``search_codebase`` — multi-query codebase file search
- ``write_plan`` — structured write to plan.md (only)
- ``task`` (via SubAgentMiddleware) — dispatches researcher subagents
- ``eval`` (via CodeInterpreterMiddleware) — parallel subagent dispatch, store results

No generic filesystem tools (ls, read_file, glob, grep, write_file,
edit_file, execute). The plan agent has targeted read access via
``read_prior_artifacts`` + ``search_codebase`` + researcher subagents,
and write access only to plan.md. It cannot browse the filesystem arbitrarily.

PLAN dispatching: research subagents explore codebase areas in parallel
via ``eval`` + ``Promise.allSettled``, matching the SPECIFY pattern.
This is the PRIMARY exploration strategy — MCP tools and ``search_codebase``
are supplemental, used only for narrow targeted lookups after broad
researcher dispatch.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.agents.artifacts import _artifact_path
from spine.agents.factory import build_phase_agent
from spine.agents.plan_tools import build_plan_agent_tools
from spine.agents.subagents import build_phase_subagents
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState


def _build_subagents(
    phase: PhaseName,
    state: WorkflowState,
    config: RunnableConfig | None,
) -> list[Any] | None:
    """Resolve subagent specs for the PLAN phase."""
    return build_phase_subagents(phase, state, config)


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

    system_prompt = _build_plan_prompt()

    agent = build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.PLAN,
        system_prompt=system_prompt,
        subagents=_build_subagents(PhaseName.PLAN, state, config),
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


def _build_plan_prompt() -> str:
    return (
        "You are the PLAN phase agent. Your job is to create a detailed "
        "technical plan from the specification, grounded in the actual "
        "codebase structure.\n\n"
        "## Your tool surface (complete list)\n"
        "- `read_prior_artifacts` — loads spec and all prior artifacts. "
        "No arguments. Call this FIRST.\n"
        "- `task` (via eval) — dispatches a `researcher` subagent. "
        "This is your PRIMARY codebase exploration tool — use it for any "
        "non-trivial codebase question.\n"
        "- `eval` — JavaScript REPL for parallel subagent dispatch and "
        "storing intermediate results.\n"
        "### Supplemental exploration (use only for narrow symbol-level questions)\n"
        "MCP codebase-index tools answer symbol-level questions in "
        "sub-milliseconds. Use these for targeted lookups AFTER dispatching "
        "researchers for broad exploration.\n"
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
        "content previews. Use for content-level queries the MCP "
        "tools don't cover.\n"
        "### Output\n"
        "- `write_plan` — writes plan.md. Call this LAST. "
        "This is the ONLY write tool.\n\n"
        "You do NOT have `ls`, `read_file`, `glob`, `grep`, `write_file`, "
        "`edit_file`, or `execute`. Do not attempt to call them. "
        "Use MCP tools, researcher subagents, and `search_codebase` "
        "for all file discovery.\n\n"
        "## Workflow (3 steps, ~4 turns)\n\n"
        "### Step 1 — Call read_prior_artifacts (1 turn)\n"
        "Call `read_prior_artifacts` with no arguments, store results:\n"
        "```js\n"
        "globalThis.ctx = JSON.parse(result);\n"
        "// ctx.description, ctx.artifacts.specify, ctx.feedback\n"
        "```\n\n"
        "### Step 2 — Dispatch researcher subagents in parallel (1 eval turn)\n"
        "Identify 2-4 areas of the codebase relevant to the specification. "
        "Dispatch one `researcher` subagent per area via `Promise.allSettled` "
        "inside a single `eval` call. Each task description must be fully "
        "self-contained — embed the work description and the specific area "
        "to investigate.\n\n"
        "**CRITICAL: Each description MUST be ≥200 characters.** "
        "Include: (1) the specification context, "
        "(2) specific file paths or modules to investigate, "
        "(3) what to look for (patterns, conventions, APIs, dependencies). "
        "Bare topic names like \"Research X\" produce empty results.\n\n"
        "Dispatch pattern:\n"
        "```js\n"
        "const desc = globalThis.ctx.description;\n"
        "const results = await Promise.allSettled([\n"
        '  tools.task({subagent_type: "researcher",\n'
        "    description: `Research area 1 for plan: ${desc}\\n"
        "Investigate: <specific module/component — file layout, existing "
        "patterns, tests>`}),\n"
        '  tools.task({subagent_type: "researcher",\n'
        "    description: `Research area 2 for plan: ${desc}\\n"
        "Investigate: <specific module/component>`}),\n"
        "]);\n"
        "globalThis.research = results;\n"
        "// Store reports:\n"
        "globalThis.reports = results.map(r => r.value);\n"
        "```\n\n"
        "BEFORE dispatching, call `read_prior_artifacts` to understand "
        "what was already discovered by the SPECIFY phase's researchers — "
        "don't re-research areas the specification already covers well.\n\n"
        "If the specification is simple and well-researched, skip researcher "
        "dispatch and go directly to Step 3.\n\n"
        "### Step 3 — Call write_plan (1 turn)\n"
        "Synthesize spec + codebase research into the six required sections "
        "and call `write_plan`:\n"
        "- `architecture_overview`: components, data flow, interfaces\n"
        "- `technology_choices`: libraries/tools with rationale\n"
        "- `module_structure`: file/module layout (all paths must be real "
        "workspace paths from research results)\n"
        "- `api_designs`: function signatures, data models, schemas\n"
        "- `implementation_order`: phases/waves with dependencies\n"
        "- `testing_strategy`: test file paths, what to add/modify\n\n"
        "## Strict rules\n"
        "- Call `read_prior_artifacts` first — always.\n"
        "- Every file path in the plan MUST come from MCP tools, "
        "researcher subagent results, or `search_codebase` results, "
        "or be a new file inside a directory confirmed to exist. "
        "Do not invent paths like `src/main.py` without verification.\n"
        "- Call `write_plan` exactly once, with all required fields.\n"
        "- Total turns: ~4. More than 5 turns without calling "
        "`write_plan` means something has gone wrong — "
        "write it with what you have.\n\n"
        "## Parallelism\n"
        "- **RESEARCHER SUBAGENTS are your PRIMARY exploration method.** "
        "Dispatch 2-4 researchers in parallel via `eval` + `Promise.allSettled` "
        "for any non-trivial codebase investigation. Never call `task` "
        "sequentially — it 4x's wall-clock time.\n"
        "- USE MCP tools for narrow, targeted symbol lookups only — "
        "after researchers have given you the broad picture.\n"
        "- FALLBACK to `search_codebase` only when MCP and researchers "
        "genuinely don't cover your need.\n\n"
        "## Token budget\n"
        "- This phase has a token budget (~500K for spec workflows). "
        "If you've made >15 `search_codebase` calls, you're over-researching — "
        "dispatch researchers instead or call `write_plan` with what you have.\n"
        "- The specification already contains substantial research from "
        "the SPECIFY phase. Do NOT re-discover what the spec already tells you.\n\n"
        "## Eval context seed\n"
        "Access session-specific context properties via `globalThis.context` "
        "preloaded in your workspace environment on first turn (e.g., "
        "use `globalThis.context.work_id` or `globalThis.context.plan_dir` inside eval).\n\n"
    )
