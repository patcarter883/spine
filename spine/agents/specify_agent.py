"""SPINE specify agent — Deep Agent for the SPECIFY phase.

Orchestrates specification writing via researcher subagents, then writes
the final specification.md via the ``write_specification`` tool.

Tool surface (complete list):
- ``read_work_context`` — loads description, feedback, prior spec (rework)
- ``write_specification`` — structured write to specification.md (only)
- ``task`` (via SubAgentMiddleware) — dispatches researcher subagents
- ``eval`` (via CodeInterpreterMiddleware) — parallel subagent dispatch

No generic filesystem tools (ls, read_file, glob, grep, write_file,
edit_file, execute). The agent cannot read arbitrary files or write
to non-artifact paths — researcher subagents do all codebase exploration.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.agents.artifacts import build_artifact_prompt
from spine.agents.factory import build_phase_agent
from spine.agents.specify_tools import build_specify_orchestrator_tools
from spine.agents.subagents import build_phase_subagents
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState


def _build_subagents(
    phase: PhaseName,
    state: WorkflowState,
    config: RunnableConfig | None,
) -> list[Any] | None:
    """Resolve subagent specs for the SPECIFY phase."""
    return build_phase_subagents(phase, state, config)


def build_specify_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the SPECIFY phase.

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

    spec_dir = f".spine/artifacts/{work_id}/specify"

    orchestrator_tools = build_specify_orchestrator_tools(
        workspace_root=workspace_root,
        work_id=work_id,
        description=description,
        work_type=work_type,
        feedback=feedback,
    )

    system_prompt = _build_specify_prompt() + build_artifact_prompt(
        state.get("artifacts", {}), PhaseName.SPECIFY.value, work_id=work_id
    )

    agent = build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.SPECIFY,
        system_prompt=system_prompt,
        subagents=_build_subagents(PhaseName.SPECIFY, state, config),
        add_summarization=True,
        extra_tools=orchestrator_tools,
        skip_filesystem_middleware=True,
    )

    return agent


def _build_specify_prompt() -> str:
    return (
        "You are the SPECIFY phase orchestrator. Your job is to produce a "
        "detailed technical specification for the given work description "
        "by dispatching researcher subagents to explore the codebase, then "
        "synthesizing their findings into a structured specification.\n\n"
        "## Your tool surface (complete list)\n"
        "- `read_work_context` — loads work description, feedback, and prior "
        "spec. No arguments. Call this FIRST.\n"
        "- `write_specification` — writes specification.md. Call this LAST. "
        "This is the ONLY write tool.\n"
        "- `task` (via eval) — dispatches a `researcher` subagent.\n"
        "- `eval` — JavaScript REPL for parallel subagent dispatch.\n\n"
        "You do NOT have `ls`, `read_file`, `glob`, `grep`, `write_file`, "
        "`edit_file`, or `execute`. Do not attempt to call them — they do "
        "not exist in your session. Codebase exploration is done by "
        "`researcher` subagents, not by you.\n\n"
        "## Workflow (3 steps, ~3 turns)\n\n"
        "### Step 1 — Call read_work_context (1 turn)\n"
        "Call `read_work_context` with no arguments. Store the result:\n"
        "```js\n"
        "globalThis.ctx = JSON.parse(result);\n"
        "// ctx.description, ctx.feedback, ctx.prior_spec, ctx.spec_dir\n"
        "```\n\n"
        "### Step 2 — Dispatch researcher subagents in parallel (1 eval turn)\n"
        "Identify 2-4 areas of the codebase relevant to the work description. "
        "Dispatch one `researcher` subagent per area via `Promise.allSettled` "
        "inside a single `eval` call. Each task description must be fully "
        "self-contained — embed the work description and the specific area "
        "to investigate.\n\n"
        "Dispatch pattern:\n"
        "```js\n"
        "const desc = globalThis.ctx.description;\n"
        "const results = await Promise.allSettled([\n"
        '  tools.task({subagent_type: "researcher",\n'
        "    description: `Research area 1 relevant to: ${desc}\\n"
        "Investigate: <specific module/component>`}),\n"
        '  tools.task({subagent_type: "researcher",\n'
        "    description: `Research area 2 relevant to: ${desc}\\n"
        "Investigate: <specific module/component>`}),\n"
        "]);\n"
        "globalThis.research = results;\n"
        "```\n\n"
        "Skip researcher dispatch if the work description is self-contained "
        "and requires no codebase knowledge (e.g. a standalone algorithm).\n\n"
        "### Step 3 — Call write_specification (1 turn)\n"
        "Synthesize research findings into the five required sections and call "
        "`write_specification`. All five fields are required:\n"
        "- `overview`: summary of what to build\n"
        "- `requirements`: functional + non-functional, as a list\n"
        "- `architecture`: design decisions and rationale\n"
        "- `interfaces`: APIs, data models, contracts\n"
        "- `success_criteria`: measurable, verifiable outcomes\n\n"
        "## Strict rules\n"
        "- Call `read_work_context` first — always.\n"
        "- Dispatch researchers before writing. Do not write the spec from "
        "the description alone without codebase research (unless trivial).\n"
        "- Call `write_specification` exactly once, with all required fields.\n"
        "- Total turns: ~3. More than 5 turns without calling "
        "`write_specification` means something has gone wrong.\n\n"
        "## Eval context seed\n"
        "Access session-specific context properties via `globalThis.context` "
        "preloaded in your workspace environment on first turn (e.g., "
        "use `globalThis.context.work_id` or `globalThis.context.spec_dir` inside eval).\n\n"
    )
