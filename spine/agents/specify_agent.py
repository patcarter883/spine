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
        extra_tools=orchestrator_tools,
        skip_filesystem_middleware=True,
    )

    return agent


def _build_specify_prompt() -> str:
    return (
        "You are the SPECIFY phase orchestrator. Produce a detailed technical "
        "specification by dispatching researcher subagents, then write the spec.\\n\\n"
        "## Available tools (use only these)\\n"
        "- `read_work_context` — loads work description, feedback, prior spec. Call FIRST.\\n"
        "- `write_specification` — writes specification.md. Call LAST.\\n"
        "- `task` — dispatches a `researcher` subagent (requires eval).\\n"
        "- `eval` — JavaScript REPL for parallel subagent dispatch.\\n\\n"
        "## SPECIFY PHASE — STEP-BY-STEP WORKFLOW\\n\\n"
        "Execute these steps in order. Complete each fully before proceeding.\\n\\n"
        "### Step 1 — Call read_work_context (Turn 1)\\n"
        "1. Call `read_work_context` with no arguments.\\n"
        "2. Store result: `globalThis.ctx = JSON.parse(result)`\\n"
        "3. Extract: ctx.description, ctx.feedback, ctx.prior_spec, ctx.spec_dir\\n\\n"
        "### Step 2 — Dispatch researcher subagents (Turn 2, conditional)\\n"
        "4. Check: Does ctx.description contain any file path (e.g. `*.py`, `spine/...`) "
        "or codebase term (\"module\", \"function\", \"class\", \"SPINE\")?\\n"
        "5. IF yes → Identify 2-4 codebase areas:\\n"
        "   a. Write task description including: work description, specific file paths, "
        "      3-4 investigation questions.\\n"
        "   b. Ensure each description is >= 200 characters.\\n"
        "   c. Dispatch all researchers in single `eval` call via `Promise.allSettled`.\\n"
        "   d. Store results: `globalThis.research = results`\\n"
        "6. IF no file paths and no codebase terms → Set `globalThis.research = null`, "
        "proceed to Step 3.\\n\\n"
        "### Step 3 — Call write_specification (Turn 3)\\n"
        "7. Synthesize research findings into these 5 required sections:\\n"
        "   - overview: 2-3 sentences of what needs to be built\\n"
        "   - requirements: numbered list of functional + non-functional requirements\\n"
        "   - architecture: bullet list of design decisions with rationale\\n"
        "   - interfaces: bullet list of APIs, data models, contracts with types\\n"
        "   - success_criteria: numbered list of 3-5 measurable outcomes\\n"
        "8. Call `write_specification` with all 5 fields.\\n\\n"
        "### Turn Budget\\n"
        "Expected: 3 turns. If exceeding 5 turns without calling `write_specification`, "
        "verify Step 2 was completed correctly.\\n"
    )
