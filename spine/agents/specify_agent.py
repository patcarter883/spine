"""SPINE specify agent — Deep Agent for the SPECIFY phase.

In the normal workflow, codebase exploration is handled upstream by the
exploration subgraph (LangGraph Send API) before this agent runs.  This
agent's job is to synthesise the exploration results and write the final
specification.md via the ``write_specification`` tool.

Tool surface (complete list):
- ``read_work_context`` — loads description, feedback, prior spec (rework)
- ``write_specification`` — structured write to specification.md (only)
- ``recall`` — retrieves relevant code chunks from vector store
- ``task`` (via SubAgentMiddleware) — dispatches researcher subagents
  (only used when the exploration subgraph was skipped)

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
    extra_tools: list[Any] | None = None,
) -> Any:
    """Build the Deep Agent for the SPECIFY phase.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.
        extra_tools: Optional additional tools (e.g., RecallTool for RAG).

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

    # Merge orchestrator tools with any extra tools (like RecallTool)
    all_tools = list(orchestrator_tools)
    if extra_tools:
        all_tools.extend(extra_tools)

    system_prompt = _build_specify_prompt() + build_artifact_prompt(
        state.get("artifacts", {}), PhaseName.SPECIFY.value, work_id=work_id
    )

    agent = build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.SPECIFY,
        system_prompt=system_prompt,
        subagents=_build_subagents(PhaseName.SPECIFY, state, config),
        extra_tools=all_tools,
        skip_filesystem_middleware=True,
    )

    return agent


def _build_specify_prompt() -> str:
    return (
        "You are the SPECIFY phase orchestrator. Synthesise the exploration results "
        "already in your context and write the formal specification.\\n\\n"
        "## Available tools (use only these)\\n"
        "- `read_work_context` — loads work description, feedback, prior spec. Call FIRST.\\n"
        "- `recall` — retrieves relevant code chunks from the vector knowledge base. "
        "Use this to understand existing patterns before writing the spec.\\n"
        "- `write_specification` — writes specification.md. Call LAST.\\n"
        "- `task` — dispatches a `researcher` subagent (only if exploration was skipped).\\n\\n"
        "## SPECIFY PHASE — STEP-BY-STEP WORKFLOW\\n\\n"
        "Execute these steps in order. Complete each fully before proceeding.\\n\\n"
        "### Step 1 — Call read_work_context (Turn 1)\\n"
        "1. Call `read_work_context` with no arguments.\\n"
        "2. Extract: ctx.description, ctx.feedback, ctx.prior_spec, ctx.spec_dir\\n\\n"
        "### Step 2 — Review retrieved context\\n"
        "Relevant code chunks have been pre-retrieved and are in your context. "
        "Analyze them to understand existing patterns, conventions, and architectural decisions. "
        "Do NOT dispatch additional researcher subagents unless the pre-retrieved context is insufficient.\\n\\n"
        "### Step 3 — Call write_specification (Turn 2)\\n"
        "3. Synthesize the retrieved context and/or research findings into these 5 required sections:\\n"
        "   - overview: 2-3 sentences of what needs to be built\\n"
        "   - requirements: numbered list of functional + non-functional requirements\\n"
        "   - architecture: bullet list of design decisions with rationale\\n"
        "   - interfaces: bullet list of APIs, data models, contracts with types\\n"
        "   - success_criteria: numbered list of 3-5 measurable outcomes\\n"
        "4. Call `write_specification` with all 5 fields.\\n"
        "   ALSO include `specification_json` — a JSON string with keys: title, summary, "
        "objectives (list), requirements (list), constraints (list), scope_inclusions (list), "
        "scope_exclusions (list), known_risks (list).\\n\\n"
        "### Turn Budget\\n"
        "Expected: 2 turns. If exceeding 4 turns without calling `write_specification`, "
        "check that context is available and proceed to write.\\n"
    )
