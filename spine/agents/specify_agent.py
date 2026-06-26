"""SPINE specify agent — Deep Agent for the SPECIFY phase.

Codebase exploration is handled upstream by the exploration subgraph
(LangGraph Send API dispatch of researcher subagents) before this agent
runs.  This agent's job is to synthesise the exploration results and write
the final specification.md via the ``write_specification`` tool.

Tool surface (complete list):
- ``write_specification`` — structured write to specification.md (only)
- ``recall`` — retrieves relevant code chunks from vector store

The work description, feedback and any prior spec (rework) are inlined into
the prompt rather than fetched via a tool, so there is no ``read_work_context``
round-trip (trace 019ec965).

No generic filesystem tools (ls, read_file, glob, grep, write_file,
edit_file, execute). The agent cannot read arbitrary files or write
to non-artifact paths — researcher subagents do all codebase exploration.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.agents.artifacts import build_artifact_prompt
from spine.agents.factory import build_phase_agent
from spine.agents.helpers import escalation_level_for_phase
from spine.agents.specify_tools import build_specify_orchestrator_tools
from spine.agents.tool_forcing import ForceToolUntilCalledMiddleware
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState


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

    # read_work_context is omitted: the work description, classification,
    # retrieved code and feedback are inlined into the user prompt by the
    # caller (and any prior spec on rework), so the tool round-trip is dead
    # weight (trace 019ec965: ~19K prompt tokens for a 29-token no-op call).
    orchestrator_tools = build_specify_orchestrator_tools(
        workspace_root=workspace_root,
        work_id=work_id,
        description=description,
        work_type=work_type,
        feedback=feedback,
        include_read_work_context=False,
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
        extra_tools=all_tools,
        skip_filesystem_middleware=True,
        # Force a tool call every turn until write_specification succeeds, so a
        # weak/quantized model can't end the turn with the spec as fenced JSON
        # text instead of calling the tool. No gate_tool here: the orchestrator
        # may legitimately call `recall` before the write.
        extra_middleware=[ForceToolUntilCalledMiddleware(final_tool="write_specification")],
        # Escalate the model on critic-driven rework (no-op without a ladder).
        escalation_level=escalation_level_for_phase(state, PhaseName.SPECIFY),
    )

    return agent


def build_specify_synthesizer(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the synthesize-only Deep Agent for the SPECIFY phase.

    Tool surface is intentionally minimal: ``read_work_context`` and
    ``write_specification`` only. No ``recall``, no ``search_codebase``,
    no researcher subagents — codebase lookups were done by the upstream
    exploration subgraph and the findings are already in the prompt.
    This stops the synthesizer from re-exploring.
    """
    work_id = state.get("work_id", "")
    workspace_root = state.get("workspace_root", ".")
    work_type = state.get("work_type", "")
    description = state.get("description", "")
    feedback_raw = state.get("feedback", [])
    feedback = [str(f) for f in feedback_raw] if feedback_raw else []

    # read_work_context is omitted: description, feedback and (on rework) the
    # prior spec are inlined into the synthesizer prompt by _synthesize_specify,
    # so the synthesizer needs only write_specification. This drops the 2-call
    # flow to a single call (trace 019ec965).
    orchestrator_tools = build_specify_orchestrator_tools(
        workspace_root=workspace_root,
        work_id=work_id,
        description=description,
        work_type=work_type,
        feedback=feedback,
        include_read_work_context=False,
    )

    system_prompt = _build_specify_synthesizer_prompt() + build_artifact_prompt(
        state.get("artifacts", {}), PhaseName.SPECIFY.value, work_id=work_id
    )

    from spine.agents.synthesis_budget import synthesis_completion_cap

    return build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.SPECIFY,
        system_prompt=system_prompt,
        extra_tools=list(orchestrator_tools),
        skip_filesystem_middleware=True,
        # The spec JSON is 2-4K tokens; without a clamp the request inherits
        # the global max_completion_tokens (30K) and a finite-window model
        # 400s once prompt + completion budget exceed the window (019eb3dd).
        completion_token_cap=synthesis_completion_cap(PhaseName.SPECIFY.value),
        # write_specification is now the synthesizer's only tool, so force it
        # from turn 1 — the model cannot stall in prose. No gate_tool: there is
        # no longer a read_work_context call to wait on. Forcing releases as
        # soon as the write succeeds so the loop ends.
        extra_middleware=[
            ForceToolUntilCalledMiddleware(final_tool="write_specification")
        ],
        # Escalate the model on critic-driven rework (no-op without a ladder).
        escalation_level=escalation_level_for_phase(state, PhaseName.SPECIFY),
    )


def _build_specify_prompt() -> str:
    return (
        "You are the SPECIFY phase orchestrator. Synthesise the exploration results "
        "already in your context and write the formal specification.\\n\\n"
        "## Available tools (use only these)\\n"
        "- `recall` — retrieves relevant code chunks from the vector knowledge base. "
        "Use this to understand existing patterns before writing the spec.\\n"
        "- `write_specification` — writes specification.md. Call LAST.\\n\\n"
        "The work description, task classification, pre-retrieved code, and any "
        "prior critic feedback (plus the prior specification on a rework pass) are "
        "provided inline in the user message below — you do NOT need to call any "
        "tool to load them. Researcher subagents that explored the codebase were "
        "dispatched by the upstream exploration subgraph; their findings are "
        "already in your context. Do not try to dispatch more.\\n\\n"
        "## SPECIFY PHASE — STEP-BY-STEP WORKFLOW\\n\\n"
        "Execute these steps in order. Complete each fully before proceeding.\\n\\n"
        "### Step 1 — Review the provided context\\n"
        "Read the work description, classification, and pre-retrieved code chunks "
        "in the user message. Analyze them to understand existing patterns, "
        "conventions, and architectural decisions. Do NOT dispatch additional "
        "researcher subagents unless the pre-retrieved context is insufficient.\\n\\n"
        "### Step 2 — Call write_specification\\n"
        "Synthesize the retrieved context and/or research findings into the structured "
        "fields below and call `write_specification` ONCE. The tool renders markdown and "
        "emits JSON for you — DO NOT author markdown, DO NOT hand-serialize JSON.\\n\\n"
        "Fields (all lists are arrays of short strings — one item per array entry):\\n"
        "- title: short specification title\\n"
        "- summary: 2-3 sentence executive summary\\n"
        "- objectives: list of high-level goals\\n"
        "- requirements: list of functional + non-functional requirements (each measurable). REQUIRED — must have at least one item.\\n"
        "- constraints: list of non-functional constraints\\n"
        "- scope_inclusions: list of areas explicitly in scope\\n"
        "- scope_exclusions: list of areas explicitly out of scope\\n"
        "- known_risks: list of open questions or risks\\n\\n"
        "### Turn Budget\\n"
        "Expected: 1 turn (optionally 1 `recall` first). If exceeding 3 turns "
        "without calling `write_specification`, proceed to write with the context "
        "already provided.\\n"
    )


def _build_specify_synthesizer_prompt() -> str:
    return (
        "You are the SPECIFY phase synthesizer. Codebase research was completed "
        "BEFORE you started — the findings are injected into your prompt below. "
        "Your job is to synthesize those findings into a structured specification "
        "and call `write_specification` ONCE. Do NOT re-explore the codebase.\\n\\n"
        "## Available tools (the ONLY tool you have)\\n"
        "- `write_specification` — writes both specification.md and specification.json.\\n\\n"
        "The work description, critic feedback, and (on a rework pass) the prior "
        "specification are all provided inline in the user message — you do NOT "
        "need to call any tool to load them.\\n\\n"
        "## Workflow (exactly 1 call)\\n\\n"
        "### Call write_specification\\n"
        "Synthesize the findings (already in your prompt) plus the work context into "
        "the structured fields below and call `write_specification` ONCE. The tool "
        "renders markdown and emits JSON for you — DO NOT author markdown, DO NOT "
        "hand-serialize JSON, DO NOT call write_file.\\n\\n"
        "Fields (all lists are arrays of short strings — one item per array entry):\\n"
        "- title: short specification title\\n"
        "- summary: 2-3 sentence executive summary\\n"
        "- objectives: list of high-level goals\\n"
        "- requirements: list of functional + non-functional requirements (each measurable). REQUIRED — must have at least one item.\\n"
        "- constraints: list of non-functional constraints\\n"
        "- scope_inclusions: list of areas explicitly in scope\\n"
        "- scope_exclusions: list of areas explicitly out of scope\\n"
        "- known_risks: list of open questions or risks\\n"
    )
