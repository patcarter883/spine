"""SPINE plan agent — Deep Agent for the PLAN phase.

Reads the specification artifact and codebase structure (pre-explored by the
exploration subgraph) then writes the technical plan via the
``write_structured_plan`` tool, emitting a flat array of feature_slices with
explicit dependencies.

Tool surface (complete list):
- ``read_prior_artifacts`` — loads specification + context in one call
- ``search_codebase`` — multi-query codebase file search
- ``write_structured_plan`` — structured write with feature_slices (only)

No generic filesystem tools (ls, read_file, glob, grep, write_file,
edit_file, execute). The plan agent has targeted read access via
``read_prior_artifacts`` + ``search_codebase``, and write access only
through ``write_structured_plan``. It cannot browse the filesystem
arbitrarily.

PLAN research strategy: codebase exploration is handled UPSTREAM by the
exploration subgraph (LangGraph Send API) before this agent runs.  The
agent synthesises those findings to produce the plan.  ``search_codebase``
and MCP tools are supplemental, used only for narrow targeted lookups that
were not covered by the upstream exploration.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.agents.artifacts import artifact_path
from spine.agents.factory import build_phase_agent
from spine.agents.plan_tools import (
    ReadPriorArtifactsTool,
    StructuredWritePlanTool,
    build_plan_agent_tools,
)
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
        extra_tools=agent_tools,
        skip_filesystem_middleware=True,
    )

    return agent


def build_plan_synthesizer(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the synthesize-only Deep Agent for the PLAN phase.

    Tool surface is intentionally minimal: ``read_prior_artifacts`` and
    ``write_structured_plan`` only. No ``search_codebase``, no researcher
    subagents — exploration was done upstream and the findings are
    already in the prompt. This stops the synthesizer from re-exploring.
    """
    work_id = state.get("work_id", "")
    workspace_root = state.get("workspace_root", ".")
    work_type = state.get("work_type", "")
    description = state.get("description", "")
    feedback_raw = state.get("feedback", [])
    feedback = [str(f) for f in feedback_raw] if feedback_raw else []

    prior_phase_dirs = _resolve_prior_phase_dirs(state, work_id)
    plan_dir = artifact_path(work_id, PhaseName.PLAN.value)

    synthesizer_tools = [
        ReadPriorArtifactsTool(
            workspace_root=workspace_root,
            work_id=work_id,
            work_type=work_type,
            description=description,
            feedback=feedback,
            plan_dir=plan_dir,
            prior_phase_dirs=prior_phase_dirs,
        ),
        StructuredWritePlanTool(
            workspace_root=workspace_root,
            plan_dir=plan_dir,
        ),
    ]

    return build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.PLAN,
        system_prompt=_build_plan_synthesizer_prompt(),
        subagents=None,
        extra_tools=synthesizer_tools,
        skip_filesystem_middleware=True,
    )


def _resolve_prior_phase_dirs(
    state: WorkflowState,
    work_id: str,
) -> dict[str, str]:
    """Map phase names to their artifact directories for phases with artifacts."""
    artifacts = state.get("artifacts", {}) or {}
    dirs: dict[str, str] = {}
    for phase, phase_artifacts in artifacts.items():
        if phase_artifacts and isinstance(phase_artifacts, dict):
            dirs[phase] = artifact_path(work_id, phase)
    return dirs


def _build_plan_prompt() -> str:
    return (
        "You are the PLAN phase agent. Create a technical plan from the "
        "specification, grounded in codebase structure. Output: flat array of "
        "feature_slices with dependencies.\\n\\n"
        "## Available tools (use only these)\\n"
        "- `read_prior_artifacts` — loads spec and all prior artifacts. Call FIRST.\\n"
        "- `search_codebase` — targeted file search for narrow lookups not in exploration.\\n"
        "- `write_structured_plan` — emits feature_slices. Call LAST.\\n\\n"
        "## WORKFLOW (2 steps, ~3 turns)\\n\\n"
        "### Step 1 — Call read_prior_artifacts (Turn 1)\\n"
        "Call with no arguments to load the specification and prior exploration results.\\n\\n"
        "### Step 2 — Review exploration results (already available)\\n"
        "Codebase research was run BEFORE this agent started via the LangGraph "
        "exploration subgraph (Send API parallel dispatch). The research findings are "
        "injected into your context alongside the specification. Synthesise them — do "
        "NOT dispatch additional researcher subagents. Use `search_codebase` only for "
        "narrow targeted lookups (specific file paths, symbol names) not already covered.\\n\\n"
        "### Step 3 — Call write_structured_plan (Turn 2-3)\\n"
        "Synthesize spec + exploration into the structured fields below and call "
        "write_structured_plan ONCE. The tool renders markdown and emits JSON — do "
        "NOT author markdown.\\n\\n"
        "Top-level fields:\\n"
        "- architecture_overview: prose paragraph (string) on how the components fit together\\n"
        "- technology_choices: list of short strings, one item per choice (rationale inline)\\n"
        "- feature_slices: list of structured slice objects (see below). REQUIRED — at least one.\\n"
        "- testing_strategy: prose paragraph (string)\\n"
        "- risks: list of short strings, one item per risk\\n"
        "- codebase_map: optional prose paragraph (string)\\n\\n"
        "Each feature_slices item is a structured object with these fields:\\n"
        "- id: unique short identifier (e.g., 'add-user-model')\\n"
        "- title: human-readable one-line summary\\n"
        "- target_files: list of every file path to create/modify\\n"
        "- execution_requirements: detailed implementation instructions (string)\\n"
        "- dependencies: list of slice IDs that must complete first\\n"
        "- acceptance_criteria: list of concrete test/verification steps\\n"
        "- complexity: 'small', 'medium', or 'large'\\n\\n"
        "## Rework handling\\n"
        "If feedback exists in prior artifacts, address every item before calling "
        "`write_structured_plan`.\\n\\n"
        "## Rules\\n"
        "- Call `read_prior_artifacts` first.\\n"
        "- Every target_file path must come from MCP, search_codebase, or "
        "confirmed existing directory.\\n"
        "- Call `write_structured_plan` exactly once with all required fields.\\n"
        "- If not called by turn 4, write with current information.\\n"
    )


def _build_plan_synthesizer_prompt() -> str:
    return (
        "You are the PLAN phase synthesizer. Codebase exploration was completed "
        "BEFORE you started — the findings are injected into your prompt below. "
        "Your job is to synthesize spec + findings into a structured plan and call "
        "`write_structured_plan` ONCE. Do NOT re-explore the codebase.\\n\\n"
        "## Available tools (the ONLY tools you have)\\n"
        "- `read_prior_artifacts` — loads the specification and prior artifacts in one call. Call FIRST.\\n"
        "- `write_structured_plan` — writes both plan.md and plan.json. Call LAST.\\n\\n"
        "## Workflow (exactly 2 calls)\\n\\n"
        "### Step 1 — Call read_prior_artifacts\\n"
        "Call with no arguments to load the specification and any prior plan artifacts.\\n\\n"
        "### Step 2 — Call write_structured_plan\\n"
        "Synthesize spec + findings (already in your prompt) into structured fields and call "
        "`write_structured_plan` ONCE. The tool renders markdown and emits JSON for you — "
        "DO NOT author markdown, DO NOT hand-serialize JSON, DO NOT call write_file.\\n\\n"
        "Top-level fields:\\n"
        "- architecture_overview: prose paragraph (string)\\n"
        "- technology_choices: list of short strings, one item per choice\\n"
        "- feature_slices: list of structured slice objects (see below). REQUIRED — at least one.\\n"
        "- testing_strategy: prose paragraph (string)\\n"
        "- risks: list of short strings, one item per risk\\n"
        "- codebase_map: optional prose paragraph (string)\\n\\n"
        "Each feature_slices item:\\n"
        "- id: unique short identifier (e.g., 'add-user-model')\\n"
        "- title: human-readable one-line summary\\n"
        "- target_files: list of every file path to create/modify\\n"
        "- execution_requirements: detailed implementation instructions (string)\\n"
        "- dependencies: list of slice IDs that must complete first\\n"
        "- acceptance_criteria: list of concrete test/verification steps\\n"
        "- complexity: 'small', 'medium', or 'large'\\n"
    )
