"""SPINE plan agent — Deep Agent for the PLAN phase.

Reads the specification artifact and codebase structure, then writes
the technical plan via the ``write_structured_plan`` tool, emitting a
flat array of feature_slices with explicit dependencies.

Tool surface (complete list):
- ``read_prior_artifacts`` — loads specification + context in one call
- ``search_codebase`` — multi-query codebase file search
- ``write_structured_plan`` — structured write with feature_slices (only)
- ``eval`` (via CodeInterpreterMiddleware) — orchestration / store results

No generic filesystem tools (ls, read_file, glob, grep, write_file,
edit_file, execute). The plan agent has targeted read access via
``read_prior_artifacts`` + ``search_codebase``, and write access only
through ``write_structured_plan``. It cannot browse the filesystem
arbitrarily.

PLAN dispatching: research subagents explore codebase areas in parallel
via ``eval`` + ``Promise.allSettled``, matching the SPECIFY pattern.
This is the PRIMARY exploration strategy — MCP tools and ``search_codebase``
are supplemental, used only for narrow targeted lookups after broad
researcher dispatch.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.agents.artifacts import artifact_path
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
            dirs[phase] = artifact_path(work_id, phase)
    return dirs


def _build_plan_prompt() -> str:
    return (
        "You are the PLAN phase agent. Create a technical plan from the "
        "specification, grounded in codebase structure. Output: flat array of "
        "feature_slices with dependencies.\\n\\n"
        "## Available tools (use only these)\\n"
        "- `read_prior_artifacts` — loads spec and all prior artifacts. Call FIRST.\\n"
        "- `task` — dispatches a `researcher` subagent via eval.\\n"
        "- `eval` — JavaScript REPL for parallel dispatch and storing results.\\n"
        "- `write_structured_plan` — emits feature_slices. Call LAST.\\n\\n"
        "## WORKFLOW (3 steps, ~4 turns)\\n\\n"
        "### Step 1 — Call read_prior_artifacts (Turn 1)\\n"
        "Call with no arguments, store: `globalThis.ctx = JSON.parse(result)`\\n\\n"
        "### Step 2 — Dispatch spec-aware researchers (Turn 2)\\n"
        "Identify 2-4 areas needing codebase mapping. For each:\\n"
        "1. Extract relevant spec section from ctx.artifacts.specify['specification.md']\\n"
        "2. Dispatch researcher with spec section + investigation task in description\\n"
        "3. Use `Promise.allSettled` for parallel dispatch in single eval call\\n"
        "4. Ensure each description is >= 300 characters with embedded spec content\\n\\n"
        "### Step 3 — Call write_structured_plan (Turn 3)\\n"
        "Synthesize spec + research into feature_slices and call write_structured_plan.\\n\\n"
        "Each slice requires these fields:\\n"
        "- id: unique short identifier (e.g., 'add-user-model')\\n"
        "- title: human-readable one-line summary\\n"
        "- target_files: list of every file path to create/modify\\n"
        "- execution_requirements: detailed implementation instructions\\n"
        "- dependencies: list of slice IDs that must complete first\\n"
        "- acceptance_criteria: concrete test/verification steps\\n"
        "- complexity: 'small', 'medium', or 'large'\\n\\n"
        "## Rework handling\\n"
        "If feedback exists in prior artifacts, address every item before calling "
        "`write_structured_plan`.\\n\\n"
        "## Rules\\n"
        "- Call `read_prior_artifacts` first.\\n"
        "- Every target_file path must come from MCP, search_codebase, or "
        "confirmed existing directory.\\n"
        "- Call `write_structured_plan` exactly once with all required fields.\\n"
        "- If not called by turn 5, write with current information.\\n"
    )
