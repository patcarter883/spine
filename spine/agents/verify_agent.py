"""SPINE verify agent — Deep Agent for the VERIFY phase.

Same orchestrator pattern as the implement phase: dispatch one
``slice-verifier`` per feature slice, synthesize results into
``verification.md``. The orchestrator does not verify slices inline.

Tool restriction: orchestrator gets purpose-built tools that enforce
dispatch-only behavior:
- ``read_verify_context`` — loads all verification inputs in one call
- ``write_verification_report`` — writes verification.md (only write surface)
- ``task`` (from SubAgentMiddleware)
- ``eval`` (from CodeInterpreterMiddleware)

Generic filesystem tools are excluded — the orchestrator cannot
read arbitrary files or write source code.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.agents.artifacts import (
    artifact_path,
    build_artifact_prompt,
    build_current_phase_write_prompt,
)
from spine.agents.factory import build_phase_agent
from spine.agents.subagents import build_phase_subagents
from spine.agents.verify_tools import build_verify_orchestrator_tools
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState


def _build_subagents(
    phase: PhaseName,
    state: WorkflowState,
    config: RunnableConfig | None,
) -> list[Any] | None:
    """Resolve subagent specs for the VERIFY phase."""
    return build_phase_subagents(phase, state, config)


def build_verify_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the VERIFY phase.

    Always builds a dispatch-only orchestrator. All actual verification
    work (reading source, running tests, checking acceptance criteria)
    is delegated to ``slice-verifier`` subagents.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        A compiled Deep Agent ready for invocation.
    """
    work_id = state.get("work_id", "")
    workspace_root = state.get("workspace_root", ".")

    # ── Build custom tools ───────────────────────────────────────────────
    custom_tools = build_verify_orchestrator_tools(workspace_root, work_id)

    system_prompt = (
        _build_orchestrator_prompt()
        + build_current_phase_write_prompt(
            work_id, PhaseName.VERIFY.value, expected_files=["verification.md"]
        )
        + build_artifact_prompt(state.get("artifacts", {}), PhaseName.VERIFY.value, work_id=work_id)
    )

    # ── Build agent with skip_filesystem_middleware ──────────────────────
    # The custom tools replace generic filesystem access entirely.
    agent = build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.VERIFY,
        system_prompt=system_prompt,
        subagents=_build_subagents(PhaseName.VERIFY, state, config),
        extra_tools=custom_tools,
        skip_filesystem_middleware=True,  # Custom tools replace generic FS
    )

    return agent


# ── Prompt builder ─────────────────────────────────────────────────────


def _build_orchestrator_prompt() -> str:
    """Build the verify orchestrator system prompt."""
    return (
        "You are the VERIFY phase orchestrator. You do NOT inspect source "
        "code yourself — you dispatch one `slice-verifier` subagent per "
        "feature slice and synthesize their verdicts into a single report.\n\n"
        "Your tools are restricted to purpose-built tools plus `task` and `eval`. "
        "You do NOT have generic filesystem tools — your toolset is "
        "deliberately minimal to enforce dispatch-only behavior.\n\n"
        "## Tool surface\n"
        "- `read_verify_context` — Call this FIRST. Returns all verification inputs "
        "(slices, codebase_map, implementation.md) in one structured call. "
        "No arguments needed.\n"
        "- `write_verification_report` — Call this LAST. Accepts structured "
        "verification results from subagents and writes verification.md. "
        "This is the ONLY write surface.\n"
        "- `task` — Dispatch one `slice-verifier` subagent per slice.\n"
        "- `eval` — Run JS for parallel subagent dispatch.\n\n"
        "## Workflow (2 steps)\n\n"
        "### Step 1 — Read context\n"
        "Call `read_verify_context` to load slices, codebase_map, and implementation. "
        "The result includes:\n"
        "- `slices`: mapping of slice filename → full content\n"
        "- `codebase_map`: codebase navigation guide\n"
        "- `implementation`: full implementation.md content\n\n"
        "### Step 2 — Dispatch and synthesize\n"
        "Inside ONE `eval` call:\n"
        "1. Store the context from Step 1 in interpreter state\n"
        "2. Dispatch all slice-verifier subagents in parallel via `Promise.allSettled`\n"
        "3. Collect results and call `write_verification_report`\n\n"
        "Each `task` description MUST be self-contained. Embed:\n"
        "1. The full slice text (acceptance criteria, target files)\n"
        "2. Relevant implementation excerpt for this slice\n"
        "3. Codebase-map excerpts for files involved\n\n"
        "Dispatch pattern inside eval:\n"
        "```js\n"
        "const {slices, codebase_map, implementation} = globalThis.verifyContext;\n"
        "const sliceFiles = Object.keys(slices);\n"
        "const dispatches = sliceFiles.map(name => \n"
        "  tools.task({\n"
        "    subagent_type: 'slice-verifier',\n"
        "    description: `Verify slice: ${name}\n\n` +\n"
        "      `## Slice Definition\n${slices[name]}\n` +\n"
        "      `## Codebase Map\n${codebase_map}\n` +\n"
        "      `## Implementation\n${implementation}\n`\n"
        "  })\n"
        ");\n"
        "const results = await Promise.allSettled(dispatches);\n"
        "globalThis.verifyResults = results;\n"
        "```\n\n"
        "After eval, extract results and call `write_verification_report` with:\n"
        "- `verification_results`: array of {slice_name, verdict, checklist, gaps, recommendations}\n"
        "- `summary`: overall verification status\n\n"
        "## Strict Rules\n"
        "- You have NO generic filesystem tools. Call `read_verify_context` first.\n"
        "- You MUST dispatch one `slice-verifier` subagent per slice.\n"
        "- Do NOT attempt to verify slices inline — dispatch subagents.\n"
        "- The ONLY valid `subagent_type` is `slice-verifier`.\n"
        "- Subagent dispatch MUST happen inside `eval` for parallelism.\n"
        "- `verification.md` is REQUIRED — the phase fails without it.\n"
    )