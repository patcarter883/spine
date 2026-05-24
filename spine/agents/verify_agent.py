"""SPINE verify agent — Deep Agent for the VERIFY phase.

Dispatch one ``slice-verifier`` subagent per feature slice, synthesize results
into ``verification.md``. The orchestrator does not verify slices inline.

Tool surface (complete list):
- ``read_verify_context`` — loads all verification inputs in one call
- ``write_verification_report`` — writes verification.md
- ``task`` (via SubAgentMiddleware) — dispatches slice-verifier subagents
- ``eval`` (via CodeInterpreterMiddleware) — parallel subagent dispatch

No generic filesystem tools are exposed — the orchestrator has targeted access
via these purpose-built tools only."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.agents.artifacts import (
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
        "You are the VERIFY phase orchestrator. Dispatch one `slice-verifier` "
        "subagent per feature slice and synthesize their verdicts into a single report.\n\n"
        "## Your Tool Surface (ONLY these tools)\n"
        "- `read_verify_context` — loads structured slices (from plan.json), codebase map, "
        "and structured implementation results (from implementation.json) in ONE call. "
        "No arguments needed. Call this FIRST.\n"
        "- `write_verification_report` — writes verification.md. Call this LAST.\n"
        "- `eval` — JavaScript REPL for subagent dispatch. Use `tools.task()` inside.\n\n"
        "### Step 1 — Call read_verify_context\n"
        "Call `read_verify_context` with no arguments. It returns:\n"
        "```json\n"
        "{\n"
        '  "slices": {"<slice_id>": {"id": "<slice_id>", "files": [...], "description": "...", ...}},\n'
        '  "codebase_map": "<string>",\n'
        '  "implementation": {"<structured implementation results dict>"}\n'
        "}\n"
        "```\n\n"
        "Store in eval: `globalThis.verifyContext = result;`\n\n"
        "### Step 2 — Dispatch Subagents In Parallel (1 eval turn)\n"
        "Complete ALL slices in a SINGLE eval call:\n"
        "```js\n"
        "const {slices, codebase_map, implementation} = globalThis.verifyContext;\n"
        "const implStr = JSON.stringify(implementation, null, 2);\n"
        "const results = await Promise.allSettled(\n"
        "  Object.entries(slices).map(([id, data]) =>\n"
        "    tools.task({\n"
        "      subagent_type: 'slice-verifier',\n"
        "      description: `Verify slice: ${id}\\n\\n` +\n"
        "        `## Slice Definition\\n${JSON.stringify(data, null, 2)}\\n` +\n"
        "        `## Codebase Map\\n${codebase_map}\\n` +\n"
        "        `## Implementation\\n${implStr}\\n`\n"
        "    })\n"
        "  )\n"
        ");\n"
        "globalThis.verifyResults = results;\n"
        "```\n\n"
        "### Step 3 — Call write_verification_report\n"
        "Parse `globalThis.verifyResults` and call `write_verification_report` with:\n"
        "- `verification_results`: array of {slice_name, verdict, checklist, gaps, recommendations}\n"
        "- `summary`: overall verification status\n\n"
        "## Completion Rules\n"
        "- Target ~3 total turns. If >5 turns without dispatching, write partial report.\n"
        "- Use PTC tools inside eval: `tools.task` (subagent dispatch).\n"
        "- Tool names are camelCase, arguments are snake_case.\n"
    )