"""SPINE verify agent — Deep Agent for the VERIFY phase.

Uses the shared :func:`build_phase_agent` factory with summarization
middleware enabled (VERIFY can be long-running across many slices).
Structured gather→verify→report workflow with parallel subagent dispatch.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.factory import build_phase_agent
from spine.agents.artifacts import build_artifact_prompt, build_current_phase_write_prompt
from spine.agents.subagents import build_phase_subagents


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

    Creates a deep agent configured for verification with summarization
    middleware for long-running multi-slice verification.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        A compiled Deep Agent ready for invocation.
    """
    work_id = state.get("work_id", "")
    tasks_path = f".spine/artifacts/{work_id}/tasks"

    system_prompt = (
        "You are a verification engineer. Review the implementation "
        "against the specification, plan, and feature slices.\n\n"
        "Your filesystem is rooted at the project workspace. "
        "Use relative paths (e.g. `src/main.py`, `.spine/artifacts/...`).\n\n"
        "## Workflow (follow this order)\n\n"
        "### Phase 1: Gather (1-2 turns)\n"
        "Batch-read ALL relevant artifacts and source files in ONE response:\n"
        "- Read tasks.md and all slice files\n"
        f"- Read the codebase map (if available): `{tasks_path}/codebase-map.md`\n"
        "- Read the implementation summary\n"
        "- Read the actual source files that were modified\n\n"
        "### Phase 2: Verify (1-2 turns) — MANDATORY PARALLEL\n"
        "When there are ≥2 slices, you MUST dispatch one slice-verifier "
        "subagent per slice via `Promise.allSettled(tools.task(...))` from "
        "eval. Do NOT verify all slices inline — parallel verification is "
        "required to keep context lean. Each subagent gets a fresh, small "
        "context instead of bloating your conversation.\n"
        "When there is exactly 1 slice: verify directly using read_file and "
        "execute.\n\n"
        "### Phase 3: Report (1 turn)\n"
        "Synthesize findings into a verification report:\n"
        "- VERIFIED or NOT VERIFIED status\n"
        "- Checklist of each feature slice and its status\n"
        "- Any gaps or issues found\n"
        "- Write verification.md to disk\n\n"
        "## Subagent Dispatch — MANDATORY FOR ≥2 SLICES\n"
        "Failure to dispatch slice-verifier subagents when there are ≥2 slices "
        "violates the workflow contract. The slice-verifier subagents have "
        "isolated contexts and return compact results. Verifying all slices "
        "inline bloats your conversation and wastes tokens.\n\n"
        "## Rules\n"
        "- Batch reads: never read one file at a time\n"
        "- Use eval for parallel subagent dispatch\n"
        "- Inspect actual code, not just the implementation summary\n"
        "- Run tests — do not assume they pass\n\n"
        "When the interpreter is available, seed it with context on your first turn:\n"
        "```js\n"
        + f'globalThis.context = {{"work_id": "{work_id}", "phase": "verify", "artifact_dir": ".spine/artifacts/{work_id}/verify"}};\n'
        + "```\n"
        + build_current_phase_write_prompt(
            work_id, PhaseName.VERIFY.value, expected_files=["verification.md"]
        )
        + build_artifact_prompt(state.get("artifacts", {}), PhaseName.VERIFY.value, work_id=work_id)
    )

    agent = build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.VERIFY,
        system_prompt=system_prompt,
        add_summarization=True,  # VERIFY can be long-running
        subagents=_build_subagents(PhaseName.VERIFY, state, config),
    )

    return agent
