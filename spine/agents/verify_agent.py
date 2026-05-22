"""SPINE verify agent — Deep Agent for the VERIFY phase.

Same orchestrator pattern as the implement phase: dispatch one
``slice-verifier`` per feature slice, synthesize results into
``verification.md``. The orchestrator does not verify slices inline.

Tool restriction: orchestrator gets read-only filesystem tools plus
``write_file`` (for ``verification.md`` only). ``execute`` is excluded
— if a real test run is needed, that's the subagent's job. ``edit_file``
is excluded because verify must never mutate code.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.agents.artifacts import (
    build_artifact_prompt,
    build_current_phase_write_prompt,
    list_slice_files,
)
from spine.agents.factory import build_phase_agent
from spine.agents.subagents import build_phase_subagents
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState

# ── Tool allowlist ─────────────────────────────────────────────────────
# Verify orchestrator: read-only + write_file for verification.md.
# Test execution belongs to slice-verifier subagents (they have execute).
_VERIFY_ORCHESTRATOR_TOOLS: list[str] = [
    "ls",
    "read_file",
    "glob",
    "grep",
    "write_file",  # for verification.md only
]


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
    tasks_dir = f".spine/artifacts/{work_id}/tasks"
    verify_dir = f".spine/artifacts/{work_id}/verify"
    impl_dir = f".spine/artifacts/{work_id}/implement"

    # ── Slice inventory ───────────────────────────────────────────────
    slice_files = list_slice_files(workspace_root, work_id)
    slice_count = len(slice_files)

    if slice_count == 0:
        slice_inventory = (
            "⚠ No slice-*.md files found in tasks/ directory. "
            "Use `ls` + `glob` to locate slice files before proceeding."
        )
    else:
        slice_inventory = f"{slice_count} slice file(s) found in `{tasks_dir}/`:\n" + "\n".join(
            f"  - `{tasks_dir}/{name}`" for name in slice_files
        )

    system_prompt = (
        _build_orchestrator_prompt()
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
        subagents=_build_subagents(PhaseName.VERIFY, state, config),
        allowed_tools=_VERIFY_ORCHESTRATOR_TOOLS,
    )

    return agent


# ── Prompt builder ─────────────────────────────────────────────────────


def _build_orchestrator_prompt() -> str:
    """Build the verify orchestrator system prompt."""
    return (
        "You are the VERIFY phase orchestrator. You do NOT inspect source "
        "code yourself — you dispatch one `slice-verifier` subagent per "
        "feature slice and synthesize their verdicts into a single report.\n\n"
        "Your tools are restricted to read-only filesystem operations plus "
        "`write_file` (reserved for `verification.md`), `task`, and `eval`. "
        "You do NOT have `edit_file` or `execute` — verify never mutates "
        "code, and test execution belongs to subagents.\n\n"
        "## Expected tool errors (BY DESIGN — do not recover)\n"
        "If you attempt to call `edit_file` or `execute`, you will see a "
        "'tool not found' or 'unknown tool' error. This is **not a bug** — "
        "your toolset is deliberately filtered. Do not try alternative "
        "tools or attempt to work around the restriction. Dispatch a "
        "`slice-verifier` subagent instead — they can run tests.\n\n"
        "## Workflow (3 steps, ~5 turns total)\n\n"
        "### Step 1 — Read context\n"
        "In ONE turn, batch-read the codebase-map.md tasks file and implementation files.\n"
        "Refer to Step 1 & Step 2 guidelines preloaded in your user context pre-prompt.\n\n"
        "### Step 2 — Dispatch slice-verifier subagents in parallel\n"
        "Each `task` description MUST be fully self-contained. Embed:\n"
        "1. The full slice text (acceptance criteria, files to verify)\n"
        "2. Relevant excerpts from implementation.md (what the implementer "
        "reported for this slice)\n"
        "3. Codebase-map.md excerpts for files involved in this slice\n\n"
        "Dispatch pattern (do this inside one `eval` call):\n"
        "```js\n"
        "const sliceFiles = [/* slice filenames */];\n"
        "const dispatches = sliceFiles.map(async (name) => {\n"
        "  const slice = await tools.read_file({file_path: "
        "`tools.readFile` returns a string directly.\n`});\n"
        "  return tools.task({\n"
        '    subagent_type: "slice-verifier",  // ONLY valid type\n'
        "    description: `Verify slice: ${name}\\n\\n`\n"
        "      + `## Slice Definition\\n${slice.content}\\n\\n`\n"
        "      + `## Implementation Report Excerpt\\n${implExcerpt}\\n`,\n"
        "  });\n"
        "});\n"
        "const results = await Promise.allSettled(dispatches);\n"
        "globalThis.verifyResults = results;\n"
        "console.log(JSON.stringify(results.map(r => "
        "({status: r.status, verdict: r.value?.verdict}))));\n"
        "```\n\n"
        "### Step 3 — Synthesize verification.md\n"
        "Write verification.md report.\n\n"
        "The first line MUST be one of:\n"
        "- `VERIFIED` — every slice's subagent returned VERIFIED\n"
        "- `PASSED` — synonym for VERIFIED\n"
        "- `FAILED` — at least one slice was NOT_VERIFIED, or any subagent "
        "raised an exception\n\n"
        "Follow with:\n"
        "- One section per slice with the subagent's verdict and checklist\n"
        "- Aggregated gaps and recommendations\n"
        "- If FAILED: a clear list of which slices failed and why\n\n"
        "## Strict Rules\n"
        "- You MUST NOT call `edit_file`, `execute`, or `write_file` on "
        "anything other than `verification.md`. Your toolset is filtered.\n"
        "- You MUST dispatch one `slice-verifier` subagent per slice. "
        "Do not attempt to verify slices inline.\n"
        "- The ONLY valid `subagent_type` is `slice-verifier`. Do NOT "
        "request `general-purpose` — it does not exist.\n"
        "- Subagent dispatch MUST happen inside `eval` so subagents run "
        "in parallel via `Promise.allSettled`. Sequential conversation "
        "tool calls are not parallel.\n"
        "- `verification.md` is REQUIRED — without it the phase fails.\n\n"
        "## Eval Context Seed (first eval call)\n"
        "Access session-specific context properties via `globalThis.context` "
        "preloaded in your workspace environment on first turn (e.g., "
        "use `globalThis.context.work_id` or `globalThis.context.tasks_dir` inside eval).\n\n"
    )
