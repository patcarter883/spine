"""SPINE gap plan agent — Deep Agent for the GAP_PLAN phase.

Produces a targeted gap remediation plan from verify feedback. Does NOT
re-explore the codebase — it relies on the codebase map and plan artifacts
already on disk from the original planning phase.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.agents.factory import build_phase_agent
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState

# ── Tool allowlist — read-only + write_file for gap_plan.md ────────────
_GAP_PLAN_ORCHESTRATOR_TOOLS: list[str] = [
    "ls",
    "read_file",
    "glob",
    "grep",
    "write_file",
]


def build_gap_plan_agent(
    state: WorkflowState,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the GAP_PLAN phase.

    The gap plan agent reads the verification report to understand what
    failed, references the original plan and codebase map for context,
    and produces a ``gap_plan.md`` with specific, targeted instructions
    for implement to fix ONLY the identified issues.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        A compiled Deep Agent ready for invocation.
    """
    work_id = state.get("work_id", "")

    from spine.agents.artifacts import artifact_path

    verify_path = artifact_path(work_id, PhaseName.VERIFY.value)
    plan_path = artifact_path(work_id, PhaseName.PLAN.value)
    tasks_path = artifact_path(work_id, PhaseName.TASKS.value)
    impl_path = artifact_path(work_id, PhaseName.IMPLEMENT.value)
    gap_plan_path = artifact_path(work_id, PhaseName.GAP_PLAN.value)

    system_prompt = (
        "You are the GAP_PLAN phase. Your job is to read the verification "
        "report and create a targeted gap remediation plan.\n\n"
        "## Context Available on Disk\n"
        f"- Verification report: `{verify_path}/verification.md`\n"
        f"- Original plan: `{plan_path}/plan.md`\n"
        f"- Codebase map: `{tasks_path}/codebase-map.md`\n"
        f"- Tasks/slices: `{tasks_path}/tasks.md`\n"
        f"- Implementation summary: `{impl_path}/implementation.md`\n\n"
        "## What You MUST Do\n"
        "1. Read the verification report first — identify every failing slice "
        "and the specific issues found.\n"
        "2. Reference the codebase map for file paths and conventions — do NOT "
        "re-explore the codebase with glob/grep/ls unless absolutely necessary.\n"
        "3. Read the original plan for context on what was supposed to be built.\n"
        "4. Produce a concise `gap_plan.md` that tells the implement phase "
        "EXACTLY what to fix. For each issue, specify:\n"
        "   - The file(s) to modify (from codebase map)\n"
        "   - The specific change needed\n"
        "   - The acceptance criteria to verify the fix\n\n"
        "## What You MUST NOT Do\n"
        "- Do NOT re-explore the codebase from scratch. Use the codebase map.\n"
        "- Do NOT produce a full new plan. This is a targeted gap-fix document.\n"
        "- Do NOT overwrite plan.md or any original plan artifacts.\n"
        "- Do NOT implement fixes yourself. Your only output is gap_plan.md.\n\n"
        "## Output\n"
        f"Write `gap_plan.md` to `{gap_plan_path}/gap_plan.md` using `write_file`.\n"
        "Start the file with `# Gap Remediation Plan` on the first line.\n"
        "This file is REQUIRED — without it the phase is treated as failed."
    )

    agent = build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.GAP_PLAN,
        system_prompt=system_prompt,
        subagents=None,
        allowed_tools=_GAP_PLAN_ORCHESTRATOR_TOOLS,
    )

    return agent
