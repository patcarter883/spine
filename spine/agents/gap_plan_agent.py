"""SPINE gap plan agent — Deep Agent for the GAP_PLAN phase.

Produces a targeted gap remediation plan from verify feedback. Does NOT
re-explore the codebase — it relies on the codebase map and plan artifacts
already on disk from the original planning phase.

Uses purpose-built tools (read_verification_findings, write_structured_gap_plan)
with skip_filesystem_middleware=True to enforce dispatch-only behavior at the
tool level. The agent cannot access generic filesystem tools.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.agents.factory import build_phase_agent
from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.gap_plan_tools import build_gap_plan_tools


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
    workspace_root = state.get("workspace_root", ".")

    system_prompt = (
        "You are the GAP_PLAN phase. Your job is to read the verification "
        "report and create a targeted gap remediation plan.\n\n"
        "## What You MUST Do\n"
        "1. Call ``read_verification_findings`` FIRST — it loads the verification "
        "report, plan, codebase map, tasks, and implementation summary in one call.\n"
        "2. Identify every failing or partially-verified slice from the verification report.\n"
        "3. For each failed slice, analyze what needs to change by referencing "
        "the codebase map and original plan.\n"
        "4. Produce a structured gap_plan.md that tells the implement phase "
        "EXACTLY what to fix. For each issue, specify:\n"
        "   - The slice_id it relates to\n"
        "   - The file(s) to modify (from codebase map)\n"
        "   - The specific change needed\n"
        "   - The acceptance criteria to verify the fix\n\n"
        "## What You MUST NOT Do\n"
        "- Do NOT re-explore the codebase from scratch. Use the codebase map.\n"
        "- Do NOT produce a full new plan. This is a targeted gap-fix document.\n"
        "- Do NOT overwrite plan.md or any original plan artifacts.\n"
        "- Do NOT implement fixes yourself. Your only output is gap_plan.md + gap_plan.json.\n\n"
        "## Tools (ONLY these)\n"
        "- `read_verification_findings` — loads all verification inputs in ONE call. No arguments.\n"
        "- `write_structured_gap_plan` — writes gap_plan.md + gap_plan.json. Call this LAST.\n\n"
        "Use ONLY these tools. Cannot access generic filesystem tools."
    )

    # Build custom orchestrator tools
    orchestrator_tools = build_gap_plan_tools(
        workspace_root=workspace_root,
        work_id=work_id,
    )

    agent = build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.GAP_PLAN,
        system_prompt=system_prompt,
        extra_tools=orchestrator_tools,
        skip_filesystem_middleware=True,
    )

    return agent
