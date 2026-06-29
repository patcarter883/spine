"""Agent builder for the project-level phase-verification agent.

One instance of this agent runs per roadmap phase (in parallel via the
Send API in project_verifier.py).  The agent:

1. Calls ``read_phase_evidence`` to load all evidence in one call.
2. Reviews each member's verification report and specification.
3. Checks for cross-member integration gaps.
4. Calls ``write_phase_verification`` with the structured verdict.

File-system read tools are available (no write access) so the agent can
spot-check implementation files when evidence from verification artifacts
is ambiguous.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.agents.factory import build_phase_agent
from spine.agents.project_verify_tools import build_project_verify_tools
from spine.agents.verify_subagent_tools import VerifyReadFileTool
from spine.models.enums import PhaseName


def build_project_verify_agent(
    *,
    project_id: str,
    phase_dict: dict,
    coverage_phase: dict,
    member_states: dict,
    workspace_root: str,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for one roadmap phase's integration verification.

    Args:
        project_id: The project slug.
        phase_dict: The phase spec dict (id, title, description, requirement_ids,
            member_work_ids).
        coverage_phase: Aggregator coverage dict for this phase (status,
            requirement_ids).
        member_states: Mapping of work_id → checkpoint state dict for each
            phase member.
        workspace_root: Absolute path to the project workspace root.
        config: LangGraph runtime config.

    Returns:
        A compiled Deep Agent ready for invocation.
    """
    phase_id = phase_dict.get("id", "unknown")
    phase_title = phase_dict.get("title", phase_id)

    pseudo_state: dict = {
        "work_id": f"project-verify-{project_id}-{phase_id}",
        "workspace_root": workspace_root,
        "artifacts": {},
        "work_type": "task",
    }

    system_prompt = (
        f"You are the PROJECT_VERIFY agent for roadmap phase '{phase_id}' "
        f"({phase_title}).\n\n"
        "## Your Task\n"
        "Determine whether this phase's requirements are truly satisfied by the "
        "combined implementation — not just whether individual work items passed "
        "their own narrow slice checks.  Look specifically for cross-member "
        "integration gaps that per-item verification cannot detect.\n\n"
        "## What You MUST Do\n"
        "1. Call ``read_phase_evidence`` FIRST — it loads member verification "
        "reports, member specifications, and files written in one call.\n"
        "2. For each phase requirement, assess whether it is satisfied by the "
        "combined work of all members.\n"
        "3. Identify integration gaps: places where members implement overlapping "
        "features differently, where member A's output is never consumed by "
        "member B, or where requirements are collectively incomplete.\n"
        "4. You MAY use ``read_file`` / ``glob`` / ``grep`` to spot-check "
        "implementation files when the verification artifacts are ambiguous.\n"
        "5. Call ``write_phase_verification`` with your structured verdict.\n\n"
        "## What You MUST NOT Do\n"
        "- Do NOT write to source files.  Only ``write_phase_verification`` "
        "is permitted as a write operation.\n"
        "- Do NOT rubber-stamp per-item verification.  The value of this pass "
        "is catching what per-item verify missed.\n\n"
        "## Verdict Scale\n"
        "- VERIFIED: All requirements satisfied, no significant integration gaps.\n"
        "- PARTIAL: Some requirements satisfied but gaps remain.\n"
        "- FAILED: Requirements not satisfied or critical integration gaps found.\n\n"
        "## Tool Surface\n"
        "- ``read_phase_evidence`` — load all evidence (call FIRST).\n"
        "- ``write_phase_verification`` — write verdict (call LAST).\n"
        "- ``read_file``, ``glob``, ``grep``, ``ls`` — read-only filesystem "
        "access for spot-checks."
    )

    tools = build_project_verify_tools(
        project_id=project_id,
        phase_dict=phase_dict,
        coverage_phase=coverage_phase,
        member_states=member_states,
        workspace_root=workspace_root,
    )
    # Spot-checks use the bounded, read-only read_file (per-file cap + global
    # wall + re-read de-dup) rather than the middleware's unbounded reader — the
    # same surface the slice-verifier got after trace 019f10bf. This is a
    # breadth survey, so the global wall is raised above the slice-verifier's.
    tools.append(
        VerifyReadFileTool(workspace_root=workspace_root, read_wall=40)
    )

    agent = build_phase_agent(
        state=pseudo_state,
        config=config,
        phase=PhaseName.PROJECT_VERIFY,
        system_prompt=system_prompt,
        extra_tools=tools,
        allowed_tools=["ls", "glob", "grep"],
    )

    return agent
