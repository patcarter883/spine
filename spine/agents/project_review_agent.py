"""Agent builder for the project-level adversarial review agent.

A single agent red-teams the completed project holistically:
- Integration gaps between members
- Requirements not truly implemented end-to-end
- Hidden design assumptions and contradictions
- Security and performance concerns not caught by per-item verify

The agent uses a plan→do split (directive + agent check) mirroring the
adversarial_subgraph pattern, coordinated by project_reviewer.py.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.agents.factory import build_phase_agent
from spine.agents.project_review_tools import build_project_review_tools
from spine.agents.verify_subagent_tools import VerifyReadFileTool
from spine.models.enums import PhaseName


def build_project_review_agent(
    *,
    project_id: str,
    spec_dict: dict,
    coverage: dict,
    member_states: dict,
    project_verification: dict | None,
    workspace_root: str,
    config: RunnableConfig | None = None,
) -> Any:
    """Build the Deep Agent for the project adversarial review.

    Args:
        project_id: The project slug.
        spec_dict: The ProjectSpec as a dict.
        coverage: Aggregator coverage output.
        member_states: Mapping of work_id → checkpoint state.
        project_verification: project_verification.json content, or None.
        workspace_root: Absolute path to workspace root.
        config: LangGraph runtime config.

    Returns:
        A compiled Deep Agent ready for invocation.
    """
    pseudo_state: dict = {
        "work_id": f"project-review-{project_id}",
        "workspace_root": workspace_root,
        "artifacts": {},
        "work_type": "critical_task",
    }

    system_prompt = (
        f"You are the PROJECT_REVIEW adversarial agent for project '{project_id}'.\n\n"
        "## Your Mission\n"
        "Red-team this completed project.  Assume the role of a hostile reviewer "
        "whose job is to find what the developers missed, what assumptions are "
        "wrong, and what will break in production.  You have seen every "
        "per-member verification report — your job is to find what they collectively "
        "MISSED, not to repeat what they said.\n\n"
        "## Attack Angles\n"
        "- **Integration gaps**: Member A implements X, member B implements Y, "
        "but they never actually integrate — examine shared interfaces, data formats, "
        "event contracts, and API boundaries.\n"
        "- **Requirement coverage gaps**: A requirement is marked 'satisfied' by "
        "exact string match, but the implementation does not actually fulfil the "
        "spirit of the requirement.  Look for superficial passes.\n"
        "- **Hidden assumptions**: The implementation assumes a runtime environment, "
        "data schema, or external service that may not exist or may behave differently.\n"
        "- **Scope creep / missing scope**: Members implemented more or less than "
        "the requirements specified.\n"
        "- **Cross-cutting concerns**: Security, observability, error handling, "
        "and data validation that no single member owns.\n"
        "- **File ownership conflicts**: Multiple members writing the same file "
        "may have clobbered each other's changes.\n\n"
        "## What You MUST Do\n"
        "1. Call ``read_project_evidence`` FIRST — it loads all evidence in one call.\n"
        "2. Use ``read_file`` / ``glob`` / ``grep`` to spot-check source files for "
        "critical claims. ``read_file`` is budgeted (~40 reads total, per-file "
        "capped, re-reads de-duplicated) — spend it on the highest-risk claims "
        "rather than reading broadly.\n"
        "3. Classify each finding by severity (CRITICAL/HIGH/MEDIUM/LOW) and "
        "provide specific evidence, not vague assertions.\n"
        "4. Assign an overall verdict: PASSED (ship-ready), NEEDS_REVISION "
        "(fixable without human judgment), or NEEDS_REVIEW (requires human "
        "judgment before proceeding).\n"
        "5. Call ``write_project_review`` with your verdict and findings.\n\n"
        "## What You MUST NOT Do\n"
        "- Do NOT write to source files.\n"
        "- Do NOT rubber-stamp passing verification results — that defeats the purpose.\n"
        "- Do NOT produce vague findings like 'may have issues'. Be specific: "
        "cite the file, the requirement, the interface mismatch.\n\n"
        "## Tools\n"
        "- ``read_project_evidence`` — load all evidence (call FIRST).\n"
        "- ``write_project_review`` — write verdict (call LAST).\n"
        "- ``read_file``, ``glob``, ``grep``, ``ls`` — read-only filesystem access."
    )

    tools = build_project_review_tools(
        project_id=project_id,
        spec_dict=spec_dict,
        coverage=coverage,
        member_states=member_states,
        project_verification=project_verification,
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
        phase=PhaseName.PROJECT_REVIEW,
        system_prompt=system_prompt,
        extra_tools=tools,
        allowed_tools=["ls", "glob", "grep"],
    )

    return agent
