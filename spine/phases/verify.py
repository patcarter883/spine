"""SPINE VERIFY phase — confirm implementation meets requirements.

The VERIFY phase is now dispatched via the Send API subgraph
(spine/workflow/subgraphs/verify_subgraph.py).  This module is
kept as a fallback for when the ``_SUBGRAPH_ENABLED`` feature flag
is turned off for VERIFY.

Phase node functions are async to avoid event-loop binding errors when
subagents inherit the parent checkpointer.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from typing import Optional

from langchain_core.runnables import RunnableConfig

from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.verify_agent import build_verify_agent
from spine.agents.helpers import extract_response
from spine.agents.retry import ainvoke_with_retry
from spine.agents.context import build_context
from spine.agents.artifacts import (
    materialize_artifacts,
    materialize_phase_artifacts,
    scan_artifact_dir,
    artifact_path,
)
from spine.workflow.registry import get_registry

logger = logging.getLogger(__name__)

_MAX_ARTIFACT_STATE_CHARS = 500

# Negated pass tokens ("NOT VERIFIED", "not passed") must never read as a pass.
# A bare `"VERIFIED" in text` substring test fails here because "NOT VERIFIED"
# contains "VERIFIED"; require a whole-word PASS/VERIFIED token and separately
# veto any negated or hedged form. The veto is deliberately broad — a false
# *pass* ships unverified work, so when in doubt we fail closed (finding #8).
# Covers: "not verified", "not fully/yet verified", "could not be verified",
# "cannot be verified", "never verified", "unverified", "not (yet) implemented",
# "incomplete".
_PASS_VERDICT_RE = re.compile(r"\b(?:verified|passed)\b", re.IGNORECASE)
_NEGATED_PASS_RE = re.compile(
    r"\b(?:not|never|cannot|can\s*not)\b[\s\w]{0,20}?"
    r"\b(?:verified|passed|implemented|complete[d]?)\b"
    r"|\bunverified\b|\bincomplete\b",
    re.IGNORECASE,
)


def _verdict_is_pass(verify_text: str) -> bool:
    """Return True only when the verify report states an affirmative pass.

    Guards against the substring trap where "NOT VERIFIED" / "not passed"
    would otherwise satisfy a naive ``"VERIFIED" in text`` check.
    """
    if not verify_text:
        return False
    if _NEGATED_PASS_RE.search(verify_text):
        return False
    return bool(_PASS_VERDICT_RE.search(verify_text))


async def call_verify(
    state: WorkflowState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """Execute the VERIFY phase (fallback path).

    When the Send API subgraph is enabled (default), this function is
    not called — the subgraph handles all verification dispatch.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        Partial state update with verification artifacts and final status.
    """
    work_id = state.get("work_id", "unknown")
    work_type = state.get("work_type", "")
    workspace_root = state.get("workspace_root", ".")

    logger.info(f"[{work_id}] VERIFY phase starting")

    try:
        agent = build_verify_agent(state, config)
        materialize_artifacts(state, workspace_root, work_id=work_id)

        verify_dir = artifact_path(work_id, PhaseName.VERIFY.value)
        impl_dir = artifact_path(work_id, PhaseName.IMPLEMENT.value)
        spec_path = artifact_path(work_id, PhaseName.SPECIFY.value)
        plan_path = artifact_path(work_id, PhaseName.PLAN.value)

        prompt = (
            "Verify that the implementation meets the requirements. "
            "Check that all feature slices are implemented correctly, "
            "the plan was followed, and the original task is complete.\n\n"
            "Prior artifacts are on disk — use `read_verify_context` to "
            "load structured slice definitions and implementation results "
            "in one call. Do NOT use `read_file`/`grep` to parse markdown.\n\n"
            "Inspect the actual code files on disk — the implementation "
            "summary may not reflect the actual state of the code.\n\n"
            f"- Specification: `{spec_path}/specification.md`\n"
            f"- Plan: `{plan_path}/plan.md`\n"
            f"- Implementation: `{impl_dir}/implementation.md`\n\n"
            "Dispatch a `slice-verifier` subagent per slice via `task` "
            "inside `eval`, then synthesize results with "
            f"`write_verification_report`. Write to `{verify_dir}/`."
        )

        ctx = build_context(state, PhaseName.VERIFY)

        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.VERIFY.value,
            work_id=work_id,
            work_type=work_type,
            context=ctx,
        )

        verify_content = extract_response(result)

        disk_artifacts = scan_artifact_dir(
            workspace_root,
            work_id,
            PhaseName.VERIFY.value,
            max_preview_chars=_MAX_ARTIFACT_STATE_CHARS,
        )

        if not disk_artifacts:
            if not verify_content or len(verify_content.strip()) < 20:
                verify_content = (
                    "Verification could not produce a meaningful report. "
                    "The agent returned insufficient output. "
                    "Manual review is required."
                )
            materialize_phase_artifacts(
                PhaseName.VERIFY.value,
                {"verification.md": verify_content},
                workspace_root,
                work_id=work_id,
            )
            disk_artifacts = {"verification.md": verify_content[:_MAX_ARTIFACT_STATE_CHARS]}

        verify_text = ""
        if disk_artifacts:
            verify_text = next(iter(disk_artifacts.values()), "")
        is_verified = _verdict_is_pass(verify_text)
        final_status = "completed" if is_verified else "needs_review"

        return {
            "artifacts": {PhaseName.VERIFY.value: disk_artifacts},
            "current_phase": PhaseName.VERIFY.value,
            "status": final_status,
            "prompt_request": None,
            "feedback": [
                {
                    "status": "passed" if is_verified else "needs_review",
                    "tier": "verify",
                    "reason": verify_text[:500],
                    "suggestions": [],
                }
            ],
        }

    except Exception as e:
        logger.error(f"[{work_id}] VERIFY phase failed: {e}", exc_info=True)
        return {
            "artifacts": {PhaseName.VERIFY.value: {}},
            "current_phase": PhaseName.VERIFY.value,
            "status": "needs_review",
            "prompt_request": {
                "message": f"VERIFY phase failed: {e}",
                "phase": PhaseName.VERIFY.value,
            },
        }


_registry = get_registry()
_registry.register(
    name=PhaseName.VERIFY.value,
    call_fn=call_verify,
    build_agent_fn=build_verify_agent,
    description="Verify implementation meets requirements",
)