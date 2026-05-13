"""SPINE VERIFY phase — confirm implementation meets requirements.

The verify Deep Agent reviews the implementation against the specification,
plan, and tasks. It confirms that feature slices have been correctly
implemented and the original requirements are satisfied.

Outputs:
    - Artifacts: verification report
    - Workflow Feedback: PASSED / NOT VERIFIED
    - Prompt Request: if human input is needed during verification
"""

from __future__ import annotations

import logging
from typing import Any

from typing import Optional

from langchain_core.runnables import RunnableConfig

from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.verify_agent import build_verify_agent
from spine.workflow.registry import get_registry

logger = logging.getLogger(__name__)


def call_verify(state: WorkflowState, config: Optional[RunnableConfig] = None) -> dict[str, Any]:
    """Execute the VERIFY phase.

    Delegates to the verify Deep Agent, which reviews all artifacts
    against the original requirements and produces a verification report.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        Partial state update with verification artifacts and final status.
    """
    description = state.get("description", "")
    work_id = state.get("work_id", "unknown")
    artifacts = state.get("artifacts", {})
    spec = artifacts.get(PhaseName.SPECIFY.value, {}).get("specification.md", "")
    plan = artifacts.get(PhaseName.PLAN.value, {}).get("plan.md", "")
    tasks_doc = artifacts.get(PhaseName.TASKS.value, {}).get("tasks.md", "")
    impl = artifacts.get(PhaseName.IMPLEMENT.value, {}).get("implementation.md", "")

    logger.info(f"[{work_id}] VERIFY phase starting")

    try:
        agent = build_verify_agent(state, config)

        prompt = (
            f"Verify that the implementation meets the requirements. "
            f"Check that all feature slices are implemented correctly, "
            f"the plan was followed, and the original task is complete.\n\n"
            f"## Original Requirements\n{description}\n\n"
        )
        if spec:
            prompt += f"## Specification\n{spec}\n\n"
        if plan:
            prompt += f"## Plan\n{plan}\n\n"
        if tasks_doc:
            prompt += f"## Feature Slices\n{tasks_doc}\n\n"
        if impl:
            prompt += f"## Implementation\n{impl}\n\n"

        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

        verify_content = _extract_response(result)

        # Provide a fallback if the agent returned nothing useful
        if not verify_content or len(verify_content.strip()) < 20:
            verify_content = (
                "Verification could not produce a meaningful report. "
                "The agent returned insufficient output. "
                "Manual review is required."
            )

        # Determine final status from verification
        is_verified = "VERIFIED" in verify_content.upper() or "PASSED" in verify_content.upper()
        final_status = "completed" if is_verified else "needs_review"

        return {
            "artifacts": {PhaseName.VERIFY.value: {"verification.md": verify_content}},
            "current_phase": PhaseName.VERIFY.value,
            "status": final_status,
            "prompt_request": None,
            "feedback": [
                {
                    "status": "passed" if is_verified else "needs_review",
                    "tier": "verify",
                    "reason": verify_content[:500],
                    "suggestions": [],
                }
            ],
        }

    except Exception as e:
        logger.error(f"[{work_id}] VERIFY phase failed: {e}", exc_info=True)
        return {
            "artifacts": {PhaseName.VERIFY.value: {}},
            "status": "failed",
            "prompt_request": {
                "message": f"VERIFY phase failed: {e}",
                "phase": PhaseName.VERIFY.value,
            },
        }


def _extract_response(result: Any) -> str:
    """Extract the text content from the agent's last message."""
    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        return getattr(last, "content", str(last))
    return ""


# ── Self-register on import ──
_registry = get_registry()
_registry.register(
    name=PhaseName.VERIFY.value,
    call_fn=call_verify,
    build_agent_fn=build_verify_agent,
    description="Verify implementation meets requirements",
)
