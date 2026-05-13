"""SPINE SPECIFY phase — generate a detailed spec from a prompt.

This phase takes a work description and produces a specification document.
It delegates to the specify Deep Agent which uses subagents for research
and documentation.

Outputs:
    - Artifacts: specification document
    - Prompt Request: if human input is needed during specification
"""

from __future__ import annotations

import logging
from typing import Any

from typing import Optional

from langchain_core.runnables import RunnableConfig

from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.specify_agent import build_specify_agent
from spine.workflow.registry import get_registry

logger = logging.getLogger(__name__)


def call_specify(state: WorkflowState, config: Optional[RunnableConfig] = None) -> dict[str, Any]:
    """Execute the SPECIFY phase.

    Delegates to the specify Deep Agent, which generates a specification
    document from the work description. If the phase is being reworked
    (retry > 0), includes prior critic feedback in the prompt.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config (contains thread_id, providers).

    Returns:
        Partial state update with artifacts and status.
    """
    description = state.get("description", "")
    work_id = state.get("work_id", "unknown")
    retry_count = state.get("retry_count", {}).get(PhaseName.SPECIFY.value, 0)
    feedback = state.get("feedback", [])

    logger.info(f"[{work_id}] SPECIFY phase starting (retry={retry_count})")

    try:
        agent = build_specify_agent(state, config)

        # Build the prompt, including feedback if reworking
        prompt = f"Create a detailed specification for the following work:\n\n{description}"
        if retry_count > 0 and feedback:
            feedback_text = "\n".join(
                f"- [{f.get('tier', 'unknown')}] {f.get('reason', '')}"
                for f in feedback
                if isinstance(f, dict)
            )
            prompt += f"\n\nPrevious review feedback (please address):\n{feedback_text}"

        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

        # Extract the specification from the agent's response
        spec_content = _extract_response(result)

        return {
            "artifacts": {PhaseName.SPECIFY.value: {"specification.md": spec_content}},
            "current_phase": PhaseName.SPECIFY.value,
            "status": "running",
            "prompt_request": None,
        }

    except Exception as e:
        logger.error(f"[{work_id}] SPECIFY phase failed: {e}", exc_info=True)
        return {
            "artifacts": {PhaseName.SPECIFY.value: {}},
            "status": "failed",
            "prompt_request": {
                "message": f"SPECIFY phase failed: {e}",
                "phase": PhaseName.SPECIFY.value,
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
    name=PhaseName.SPECIFY.value,
    call_fn=call_specify,
    build_agent_fn=build_specify_agent,
    description="Generate a detailed specification from a work description",
)
