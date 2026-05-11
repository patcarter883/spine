"""Critic gate middleware for Deep Agents agent loop.

SPINE's critic gate runs at two levels:
1. Agent loop level (this middleware): catches the agent's plan output,
   runs critic review, and either approves or injects revision feedback.
2. State machine level (should_continue): enforces the invariant that
   EXECUTION never starts without an approved plan.

This middleware enables fast iterative refinement within a single planning
invocation, while the state machine provides the structural guarantee.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware

logger = logging.getLogger(__name__)


class CriticGateMiddleware(AgentMiddleware):
    """SPINE critic gate as DA-compatible middleware.

    Inspects agent output after model calls during the PLANNING phase.
    If the agent signals plan completion, runs critic review. Returns
    a state update to inject revision feedback if the plan is rejected.

    Usage::

        middleware = [CriticGateMiddleware(llm_provider=llm)]
        agent = create_deep_agent(model=model, middleware=middleware, ...)
    """

    name = "CriticGateMiddleware"

    def __init__(self, llm_provider: Any = None) -> None:
        self._llm = llm_provider

    def after_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        """Inspect model output after each call during PLANNING phase.

        The DA AgentMiddleware.after_model hook receives (state, runtime).
        The model's response is already appended to state["messages"] before
        this hook fires, so we inspect the last message in state.

        If the agent signals plan completion (PLAN_COMPLETE marker),
        runs critic review. If the critic rejects, injects revision
        feedback as a new user message.
        """
        # Only run critic during PLANNING phase
        spine_phase = state.get("spine_phase", "")
        if spine_phase != "PLANNING":
            return None

        # Check if the agent has produced a plan
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        if not last_msg:
            return None

        content = ""
        if hasattr(last_msg, "content"):
            content = last_msg.content or ""
        elif isinstance(last_msg, dict):
            content = last_msg.get("content", "")

        if "PLAN_COMPLETE" not in content:
            return None

        # Run critic review
        critic_result = self._run_critic(content)
        logger.info("Critic gate result: %s", critic_result)

        if critic_result == "APPROVED":
            # Store result in state for state machine to read
            return {"critic_gate_result": "APPROVED"}

        # Inject critic feedback for revision
        feedback_msg = {
            "role": "user",
            "content": f"Critic feedback: {critic_result}. Please revise the plan. "
                       f"Address the specific issues raised and resubmit with PLAN_COMPLETE.",
        }
        new_messages = list(messages) + [feedback_msg]
        return {"messages": new_messages, "critic_gate_result": critic_result}

    def _run_critic(self, plan_text: str) -> str:
        """Run critic review on the plan.

        Uses the LLM provider if available, otherwise does a basic
        heuristic check.
        """
        if self._llm is not None and hasattr(self._llm, "chat_model"):
            try:
                model = self._llm.chat_model
                from langchain_core.messages import HumanMessage
                critic_prompt = (
                    "You are a software architecture critic. Review this plan "
                    "for correctness, completeness, and feasibility.\n\n"
                    f"PLAN:\n{plan_text}\n\n"
                    "Respond with exactly one word: APPROVED, NEEDS_REVISION, "
                    "or REJECTED. If not APPROVED, explain what needs to change."
                )
                result = model.invoke([HumanMessage(content=critic_prompt)])
                content = result.content.strip()
                # Extract the verdict word
                for verdict in ("APPROVED", "NEEDS_REVISION", "REJECTED"):
                    if verdict in content.upper():
                        return verdict
                return "NEEDS_REVISION"
            except Exception as e:
                logger.warning("Critic LLM invocation failed: %s", e)
                return "NEEDS_REVISION"

        # Heuristic fallback: approve if plan has structure
        if len(plan_text) > 100 and "slice" in plan_text.lower():
            return "APPROVED"
        return "NEEDS_REVISION"


__all__ = ["CriticGateMiddleware"]
