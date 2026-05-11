"""Step limit notification middleware for Deep Agents agent loop.

Provides a configurable step limit that runs inside the DA agent loop.
When the model call count exceeds the limit, injects a notification message
into the conversation so the agent can wrap up before hitting the
LangGraph recursion limit.

This replaces the inline lambda hooks in the old state_machine.py.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware

logger = logging.getLogger(__name__)


class StepLimitMiddleware(AgentMiddleware):
    """Step limit notification as DA-compatible middleware.

    Tracks model call count in graph state. When the count exceeds the
    configured limit, injects a user message asking the agent to wrap up.

    Usage::

        middleware = [StepLimitMiddleware(max_steps=50)]
        agent = create_deep_agent(model=model, middleware=middleware, ...)
    """

    name = "StepLimitMiddleware"

    def __init__(self, max_steps: int = 50) -> None:
        self._max_steps = max_steps

    def after_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        """Check step count after each model call. Notify if near limit.

        The DA AgentMiddleware.after_model hook receives (state, runtime).
        We increment our counter and inject wrap-up messages when needed.
        """
        step_count = state.get("model_call_count", 0) + 1
        updates: dict[str, Any] = {"model_call_count": step_count}

        # Warn at 80% of limit
        if step_count == int(self._max_steps * 0.8):
            logger.warning("Step limit approaching: %d/%d", step_count, self._max_steps)
            messages = list(state.get("messages", []))
            messages.append({
                "role": "user",
                "content": (
                    f"NOTIFICATION: You have used {step_count} of {self._max_steps} "
                    f"allowed steps. Please wrap up your current work and provide "
                    f"a summary of what you've accomplished."
                ),
            })
            updates["messages"] = messages

        # Hard limit — tell agent to stop immediately
        elif step_count >= self._max_steps:
            logger.warning("Step limit reached: %d/%d", step_count, self._max_steps)
            messages = list(state.get("messages", []))
            messages.append({
                "role": "user",
                "content": (
                    f"STOP: You have reached the maximum of {self._max_steps} steps. "
                    f"Provide your final output immediately — do not make any more tool calls."
                ),
            })
            updates["messages"] = messages

        return updates


__all__ = ["StepLimitMiddleware"]
