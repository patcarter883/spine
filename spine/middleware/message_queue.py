"""Message queue injection middleware for Deep Agents agent loop.

Enables mid-run message injection: external code can push messages into
the agent's conversation at any time via a shared message queue. This is
useful for:
- Injecting updated context mid-execution
- Adding human feedback during long-running tasks
- Providing notifications about external events

The queue is stored in graph state as ``pending_messages``. Before each
model call, this middleware drains the queue and appends the messages.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware

logger = logging.getLogger(__name__)


class MessageQueueMiddleware(AgentMiddleware):
    """Mid-run message injection as DA-compatible middleware.

    Checks for pending messages in graph state before each model call.
    If found, drains the queue and appends them to the conversation.

    Usage::

        middleware = [MessageQueueMiddleware()]
        agent = create_deep_agent(model=model, middleware=middleware, ...)

        # To inject a message during execution:
        state["pending_messages"] = [{
            "role": "user",
            "content": "Additional context: the database schema has changed."
        }]
    """

    name = "MessageQueueMiddleware"

    def before_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
        """Drain pending messages before the model call."""
        pending = state.get("pending_messages", [])
        if not pending:
            return None

        logger.info("Injecting %d pending message(s) into conversation", len(pending))
        messages = list(state.get("messages", []))
        messages.extend(pending)

        return {
            "messages": messages,
            "pending_messages": [],  # Clear the queue
        }


__all__ = ["MessageQueueMiddleware"]
