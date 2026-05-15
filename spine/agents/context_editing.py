"""SPINE context editing middleware — trims old tool outputs.

DA's built-in SummarizationMiddleware triggers at a configurable token
threshold (default 80K for SPINE). Between triggers, tool results
accumulate in full. This middleware trims old tool results earlier,
keeping the conversation lean and reducing peak KV cache pressure.

Strategy: When tool result count exceeds `max_full_tool_results`, replace
old tool call results with a compact placeholder. This preserves the
conversation structure (the agent knows it read a file) but removes
the potentially large file content from context.

The offloaded conversation history (written by DA SummarizationMiddleware
to /conversation_history/{thread_id}.md) serves as swap space — the
agent can page back by reading that file if the placeholder strips
out a crucial detail.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ToolOutputTrimmer:
    """Trims old tool outputs from the conversation to keep context lean.

    Replaces old tool result content with a compact placeholder when
    the tool result count exceeds the threshold. Only trims tool results
    (ToolMessage), not human or AI messages.

    Design: treats context as L1 cache. Evicted content lives in the
    offloaded conversation history (swap) and can be paged back via
    read_file if needed.
    """

    def __init__(
        self,
        max_full_tool_results: int = 20,
        placeholder: str = "[evicted — recover from eval or re-read only if essential]",
    ) -> None:
        self.max_full_tool_results = max_full_tool_results
        self.placeholder = placeholder

    async def awrap_model_call(self, request, handler):
        """Trim old tool results before each model call."""
        messages = request.messages

        # Count tool results in the message list
        tool_result_indices = []
        for i, msg in enumerate(messages):
            if hasattr(msg, "type") and msg.type == "tool":
                tool_result_indices.append(i)

        # If within budget, pass through unchanged
        if len(tool_result_indices) <= self.max_full_tool_results:
            return await handler(request)

        # Trim old results — keep the last N in full
        trim_count = len(tool_result_indices) - self.max_full_tool_results
        trimmed_messages = list(messages)
        for idx in tool_result_indices[:trim_count]:
            msg = trimmed_messages[idx]
            content = self.placeholder
            if hasattr(msg, "content") and isinstance(msg.content, str):
                hint = msg.content[:100].split("\n")[0]
                if hint and len(hint) > 10:
                    content = f"[evicted: {hint}... — recover from eval or re-read only if essential]"
            try:
                trimmed_messages[idx] = msg.__class__(
                    content=content,
                    tool_call_id=msg.tool_call_id,
                    name=getattr(msg, "name", None),
                )
            except Exception:
                pass

        return await handler(request.override(messages=trimmed_messages))
