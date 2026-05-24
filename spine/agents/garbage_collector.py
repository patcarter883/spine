"""Agentic garbage collection — commit_findings_and_clear_search tool."""

import logging

from langchain_core.tools import tool
from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage

logger = logging.getLogger(__name__)

EVICTION_ANCHOR = (
    "SUCCESS: Findings saved to the system scratchpad. "
    "SYSTEM ALERT: All previous search history has been successfully "
    "evicted from the context window. You now have a clean context. "
    "Only the scratchpad (below) and this message remain from prior turns. "
    "Use the scratchpad to pick up where you left off."
)


@tool
def commit_findings_and_clear_search(note: str, relevant_code: str) -> str:
    """Save critical findings to a persistent scratchpad and trigger context eviction.

    When you call this tool, the system will:
    1. Append your findings to the persistent scratchpad
    2. Delete ALL prior search history from the context window
    3. Preserve only the essential: scratchpad, this message, and new messages

    Use this tool when you've gathered enough information and your context
    window is getting full. After calling, you start fresh but with all
    your key findings preserved in the scratchpad.
    """
    return EVICTION_ANCHOR


def calculate_safe_eviction(messages: list) -> list[RemoveMessage]:
    """Return RemoveMessage objects that safely evict old search history.

    SAFETY GUARANTEE: Preserves the Boundary AIMessage and ALL sibling
    ToolMessages from the same turn, preventing parallel-tool parser crashes.

    ALGORITHM:
    1. Scan messages backward to find the last ToolMessage whose content
       matches EVICTION_ANCHOR.
    2. From that ToolMessage, scan backward to find the Boundary AIMessage
       (the AIMessage whose tool_calls list includes the commit_findings tool).
    3. Collect ALL ToolMessages whose tool_call_id appears in the Boundary's
       tool_calls list (these are siblings — must be preserved).
    4. Generate RemoveMessage for every AIMessage and ToolMessage that
       occurred BEFORE the Boundary AIMessage.
    5. Do NOT remove: the Boundary, sibling ToolMessages, any SystemMessage,
       or any HumanMessage.

    Returns:
        List of RemoveMessage objects. Empty list = no eviction needed.
    """
    anchor_idx: int | None = None
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, ToolMessage) and msg.content == EVICTION_ANCHOR:
            anchor_idx = i
            break

    if anchor_idx is None:
        return []

    boundary_idx: int | None = None
    for i in range(anchor_idx - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, AIMessage) or not msg.tool_calls:
            continue
        for tc in msg.tool_calls:
            if tc.get("name") == "commit_findings_and_clear_search":
                boundary_idx = i
                break
        if boundary_idx is not None:
            break

    if boundary_idx is None:
        return []

    preserved_ids: set[str] = set()
    for tc in messages[boundary_idx].tool_calls:
        preserved_ids.add(tc.get("id", ""))

    to_remove: list[RemoveMessage] = []
    for i in range(boundary_idx):
        msg = messages[i]
        if isinstance(msg, (AIMessage, ToolMessage)):
            if hasattr(msg, "id") and msg.id:
                to_remove.append(RemoveMessage(id=msg.id))

    return to_remove
