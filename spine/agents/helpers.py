"""SPINE agent helpers — shared utilities for all agent builders and phases.

Every agent builder (specify, plan, implement, verify, critic) and every
phase function had identical copies of ``_resolve_model``,
``_debug_enabled``, and ``_extract_response``.  Consolidate them here
to eliminate duplication and ensure consistent behavior.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.runnables import RunnableConfig


def resolve_model(config: RunnableConfig | None) -> str:
    """Resolve the LLM model identifier from config or SpineConfig.

    Checks the LangGraph runtime config first (set by the dispatcher),
    then falls back to the ``SpineConfig`` loaded from
    ``.spine/config.yaml``.

    Args:
        config: LangGraph runtime config (may contain ``configurable.model``).

    Returns:
        A model string like ``openrouter:z-ai/glm-4.5-air:free``.

    Raises:
        ValueError: If no model is configured and ``SPINE_MODEL`` is unset.
    """
    if config and config.get("configurable", {}).get("model"):
        return config["configurable"]["model"]
    from spine.config import SpineConfig

    return SpineConfig.load().resolve_model()


def debug_enabled() -> bool:
    """Check if LLM debug logging is enabled via the SPINE_DEBUG_LLM env var."""
    return os.getenv("SPINE_DEBUG_LLM", "").strip().lower() in ("1", "true", "yes")


def extract_response(result: dict[str, Any]) -> str:
    """Extract the text content from a Deep Agent's last message.

    Args:
        result: The agent result dict (has ``"messages"`` key).

    Returns:
        The content string of the final message, or empty string if none.
    """
    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        return getattr(last, "content", str(last))
    return ""
