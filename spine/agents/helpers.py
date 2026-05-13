"""SPINE agent helpers — shared utilities for all agent builders and phases.

Every agent builder (specify, plan, implement, verify, critic) and every
phase function had identical copies of ``_resolve_model``,
``_debug_enabled``, and ``_extract_response``.  Consolidate them here
to eliminate duplication and ensure consistent behavior.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableConfig


def resolve_model(
    config: RunnableConfig | None,
    session_id: str | None = None,
) -> str | BaseChatModel:
    """Resolve the LLM model identifier from config or SpineConfig.

    When the resolved model is an OpenRouter model and a ``session_id`` is
    provided, returns a pre-built :class:`ChatOpenRouter` instance with
    ``session_id`` set — this lets OpenRouter group all requests for a work
    item into a single session on the dashboard.

    When the model is not OpenRouter, or when ``session_id`` is ``None``,
    returns the model string for Deep Agents' built-in resolution.

    Args:
        config: LangGraph runtime config (may contain ``configurable.model``).
        session_id: Optional session identifier (typically the work_id) to
            pass to OpenRouter for request grouping. Ignored for non-OpenRouter
            providers.

    Returns:
        A model string like ``"openrouter:z-ai/glm-4.5-air:free"`` or a
        pre-built ``BaseChatModel`` instance when session tracking is needed.
    """
    model_spec = _model_spec_from_config(config)

    # Only build a ChatOpenRouter instance if both conditions are met:
    # 1. The model spec points to OpenRouter
    # 2. A session_id was provided (work_id from the workflow)
    if session_id and model_spec.startswith("openrouter:"):
        return _build_openrouter_model(model_spec, session_id)

    return model_spec


def _model_spec_from_config(config: RunnableConfig | None) -> str:
    """Extract the model spec string from config or SpineConfig."""
    if config and config.get("configurable", {}).get("model"):
        return config["configurable"]["model"]
    from spine.config import SpineConfig

    return SpineConfig.load().resolve_model()


def _build_openrouter_model(model_spec: str, session_id: str) -> BaseChatModel:
    """Build a ChatOpenRouter instance with session_id set.

    Applies the DA ProviderProfile for OpenRouter (app_url, app_title,
    openrouter_provider defaults) before constructing the model, so we
    don't lose the attribution headers and Azure-ignore rule that the
    string-based ``init_chat_model`` path would normally provide.

    Args:
        model_spec: Full model spec like ``"openrouter:z-ai/glm-4.5-air:free"``.
        session_id: Work item ID for OpenRouter request grouping.

    Returns:
        A configured ``ChatOpenRouter`` instance.
    """
    from deepagents.profiles.provider import apply_provider_profile

    from langchain_openrouter import ChatOpenRouter

    # Strip the "openrouter:" prefix to get the raw model name
    model_name = model_spec.removeprefix("openrouter:")

    # OpenRouter limits session_id to 128 characters
    truncated_session_id = session_id[:128]

    # Apply DA ProviderProfile kwargs (app_url, app_title,
    # openrouter_provider, etc.) so we don't lose defaults that the
    # string-based init_chat_model path would normally inject.
    profile_kwargs = apply_provider_profile(model_spec)

    return ChatOpenRouter(
        model=model_name,
        session_id=truncated_session_id,
        **profile_kwargs,
    )


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
