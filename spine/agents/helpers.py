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

    When the resolved model starts with ``openai:`` and the provider config
    includes ``base_url`` (i.e. a local/OpenAI-compatible server), returns a
    pre-built :class:`ChatOpenAI` with ``base_url`` and ``api_key`` wired
    in — without this, ``init_chat_model()`` creates a default that looks
    for ``OPENAI_API_KEY`` in the environment, causing a "missing
    credentials" error.

    Otherwise returns the model string for Deep Agents' built-in resolution.

    Args:
        config: LangGraph runtime config (may contain ``configurable.model``).
        session_id: Optional session identifier (typically the work_id) to
            pass to OpenRouter for request grouping. Ignored for non-OpenRouter
            providers.

    Returns:
        A model string like ``"openrouter:z-ai/glm-4.5-air:free"`` or a
        pre-built ``BaseChatModel`` instance when extra config is needed.
    """
    model_spec = _model_spec_from_config(config)

    # Only build a pre-built model when the provider needs extra kwargs
    # (base_url, api_key, session_id, etc.) that the string-based
    # init_chat_model path would silently drop.
    if session_id and model_spec.startswith("openrouter:"):
        return _build_openrouter_model(model_spec, session_id)

    # For local/OpenAI-compatible servers with custom base_url + api_key,
    # we must build a ChatOpenAI instance ourselves — otherwise
    # init_chat_model() creates one that falls back to OPENAI_API_KEY
    # env var (which isn't set for local servers), producing a
    # "missing credentials" error.
    #
    # Only do this when the model spec came from the active provider
    # (i.e. matches what's in config.yaml).  If the caller explicitly
    # set config["configurable"]["model"] to a different provider
    # (e.g. "openai:gpt-4o-mini" for cloud OpenAI), we must NOT
    # apply the local server's base_url/api_key to it.
    if model_spec.startswith("openai:"):
        provider_cfg = _active_provider_config()
        if (
            provider_cfg
            and provider_cfg.get("base_url")
            and provider_cfg.get("model") == model_spec
        ):
            return _build_local_model(model_spec, provider_cfg)

    return model_spec


def _model_spec_from_config(config: RunnableConfig | None) -> str:
    """Extract the model spec string from config or SpineConfig."""
    if config and config.get("configurable", {}).get("model"):
        return config["configurable"]["model"]
    from spine.config import SpineConfig

    return SpineConfig.load().resolve_model()


def _active_provider_config() -> dict[str, Any] | None:
    """Return the full config dict for the first enabled LLM provider.

    Delegates to :meth:`SpineConfig.resolve_active_provider` so that
    ``base_url``, ``api_key``, ``temperature``, and other provider fields
    are available for building pre-built model instances.
    """
    from spine.config import SpineConfig

    return SpineConfig.load().resolve_active_provider()


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


def _build_local_model(model_spec: str, provider_cfg: dict[str, Any]) -> BaseChatModel:
    """Build a ChatOpenAI instance for a local/OpenAI-compatible server.

    When the provider config includes ``base_url`` and ``api_key`` (e.g. a
    local vLLM instance), we construct a :class:`ChatOpenAI` directly so
    those fields are wired in.  Without this, ``init_chat_model("openai:…")``
    creates a default ``ChatOpenAI`` that looks for ``OPENAI_API_KEY`` in the
    environment — which doesn't exist for local servers, causing a
    "missing credentials" error.

    Args:
        model_spec: Full model spec like ``"openai:model"``.
        provider_cfg: The full provider dict from config (has ``base_url``,
            ``api_key``, ``temperature``, etc.).

    Returns:
        A configured ``ChatOpenAI`` instance pointed at the local server.
    """
    from langchain_openai import ChatOpenAI

    model_name = model_spec.removeprefix("openai:")

    kwargs: dict[str, Any] = {
        "model": model_name,
        "base_url": provider_cfg["base_url"],
    }

    # api_key is required by ChatOpenAI even for local servers that ignore it
    if api_key := provider_cfg.get("api_key"):
        kwargs["api_key"] = api_key

    # Pass through optional tuning fields if present
    for key in ("temperature", "max_tokens", "max_retries", "request_timeout"):
        if key in provider_cfg:
            kwargs[key] = provider_cfg[key]

    return ChatOpenAI(**kwargs)


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
