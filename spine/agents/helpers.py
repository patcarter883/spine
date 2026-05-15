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
    phase: str | None = None,
) -> str | BaseChatModel:
    """Resolve the LLM model identifier from config or SpineConfig.

    Supports per-phase and per-subagent model overrides.  When ``phase`` is
    provided, checks ``SpineConfig.providers.phases.<phase>.model`` before
    falling back to the default provider resolution.

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
        phase: Optional phase or phase/subagent path for model override
            resolution (e.g. ``"implement"`` or
            ``"implement/subagents/slice-implementer"``).

    Returns:
        A model string like ``"openrouter:z-ai/glm-4.5-air:free"`` or a
        pre-built ``BaseChatModel`` instance when extra config is needed.
    """
    model_spec = _model_spec_from_config(config, phase=phase)

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


def _model_spec_from_config(config: RunnableConfig | None, phase: str | None = None) -> str:
    """Extract the model spec string from config or SpineConfig.

    Checks ``config["configurable"]["model"]`` first, then delegates to
    ``SpineConfig.resolve_model(phase=phase)`` which handles per-phase
    overrides and the default provider resolution.

    Args:
        config: LangGraph runtime config.
        phase: Optional phase or phase/subagent path for override resolution.

    Returns:
        A model spec string like ``"openrouter:z-ai/glm-4.5-air:free"``.
    """
    if config and config.get("configurable", {}).get("model"):
        return config["configurable"]["model"]
    from spine.config import SpineConfig

    return SpineConfig.load().resolve_model(phase=phase)


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

    Sets a default ``request_timeout`` of 300 seconds (5 minutes) to
    prevent hung connections from blocking the workflow indefinitely.
    Provider config can override via ``providers.llm[].request_timeout``.

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

    # ── Resolve request_timeout ──────────────────────────────────────
    # Default: 300s (5 min).  Provider config can override this via
    # providers.llm[].request_timeout.  Without a timeout, hung
    # connections (e.g. OpenRouter dropping mid-stream) can block
    # the workflow for 30+ minutes waiting for OS-level TCP timeouts.
    # Note: ChatOpenRouter expects milliseconds, not seconds.
    timeout_ms = _resolve_timeout_from_config(default=300) * 1000

    # ── Resolve max_completion_tokens ────────────────────────────────
    # When max_completion_tokens is not set, reasoning models (e.g.
    # DeepSeek-v4-flash) can consume their entire output budget on
    # chain-of-thought tokens, leaving the visible content truncated
    # mid-generation.  Setting an explicit limit ensures the model
    # allocates enough output budget to produce complete artifacts.
    #
    # Provider config can override via providers.llm[].max_completion_tokens
    # or providers.llm[].max_tokens.  max_completion_tokens is preferred
    # (it includes reasoning tokens in the budget, giving the model full
    # control over allocation).
    provider_cfg = _active_provider_config() or {}
    max_completion_tokens = provider_cfg.get("max_completion_tokens")
    max_tokens = provider_cfg.get("max_tokens")

    model_kwargs: dict[str, Any] = {
        "model": model_name,
        "session_id": truncated_session_id,
        "request_timeout": timeout_ms,
        **profile_kwargs,
    }
    if max_completion_tokens is not None:
        model_kwargs["max_completion_tokens"] = int(max_completion_tokens)
    elif max_tokens is not None:
        # Fall back to max_tokens if max_completion_tokens isn't set
        model_kwargs["max_tokens"] = int(max_tokens)

    return ChatOpenRouter(**model_kwargs)


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
    for key in ("temperature", "max_tokens", "max_completion_tokens", "max_retries", "request_timeout"):
        if key in provider_cfg:
            kwargs[key] = provider_cfg[key]

    # ── Default request_timeout ───────────────────────────────────────
    # If not explicitly configured, default to 300s (5 min) to prevent
    # hung connections from blocking the workflow for 30+ minutes.
    if "request_timeout" not in kwargs:
        kwargs["request_timeout"] = 300

    # ── Enable stream_usage for token counting ───────────────────────
    # Without stream_usage=True, ChatOpenAI does not send
    # `stream_options: {"include_usage": true}` to the OpenAI-compatible
    # server.  The server then omits the final usage chunk, and
    # AIMessage.usage_metadata is None.  This breaks LangSmith token
    # tracing AND the SPINE token budget tracker.  All local inference
    # engines (vLLM, SGLang, hipfire) support this OpenAI API option.
    kwargs.setdefault("stream_usage", True)

    return ChatOpenAI(**kwargs)


def debug_enabled() -> bool:
    """Check if LLM debug logging is enabled via the SPINE_DEBUG_LLM env var."""
    return os.getenv("SPINE_DEBUG_LLM", "").strip().lower() in ("1", "true", "yes")


def _resolve_timeout_from_config(default: int = 300) -> int:
    """Resolve the request_timeout in seconds from provider config.

    Checks the active provider config for a ``request_timeout`` field.
    Falls back to ``default`` when not configured.  The return value is
    always in **seconds** — callers must convert to milliseconds when
    needed by the underlying client.

    Args:
        default: Default timeout in seconds when not configured.

    Returns:
        Timeout value in seconds.
    """
    provider_cfg = _active_provider_config()
    if provider_cfg and "request_timeout" in provider_cfg:
        try:
            return int(provider_cfg["request_timeout"])
        except (ValueError, TypeError):
            pass
    return default


def extract_response(result: dict[str, Any]) -> str:
    """Extract the text content from a Deep Agent's last message.

    For thinking/reasoning models (e.g. DeepSeek-v4-flash), the final
    message content may be chain-of-thought reasoning rather than
    structured output.  We detect this pattern and return an empty string
    to avoid polluting artifacts with leaked reasoning.

    Args:
        result: The agent result dict (has ``"messages"`` key).

    Returns:
        The content string of the final message, or empty string if the
        content appears to be leaked reasoning or is absent.
    """
    messages = result.get("messages", [])
    if messages:
        last = messages[-1]
        content = getattr(last, "content", str(last))
        if not content:
            return ""
        # ── Detect leaked thinking-model reasoning ─────────────────────
        # Thinking models sometimes put their chain-of-thought in the
        # content field instead of reasoning_content.  Common patterns:
        #   - "Now let me check..." / "Let me look at..." / "I should..."
        #   - "Good, I can see..." / "The problem is..."
        #   - Starts with a lowercase letter (structured artifacts start
        #     with a heading or title case)
        stripped = content.strip()
        if stripped and not stripped[0].isupper() and not stripped[0] in ("#", "*", "-", "|", "`", "[", '"'):
            # Looks like reasoning, not a structured artifact
            return ""
        return content
    return ""
