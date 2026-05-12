"""Deep Agents model provider for direct LangChain chat model integration.

Replaces OpenCodeAgentProvider for local and cloud models where the
subprocess wrapper adds latency and protocol fragility. Uses LangChain's
init_chat_model() to construct a BaseChatModel that Deep Agents can consume
directly via create_deep_agent(model=...).

Configuration (spine.yaml)::

    providers:
      llm:
        # OpenRouter — api_key falls back to OPENROUTER_API_KEY env var
        - name: openrouter-da
          type: deepagents-model
          config:
            model: "openrouter:anthropic/claude-sonnet-4-5"
            # api_key: omitted → reads OPENROUTER_API_KEY from env
            temperature: 0.3
            max_tokens: 8192

        # Local vLLM / Ollama (openai-compatible endpoint)
        - name: local-vllm
          type: deepagents-model
          config:
            model: "openai:Qwen/Qwen3-32B"
            base_url: "http://localhost:8000/v1"
            api_key: "dummy"
            temperature: 0.3
            max_tokens: 8192

The ``model`` key uses LangChain's ``provider:model`` format:
  - ``openrouter:``   → langchain-openrouter (ChatOpenRouter)
  - ``openai:``       → langchain-openai   (ChatOpenAI)
  - ``anthropic:``    → langchain-anthropic (ChatAnthropic)
  - ``ollama:``       → langchain-ollama    (ChatOllama)

Any LangChain-supported provider prefix works; just ensure the
corresponding langchain-* integration package is installed.

Debug logging:
    Set ``SPINE_DEBUG_MODEL_IO=1`` to log all model input/output
    to ``.spine/debug/model_io/``. Each invoke() call produces a pair
    of JSON files (``_in.json`` / ``_out.json``) with timestamps,
    phase context, and full message content.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from .base import Provider, ProviderType

logger = logging.getLogger(__name__)


class DeepAgentsModelProvider(Provider):
    """Provider that creates LangChain chat models for direct DA integration.

    This provider does NOT implement AgentProvider.execute().  Instead, it
    exposes a ``chat_model`` property that returns a ``BaseChatModel`` suitable
    for passing to ``create_deep_agent(model=...)``.  This bypasses OpenCode
    entirely for local models, eliminating the 3-24 token problem caused by
    protocol mismatch between OpenCode's ACP and vLLM's OpenAI endpoint.

    Usage in phase functions::

        providers = _get_providers_from_config(config)
        da_model = providers.get("llm")  # DeepAgentsModelProvider instance
        agent = create_deep_agent(model=da_model.chat_model, ...)
    """

    provider_type = ProviderType.DEEPAGENTS_MODEL

    def __init__(self) -> None:
        self._config: dict[str, Any] = {}
        self._chat_model: BaseChatModel | None = None

    @property
    def name(self) -> str:
        return "deepagents-model"

    # Map of provider prefix → env var that holds the API key.
    # LangChain providers read these automatically when api_key is
    # not passed explicitly, but we resolve them here so that a
    # single ``api_key`` config key (or its absence) works across
    # all backends.
    _PROVIDER_ENV_KEYS: dict[str, str] = {
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "google_genai": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
        "mistralai": "MISTRAL_API_KEY",
        "xai": "XAI_API_KEY",
    }

    def configure(self, config: dict[str, Any]) -> None:
        """Initialise the LangChain chat model from config.

        Expected keys (all optional except ``model``):
            model      – provider:model string (e.g. "openrouter:anthropic/claude-sonnet-4-5")
            api_key    – API key; falls back to provider-specific env var
                         (OPENROUTER_API_KEY, OPENAI_API_KEY, etc.)
            base_url   – custom endpoint override
            temperature, max_tokens, top_p, reasoning – forwarded as-is
        """
        self._config = config.copy()
        model_str = config.get("model", "openai:gpt-4")
        base_url = config.get("base_url")

        # Resolve api_key: explicit config > env var for provider > not set.
        # When api_key is not set at all, LangChain's own init_chat_model
        # will read the provider's default env var automatically.
        # Treat empty string the same as missing — both should fall back
        # to the provider-specific env var.
        api_key = config.get("api_key") or None
        if api_key is None:
            provider_prefix = model_str.split(":")[0] if ":" in model_str else ""
            env_var = self._PROVIDER_ENV_KEYS.get(provider_prefix)
            if env_var:
                api_key = os.environ.get(env_var)

        kwargs: dict[str, Any] = {}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key

        # Forward common generation params
        for key in ("temperature", "max_tokens", "reasoning", "top_p"):
            if key in config:
                kwargs[key] = config[key]

        self._chat_model = init_chat_model(model_str, **kwargs)
        logger.info(
            "DeepAgentsModelProvider configured: model=%s base_url=%s",
            model_str, base_url,
        )

    @property
    def chat_model(self) -> BaseChatModel | None:
        """Return the initialised LangChain chat model.

        When ``SPINE_DEBUG_MODEL_IO=1``, the model is wrapped with
        :class:`ModelIOLogger` which logs every invoke() call to
        ``.spine/debug/model_io/``.
        """
        if self._chat_model is None:
            return None

        # Wrap with debug logger if enabled
        from ..debug.model_io import ModelIOLogger, is_debug_enabled
        if is_debug_enabled():
            return ModelIOLogger.wrap_if_enabled(self._chat_model)

        return self._chat_model

    def validate(self) -> bool:
        """Health-check: try a minimal invoke to confirm connectivity."""
        if self._chat_model is None:
            return False
        try:
            self._chat_model.invoke("ping")
            return True
        except Exception:
            return False

    @property
    def enabled(self) -> bool:
        return self._chat_model is not None

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text using the underlying chat model."""
        if self._chat_model is None:
            raise RuntimeError("DeepAgentsModelProvider not configured")
        response = await self._chat_model.ainvoke(prompt, **kwargs)
        return response.content if hasattr(response, "content") else str(response)


__all__ = ["DeepAgentsModelProvider"]
