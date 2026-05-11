"""Deep Agents model provider for direct LangChain chat model integration.

Replaces OpenCodeAgentProvider for local models (vLLM, Ollama) where the
subprocess wrapper adds latency and protocol fragility. Uses LangChain's
init_chat_model() to construct a BaseChatModel that Deep Agents can consume
directly via create_deep_agent(model=...).

Configuration (spine.yaml)::

    providers:
      llm:
        - name: local-vllm
          type: deepagents-model
          config:
            model: "openai:Qwen/Qwen3-32B"
            base_url: "http://localhost:8000/v1"
            api_key: "dummy"
            temperature: 0.3
            max_tokens: 8192
"""

from __future__ import annotations

import logging
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

    def configure(self, config: dict[str, Any]) -> None:
        """Initialise the LangChain chat model from config.

        Expected keys (all optional except ``model``):
            model      – provider:model string (e.g. "openai:Qwen/Qwen3-32B")
            base_url   – OpenAI-compatible endpoint
            api_key    – API key (use "dummy" for local vLLM)
            temperature, max_tokens, top_p, reasoning – forwarded as-is
        """
        self._config = config.copy()
        model_str = config.get("model", "openai:gpt-4")
        base_url = config.get("base_url")
        api_key = config.get("api_key", "dummy")

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
        """Return the initialised LangChain chat model."""
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
