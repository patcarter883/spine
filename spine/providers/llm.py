"""LLM Provider implementations with graceful error handling and validation."""

import asyncio
import functools
import logging
import socket
from abc import abstractmethod
from typing import Any, AsyncIterator, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from .base import Provider, ProviderType

logger = logging.getLogger(__name__)

# Default timeout for LLM HTTP calls (connect, read) in seconds
DEFAULT_TIMEOUT = (10, 30)  # (connect_timeout, read_timeout)


def _add_timeout(kwargs: dict[str, Any], timeout: tuple[int, int] | int | None = None) -> dict[str, Any]:
    """Merge timeout into kwargs for HTTP-based LLM calls.

    Only adds the 'timeout' key if it is not already present and
    the underlying client supports it (openai >= 1.0).
    """
    if "timeout" in kwargs:
        return kwargs
    kwargs = dict(kwargs)
    kwargs.setdefault("timeout", timeout or DEFAULT_TIMEOUT)
    return kwargs


def _run_with_socket_timeout(fn, timeout: tuple[int, int] | int | None = None):
    """Execute *fn* with a temporary socket-level timeout so that
    synchronous HTTP clients that do not honour a *request* timeout
    still bail out after ~30 s by default.
    """
    timeout_val = timeout or DEFAULT_TIMEOUT
    # If a tuple was passed, use the read portion for socket timeout
    if isinstance(timeout_val, tuple):
        socket_timeout = timeout_val[1]
    else:
        socket_timeout = timeout_val

    old = socket.getdefaulttimeout()
    socket.setdefaulttimeout(socket_timeout)
    try:
        return fn()
    finally:
        socket.setdefaulttimeout(old)


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    content: str
    usage: dict[str, int]
    finish_reason: str
    model: str
    request_id: str


class LLMProvider(Provider):
    """Base class for LLM providers."""
    provider_type = ProviderType.LLM

    @abstractmethod
    def generate_sync(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt (sync implementation)."""
        pass

    @abstractmethod
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream text generation."""
        pass

    def generate(self, prompt: str, timeout: float = 60.0, **kwargs) -> str:
        """Generate text from prompt (sync - calls generate_sync with optional timeout)."""
        import concurrent.futures
        from functools import partial

        if timeout is None or timeout <= 0:
            return self.generate_sync(prompt, **kwargs)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            try:
                future = executor.submit(self.generate_sync, prompt, **kwargs)
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                logger.warning(f"LLM call timed out after {timeout}s for prompt: {prompt[:80]}...")
                raise TimeoutError(f"LLM call timed out after {timeout} seconds")

    async def generate_async(self, prompt: str, timeout: float = 60.0, **kwargs) -> str:
        """Async generate - runs sync in thread pool by default with timeout."""
        loop = asyncio.get_event_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self.generate_sync(prompt, **kwargs)),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Async LLM call timed out after {timeout}s for prompt: {prompt[:80]}...")
            raise TimeoutError(f"LLM call timed out after {timeout} seconds")

    async def generate_with_confidence(
        self, prompt: str, **kwargs
    ) -> Tuple[LLMResponse, float]:
        """Return response with self-reported confidence. Override in subclasses."""
        response = LLMResponse(
            content=self.generate_sync(prompt, **kwargs),
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            finish_reason="stop",
            model="unknown",
            request_id=""
        )

        confidence_prompt = f"""Rate your confidence (0.0-1.0) in this answer:
Question: {prompt}
Answer: {response.content}

Return only a number:"""

        conf_response = self.generate_sync(confidence_prompt, max_tokens=10)
        try:
            confidence = float(conf_response.strip())
        except ValueError:
            confidence = 0.8

        return (response, confidence)


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, api_key: str = "", model: str = "gpt-4"):
        self._api_key = api_key
        self._model = model
        self._client = None

    def configure(self, config: dict[str, Any]) -> None:
        api_key = config.get("api_key", self._api_key)
        model = config.get("model", self._model)

        if not api_key:
            logger.warning(
                "OpenAIProvider: No API key configured. "
                "Provider will fall back to stub execution. "
                "Set 'providers.llm.*.config.api_key' in your config."
            )
            self._client = None
            return

        try:
            import openai
            self._client = openai.OpenAI(api_key=api_key)
            self._model = model
        except Exception as e:
            logger.warning(
                f"OpenAIProvider: Failed to create client: {e}. "
                "Provider will fall back to stub execution."
            )
            self._client = None

    def validate(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.models.list()
            return True
        except Exception:
            return False

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def generate_sync(self, prompt: str, **kwargs) -> str:
        if self._client is None:
            raise RuntimeError(
                "OpenAIProvider: No client available. "
                "Check API key configuration or fall back to stub mode."
            )
        kwargs = _add_timeout(kwargs)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content

    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        if self._client is None:
            raise RuntimeError(
                "OpenAIProvider: No client available. "
                "Check API key configuration or fall back to stub mode."
            )
        kwargs = _add_timeout(kwargs)
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""

    def __init__(self, model: str = "qwen3:32b", base_url: str = "http://localhost:11434"):
        self._model = model
        self._base_url = base_url
        self._client = None

    def configure(self, config: dict[str, Any]) -> None:
        base_url = config.get("base_url", self._base_url)
        model = config.get("model", self._model)

        try:
            import ollama
            self._client = ollama.Client(host=base_url)
            self._model = model
        except Exception as e:
            logger.warning(
                f"OllamaProvider: Failed to create client at {base_url}: {e}. "
                "Provider will fall back to stub execution. "
                "Ensure Ollama is running and accessible."
            )
            self._client = None

    def validate(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.list()
            return True
        except Exception:
            return False

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def generate_sync(self, prompt: str, **kwargs) -> str:
        if self._client is None:
            raise RuntimeError(
                "OllamaProvider: No client available. "
                "Check Ollama server is running or fall back to stub mode."
            )
        kwargs = _add_timeout(kwargs)
        response = self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response["message"]["content"]

    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        if self._client is None:
            raise RuntimeError(
                "OllamaProvider: No client available. "
                "Check Ollama server is running or fall back to stub mode."
            )
        kwargs = _add_timeout(kwargs)
        stream = self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs
        )
        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                yield chunk["message"]["content"]


class OpenRouterProvider(LLMProvider):
    """OpenRouter LLM provider using OpenAI-compatible API."""

    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str = "",
        model: str = "openai/gpt-4",
        base_url: str = DEFAULT_BASE_URL
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._client = None

        # Validate api_key early - OpenAI SDK doesn't validate on construction
        if not api_key:
            logger.warning(
                "OpenRouterProvider: No API key provided at construction. "
                "Provider will not be enabled. "
                "Set 'providers.llm.*.config.api_key' in your config."
            )
            return

        try:
            import openai
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url
            )
        except Exception as e:
            logger.warning(
                f"OpenRouterProvider: Failed to create client: {e}. "
                "Provider will fall back to stub execution."
            )
            self._client = None

    def configure(self, config: dict[str, Any]) -> None:
        api_key = config.get("api_key", self._api_key)
        base_url = config.get("base_url", self._base_url)
        model = config.get("model", self._model)

        if not api_key:
            logger.warning(
                "OpenRouterProvider: No API key in config. "
                "Provider will fall back to stub execution."
            )
            self._client = None
            return

        try:
            import openai
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            self._model = model
        except Exception as e:
            logger.warning(
                f"OpenRouterProvider: Failed to create client: {e}. "
                "Provider will fall back to stub execution."
            )
            self._client = None

    def validate(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.models.list()
            return True
        except Exception:
            return False

    @property
    def name(self) -> str:
        return f"openrouter:{self._model}"

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def generate_sync(self, prompt: str, **kwargs) -> str:
        if self._client is None:
            raise RuntimeError(
                "OpenRouterProvider: No client available. "
                "Check API key configuration or fall back to stub mode."
            )
        kwargs = _add_timeout(kwargs)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content

    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        if self._client is None:
            raise RuntimeError(
                "OpenRouterProvider: No client available. "
                "Check API key configuration or fall back to stub mode."
            )
        kwargs = _add_timeout(kwargs)
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class LocalOpenAIProvider(LLMProvider):
    """Local OpenAI-compatible LLM provider.

    Connects to any OpenAI-compatible endpoint (e.g., llama.cpp, vLLM, text-generation-webui).
    """

    def __init__(
        self,
        api_key: str = "not-required",
        model: str = "local-model",
        base_url: str = "http://localhost:8000/v1"
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._client = None

    def configure(self, config: dict[str, Any]) -> None:
        api_key = config.get("api_key", self._api_key)
        base_url = config.get("base_url", self._base_url)
        model = config.get("model", self._model)

        try:
            import openai
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            self._model = model
        except Exception as e:
            logger.warning(
                f"LocalOpenAIProvider: Failed to create client at {base_url}: {e}. "
                "Provider will fall back to stub execution. "
                "Ensure local server is running and accessible."
            )
            self._client = None

    def validate(self) -> bool:
        if self._client is None:
            return False
        try:
            self._client.models.list()
            return True
        except Exception:
            return False

    @property
    def name(self) -> str:
        return f"local-openai:{self._model}"

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def generate_sync(self, prompt: str, **kwargs) -> str:
        if self._client is None:
            raise RuntimeError(
                "LocalOpenAIProvider: No client available. "
                "Check local server is running or fall back to stub mode."
            )
        kwargs = _add_timeout(kwargs)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content

    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        if self._client is None:
            raise RuntimeError(
                "LocalOpenAIProvider: No client available. "
                "Check local server is running or fall back to stub mode."
            )
        kwargs = _add_timeout(kwargs)
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
