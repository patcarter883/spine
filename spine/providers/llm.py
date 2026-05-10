"""LLM Provider implementations with graceful error handling and validation."""

import asyncio
import logging
import socket
from abc import abstractmethod
from typing import Any, AsyncIterator, Tuple
from dataclasses import dataclass
from .base import Provider, ProviderType

logger = logging.getLogger(__name__)

# Default timeout for LLM HTTP calls (connect, read) in seconds
DEFAULT_TIMEOUT = (10, 300)  # (connect_timeout, read_timeout)

# Streaming-specific timeouts
# Per-chunk read timeout for streaming: set high so long-running completions
# are not killed between chunks. TTFB timeout handles initial responsiveness.
STREAM_READ_TIMEOUT = 300  # 5 minutes per chunk - generous for slow token generation

# Default time-to-first-byte timeout in seconds
# Applies ONLY to the initial connection + first token.
# Once tokens start flowing, the stream stays open indefinitely.
TTFB_TIMEOUT = 30.0


class TTFBTimeoutError(TimeoutError):
    """Raised when the LLM fails to produce its first token within the TTFB deadline.

    This is distinct from a general TimeoutError: it specifically means the
    model did not start streaming any output within the configured window,
    which typically indicates network issues, server overload, or model
    cold-start latency.
    """

    def __init__(self, timeout_seconds: float, provider_name: str = ""):
        self.timeout_seconds = timeout_seconds
        self.provider_name = provider_name
        msg = (
            f"TTFB timeout: {provider_name or 'LLM'} did not produce "
            f"any output within {timeout_seconds:.1f}s (time-to-first-byte). "
            f"This may indicate network issues, server overload, or model cold-start."
        )
        super().__init__(msg)


def _add_timeout(kwargs: dict[str, Any], timeout: tuple[int, int] | int | None = None, streaming: bool = False) -> dict[str, Any]:
    """Merge timeout into kwargs for HTTP-based LLM calls.

    Only adds the 'timeout' key if it is not already present and
    the underlying client supports it (openai >= 1.0).

    When *streaming* is True, uses STREAM_READ_TIMEOUT for the per-chunk
    read timeout so long-running completions are not killed between chunks.
    """
    if "timeout" in kwargs:
        return kwargs
    kwargs = dict(kwargs)
    if streaming and timeout is None:
        kwargs.setdefault("timeout", (DEFAULT_TIMEOUT[0], STREAM_READ_TIMEOUT))
    else:
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
    async def stream(self, prompt: str, ttfb_timeout: float = TTFB_TIMEOUT, **kwargs) -> AsyncIterator[str]:
        """Stream text generation.

        Args:
            prompt: The user prompt.
            ttfb_timeout: Maximum seconds to wait for the first token (TTFB).
                          Defaults to TTFB_TIMEOUT (30 s). Once streaming
                          begins, subsequent chunks have no timeout.
            **kwargs: Provider-specific parameters.
        """
        pass

    def generate(self, prompt: str, timeout: float = 60.0, **kwargs) -> str:
        """Generate text from prompt (sync - calls generate_sync with optional timeout)."""
        import concurrent.futures

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
                timeout=timeout or 60.0,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Async LLM call timed out after {timeout}s for prompt: {prompt[:80]}...")
            raise TimeoutError(f"LLM call timed out after {timeout} seconds")

    async def _stream_with_ttfb_timeout(
        self,
        prompt: str,
        ttfb_timeout: float = TTFB_TIMEOUT,
        **kwargs
    ) -> AsyncIterator[str]:
        """Yield chunks from an LLM stream with TTFB timeout on the first chunk.

        This is a convenience method that providers can call from their
        ``stream()`` implementation.  It wraps the first-chunk retrieval
        in ``asyncio.wait_for()`` so that network issues, server overload,
        or model cold-start latency are caught with a clear error.

        After the first chunk arrives, subsequent chunks are yielded with
        **no timeout**, allowing long-running completions to complete
        naturally.

        Args:
            prompt: The user prompt.
            ttfb_timeout: Maximum seconds to wait for the first token.
                          Defaults to ``TTFB_TIMEOUT`` (30 s).
            **kwargs: Passed through to ``self._raw_stream()``.

        Yields:
            Content strings from the model stream.
        """
        loop = asyncio.get_event_loop()

        # Get the raw sync stream from the provider
        raw_stream = self._raw_stream(prompt, **kwargs)
        stream_iter = iter(raw_stream)

        try:
            # TTFB: wait for the first item, but only up to ttfb_timeout
            first_item = await asyncio.wait_for(
                loop.run_in_executor(None, next, stream_iter),
                timeout=ttfb_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"TTFB timeout after {ttfb_timeout:.1f}s for "
                f"{self.name} with prompt: {prompt[:80]}..."
            )
            raise TTFBTimeoutError(
                timeout_seconds=ttfb_timeout,
                provider_name=self.name,
            )
        except StopIteration:
            # Empty stream -- nothing to yield
            return

        # Yield the content of the first chunk
        content = self._extract_chunk_content(first_item)
        if content:
            yield content

        # Remaining chunks: no timeout between them
        for item in stream_iter:
            content = self._extract_chunk_content(item)
            if content:
                yield content

    def _raw_stream(self, prompt: str, **kwargs):
        """Return the raw sync streaming iterator from the LLM client.

        Subclasses MUST override this to create the actual streaming call
        (e.g. ``self._client.chat.completions.create(..., stream=True)``).
        Returns an iterable of provider-specific chunk objects.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _raw_stream() "
            "to return a sync streaming iterator."
        )

    @staticmethod
    def _extract_chunk_content(chunk: Any) -> str | None:
        """Extract text content from a raw stream chunk.

        Subclasses override this to handle provider-specific chunk formats.
        Returns the text content string, or None if this chunk has no content.
        """
        # Default: OpenAI/OpenAI-compatible chunk format
        if hasattr(chunk, "choices") and chunk.choices:
            delta = getattr(chunk.choices[0], "delta", None)
            if delta and getattr(delta, "content", None):
                return delta.content
        return None

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

    def _raw_stream(self, prompt: str, **kwargs):
        """Create the raw OpenAI streaming response."""
        if self._client is None:
            raise RuntimeError(
                "OpenAIProvider: No client available. "
                "Check API key configuration or fall back to stub mode."
            )
        kwargs = _add_timeout(kwargs, streaming=True)
        return self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs
        )

    @staticmethod
    def _extract_chunk_content(chunk: Any) -> str | None:
        """Extract text from an OpenAI stream chunk."""
        if hasattr(chunk, "choices") and chunk.choices:
            delta = getattr(chunk.choices[0], "delta", None)
            if delta and getattr(delta, "content", None):
                return delta.content
        return None

    async def stream(self, prompt: str, ttfb_timeout: float = TTFB_TIMEOUT, **kwargs) -> AsyncIterator[str]:
        """Stream text generation with TTFB timeout.

        Args:
            prompt: The user prompt.
            ttfb_timeout: Maximum seconds to wait for the first token.
                          Defaults to TTFB_TIMEOUT (30 s). Once the first
                          token arrives, the stream continues indefinitely.
            **kwargs: Passed through to the OpenAI API (temperature, max_tokens, etc.)
        """
        async for chunk in self._stream_with_ttfb_timeout(
            prompt, ttfb_timeout=ttfb_timeout, **kwargs
        ):
            yield chunk


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

    def _raw_stream(self, prompt: str, **kwargs):
        """Create the raw Ollama streaming response."""
        if self._client is None:
            raise RuntimeError(
                "OllamaProvider: No client available. "
                "Check Ollama server is running or fall back to stub mode."
            )
        kwargs = _add_timeout(kwargs, streaming=True)
        return self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs
        )

    @staticmethod
    def _extract_chunk_content(chunk: Any) -> str | None:
        """Extract text from an Ollama stream chunk (dict format)."""
        if isinstance(chunk, dict) and "message" in chunk:
            return chunk["message"].get("content")
        return None

    async def stream(self, prompt: str, ttfb_timeout: float = TTFB_TIMEOUT, **kwargs) -> AsyncIterator[str]:
        """Stream text generation with TTFB timeout.

        Args:
            prompt: The user prompt.
            ttfb_timeout: Maximum seconds to wait for the first token.
            **kwargs: Passed through to the Ollama API.
        """
        async for chunk in self._stream_with_ttfb_timeout(
            prompt, ttfb_timeout=ttfb_timeout, **kwargs
        ):
            yield chunk


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

    def _raw_stream(self, prompt: str, **kwargs):
        """Create the raw OpenRouter streaming response."""
        if self._client is None:
            raise RuntimeError(
                "OpenRouterProvider: No client available. "
                "Check API key configuration or fall back to stub mode."
            )
        kwargs = _add_timeout(kwargs, streaming=True)
        return self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs
        )

    async def stream(self, prompt: str, ttfb_timeout: float = TTFB_TIMEOUT, **kwargs) -> AsyncIterator[str]:
        """Stream text generation with TTFB timeout.

        Args:
            prompt: The user prompt.
            ttfb_timeout: Maximum seconds to wait for the first token.
            **kwargs: Passed through to the OpenRouter API.
        """
        async for chunk in self._stream_with_ttfb_timeout(
            prompt, ttfb_timeout=ttfb_timeout, **kwargs
        ):
            yield chunk


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

    def _raw_stream(self, prompt: str, **kwargs):
        """Create the raw local OpenAI streaming response."""
        if self._client is None:
            raise RuntimeError(
                "LocalOpenAIProvider: No client available. "
                "Check local server is running or fall back to stub mode."
            )
        kwargs = _add_timeout(kwargs, streaming=True)
        return self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs
        )

    async def stream(self, prompt: str, ttfb_timeout: float = TTFB_TIMEOUT, **kwargs) -> AsyncIterator[str]:
        """Stream text generation with TTFB timeout.

        Args:
            prompt: The user prompt.
            ttfb_timeout: Maximum seconds to wait for the first token.
            **kwargs: Passed through to the local OpenAI-compatible API.
        """
        async for chunk in self._stream_with_ttfb_timeout(
            prompt, ttfb_timeout=ttfb_timeout, **kwargs
        ):
            yield chunk
