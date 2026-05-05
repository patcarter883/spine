"""LLM Provider implementations."""

import asyncio
from abc import abstractmethod
from typing import Any, Optional, AsyncIterator, Tuple
from dataclasses import dataclass
from .base import Provider, ProviderType


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
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt (sync - calls generate_sync)."""
        return self.generate_sync(prompt, **kwargs)
    
    async def generate_async(self, prompt: str, **kwargs) -> str:
        """Async generate - runs sync in thread pool by default."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.generate_sync(prompt, **kwargs))
    
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
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self._api_key = api_key
        self._model = model
        self._client = None
    
    def configure(self, config: dict[str, Any]) -> None:
        import openai
        self._client = openai.OpenAI(api_key=config.get("api_key", self._api_key))
        self._model = config.get("model", self._model)
    
    def validate(self) -> bool:
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
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content
    
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
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
        import ollama
        self._client = ollama.Client(host=config.get("base_url", self._base_url))
        self._model = config.get("model", self._model)
    
    def validate(self) -> bool:
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
        response = self._client.chat(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response["message"]["content"]
    
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
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
        api_key: str, 
        model: str = "openai/gpt-4", 
        base_url: str = DEFAULT_BASE_URL
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._client = None
    
    def configure(self, config: dict[str, Any]) -> None:
        import openai
        self._client = openai.OpenAI(
            api_key=config.get("api_key", self._api_key),
            base_url=config.get("base_url", self._base_url)
        )
        self._model = config.get("model", self._model)
    
    def validate(self) -> bool:
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
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content
    
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
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
        import openai
        self._client = openai.OpenAI(
            api_key=config.get("api_key", self._api_key),
            base_url=config.get("base_url", self._base_url)
        )
        self._model = config.get("model", self._model)
    
    def validate(self) -> bool:
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
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content
    
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content