"""Tests for reasoning_effort functionality in LLM providers.

Tests cover:
- reasoning_effort stored from config
- per-request override works
- reasoning_effort passed to API calls via mock
- invalid values handled gracefully
- OllamaProvider ignores reasoning_effort (OpenAI-specific)
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock, call
from collections import deque

import pytest

# Mock openai module before importing providers that use it
_mock_openai = MagicMock()
_mock_openai.OpenAI = MagicMock()
_mock_openai.OpenAI.return_value = MagicMock()
sys.modules["openai"] = _mock_openai

# Mock ollama module for OllamaProvider
_mock_ollama = MagicMock()
_mock_ollama.Client = MagicMock()
sys.modules["ollama"] = _mock_ollama

# Now import the module under test
spec = __import__("importlib").util.spec_from_file_location(
    "spine.providers.llm", "spine/providers/llm.py"
)
llm_module = __import__("importlib").util.module_from_spec(spec)
spec.loader.exec_module(llm_module)

LLMProvider = llm_module.LLMProvider
LLMResponse = llm_module.LLMResponse
OpenAIProvider = llm_module.OpenAIProvider
OllamaProvider = llm_module.OllamaProvider
OpenRouterProvider = llm_module.OpenRouterProvider
LocalOpenAIProvider = llm_module.LocalOpenAIProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_openai_client():
    """Create a mock OpenAI client with a mock chat.completions.create."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="mocked response"))]
    )
    return mock_client


def _configure_openai(provider, config_dict):
    """Configure an OpenAI-based provider with a mocked client."""
    mock_client = _make_mock_openai_client()
    # Patch the openai.OpenAI constructor to return our mock
    with patch.dict(sys.modules, {"openai": MagicMock(OpenAI=MagicMock(return_value=mock_client))}):
        provider.configure(config_dict)
    return mock_client


# ---------------------------------------------------------------------------
# Test: reasoning_effort stored from config
# ---------------------------------------------------------------------------

class TestReasoningEffortStoredFromConfig:
    """Test that reasoning_effort is stored in _reasoning_effort from config."""

    def test_openai_stores_reasoning_effort_high(self):
        provider = OpenAIProvider(api_key="test", model="gpt-4")
        _configure_openai(provider, {"api_key": "test", "reasoning_effort": "high"})
        assert provider._reasoning_effort == "high"

    def test_openai_stores_reasoning_effort_low(self):
        provider = OpenAIProvider(api_key="test", model="gpt-4")
        _configure_openai(provider, {"api_key": "test", "reasoning_effort": "low"})
        assert provider._reasoning_effort == "low"

    def test_openai_stores_reasoning_effort_medium(self):
        provider = OpenAIProvider(api_key="test", model="gpt-4")
        _configure_openai(provider, {"api_key": "test", "reasoning_effort": "medium"})
        assert provider._reasoning_effort == "medium"

    def test_openai_defaults_reasoning_effort_to_none(self):
        provider = OpenAIProvider(api_key="test", model="gpt-4")
        _configure_openai(provider, {"api_key": "test"})
        assert provider._reasoning_effort is None

    def test_openrouter_stores_reasoning_effort(self):
        provider = OpenRouterProvider(api_key="test", model="openai/gpt-4")
        _configure_openai(provider, {"api_key": "test", "reasoning_effort": "high"})
        assert provider._reasoning_effort == "high"

    def test_openrouter_defaults_reasoning_effort_to_none(self):
        provider = OpenRouterProvider(api_key="test", model="openai/gpt-4")
        _configure_openai(provider, {"api_key": "test"})
        assert provider._reasoning_effort is None

    def test_local_openai_stores_reasoning_effort(self):
        provider = LocalOpenAIProvider(api_key="test", model="local-model")
        _configure_openai(provider, {"api_key": "test", "reasoning_effort": "low"})
        assert provider._reasoning_effort == "low"

    def test_local_openai_defaults_reasoning_effort_to_none(self):
        provider = LocalOpenAIProvider(api_key="test", model="local-model")
        _configure_openai(provider, {"api_key": "test"})
        assert provider._reasoning_effort is None

    def test_ollama_does_not_store_reasoning_effort(self):
        """OllamaProvider.configure does not set _reasoning_effort (OpenAI-specific)."""
        provider = OllamaProvider(model="qwen")
        _configure_openai(provider, {"base_url": "http://localhost:11434"})
        assert not hasattr(provider, "_reasoning_effort") or provider._reasoning_effort is None


# ---------------------------------------------------------------------------
# Test: per-request override works
# ---------------------------------------------------------------------------

class TestReasoningEffortPerRequestOverride:
    """Test that per-request reasoning_effort overrides config-level setting."""

    def test_generate_sync_override_high_over_config_low(self):
        provider = OpenAIProvider(api_key="test", model="gpt-4")
        mock_client = _configure_openai(provider, {"api_key": "test", "reasoning_effort": "low"})
        provider.generate_sync("hello", reasoning_effort="high")
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "high"

    def test_generate_sync_override_low_over_config_high(self):
        provider = OpenAIProvider(api_key="test", model="gpt-4")
        mock_client = _configure_openai(provider, {"api_key": "test", "reasoning_effort": "high"})
        provider.generate_sync("hello", reasoning_effort="low")
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "low"

    def test_generate_sync_none_uses_config(self):
        provider = OpenAIProvider(api_key="test", model="gpt-4")
        mock_client = _configure_openai(provider, {"api_key": "test", "reasoning_effort": "medium"})
        provider.generate_sync("hello", reasoning_effort=None)
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "medium"

    def test_generate_sync_none_no_config(self):
        provider = OpenAIProvider(api_key="test", model="gpt-4")
        mock_client = _configure_openai(provider, {"api_key": "test"})
        provider.generate_sync("hello", reasoning_effort=None)
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["reasoning_effort"] is None

    @pytest.mark.asyncio
    async def test_stream_override_high_over_config_low(self):
        provider = OpenAIProvider(api_key="test", model="gpt-4")
        mock_client = _configure_openai(provider, {"api_key": "test", "reasoning_effort": "low"})

        # Set up streaming mock
        mock_chunk1 = MagicMock(choices=[MagicMock(delta=MagicMock(content="hello"))])
        mock_chunk2 = MagicMock(choices=[MagicMock(delta=MagicMock(content=" world"))])
        mock_client.chat.completions.create.return_value = iter([mock_chunk1, mock_chunk2])

        async for _ in provider.stream("hello", reasoning_effort="high"):
            pass
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "high"

    @pytest.mark.asyncio
    async def test_stream_none_uses_config(self):
        provider = OpenAIProvider(api_key="test", model="gpt-4")
        mock_client = _configure_openai(provider, {"api_key": "test", "reasoning_effort": "high"})

        mock_chunk = MagicMock(choices=[MagicMock(delta=MagicMock(content="hi"))])
        mock_client.chat.completions.create.return_value = iter([mock_chunk])

        async for _ in provider.stream("hello", reasoning_effort=None):
            pass
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "high"

    def test_openrouter_generate_sync_override(self):
        provider = OpenRouterProvider(api_key="test", model="openai/gpt-4")
        mock_client = _configure_openai(provider, {"api_key": "test", "reasoning_effort": "low"})
        provider.generate_sync("hello", reasoning_effort="high")
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "high"

    def test_local_openai_generate_sync_override(self):
        provider = LocalOpenAIProvider(api_key="test", model="local-model")
        mock_client = _configure_openai(provider, {"api_key": "test", "reasoning_effort": "low"})
        provider.generate_sync("hello", reasoning_effort="high")
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "high"


# ---------------------------------------------------------------------------
# Test: reasoning_effort passed to API calls via mock
# ---------------------------------------------------------------------------

class TestReasoningEffortPassedToAPICalls:
    """Test that reasoning_effort is actually passed through to the API call."""

    def test_openai_generate_sync_passes_reasoning_effort(self):
        provider = OpenAIProvider(api_key="test", model="gpt-4")
        mock_client = _configure_openai(provider, {"api_key": "test"})
        provider.generate_sync("hello", reasoning_effort="medium")
        mock_client.chat.completions.create.assert_called_once()
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["reasoning_effort"] == "medium"

    @pytest.mark.asyncio
    async def test_openai_stream_passes_reasoning_effort(self):
        provider = OpenAIProvider(api_key="test", model="gpt-4")
        mock_client = _configure_openai(provider, {"api_key": "test"})

        mock_chunk = MagicMock(choices=[MagicMock(delta=MagicMock(content="x"))])
        mock_client.chat.completions.create.return_value = iter([mock_chunk])

        async for _ in provider.stream("hello", reasoning_effort="high"):
            pass
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["reasoning_effort"] == "high"
        assert kwargs["stream"] is True

    def test_openrouter_generate_sync_passes_reasoning_effort(self):
        provider = OpenRouterProvider(api_key="test", model="openai/gpt-4")
        mock_client = _configure_openai(provider, {"api_key": "test"})
        provider.generate_sync("hello", reasoning_effort="low")
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["reasoning_effort"] == "low"

    def test_local_openai_generate_sync_passes_reasoning_effort(self):
        provider = LocalOpenAIProvider(api_key="test", model="local-model")
        mock_client = _configure_openai(provider, {"api_key": "test"})
        provider.generate_sync("hello", reasoning_effort="high")
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["reasoning_effort"] == "high"

    def test_ollama_generate_sync_ignores_reasoning_effort(self):
        """OllamaProvider should not pass reasoning_effort to its API call."""
        mock_ollama_client = MagicMock()
        mock_ollama_client.chat.return_value = {"message": {"content": "ollama response"}}

        provider = OllamaProvider(model="qwen")
        with patch.dict(sys.modules, {"ollama": MagicMock(Client=MagicMock(return_value=mock_ollama_client))}):
            provider.configure({"base_url": "http://localhost:11434"})

        provider.generate_sync("hello", reasoning_effort="high")
        # Ollama's chat should NOT receive reasoning_effort
        _, kwargs = mock_ollama_client.chat.call_args
        assert "reasoning_effort" not in kwargs

    def test_base_llm_provider_generate_passes_reasoning_effort(self):
        """LLMProvider.generate() delegates to generate_sync with reasoning_effort."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate_sync = MagicMock(return_value="response")

        # Call the base class generate method directly
        LLMProvider.generate(mock_provider, "hello", reasoning_effort="high")
        mock_provider.generate_sync.assert_called_once_with("hello", reasoning_effort="high")

    @pytest.mark.asyncio
    async def test_base_llm_provider_generate_async_passes_reasoning_effort(self):
        """LLMProvider.generate_async() delegates to generate_sync with reasoning_effort."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate_sync = MagicMock(return_value="response")

        result = await LLMProvider.generate_async(mock_provider, "hello", reasoning_effort="medium")
        mock_provider.generate_sync.assert_called_once_with("hello", reasoning_effort="medium")
        assert result == "response"

    @pytest.mark.asyncio
    async def test_base_llm_provider_generate_with_confidence_passes_reasoning_effort(self):
        """LLMProvider.generate_with_confidence() passes reasoning_effort to internal calls."""
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.generate_sync = MagicMock(side_effect=[
            "answer",  # first call: generate
            "0.9",     # second call: confidence prompt
        ])

        response, confidence = await LLMProvider.generate_with_confidence(
            mock_provider, "hello", reasoning_effort="high"
        )
        assert response.content == "answer"
        assert confidence == 0.9
        # Should have been called twice: once for content, once for confidence
        assert mock_provider.generate_sync.call_count == 2
        # First call should have reasoning_effort='high'
        first_call_kwargs = mock_provider.generate_sync.call_args_list[0].kwargs
        assert first_call_kwargs.get("reasoning_effort") == "high"


# ---------------------------------------------------------------------------
# Test: invalid values handled gracefully
# ---------------------------------------------------------------------------

class TestReasoningEffortInvalidValues:
    """Test that invalid reasoning_effort values are handled gracefully."""

    def test_invalid_value_passed_to_api(self):
        """Invalid reasoning_effort values are passed through to the API.
        
        The provider doesn't validate - it's the API's job to reject invalid values.
        This test verifies the value is still passed (not silently dropped).
        """
        provider = OpenAIProvider(api_key="test", model="gpt-4")
        mock_client = _configure_openai(provider, {"api_key": "test"})
        provider.generate_sync("hello", reasoning_effort="invalid_value")
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["reasoning_effort"] == "invalid_value"

    def test_empty_string_reasoning_effort_passed_through(self):
        """Empty string is passed through (API will handle validation)."""
        provider = OpenAIProvider(api_key="test", model="gpt-4")
        mock_client = _configure_openai(provider, {"api_key": "test"})
        provider.generate_sync("hello", reasoning_effort="")
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["reasoning_effort"] == ""

    def test_numeric_reasoning_effort_passed_through(self):
        """Non-string values are passed through (API handles validation)."""
        provider = OpenAIProvider(api_key="test", model="gpt-4")
        mock_client = _configure_openai(provider, {"api_key": "test"})
        provider.generate_sync("hello", reasoning_effort=123)
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["reasoning_effort"] == 123

    def test_reasoning_effort_none_is_valid(self):
        """None is a valid reasoning_effort value (means no effort specified)."""
        provider = OpenAIProvider(api_key="test", model="gpt-4")
        mock_client = _configure_openai(provider, {"api_key": "test"})
        provider.generate_sync("hello", reasoning_effort=None)
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["reasoning_effort"] is None

    def test_reasoning_effort_not_in_ollama_kwargs(self):
        """OllamaProvider should never include reasoning_effort in API call kwargs."""
        mock_ollama_client = MagicMock()
        mock_ollama_client.chat.return_value = {"message": {"content": "response"}}

        provider = OllamaProvider(model="qwen")
        with patch.dict(sys.modules, {"ollama": MagicMock(Client=MagicMock(return_value=mock_ollama_client))}):
            provider.configure({"base_url": "http://localhost:11434"})

        # Even when explicitly passed, Ollama should not include it
        provider.generate_sync("hello", reasoning_effort="high")
        _, kwargs = mock_ollama_client.chat.call_args
        assert "reasoning_effort" not in kwargs

    @pytest.mark.asyncio
    async def test_stream_with_invalid_reasoning_effort(self):
        """Invalid reasoning_effort still gets passed through in stream calls."""
        provider = OpenAIProvider(api_key="test", model="gpt-4")
        mock_client = _configure_openai(provider, {"api_key": "test"})

        mock_chunk = MagicMock(choices=[MagicMock(delta=MagicMock(content="x"))])
        mock_client.chat.completions.create.return_value = iter([mock_chunk])

        async for _ in provider.stream("hello", reasoning_effort="bad_value"):
            pass
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["reasoning_effort"] == "bad_value"


# ---------------------------------------------------------------------------
# Test: all valid reasoning_effort values
# ---------------------------------------------------------------------------

class TestValidReasoningEffortValues:
    """Test that all valid reasoning_effort values work correctly."""

    @pytest.mark.parametrize("valid_value", ["low", "medium", "high", None])
    def test_all_valid_values_passed_to_openai(self, valid_value):
        provider = OpenAIProvider(api_key="test", model="gpt-4")
        mock_client = _configure_openai(provider, {"api_key": "test"})
        provider.generate_sync("test prompt", reasoning_effort=valid_value)
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["reasoning_effort"] == valid_value

    @pytest.mark.parametrize("valid_value", ["low", "medium", "high", None])
    def test_all_valid_values_passed_to_openrouter(self, valid_value):
        provider = OpenRouterProvider(api_key="test", model="openai/gpt-4")
        mock_client = _configure_openai(provider, {"api_key": "test"})
        provider.generate_sync("test prompt", reasoning_effort=valid_value)
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["reasoning_effort"] == valid_value

    @pytest.mark.parametrize("valid_value", ["low", "medium", "high", None])
    def test_all_valid_values_passed_to_local_openai(self, valid_value):
        provider = LocalOpenAIProvider(api_key="test", model="local-model")
        mock_client = _configure_openai(provider, {"api_key": "test"})
        provider.generate_sync("test prompt", reasoning_effort=valid_value)
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["reasoning_effort"] == valid_value


# ---------------------------------------------------------------------------
# Test: model and other config not affected by reasoning_effort
# ---------------------------------------------------------------------------

class TestReasoningEffortConfigIsolation:
    """Test that reasoning_effort doesn't interfere with other config options."""

    def test_openai_model_not_affected_by_reasoning_effort(self):
        provider = OpenAIProvider(api_key="test", model="gpt-4")
        mock_client = _configure_openai(provider, {"api_key": "test", "model": "gpt-4o", "reasoning_effort": "high"})
        provider.generate_sync("hello")
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["model"] == "gpt-4o"

    def test_openai_base_url_not_affected_by_reasoning_effort(self):
        provider = OpenRouterProvider(api_key="test", model="openai/gpt-4")
        mock_client = _configure_openai(
            provider,
            {"api_key": "test", "base_url": "https://custom.api/v1", "reasoning_effort": "low"}
        )
        provider.generate_sync("hello")
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["model"] == "openai/gpt-4"

    def test_openai_extra_kwargs_passed_through(self):
        """Additional kwargs should be passed through alongside reasoning_effort."""
        provider = OpenAIProvider(api_key="test", model="gpt-4")
        mock_client = _configure_openai(provider, {"api_key": "test"})
        provider.generate_sync("hello", reasoning_effort="high", temperature=0.7, max_tokens=100)
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["reasoning_effort"] == "high"
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 100
