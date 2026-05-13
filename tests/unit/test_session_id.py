"""Tests for OpenRouter session_id wiring.

Verifies that resolve_model produces a ChatOpenRouter with session_id
when the model is OpenRouter and a work_id is provided, and falls back
to a string when conditions are not met.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestResolveModelSessionId:
    """Verify session_id wiring in resolve_model."""

    def test_openrouter_with_session_id_returns_chat_model(self) -> None:
        """resolve_model should return a ChatOpenRouter instance when model
        is OpenRouter and session_id is provided."""
        from langchain_core.language_models.chat_models import BaseChatModel

        from spine.agents.helpers import resolve_model

        config = {"configurable": {"model": "openrouter:z-ai/glm-4.5-air:free"}}

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-12345"}):
            model = resolve_model(config, session_id="work-abc123")

        assert isinstance(model, BaseChatModel)
        # Verify it's actually a ChatOpenRouter with session_id set
        assert hasattr(model, "session_id")
        assert model.session_id == "work-abc123"

    def test_openrouter_without_session_id_returns_string(self) -> None:
        """resolve_model should return a string when model is OpenRouter
        but no session_id is provided."""
        from spine.agents.helpers import resolve_model

        config = {"configurable": {"model": "openrouter:z-ai/glm-4.5-air:free"}}

        model = resolve_model(config, session_id=None)

        assert isinstance(model, str)
        assert model == "openrouter:z-ai/glm-4.5-air:free"

    def test_non_openrouter_with_session_id_returns_string(self) -> None:
        """resolve_model should return a string when model is NOT OpenRouter,
        even if session_id is provided."""
        from spine.agents.helpers import resolve_model

        config = {"configurable": {"model": "openai:gpt-4o-mini"}}

        model = resolve_model(config, session_id="work-abc123")

        assert isinstance(model, str)
        assert model == "openai:gpt-4o-mini"

    def test_session_id_truncated_to_128_chars(self) -> None:
        """OpenRouter limits session_id to 128 characters."""
        from spine.agents.helpers import resolve_model

        config = {"configurable": {"model": "openrouter:z-ai/glm-4.5-air:free"}}
        long_id = "x" * 200

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-12345"}):
            model = resolve_model(config, session_id=long_id)

        assert hasattr(model, "session_id")
        assert len(model.session_id) == 128
        assert model.session_id == "x" * 128

    def test_model_name_stripped_of_prefix(self) -> None:
        """The ChatOpenRouter should use the model name without the
        'openrouter:' prefix."""
        from spine.agents.helpers import resolve_model

        config = {"configurable": {"model": "openrouter:anthropic/claude-sonnet-4-5"}}

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-12345"}):
            model = resolve_model(config, session_id="work-123")

        assert model.model_name == "anthropic/claude-sonnet-4-5"

    def test_no_config_falls_back_to_spineconfig(self) -> None:
        """When config is None, resolve_model should fall back to SpineConfig.

        For local providers with base_url, this returns a pre-built ChatOpenAI.
        For cloud providers (no base_url), this returns a string.
        """
        from langchain_core.language_models.chat_models import BaseChatModel

        from spine.agents.helpers import resolve_model

        model = resolve_model(None, session_id=None)
        # The active provider in config.yaml is a local vLLM server with
        # base_url, so resolve_model should return a pre-built ChatOpenAI.
        assert isinstance(model, BaseChatModel)

    def test_work_id_as_session_id(self) -> None:
        """Typical usage: state.get('work_id') passed as session_id."""
        from langchain_core.language_models.chat_models import BaseChatModel

        from spine.agents.helpers import resolve_model

        config = {"configurable": {"model": "openrouter:z-ai/glm-4.5-air:free"}}
        work_id = "a1b2c3d4"  # 8-char UUID prefix used by dispatcher

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-12345"}):
            model = resolve_model(config, session_id=work_id)

        assert isinstance(model, BaseChatModel)
        assert model.session_id == work_id

    def test_provider_profile_kwargs_applied(self) -> None:
        """_build_openrouter_model should apply DA ProviderProfile kwargs."""
        from spine.agents.helpers import _build_openrouter_model

        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key-12345"}):
            with patch(
                "deepagents.profiles.provider.apply_provider_profile",
                return_value={"app_url": "https://spine.dev", "app_title": "SPINE"},
            ) as mock_apply:
                model = _build_openrouter_model(
                    "openrouter:z-ai/glm-4.5-air:free",
                    "work-123",
                )

        # Verify apply_provider_profile was called with the model spec
        mock_apply.assert_called_once_with("openrouter:z-ai/glm-4.5-air:free")
        # Verify the profile kwargs were passed to ChatOpenRouter
        assert model.app_url == "https://spine.dev"
        assert model.app_title == "SPINE"


class TestLocalProviderModel:
    """Verify that local providers with base_url get pre-built ChatOpenAI models."""

    def test_local_provider_with_base_url_returns_chat_model(self) -> None:
        """resolve_model should return a ChatOpenAI when the active provider
        has base_url and the model spec matches."""
        from unittest.mock import MagicMock

        from langchain_core.language_models.chat_models import BaseChatModel

        from spine.agents.helpers import resolve_model

        # No config override — falls back to SpineConfig, which has the
        # local vLLM provider with base_url.
        model = resolve_model(None, session_id=None)
        assert isinstance(model, BaseChatModel)
        # Verify the model points at the local server
        assert model.openai_api_base == "http://localhost:8000/v1"

    def test_local_provider_config_mismatch_returns_string(self) -> None:
        """resolve_model should return a string when the model spec does NOT
        match the active provider's model (e.g. explicit config override)."""
        from spine.agents.helpers import resolve_model

        # Explicitly request a different model than the local provider
        config = {"configurable": {"model": "openai:gpt-4o-mini"}}
        model = resolve_model(config, session_id=None)
        assert isinstance(model, str)
        assert model == "openai:gpt-4o-mini"

    def test_local_provider_wires_api_key(self) -> None:
        """The pre-built ChatOpenAI should use the api_key from config."""
        from spine.agents.helpers import resolve_model

        model = resolve_model(None, session_id=None)
        # ChatOpenAI stores it as openai_api_key
        assert model.openai_api_key.get_secret_value() == "vllm"
