"""Tests for per-phase model resolution and subagent factory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from spine.config import SpineConfig


class TestPhaseAwareModelResolution:
    """Test SpineConfig.resolve_model(phase=) with providers.phases."""

    def test_default_no_phase(self) -> None:
        cfg = SpineConfig(
            providers={
                "llm": [{"name": "default", "model": "openrouter:default-model", "enabled": True}],
            }
        )
        assert cfg.resolve_model() == "openrouter:default-model"

    def test_phase_override(self) -> None:
        cfg = SpineConfig(
            providers={
                "llm": [{"name": "default", "model": "openrouter:default-model", "enabled": True}],
                "phases": {
                    "implement": {"model": "openrouter:impl-model"},
                },
            }
        )
        assert cfg.resolve_model(phase="implement") == "openrouter:impl-model"

    def test_subagent_override_beats_phase(self) -> None:
        """Most specific key wins over parent phase key."""
        cfg = SpineConfig(
            providers={
                "llm": [{"name": "default", "model": "openrouter:default-model", "enabled": True}],
                "phases": {
                    "implement": {"model": "openrouter:impl-model"},
                    "implement/subagents/slice-implementer": {"model": "openrouter:mini-model"},
                },
            }
        )
        assert (
            cfg.resolve_model(phase="implement/subagents/slice-implementer")
            == "openrouter:mini-model"
        )

    def test_unknown_subagent_falls_back_to_phase(self) -> None:
        """Subagent path without own key inherits parent phase model."""
        cfg = SpineConfig(
            providers={
                "llm": [{"name": "default", "model": "openrouter:default-model", "enabled": True}],
                "phases": {
                    "implement": {"model": "openrouter:impl-model"},
                },
            }
        )
        assert cfg.resolve_model(phase="implement/subagents/unknown") == "openrouter:impl-model"

    def test_unknown_phase_falls_back_to_default(self) -> None:
        """Phase without override uses default provider."""
        cfg = SpineConfig(
            providers={
                "llm": [{"name": "default", "model": "openrouter:default-model", "enabled": True}],
                "phases": {
                    "implement": {"model": "openrouter:impl-model"},
                },
            }
        )
        assert cfg.resolve_model(phase="plan") == "openrouter:default-model"


class TestPhaseAwareProviderConfig:
    """Test SpineConfig.resolve_provider_config(phase=)."""

    def test_default_no_phase(self) -> None:
        """Without phase, returns the default provider config."""
        cfg = SpineConfig(
            providers={
                "llm": [
                    {
                        "name": "default",
                        "model": "openai:default-model",
                        "base_url": "http://localhost:8000/v1",
                        "api_key": "dummy",
                        "temperature": 0.7,
                        "enabled": True,
                    },
                ],
            }
        )
        result = cfg.resolve_provider_config()
        assert result["base_url"] == "http://localhost:8000/v1"
        assert result["api_key"] == "dummy"
        assert result["temperature"] == 0.7

    def test_phase_provider_reference(self) -> None:
        """Phase with ``provider: name`` inherits that provider's settings."""
        cfg = SpineConfig(
            providers={
                "llm": [
                    {"name": "default", "model": "openrouter:default", "enabled": True},
                    {
                        "name": "local-huge",
                        "model": "openai:big-model",
                        "base_url": "http://gpu1:8000/v1",
                        "api_key": "huge-key",
                        "temperature": 0.3,
                        "request_timeout": 600,
                        "enabled": True,
                    },
                ],
                "phases": {
                    "implement": {
                        "provider": "local-huge",
                    },
                },
            }
        )
        result = cfg.resolve_provider_config(phase="implement")
        assert result["base_url"] == "http://gpu1:8000/v1"
        assert result["api_key"] == "huge-key"
        assert result["temperature"] == 0.3
        assert result["request_timeout"] == 600

    def test_phase_provider_reference_with_override(self) -> None:
        """Phase provider reference + inline override: phase key wins."""
        cfg = SpineConfig(
            providers={
                "llm": [
                    {"name": "default", "model": "openrouter:default", "enabled": True},
                    {
                        "name": "local-cheap",
                        "model": "openai:mini",
                        "base_url": "http://localhost:8000/v1",
                        "api_key": "vllm",
                        "temperature": 0.7,
                        "enabled": True,
                    },
                ],
                "phases": {
                    "verify": {
                        "provider": "local-cheap",
                        "temperature": 0.1,  # override
                        "request_timeout": 120,  # added
                    },
                },
            }
        )
        result = cfg.resolve_provider_config(phase="verify")
        assert result["base_url"] == "http://localhost:8000/v1"  # inherited
        assert result["api_key"] == "vllm"  # inherited
        assert result["temperature"] == 0.1  # overridden
        assert result["request_timeout"] == 120  # added

    def test_phase_direct_keys_no_provider_ref(self) -> None:
        """Phase with direct keys (no provider ref) merges onto default."""
        cfg = SpineConfig(
            providers={
                "llm": [
                    {
                        "name": "default",
                        "model": "openai:default",
                        "base_url": "http://default:8000/v1",
                        "api_key": "default-key",
                        "enabled": True,
                    },
                ],
                "phases": {
                    "verify": {
                        "model": "openai:tiny-model",
                        "base_url": "http://other:8000/v1",
                        "api_key": "other-key",
                        "temperature": 0.1,
                    },
                },
            }
        )
        result = cfg.resolve_provider_config(phase="verify")
        assert result["base_url"] == "http://other:8000/v1"  # overridden
        assert result["api_key"] == "other-key"  # overridden
        assert result["temperature"] == 0.1  # added

    def test_subagent_falls_back_to_parent_phase_provider(self) -> None:
        """Subagent with no own ``provider`` ref inherits parent phase's."""
        cfg = SpineConfig(
            providers={
                "llm": [
                    {"name": "default", "model": "openrouter:default", "enabled": True},
                    {
                        "name": "fast-local",
                        "model": "openai:fast",
                        "base_url": "http://fast:8000/v1",
                        "api_key": "fast",
                        "enabled": True,
                    },
                ],
                "phases": {
                    "implement": {
                        "provider": "fast-local",
                    },
                    "implement/subagents/slice-implementer": {
                        "model": "openai:even-faster",
                    },
                },
            }
        )
        result = cfg.resolve_provider_config(phase="implement/subagents/slice-implementer")
        # inherits parent phase's provider settings
        assert result["base_url"] == "http://fast:8000/v1"
        assert result["api_key"] == "fast"

    def test_subagent_own_provider_ref(self) -> None:
        """Subagent with its own ``provider`` ref uses that one."""
        cfg = SpineConfig(
            providers={
                "llm": [
                    {"name": "default", "model": "openrouter:default", "enabled": True},
                    {
                        "name": "fast-local",
                        "model": "openai:fast",
                        "base_url": "http://fast:8000/v1",
                        "api_key": "fast",
                        "enabled": True,
                    },
                    {
                        "name": "even-faster-local",
                        "model": "openai:tiny",
                        "base_url": "http://tiny:8000/v1",
                        "api_key": "tiny",
                        "enabled": True,
                    },
                ],
                "phases": {
                    "implement": {
                        "provider": "fast-local",
                    },
                    "implement/subagents/slice-implementer": {
                        "provider": "even-faster-local",
                    },
                },
            }
        )
        result = cfg.resolve_provider_config(phase="implement/subagents/slice-implementer")
        assert result["base_url"] == "http://tiny:8000/v1"
        assert result["api_key"] == "tiny"

    def test_unknown_phase_falls_back_to_default_provider(self) -> None:
        """Phase with no override gets the default provider's settings."""
        cfg = SpineConfig(
            providers={
                "llm": [
                    {
                        "name": "default",
                        "model": "openai:default",
                        "base_url": "http://default:8000/v1",
                        "api_key": "dummy",
                        "enabled": True,
                    },
                ],
                "phases": {
                    "implement": {"provider": "local-huge"},
                },
            }
        )
        result = cfg.resolve_provider_config(phase="plan")
        assert result["base_url"] == "http://default:8000/v1"

    def test_empty_providers_returns_empty_dict(self) -> None:
        """No providers configured → empty dict returned gracefully."""
        cfg = SpineConfig(providers={})
        result = cfg.resolve_provider_config(phase="implement")
        assert result == {}

    def test_all_disabled_returns_empty_dict(self) -> None:
        """All providers disabled → empty dict."""
        cfg = SpineConfig(
            providers={
                "llm": [
                    {"name": "one", "model": "openai:m1", "enabled": False},
                    {"name": "two", "model": "openai:m2", "enabled": False},
                ],
            }
        )
        result = cfg.resolve_provider_config()
        assert result == {}

    def test_provider_name_lookup_miss(self) -> None:
        """``provider`` ref to a non-existent name falls back to default."""
        cfg = SpineConfig(
            providers={
                "llm": [
                    {
                        "name": "default",
                        "model": "openai:default",
                        "base_url": "http://default:8000/v1",
                        "enabled": True,
                    },
                ],
                "phases": {
                    "implement": {"provider": "nonexistent"},
                },
            }
        )
        result = cfg.resolve_provider_config(phase="implement")
        assert result["base_url"] == "http://default:8000/v1"


class TestSubagentFactory:
    """Test spine.agents.subagents module."""

    def test_response_models_defined(self) -> None:
        from spine.agents.subagents import SUBAGENT_RESPONSE_MODELS

        assert "researcher" in SUBAGENT_RESPONSE_MODELS
        assert "slice-implementer" in SUBAGENT_RESPONSE_MODELS
        assert "slice-verifier" in SUBAGENT_RESPONSE_MODELS

    def test_tool_restrictions(self) -> None:
        from spine.agents.subagents import SUBAGENT_TOOLS

        # Researcher is read-only
        researcher_tools = SUBAGENT_TOOLS["researcher"]
        assert "write_file" not in researcher_tools
        assert "edit_file" not in researcher_tools
        assert "read_edit_lint" not in researcher_tools
        assert "execute" not in researcher_tools
        assert "ast_extract_symbol" in researcher_tools

        # Implementer writes via the linted compound tool, NOT raw write/edit.
        impl_tools = SUBAGENT_TOOLS["slice-implementer"]
        assert "write_file" not in impl_tools
        assert "edit_file" not in impl_tools
        assert "read_edit_lint" in impl_tools
        assert "execute" in impl_tools
        # The implementer's surface is intentionally narrow: the broad keyword
        # search and redundant symbol-extractor are NOT bound, so it can't
        # "research half the codebase" (trace 019e784c). Targeted lookups go
        # through the injected codebase_query wrapper, not these.
        assert "search_codebase" not in impl_tools
        assert "ast_extract_symbol" not in impl_tools
        assert "read_file" in impl_tools

        # Verifier can execute (tests/lint) but not write
        verifier_tools = SUBAGENT_TOOLS["slice-verifier"]
        assert "execute" in verifier_tools
        assert "write_file" not in verifier_tools
        assert "read_edit_lint" not in verifier_tools

    def test_build_subagent_spec_rejects_unknown(self) -> None:
        from spine.agents.subagents import build_subagent_spec
        from spine.models.enums import PhaseName
        from spine.models.state import WorkflowState

        with pytest.raises(ValueError, match="Unknown subagent"):
            build_subagent_spec(
                name="nonexistent",
                phase=PhaseName.IMPLEMENT,
                state=WorkflowState(work_id="test"),
                config=None,
            )
