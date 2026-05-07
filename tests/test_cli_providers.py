"""Tests for CLI provider loading."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spine.cli import load_config, load_providers, create_provider, get_primary_provider
from spine.providers.base import ProviderConfig


class TestLoadConfig:
    def test_load_config_existing_file(self):
        """Test loading config from existing file."""
        yaml_content = """
spine:
  checkpoint_path: .spine/spine.db
providers:
  llm:
    - name: primary
      type: ollama
      model: qwen3:32b
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(yaml_content)
            
            result = load_config(str(config_path))
            
            assert "spine" in result
            assert "providers" in result
            assert result["spine"]["checkpoint_path"] == ".spine/spine.db"

    def test_load_config_missing_file(self):
        """Test loading config when file doesn't exist."""
        result = load_config("/nonexistent/config.yaml")
        assert result == {}

    def test_load_config_empty_file(self):
        """Test loading config from empty file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text("")
            
            result = load_config(str(config_path))
            assert result == {}


class TestCreateProvider:
    def test_create_ollama_provider(self):
        """Test creating Ollama provider."""
        cfg = ProviderConfig(
            name="test-ollama",
            type="ollama",
            config={"model": "llama3", "base_url": "http://localhost:11434"}
        )
        
        provider = create_provider(cfg)
        
        assert provider is not None
        assert provider.name == "ollama:llama3"

    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        cfg = ProviderConfig(
            name="test-openai",
            type="openai",
            config={"api_key": "test-key", "model": "gpt-4"}
        )
        
        provider = create_provider(cfg)
        
        assert provider is not None
        assert provider.name == "openai:gpt-4"

    def test_create_unknown_provider(self):
        """Test creating unknown provider type returns None."""
        cfg = ProviderConfig(
            name="test-unknown",
            type="unknown_type",
            config={}
        )
        
        provider = create_provider(cfg)
        
        assert provider is None

    def test_create_local_openai_provider(self):
        """Test creating LocalOpenAI provider with base_url from config."""
        cfg = ProviderConfig(
            name="test-local-openai",
            type="local-openai",
            config={"base_url": "http://localhost:8000/v1", "model": "custom-model"}
        )
        
        provider = create_provider(cfg)
        
        assert provider is not None
        assert provider.name == "local-openai:custom-model"
        assert provider._base_url == "http://localhost:8000/v1"

    def test_create_openrouter_provider(self):
        """Test creating OpenRouter provider with api_key from config."""
        cfg = ProviderConfig(
            name="test-openrouter",
            type="openrouter",
            config={"api_key": "test-key", "model": "openai/gpt-4", "base_url": "https://openrouter.ai/api/v1"}
        )
        
        provider = create_provider(cfg)
        
        assert provider is not None
        assert provider.name == "openrouter:openai/gpt-4"
        assert provider._base_url == "https://openrouter.ai/api/v1"


class TestLoadProviders:
    def test_load_providers_from_yaml(self):
        """Test loading providers from YAML config."""
        yaml_content = """
providers:
  llm:
    - name: primary
      type: ollama
      config:
        model: qwen3:32b
      priority: 1
    - name: fallback
      type: openai
      config:
        api_key: test-key
      priority: 2
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "spine.yaml"
            config_path.write_text(yaml_content)
            
            providers = load_providers(str(config_path))
            
            assert "llm" in providers
            assert len(providers["llm"]) == 2
            assert providers["llm"][0][0] == "primary"
            assert providers["llm"][1][0] == "fallback"
            # Check that priority is included in tuple
            assert providers["llm"][0][2] == 1
            assert providers["llm"][1][2] == 2

    def test_load_providers_no_config(self):
        """Test loading providers when no config exists."""
        providers = load_providers("/nonexistent/spine.yaml")
        
        assert providers == {}

    def test_load_providers_disabled_provider(self):
        """Test that disabled providers are skipped."""
        yaml_content = """
providers:
  llm:
    - name: disabled
      type: ollama
      enabled: false
      config: {}
    - name: enabled
      type: ollama
      enabled: true
      config: {}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "spine.yaml"
            config_path.write_text(yaml_content)
            
            providers = load_providers(str(config_path))
            
            # Only enabled provider should be loaded
            assert len(providers.get("llm", [])) == 1
            assert providers["llm"][0][0] == "enabled"

    def test_load_providers_local_openai_and_openrouter(self):
        """Test loading LocalOpenAI and OpenRouter providers from config."""
        yaml_content = """
providers:
  llm:
    - name: primary
      type: local-openai
      enabled: true
      priority: 0
      config:
        base_url: http://localhost:8000/v1
        model: poolside/laguna-m.1:free
    - name: fallback
      type: openrouter
      enabled: true
      priority: 1
      config:
        api_key: test-api-key
        model: poolside/laguna-m.1:free
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "spine.yaml"
            config_path.write_text(yaml_content)
            
            providers = load_providers(str(config_path))
            
            assert "llm" in providers
            assert len(providers["llm"]) == 2
            
            # Check LocalOpenAI provider
            name, provider, priority = providers["llm"][0]
            assert name == "primary"
            assert priority == 0
            assert "local-openai" in provider.name
            assert provider._base_url == "http://localhost:8000/v1"
            
            # Check OpenRouter provider
            name, provider, priority = providers["llm"][1]
            assert name == "fallback"
            assert priority == 1
            assert "openrouter" in provider.name

    def test_load_config_env_var_expansion(self):
        """Test that environment variables are expanded in config values."""
        import os
        os.environ["TEST_API_KEY"] = "my-secret-key"
        
        yaml_content = """
providers:
  llm:
    - name: test
      type: openrouter
      config:
        api_key: ${TEST_API_KEY}
        model: gpt-4
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "spine.yaml"
            config_path.write_text(yaml_content)
            
            config = load_config(str(config_path))
            
            assert config["providers"]["llm"][0]["config"]["api_key"] == "my-secret-key"
            
        del os.environ["TEST_API_KEY"]


class TestGetPrimaryProvider:
    def test_get_primary_provider(self):
        """Test getting primary provider from loaded providers."""
        mock_provider = MagicMock()
        mock_provider.name = "ollama:qwen3"
        
        providers_by_type = {
            "llm": [("primary", mock_provider, 1)]
        }
        
        provider = get_primary_provider(providers_by_type, "llm")
        
        assert provider is mock_provider

    def test_get_primary_provider_empty(self):
        """Test getting primary provider when none exist."""
        providers_by_type = {}
        
        provider = get_primary_provider(providers_by_type, "llm")
        
        assert provider is None