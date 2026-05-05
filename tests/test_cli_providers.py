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