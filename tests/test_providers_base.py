"""Tests for provider base interfaces."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "spine.providers.base", "spine/providers/base.py"
)
base_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_module)

PluginLoader = base_module.PluginLoader
ProviderConfig = base_module.ProviderConfig
ProviderRegistry = base_module.ProviderRegistry


class TestProviderConfig:
    def test_init_with_defaults(self):
        config = ProviderConfig(name="test", type="llm")
        assert config.name == "test"
        assert config.type == "llm"
        assert config.enabled is True
        assert config.priority == 0
        assert config.config == {}

    def test_init_with_all_fields(self):
        config = ProviderConfig(
            name="openai",
            type="llm",
            enabled=False,
            priority=5,
            config={"api_key": "test", "model": "gpt-4"},
        )
        assert config.name == "openai"
        assert config.type == "llm"
        assert config.enabled is False
        assert config.priority == 5
        assert config.config == {"api_key": "test", "model": "gpt-4"}

    def test_config_defaults_to_empty_dict(self):
        config = ProviderConfig(name="test", type="llm")
        assert config.config == {}


class TestProviderRegistryLoadProviders:
    def test_load_providers_from_yaml(self):
        yaml_content = """
providers:
  llm:
    - name: primary
      type: openai
      config:
        api_key: test-key
        model: gpt-4
      priority: 1
      enabled: true
    - name: fallback
      type: ollama
      config:
        host: localhost
      priority: 2
  memory:
    - name: session
      type: sqlite
      config:
        path: .spine/sessions.db
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "spine.yaml"
            config_path.write_text(yaml_content)

            registry = ProviderRegistry()
            configs = registry.load_providers(str(config_path))

            assert len(configs) == 3
            assert configs[0].name == "primary"
            assert configs[0].type == "openai"
            assert configs[0].priority == 1
            assert configs[0].enabled is True
            assert configs[0].config == {"api_key": "test-key", "model": "gpt-4"}

            assert configs[1].name == "fallback"
            assert configs[1].priority == 2

            assert configs[2].name == "session"
            assert configs[2].type == "sqlite"

    def test_load_providers_empty_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "spine.yaml"
            config_path.write_text("other: value\n")

            registry = ProviderRegistry()
            configs = registry.load_providers(str(config_path))
            assert configs == []

    def test_load_providers_missing_file(self):
        registry = ProviderRegistry()
        configs = registry.load_providers("/nonexistent/spine.yaml")
        assert configs == []


class TestPluginLoader:
    def test_init_with_default_dirs(self):
        loader = PluginLoader()
        assert loader.plugin_dirs == ["./spine-plugins"]
        assert isinstance(loader.registry, ProviderRegistry)

    def test_init_with_custom_dirs(self):
        loader = PluginLoader(plugin_dirs=["/custom/plugins", "/opt/spine-plugins"])
        assert loader.plugin_dirs == ["/custom/plugins", "/opt/spine-plugins"]

    def test_discover_plugins_finds_manifest(self):
        yaml_content = """
providers:
  - type: test_llm
    class: spine.providers.test.TestLLMProvider
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_dir = Path(tmpdir) / "spine-plugins"
            plugin_dir.mkdir()
            manifest_path = plugin_dir / "spine-plugin.yaml"
            manifest_path.write_text(yaml_content)

            loader = PluginLoader(plugin_dirs=[str(plugin_dir)])

            mock_provider = MagicMock()
            import sys
            sys.modules["spine.providers.test"] = MagicMock(TestLLMProvider=mock_provider)

            loader.discover_plugins()

            assert "test_llm" in loader.registry._factories
            assert loader.registry.get_factory("test_llm") is mock_provider

    def test_discover_plugins_no_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = PluginLoader(plugin_dirs=[tmpdir])
            loader.discover_plugins()

    def test_register_factory(self):
        loader = PluginLoader()
        mock_provider = MagicMock()
        loader.register_factory("custom_type", mock_provider)

        assert "custom_type" in loader.registry._factories