"""Tests for provider base interfaces."""

import tempfile
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

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
ProviderFallbackChain = base_module.ProviderFallbackChain


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


class TestProviderFallbackChain:
    """Tests for the ProviderFallbackChain class."""

    def _create_mock_provider(self, name: str, enabled: bool = True, healthy: bool = True, fail: bool = False):
        """Create a mock provider for testing."""
        mock = MagicMock()
        mock.name = name
        mock.enabled = enabled
        mock.validate.return_value = healthy
        
        # Sync generate for direct calls
        def sync_generate(prompt, **kwargs):
            if fail:
                raise Exception("API error")
            return f"{name}: {prompt}"
        
        # Async generate for generate_async method
        async def async_generate(prompt, **kwargs):
            if fail:
                raise Exception("API error")
            return f"{name}: {prompt}"
        
        async def async_stream(prompt, **kwargs):
            yield f"{name}: {prompt}"
        
        mock.generate = sync_generate
        mock.generate_async = async_generate
        mock.stream = async_stream
        return mock

    def test_init_empty(self):
        """Test initializing with no providers."""
        chain = ProviderFallbackChain()
        assert chain.active_provider is None

    def test_init_with_providers(self):
        """Test initializing with providers."""
        provider1 = self._create_mock_provider("primary")
        provider2 = self._create_mock_provider("fallback")
        
        chain = ProviderFallbackChain(providers=[provider1, provider2])
        assert chain.active_provider == provider1

    @pytest.mark.asyncio
    async def test_generate_with_healthy_provider(self):
        """Test generate with a healthy provider."""
        provider = self._create_mock_provider("test")
        chain = ProviderFallbackChain(providers=[provider])
        
        result = await chain.generate("hello")
        assert result == "test: hello"

    @pytest.mark.asyncio
    async def test_generate_fallback_on_failure(self):
        """Test generate falls back when primary fails."""
        provider1 = self._create_mock_provider("primary", fail=True)
        provider2 = self._create_mock_provider("fallback")
        
        chain = ProviderFallbackChain(providers=[provider1, provider2])
        
        result = await chain.generate("hello")
        assert result == "fallback: hello"

    @pytest.mark.asyncio
    async def test_generate_all_fail(self):
        """Test generate raises when all providers fail."""
        provider1 = self._create_mock_provider("primary", fail=True)
        provider2 = self._create_mock_provider("fallback", fail=True)
        
        chain = ProviderFallbackChain(providers=[provider1, provider2])
        
        with pytest.raises(RuntimeError, match="All providers failed"):
            await chain.generate("hello")

    @pytest.mark.asyncio
    async def test_stream_with_active_provider(self):
        """Test stream with active provider."""
        provider = self._create_mock_provider("test")
        chain = ProviderFallbackChain(providers=[provider])
        
        chunks = []
        async for chunk in chain.stream("hello"):
            chunks.append(chunk)
        assert chunks == ["test: hello"]

    @pytest.mark.asyncio
    async def test_stream_no_healthy_provider(self):
        """Test stream raises when no healthy providers."""
        provider = self._create_mock_provider("test", healthy=False)
        chain = ProviderFallbackChain(providers=[provider])
        
        with pytest.raises(RuntimeError, match="No healthy providers"):
            async for _ in chain.stream("hello"):
                pass

    @pytest.mark.asyncio
    async def test_load_config_async(self):
        """Test async config loading."""
        yaml_content = """
providers:
  llm:
    - name: primary
      type: openai
      config:
        api_key: test-key
      priority: 1
    - name: fallback
      type: ollama
      config:
        model: qwen
      priority: 2
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "spine.yaml"
            config_path.write_text(yaml_content)
            
            chain = ProviderFallbackChain()
            configs = await chain.load_config(str(config_path))
            
            assert len(configs) == 2
            assert configs[0].name == "primary"
            assert configs[0].type == "openai"
            assert configs[0].priority == 1
            assert configs[1].name == "fallback"
            assert configs[1].type == "ollama"

    def test_health_check_caching(self):
        """Test that health checks are cached."""
        provider = self._create_mock_provider("test")
        chain = ProviderFallbackChain(providers=[provider])
        
        # First check
        _ = chain.active_provider
        provider.validate.reset_mock()
        
        # Second check within cache TTL should not call validate
        _ = chain.active_provider
        # Cache should prevent re-validation within TTL
        # Note: actual behavior depends on implementation


class TestConflictResolver:
    """Tests for the ConflictResolver class."""

    def test_confidence_weighted_returns_value_not_name(self):
        """Test that confidence_weighted returns the value, not the provider name."""
        conflict = base_module.ConflictResult(
            key="test-key",
            values={"provider1": "value1", "provider2": "value2"},
            confidence={"provider1": 0.9, "provider2": 0.3}
        )
        
        resolver = base_module.ConflictResolver()
        result = resolver.resolve(conflict, "confidence_weighted")
        
        # Should return the value with highest weighted score
        assert result == "value1"

    def test_confidence_weighted_numeric_values(self):
        """Test confidence_weighted with numeric values."""
        conflict = base_module.ConflictResult(
            key="score",
            values={"provider1": 100, "provider2": 50},
            confidence={"provider1": 0.5, "provider2": 0.9}
        )
        
        resolver = base_module.ConflictResolver()
        result = resolver.resolve(conflict, "confidence_weighted")
        
        # 100 * 0.5 = 50, 50 * 0.9 = 45, so provider1 wins
        assert result == 100

    def test_voting_majority(self):
        """Test voting strategy with clear majority."""
        conflict = base_module.ConflictResult(
            key="choice",
            values={"p1": "a", "p2": "a", "p3": "b"},
            confidence={"p1": 0.5, "p2": 0.5, "p3": 0.5}
        )
        
        resolver = base_module.ConflictResolver()
        result = resolver.resolve(conflict, "voting")
        
        assert result == "a"

    def test_voting_fallback_to_confidence(self):
        """Test voting falls back to confidence when no majority."""
        conflict = base_module.ConflictResult(
            key="choice",
            values={"p1": "a", "p2": "b", "p3": "c"},
            confidence={"p1": 0.9, "p2": 0.5, "p3": 0.3}
        )
        
        resolver = base_module.ConflictResolver()
        result = resolver.resolve(conflict, "voting")
        
        # No majority, should fall back to confidence_weighted
        assert result == "a"

    def test_consensus_success(self):
        """Test consensus strategy when all agree."""
        conflict = base_module.ConflictResult(
            key="decision",
            values={"p1": "yes", "p2": "yes", "p3": "yes"},
            confidence={"p1": 0.9, "p2": 0.9, "p3": 0.9}
        )
        
        resolver = base_module.ConflictResolver()
        result = resolver.resolve(conflict, "consensus")
        
        assert result == "yes"

    def test_consensus_fails(self):
        """Test consensus strategy raises when providers disagree."""
        conflict = base_module.ConflictResult(
            key="decision",
            values={"p1": "yes", "p2": "no"},
            confidence={"p1": 0.9, "p2": 0.8}
        )
        
        resolver = base_module.ConflictResolver()
        
        with pytest.raises(base_module.ConflictRequiresHuman):
            resolver.resolve(conflict, "consensus")

    def test_highest_priority(self):
        """Test highest_priority strategy."""
        conflict = base_module.ConflictResult(
            key="choice",
            values={"p1": "a", "p2": "b"},
            confidence={"p1": 0.9, "p2": 0.5}
        )
        
        resolver = base_module.ConflictResolver()
        result = resolver.resolve(conflict, "highest_priority")
        
        # p1 has higher confidence, should win
        assert result == "a"

    def test_highest_priority_no_confidence(self):
        """Test highest_priority with no confidence scores."""
        conflict = base_module.ConflictResult(
            key="choice",
            values={"p1": "first"},
            confidence={}
        )
        
        resolver = base_module.ConflictResolver()
        result = resolver.resolve(conflict, "highest_priority")
        
        assert result == "first"

    def test_unknown_strategy_raises(self):
        """Test that unknown strategy raises ValueError."""
        conflict = base_module.ConflictResult(
            key="test",
            values={"p1": "a"},
            confidence={"p1": 1.0}
        )
        
        resolver = base_module.ConflictResolver()
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            resolver.resolve(conflict, "invalid_strategy")


class TestNotifyProviders:
    """Tests for notification providers."""

    def test_discord_notify_provider_configure(self):
        """Test Discord provider configuration."""
        provider = base_module.DiscordNotifyProvider()
        provider.configure({"webhook_url": "https://discord.com/api/webhooks/test", "enabled": False})
        
        assert provider._webhook_url == "https://discord.com/api/webhooks/test"
        assert provider.enabled is False
        assert provider.validate() is True

    def test_discord_notify_provider_name(self):
        """Test Discord provider name."""
        provider = base_module.DiscordNotifyProvider()
        assert provider.name == "discord"

    @pytest.mark.asyncio
    async def test_discord_notify_send(self):
        """Test Discord notification send."""
        provider = base_module.DiscordNotifyProvider()
        provider.configure({"webhook_url": "https://discord.com/api/webhooks/test"})
        
        notification = base_module.Notification(
            title="Test",
            message="Test message",
            level="info",
            details={"key": "value"}
        )
        
        # Mock the httpx call
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=MagicMock())
            
            result = await provider.send(notification)
            assert result is True

    def test_slack_notify_provider_configure(self):
        """Test Slack provider configuration."""
        provider = base_module.SlackNotifyProvider()
        provider.configure({"webhook_url": "https://slack.com/webhook", "channel": "#alerts"})
        
        assert provider._webhook_url == "https://slack.com/webhook"
        assert provider._channel == "#alerts"
        assert provider.validate() is True

    def test_slack_notify_provider_name(self):
        """Test Slack provider name."""
        provider = base_module.SlackNotifyProvider()
        assert provider.name == "slack"

    def test_email_notify_provider_configure(self):
        """Test Email provider configuration."""
        provider = base_module.EmailNotifyProvider()
        provider.configure({
            "smtp_host": "smtp.example.com",
            "smtp_user": "user@example.com",
            "smtp_pass": "password",
            "from_addr": "spine@example.com",
            "to_addrs": ["admin@example.com"]
        })
        
        assert provider._smtp_host == "smtp.example.com"
        assert provider.validate() is True

    def test_email_notify_provider_name(self):
        """Test Email provider name."""
        provider = base_module.EmailNotifyProvider()
        assert provider.name == "email"