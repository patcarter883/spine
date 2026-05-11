"""Base provider interfaces."""

import importlib
from collections import Counter
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Type, AsyncIterator


@dataclass
class ProviderConfig:
    """Configuration for a provider instance."""

    name: str
    type: str
    enabled: bool = True
    priority: int = 0
    config: dict[str, Any] = field(default_factory=dict)


class ProviderType(str, Enum):
    """Types of providers in SPINE."""
    LLM = "llm"
    MEMORY = "memory"
    TOOLS = "tools"
    STORAGE = "storage"
    NOTIFY = "notify"
    AGENT = "agent"
    DEEPAGENTS_MODEL = "deepagents-model"


class Provider(ABC):
    """Base interface for all pluggable providers."""
    
    @abstractmethod
    def configure(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Health check."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return provider name for logging."""
        pass
    
    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Return whether provider is enabled."""
        pass
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt. Default sync fallback."""
        return self._generate_sync(prompt, **kwargs)
    
    async def stream(self, prompt: str, ttfb_timeout: float = 30.0, **kwargs) -> AsyncIterator[str]:
        """Stream text generation. Default yields entire response.
        
        Args:
            prompt: Text prompt to generate from.
            ttfb_timeout: Timeout in seconds for the first chunk.
            **kwargs: Additional provider-specific arguments.
        """
        yield await self.generate(prompt, **kwargs)
    
    def _generate_sync(self, prompt: str, **kwargs) -> str:
        """Sync fallback for providers that don't implement async."""
        raise NotImplementedError("Provider must implement generate() or _generate_sync()")


@dataclass
class Notification:
    """Multi-channel notification data."""
    title: str
    message: str
    level: str = "info"
    details: dict[str, Any] = field(default_factory=dict)
    actions: list[dict[str, str]] = field(default_factory=list)


class NotifyProvider(Provider):
    """Multi-channel notifications (email, slack, webhook, etc.)."""

    provider_type = ProviderType.NOTIFY

    @abstractmethod
    async def send(self, notification: Notification) -> bool:
        """Send a notification via the configured channel."""
        pass

    @abstractmethod
    async def ask(self, question: str, options: list[str]) -> str:
        """Request human input via notification channel."""
        pass


class ProviderRegistry:
    """Registry for all providers."""

    def __init__(self):
        self._providers: dict[str, Provider] = {}

    def register(self, name: str, provider: Provider) -> None:
        """Register a provider."""
        self._providers[name] = provider

    def get(self, name: str) -> Optional[Provider]:
        """Get a provider by name."""
        return self._providers.get(name)

    def get_all(self, provider_type: Optional[ProviderType] = None) -> list[Provider]:
        """Get all providers, optionally filtered by type."""
        if provider_type is None:
            return list(self._providers.values())
        return [p for p in self._providers.values() if hasattr(p, "provider_type")
                and p.provider_type == provider_type]

    def load_providers(self, config_path: str = "spine.yaml") -> list[ProviderConfig]:
        """Load providers from YAML config file."""
        path = Path(config_path)
        if not path.exists():
            return []

        with open(path) as f:
            config = yaml.safe_load(f) or {}

        instances: list[ProviderConfig] = []
        for provider_type, provider_list in config.get("providers", {}).items():
            for instance in provider_list:
                cfg = ProviderConfig(
                    name=instance["name"],
                    type=instance["type"],
                    enabled=instance.get("enabled", True),
                    priority=instance.get("priority", 0),
                    config=instance.get("config", {}),
                )
                instances.append(cfg)

        return instances

    def register_factory(self, provider_type: str, factory: Type["Provider"]) -> None:
        """Register a provider factory for dynamic instantiation."""
        if not hasattr(self, "_factories"):
            self._factories: dict[str, Type[Provider]] = {}
        self._factories[provider_type] = factory

    def get_factory(self, provider_type: str) -> Optional[Type["Provider"]]:
        """Get a registered factory by provider type."""
        return getattr(self, "_factories", {}).get(provider_type)


class PluginLoader:
    """Discover and load provider plugins from spine-plugin.yaml manifests."""

    def __init__(self, plugin_dirs: list[str] | None = None):
        self.plugin_dirs = plugin_dirs or ["./spine-plugins"]
        self.registry = ProviderRegistry()

    def discover_plugins(self) -> None:
        """Find and load all plugins from configured directories."""
        for plugin_dir in self.plugin_dirs:
            manifest_path = Path(plugin_dir) / "spine-plugin.yaml"
            if manifest_path.exists():
                self._load_plugin(manifest_path)

    def _load_plugin(self, manifest_path: Path) -> None:
        """Load a plugin from its manifest file."""
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f) or {}

        for provider_def in manifest.get("providers", []):
            module_path, class_name = provider_def["class"].rsplit(".", 1)
            module = importlib.import_module(module_path)
            provider_class = getattr(module, class_name)

            self.registry.register_factory(provider_def["type"], provider_class)

    def register_factory(self, provider_type: str, factory: Type[Provider]) -> None:
        """Register a provider factory directly."""
        self.registry.register_factory(provider_type, factory)


class ProviderFallbackChain:
    """Manages multiple providers with priority-based fallback.
    
    Routes requests through providers in priority order, automatically
    failing over on errors or health check failures.
    """
    
    def __init__(self, providers: Optional[list[Provider]] = None, config_path: Optional[str] = None):
        """Initialize the fallback chain.
        
        Args:
            providers: List of providers, sorted by priority (highest first).
            config_path: Optional path for async config loading.
        """
        self._providers: list[Provider] = []
        self._healthy_cache: dict[str, tuple[bool, float]] = {}
        self._cache_ttl = 30.0  # seconds
        self._provider_type: Optional[str] = None
        
        if providers:
            self.add_providers(providers)
        
        if config_path:
            self._config_path = config_path
    
    def add_providers(self, providers: list[Provider]) -> None:
        """Add providers to the chain, sorted by priority."""
        for provider in providers:
            if hasattr(provider, "priority"):
                self._providers.append(provider)
            else:
                self._providers.append(provider)
        # Sort by enabled status first, then by priority (assuming priority attr or just use order)
        self._providers = [p for p in self._providers if getattr(p, "enabled", True)]
    
    @property
    def active_provider(self) -> Optional[Provider]:
        """Return the currently active healthy provider."""
        import time
        for provider in self._providers:
            name = provider.name
            cached = self._healthy_cache.get(name)
            if cached and time.time() - cached[1] < self._cache_ttl:
                if cached[0]:
                    return provider
            if self._check_health(provider):
                self._healthy_cache[name] = (True, time.time())
                return provider
        return None
    
    def _check_health(self, provider: Provider) -> bool:
        """Check if a provider is healthy."""
        try:
            return provider.validate()
        except Exception:
            return False
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate using the best available provider with fallback."""
        import asyncio
        for provider in self._providers:
            if not provider.enabled:
                continue
            try:
                # Try async generate first, fall back to sync
                if hasattr(provider, 'generate_async'):
                    return await provider.generate_async(prompt, **kwargs)
                # Run sync generate in executor
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: provider.generate(prompt, **kwargs))
            except Exception:
                # Mark provider as unhealthy
                self._healthy_cache[provider.name] = (False, 0)
                continue
        raise RuntimeError("All providers failed to generate")
    
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """Stream using the active provider.

        Passes through all kwargs (including *ttfb_timeout*) to the
        underlying provider's ``stream()`` method.
        """
        provider = self.active_provider
        if not provider:
            raise RuntimeError("No healthy providers available")
        async for chunk in provider.stream(prompt, **kwargs):
            yield chunk
    
    async def load_config(self, config_path: str) -> list[ProviderConfig]:
        """Async load providers from YAML config file.
        
        Args:
            config_path: Path to the YAML configuration file.
            
        Returns:
            List of ProviderConfig objects.
        """
        import asyncio
        path = Path(config_path)
        
        def _load_yaml():
            if not path.exists():
                return []
            with open(path) as f:
                config = yaml.safe_load(f) or {}
            instances: list[ProviderConfig] = []
            for provider_type, provider_list in config.get("providers", {}).items():
                for instance in provider_list:
                    cfg = ProviderConfig(
                        name=instance["name"],
                        type=instance["type"],
                        enabled=instance.get("enabled", True),
                        priority=instance.get("priority", 0),
                        config=instance.get("config", {}),
                    )
                    instances.append(cfg)
            return instances
        
        return await asyncio.to_thread(_load_yaml)


@dataclass
class ConflictResult:
    """Result from multiple providers with potentially conflicting values."""

    key: str
    values: dict[str, Any]
    confidence: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


class ConflictRequiresHuman(Exception):
    """Raised when conflict cannot be automatically resolved."""

    def __init__(self, key: str, message: str = ""):
        self.key = key
        super().__init__(message or f"Conflict on {key} requires human resolution")


class ConflictResolver:
    """Resolves conflicts between provider results using weighted strategies."""

    def resolve(
        self, conflict: ConflictResult, strategy: str = "confidence_weighted"
    ) -> Any:
        """Apply resolution strategy to conflict."""
        strategies = {
            "confidence_weighted": self._confidence_weighted,
            "voting": self._voting,
            "consensus": self._consensus,
            "highest_priority": self._highest_priority,
        }
        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        return strategies[strategy](conflict)

    def _confidence_weighted(self, conflict: ConflictResult) -> Any:
        """Weight results by provider confidence scores.
        
        Returns the value with the highest weighted score, not the provider name.
        """
        weighted = {
            name: self._compute_weight(val, conflict.confidence.get(name, 0.0))
            for name, val in conflict.values.items()
        }
        # Get the provider name with max weight, then return their value
        winner_name = max(weighted.items(), key=lambda x: x[1])[0]
        return conflict.values[winner_name]

    def _compute_weight(self, value: Any, confidence: float) -> float:
        """Compute weight for a value based on confidence."""
        if isinstance(value, (int, float)):
            return float(value) * confidence
        return confidence

    def _voting(self, conflict: ConflictResult) -> Any:
        """Simple majority vote among providers."""
        counts = Counter(conflict.values.values())
        winner, count = counts.most_common(1)[0]
        if count > len(conflict.values) / 2:
            return winner
        return self._confidence_weighted(conflict)

    def _consensus(self, conflict: ConflictResult) -> Any:
        """Require all providers to agree, otherwise escalate."""
        unique_values = set(conflict.values.values())
        if len(unique_values) == 1:
            return list(unique_values)[0]
        raise ConflictRequiresHuman(conflict.key)

    def _highest_priority(self, conflict: ConflictResult) -> Any:
        """Trust provider with highest priority (highest confidence score)."""
        if not conflict.confidence:
            return list(conflict.values.values())[0]
        winner = max(conflict.confidence.keys(), key=lambda k: conflict.confidence[k])
        return conflict.values[winner]


class DiscordNotifyProvider(NotifyProvider):
    """Discord webhook notification provider."""

    def __init__(self):
        self._webhook_url: str | None = None
        self._enabled_flag = True

    def configure(self, config: dict[str, Any]) -> None:
        self._webhook_url = config.get("webhook_url")
        self._enabled_flag = config.get("enabled", True)

    def validate(self) -> bool:
        return bool(self._webhook_url)

    @property
    def name(self) -> str:
        return "discord"

    @property
    def enabled(self) -> bool:
        return self._enabled_flag

    async def send(self, notification: Notification) -> bool:
        import httpx

        color_map = {
            "info": 0x3498db,
            "warning": 0xf39c12,
            "error": 0xe74c3c,
            "success": 0x2ecc71,
        }
        color = color_map.get(notification.level, 0x95a5a6)

        async with httpx.AsyncClient() as client:
            await client.post(
                self._webhook_url,
                json={
                    "embeds": [
                        {
                            "title": notification.title,
                            "description": notification.message,
                            "color": color,
                            "fields": [
                                {"name": k, "value": str(v), "inline": True}
                                for k, v in notification.details.items()
                            ]
                            if notification.details
                            else [],
                        }
                    ]
                },
            )
        return True

    async def ask(self, question: str, options: list[str]) -> str:
        await self.send(
            Notification(
                title="Question",
                message=question,
                actions=[{"label": opt, "value": opt} for opt in options],
            )
        )
        raise NotImplementedError("Interactive question requires additional setup")


class SlackNotifyProvider(NotifyProvider):
    """Slack webhook notification provider."""

    def __init__(self):
        self._webhook_url: str | None = None
        self._enabled_flag = True
        self._channel: str | None = None

    def configure(self, config: dict[str, Any]) -> None:
        self._webhook_url = config.get("webhook_url")
        self._enabled_flag = config.get("enabled", True)
        self._channel = config.get("channel", "#general")

    def validate(self) -> bool:
        return bool(self._webhook_url)

    @property
    def name(self) -> str:
        return "slack"

    @property
    def enabled(self) -> bool:
        return self._enabled_flag

    async def send(self, notification: Notification) -> bool:
        import httpx

        color_map = {
            "info": "#3498db",
            "warning": "#f39c12",
            "error": "#e74c3c",
            "success": "#2ecc71",
        }
        color = color_map.get(notification.level, "#95a5a6")

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": notification.title, "emoji": True}
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": notification.message}
            }
        ]
        
        # Add detail fields
        for key, value in notification.details.items():
            blocks.append({
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*{key}:*"},
                    {"type": "mrkdwn", "text": str(value)}
                ]
            })

        async with httpx.AsyncClient() as client:
            await client.post(
                self._webhook_url,
                json={
                    "channel": self._channel,
                    "attachments": [{
                        "color": color,
                        "blocks": blocks
                    }]
                },
            )
        return True

    async def ask(self, question: str, options: list[str]) -> str:
        await self.send(
            Notification(
                title="Question",
                message=question,
                actions=[{"label": opt, "value": opt} for opt in options],
            )
        )
        raise NotImplementedError("Interactive question requires Slack app configuration")


class EmailNotifyProvider(NotifyProvider):
    """Email notification provider via SMTP."""

    def __init__(self):
        self._smtp_host: str | None = None
        self._smtp_port: int = 587
        self._smtp_user: str | None = None
        self._smtp_pass: str | None = None
        self._from_addr: str | None = None
        self._to_addrs: list[str] = []
        self._enabled_flag = True

    def configure(self, config: dict[str, Any]) -> None:
        self._smtp_host = config.get("smtp_host")
        self._smtp_port = config.get("smtp_port", 587)
        self._smtp_user = config.get("smtp_user")
        self._smtp_pass = config.get("smtp_pass")
        self._from_addr = config.get("from_addr")
        self._to_addrs = config.get("to_addrs", [])
        self._enabled_flag = config.get("enabled", True)

    def validate(self) -> bool:
        return all([self._smtp_host, self._smtp_user, self._smtp_pass, self._from_addr, self._to_addrs])

    @property
    def name(self) -> str:
        return "email"

    @property
    def enabled(self) -> bool:
        return self._enabled_flag

    async def send(self, notification: Notification) -> bool:
        import aiosmtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        subject = f"[{notification.level.upper()}] {notification.title}"
        
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self._from_addr
        msg["To"] = ", ".join(self._to_addrs)

        # Format message body
        body = notification.message
        if notification.details:
            body += "\n\nDetails:\n" + "\n".join(f"  {k}: {v}" for k, v in notification.details.items())

        msg.attach(MIMEText(body, "plain"))

        await aiosmtplib.send(
            msg,
            hostname=self._smtp_host,
            port=self._smtp_port,
            username=self._smtp_user,
            password=self._smtp_pass,
            start_tls=True,
        )
        return True

    async def ask(self, question: str, options: list[str]) -> str:
        await self.send(
            Notification(
                title="Question",
                message=question + "\n\nReply with your choice: " + ", ".join(options),
            )
        )
        # Email doesn't support interactive response - would need IMAP polling
        raise NotImplementedError("Interactive question requires IMAP polling setup")