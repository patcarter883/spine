"""Base provider interfaces."""

import importlib
from collections import Counter
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Type


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
        """Weight results by provider confidence scores."""
        weighted = {
            name: self._compute_weight(val, conflict.confidence.get(name, 0.0))
            for name, val in conflict.values.items()
        }
        return max(weighted.items(), key=lambda x: x[1])[0]

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