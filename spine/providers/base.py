"""Base provider interfaces."""

import importlib
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