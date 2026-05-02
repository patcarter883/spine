"""Storage Provider implementations."""

from abc import abstractmethod
from typing import Any, Optional
from .base import Provider, ProviderType
import os
import shutil


class StorageProvider(Provider):
    """Base class for storage providers."""
    provider_type = ProviderType.STORAGE
    
    @abstractmethod
    def read(self, path: str) -> bytes:
        """Read file contents."""
        pass
    
    @abstractmethod
    def write(self, path: str, contents: bytes) -> None:
        """Write file contents."""
        pass
    
    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if path exists."""
        pass


class LocalStorageProvider(StorageProvider):
    """Local filesystem storage provider."""
    
    def __init__(self, base_path: str = ".spine"):
        self._base_path = base_path
    
    def configure(self, config: dict[str, Any]) -> None:
        self._base_path = config.get("base_path", self._base_path)
    
    def validate(self) -> bool:
        return os.path.isdir(self._base_path) if os.path.exists(self._base_path) else True
    
    @property
    def name(self) -> str:
        return f"local:{self._base_path}"
    
    @property
    def enabled(self) -> bool:
        return True
    
    def _full_path(self, path: str) -> str:
        return os.path.join(self._base_path, path)
    
    def read(self, path: str) -> bytes:
        full_path = self._full_path(path)
        with open(full_path, "rb") as f:
            return f.read()
    
    def write(self, path: str, contents: bytes) -> None:
        full_path = self._full_path(path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "wb") as f:
            f.write(contents)
    
    def exists(self, path: str) -> bool:
        return os.path.exists(self._full_path(path))