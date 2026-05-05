"""Storage Provider implementations."""

from abc import abstractmethod
from typing import Any, Optional
from .base import Provider, ProviderType
import os
import shutil


class FileWriteGuard:
    """Guards file writes with reservation system and diff verification.
    
    Ensures only the reserving agent can write to reserved files,
    and optionally verifies diffs before release.
    """

    def __init__(self, resource_manager: Optional[Any] = None, storage_provider: Optional["StorageProvider"] = None):
        self._resource_manager = resource_manager
        self._storage = storage_provider
        self._write_log: list[dict] = []

    def check_reservation(self, agent_id: str, path: str) -> bool:
        """Check if agent has active reservation for path."""
        if not self._resource_manager:
            return True  # No guard if no resource manager
        
        if not self._resource_manager.is_reserved(path):
            return True  # Not reserved, allow access
        
        # Check if this agent has the reservation
        for reserved_agent, reservation in self._resource_manager._reservations.items():
            if reserved_agent == agent_id and path in reservation.paths:
                return True
        return False

    def capture_original(self, agent_id: str, path: str, contents: str) -> None:
        """Capture original content before modification for diff verification."""
        if self._resource_manager:
            self._resource_manager.capture_original(agent_id, path, contents)

    def verify_diff(self, agent_id: str, path: str, new_contents: str) -> list[str]:
        """Verify diff before writing."""
        if self._resource_manager:
            return self._resource_manager.verify_diff(agent_id, path, new_contents)
        return []

    def guarded_write(self, agent_id: str, path: str, contents: bytes) -> dict[str, Any]:
        """Write with reservation and diff verification.
        
        Returns:
            Dict with 'success', 'diff', and 'error' keys.
        """
        # Check reservation
        if not self.check_reservation(agent_id, path):
            return {
                "success": False,
                "error": f"Agent {agent_id} does not have reservation for {path}",
            }
        
        # Capture original before modification
        if self._storage and self._storage.exists(path):
            try:
                original = self._storage.read(path).decode('utf-8', errors='replace')
                self.capture_original(agent_id, path, original)
            except Exception:
                pass  # New file, no original to capture
        
        # Verify diff if we have original content
        original = ""
        if self._storage and self._storage.exists(path):
            try:
                original = self._storage.read(path).decode('utf-8', errors='replace')
            except Exception:
                pass
        
        new_content_str = contents.decode('utf-8', errors='replace')
        diff = self.verify_diff(agent_id, path, new_content_str)
        
        # Log the write
        write_record = {
            "agent": agent_id,
            "path": path,
            "timestamp": self._get_timestamp(),
            "diff": diff,
        }
        self._write_log.append(write_record)
        
        return {
            "success": True,
            "diff": diff,
            "original_length": len(original),
            "new_length": len(new_content_str),
        }

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


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