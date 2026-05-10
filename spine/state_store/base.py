"""Abstract base class for thread state storage."""

from abc import ABC, abstractmethod
from typing import Optional


class StateStoreError(Exception):
    """Base exception for state store operations."""


class ThreadNotFoundError(StateStoreError):
    """Raised when a thread is not found in the store."""


class StateStore(ABC):
    """Abstract interface for thread state persistence.

    Implementations must provide atomic read/write of thread state.
    """

    @abstractmethod
    def get_thread_ids(self) -> list[str]:
        """Return all known thread IDs."""
        ...

    @abstractmethod
    def get_state(self, thread_id: str) -> Optional[dict]:
        """Return the latest state for a thread, or None if not found."""
        ...

    @abstractmethod
    def thread_exists(self, thread_id: str) -> bool:
        """Check if a thread exists in the store."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close any open connections."""
        ...
