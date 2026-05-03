"""Memory Provider implementations."""

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from .base import Provider, ProviderType
import json
import os


@dataclass
class MemoryEntry:
    """Dataclass for memory persistence layer."""
    key: str
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    ttl: Optional[int] = None
    embeddings: Optional[List[float]] = None


class MemoryProvider(Provider):
    """Base class for memory providers."""
    provider_type = ProviderType.MEMORY
    
    @abstractmethod
    def get(self, key: str) -> Any:
        """Get value by key."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set value by key."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete value by key."""
        pass


class SQLiteProvider(MemoryProvider):
    """SQLite-based memory provider."""
    
    def __init__(self, db_path: str = ".spine/memory.db"):
        self._db_path = db_path
        self._conn = None
    
    def configure(self, config: dict[str, Any]) -> None:
        import sqlite3
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.commit()
    
    def validate(self) -> bool:
        return self._conn is not None
    
    @property
    def name(self) -> str:
        return f"sqlite:{self._db_path}"
    
    @property
    def enabled(self) -> bool:
        return self._conn is not None
    
    def get(self, key: str) -> Any:
        cursor = self._conn.execute("SELECT value FROM memory WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            return json.loads(row[0])
        return None
    
    def set(self, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO memory (key, value) VALUES (?, ?)",
            (key, json.dumps(value))
        )
        self._conn.commit()
    
    def delete(self, key: str) -> None:
        self._conn.execute("DELETE FROM memory WHERE key = ?", (key,))
        self._conn.commit()