"""Thread model with UUID4-based ID generation."""

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


def generate_thread_id() -> str:
    """Generate a unique thread ID using UUID4."""
    return str(uuid.uuid4())


@dataclass
class Thread:
    """Represents a workflow thread with a unique UUID4 identifier.

    Attributes:
        thread_id: UUID4 string (auto-generated if not provided).
        created_at: ISO 8601 timestamp of creation.
        requirement: The requirement/description for this thread.
        status: Current thread status (INIT, PLANNING, EXECUTION, etc.).
        metadata: Optional dictionary for extensible attributes.
    """
    thread_id: str = field(default_factory=generate_thread_id)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    requirement: str = ""
    status: str = "INIT"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.thread_id:
            self.thread_id = generate_thread_id()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Thread":
        return cls(
            thread_id=data.get("thread_id", generate_thread_id()),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            requirement=data.get("requirement", ""),
            status=data.get("status", "INIT"),
            metadata=data.get("metadata", {}),
        )


__all__ = ["Thread", "generate_thread_id"]
