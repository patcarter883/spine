"""Semantic memory system with embeddings for pattern similarity and contextual recall."""

import json
import os
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Optional, List, Dict


@dataclass
class Memory:
    """A semantic memory with optional embedding vector."""
    memory_id: str
    content: str
    context: str = ""
    embedding: Optional[List[float]] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Memory":
        return cls(**data)


class Hivemind:
    """
    Semantic memory store with embedding-based similarity search.

    Provides pattern similarity matching and contextual recall for
    swarm coordination and learning systems.
    """

    def __init__(self, memory_path: str = ".spine/memory"):
        self.memory_path = memory_path
        self.memories: Dict[str, Memory] = {}
        self._ensure_memory_dir()

    def _ensure_memory_dir(self) -> None:
        """Ensure memory directory exists."""
        os.makedirs(self.memory_path, exist_ok=True)

    def _load_memories(self) -> None:
        """Load memories from disk."""
        memories_file = os.path.join(self.memory_path, "memories.json")
        if os.path.exists(memories_file):
            with open(memories_file, "r") as f:
                data = json.load(f)
                for mem_data in data.get("memories", []):
                    memory = Memory.from_dict(mem_data)
                    self.memories[memory.memory_id] = memory

    def _save_memories(self) -> None:
        """Save memories to disk."""
        memories_file = os.path.join(self.memory_path, "memories.json")
        data = {
            "version": "1.0",
            "memories": [m.to_dict() for m in self.memories.values()]
        }
        with open(memories_file, "w") as f:
            json.dump(data, f, indent=2)

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = math.sqrt(sum(x * x for x in a))
        magnitude_b = math.sqrt(sum(x * x for x in b))
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        return dot_product / (magnitude_a * magnitude_b)

    def add_memory(
        self,
        content: str,
        context: str = "",
        memory_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Memory:
        """
        Add a memory to the semantic store.

        Args:
            content: The memory content
            context: Optional context for recall
            memory_id: Optional ID (auto-generated if not provided)
            embedding: Optional embedding vector for similarity search
            tags: Optional tags for categorization
            metadata: Optional metadata dictionary

        Returns:
            The created Memory object
        """
        mid = memory_id or f"mem_{len(self.memories) + 1:04d}"
        memory = Memory(
            memory_id=mid,
            content=content,
            context=context,
            embedding=embedding,
            tags=tags or [],
            metadata=metadata or {}
        )
        self.memories[mid] = memory
        self._save_memories()
        return memory

    def query_similarity(
        self,
        query: str,
        embedding: Optional[List[float]] = None,
        threshold: float = 0.5,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Query memories by embedding similarity.

        Args:
            query: Query text (used as fallback when no embedding)
            embedding: Query embedding vector for similarity search
            threshold: Minimum similarity score (0.0-1.0)
            limit: Maximum number of results

        Returns:
            List of matching memories with similarity scores
        """
        results = []

        for memory in self.memories.values():
            if memory.embedding is None:
                if query.lower() in memory.content.lower():
                    results.append({
                        "memory": memory,
                        "score": 1.0
                    })
            elif embedding is not None:
                score = self._cosine_similarity(embedding, memory.embedding)
                if score >= threshold:
                    results.append({
                        "memory": memory,
                        "score": score
                    })

        results.sort(key=lambda x: x["score"], reverse=True)  # type: ignore[arg-type,return-value]
        return results[:limit]

    def get_insights(
        self,
        tags: Optional[List[str]] = None,
        limit: int = 20
    ) -> List[Memory]:
        """
        Get insights from stored memories.

        Args:
            tags: Optional filter by tags
            limit: Maximum number of insights to return

        Returns:
            List of relevant memories
        """
        memories = list(self.memories.values())

        if tags:
            memories = [m for m in memories if any(t in m.tags for t in tags)]

        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories[:limit]

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        """Get a memory by ID."""
        return self.memories.get(memory_id)

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        if memory_id in self.memories:
            del self.memories[memory_id]
            self._save_memories()
            return True
        return False

    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        context: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Memory]:
        """Update an existing memory."""
        memory = self.memories.get(memory_id)
        if memory is None:
            return None

        if content is not None:
            memory.content = content
        if context is not None:
            memory.context = context
        if embedding is not None:
            memory.embedding = embedding
        if tags is not None:
            memory.tags = tags
        if metadata is not None:
            memory.metadata = metadata

        self._save_memories()
        return memory

    def clear(self) -> None:
        """Clear all memories."""
        self.memories.clear()
        self._save_memories()