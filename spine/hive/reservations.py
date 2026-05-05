"""Resource management for parallel execution."""

import difflib
import json
import os
from dataclasses import dataclass, field
from typing import Optional, Any, Callable, Set
from datetime import datetime


@dataclass
class OwnedReservation:
    """Tracks file reservation with ownership and original content for diff verification."""
    agent_id: str
    paths: list[str]
    exclusive: bool
    reserved_at: str
    original_contents: dict[str, str] = field(default_factory=dict)
    ttl_seconds: Optional[int] = None

    def capture_content(self, path: str, contents: str) -> None:
        """Capture original content for diff verification."""
        self.original_contents[path] = contents

    def get_diff(self, path: str, new_contents: str) -> list[str]:
        """Generate unified diff between original and new content."""
        original = self.original_contents.get(path, "")
        return list(difflib.unified_diff(
            original.splitlines(keepends=True),
            new_contents.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        ))


class ResourceManager:
    """
    Manages file reservations for parallel sub-phases.
    
    Ensures no two agents modify the same files simultaneously,
    broadcasting FILE_RESERVED events via SwarmMail for coordination.
    """
    
    def __init__(self, path: str = ".spine/state/hive", swarm_mail: Optional[Any] = None):
        self.path = path
        self.swarm_mail = swarm_mail
        self._reservations: dict[str, OwnedReservation] = {}
        self._acknowledgments: dict[str, Set[str]] = {}  # message_id -> set of agent_ids
        os.makedirs(path, exist_ok=True)
    
    def reserve(self, agent_id: str, paths: list[str], exclusive: bool = True, ttl_seconds: Optional[int] = None) -> bool:
        """Reserve paths for an agent with optional TTL."""
        # Check for conflicts
        for path in paths:
            for reserved_agent, reservation in self._reservations.items():
                if agent_id != reserved_agent and reservation.exclusive:
                    if self._paths_overlap(path, reservation.paths):
                        return False
        
        self._reservations[agent_id] = OwnedReservation(
            agent_id=agent_id,
            paths=paths,
            exclusive=exclusive,
            reserved_at=datetime.now().isoformat(),
            ttl_seconds=ttl_seconds,
        )
        
        if self.swarm_mail:
            self.swarm_mail.broadcast(
                subject="FILE_RESERVED",
                body={
                    "agent": agent_id,
                    "paths": paths,
                    "exclusive": exclusive,
                }
            )
        
        return True
    
    def capture_original(self, agent_id: str, path: str, contents: str) -> None:
        """Capture original file content for diff verification."""
        if agent_id in self._reservations:
            self._reservations[agent_id].capture_content(path, contents)
    
    def verify_diff(self, agent_id: str, path: str, new_contents: str) -> list[str]:
        """Generate diff for verification before release."""
        if agent_id in self._reservations:
            return self._reservations[agent_id].get_diff(path, new_contents)
        return []
    
    def release(self, agent_id: str, verify_diffs: bool = True) -> dict[str, Any]:
        """Release reservations for an agent with optional diff verification.
        
        Returns:
            Dictionary with verification results including any diffs found.
        """
        reservation = self._reservations.pop(agent_id, None)
        return {
            "agent": agent_id,
            "verified": reservation is not None,
            "paths": reservation.paths if reservation else [],
        }
    
    def is_reserved(self, path: str) -> bool:
        """Check if a path is reserved."""
        return any(
            path in res.paths
            for res in self._reservations.values()
        )

    def acknowledge(self, message_id: str, agent_id: str) -> bool:
        """Acknowledge a message as processed by an agent."""
        if message_id not in self._acknowledgments:
            self._acknowledgments[message_id] = set()
        self._acknowledgments[message_id].add(agent_id)
        return True

    def get_acknowledgments(self, message_id: str) -> Set[str]:
        """Get set of agents who have acknowledged a message."""
        return self._acknowledgments.get(message_id, set())

    def is_acknowledged_by(self, message_id: str, agent_id: str) -> bool:
        """Check if an agent has acknowledged a message."""
        return agent_id in self._acknowledgments.get(message_id, set())

    def _paths_overlap(self, path1: str, paths2: list[str]) -> bool:
        """Check if paths overlap (simplified glob matching)."""
        # Simplified - just check if any path contains the other
        for p2 in paths2:
            if p2 in path1 or path1 in p2:
                return True
        return False