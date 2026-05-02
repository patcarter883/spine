"""Resource management for parallel execution."""

import json
import os
from typing import Optional, Any
from datetime import datetime


class ResourceManager:
    """
    Manages file reservations for parallel sub-phases.
    
    Ensures no two agents modify the same files simultaneously,
    broadcasting FILE_RESERVED events via SwarmMail for coordination.
    """
    
    def __init__(self, path: str = ".spine/state/hive", swarm_mail: Optional[Any] = None):
        self.path = path
        self.swarm_mail = swarm_mail
        self._reservations: dict[str, dict] = {}
        os.makedirs(path, exist_ok=True)
    
    def reserve(self, agent_id: str, paths: list[str], exclusive: bool = True) -> bool:
        """Reserve paths for an agent."""
        # Check for conflicts
        for path in paths:
            for reserved_agent, reservation in self._reservations.items():
                if agent_id != reserved_agent and reservation.get("exclusive"):
                    if self._paths_overlap(path, reservation.get("paths", [])):
                        return False
        
        self._reservations[agent_id] = {
            "paths": paths,
            "exclusive": exclusive,
            "reserved_at": datetime.now().isoformat()
        }
        
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
    
    def release(self, agent_id: str) -> None:
        """Release reservations for an agent."""
        self._reservations.pop(agent_id, None)
    
    def is_reserved(self, path: str) -> bool:
        """Check if a path is reserved."""
        return any(
            path in res.get("paths", []) 
            for res in self._reservations.values()
        )
    
    def _paths_overlap(self, path1: str, paths2: list[str]) -> bool:
        """Check if paths overlap (simplified glob matching)."""
        # Simplified - just check if any path contains the other
        for p2 in paths2:
            if p2 in path1 or path1 in p2:
                return True
        return False