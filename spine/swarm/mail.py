"""Swarm Mail actor-model communication system."""

import json
import os
from datetime import datetime
from typing import Any, Optional, List, Dict


class _DefaultResourceManager:
    """Default ResourceManager for SwarmMail when none provided."""

    def __init__(self):
        self._reservations: dict[str, dict] = {}

    def reserve(self, agent_id: str, paths: list[str], exclusive: bool = True) -> bool:
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
        return True

    def release(self, agent_id: str) -> None:
        self._reservations.pop(agent_id, None)

    def _paths_overlap(self, path1: str, paths2: list[str]) -> bool:
        for p2 in paths2:
            if p2 in path1 or path1 in p2:
                return True
        return False


class SwarmMail:
    """Actor-model coordination with durable state.

    Provides message passing between agents with persistent event logging
    to .spine/events/swarm.log as JSONL.
    """

    def __init__(self, agent_id: str, event_path: str = ".spine/events", resource_manager: Optional[Any] = None):
        self.agent_id = agent_id
        self.event_path = event_path
        self.resource_manager = resource_manager or _DefaultResourceManager()
        os.makedirs(event_path, exist_ok=True)
        self._log_file = os.path.join(event_path, "swarm.log")
    
    def handle_plan_for_review(self, plan: dict[str, Any], critic: Any) -> dict[str, Any]:
        """Process a PLAN_FOR_REVIEW message and get critic review."""
        result = critic.execute({"plan": plan}, "review")
        return {
            "type": "PLAN_REVIEWED",
            "plan_id": plan.get("id"),
            "approved": result.get("approved", False),
            "issues": result.get("issues", []),
            "from": self.agent_id
        }
    
    def handle_task_assignment(self, task_id: str, task_data: dict[str, Any], worker: Any) -> None:
        """Process a TASK_ASSIGNMENT message and dispatch to worker."""
        pass
    
    def get_unread_messages(self, processed_ids: Optional[set] = None) -> List[Dict[str, Any]]:
        """Get unread messages, optionally filtered by already-processed IDs."""
        messages = self.inbox()
        if processed_ids:
            return [m for m in messages if m.get("id") not in processed_ids]
        return messages

    def _log_event(self, event: Dict[str, Any]) -> None:
        """Append event to JSONL log file."""
        event["timestamp"] = datetime.now().isoformat()
        with open(self._log_file, "a") as f:
            f.write(json.dumps(event) + "\n")

    def send(self, to: str, subject: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Send message between agents, persisted to .spine/events/.

        Args:
            to: Target agent role (e.g., "planner", "coder")
            subject: Message subject for routing
            body: Message payload

        Returns:
            The logged event dictionary
        """
        event = {
            "type": "message_sent",
            "from": self.agent_id,
            "to": to,
            "subject": subject,
            "body": body,
        }
        self._log_event(event)
        return event

    def broadcast(self, subject: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Broadcast message to all agents, persisted to .spine/events/.

        Args:
            subject: Message subject for routing
            body: Message payload

        Returns:
            The logged event dictionary
        """
        event = {
            "type": "message_broadcast",
            "from": self.agent_id,
            "subject": subject,
            "body": body,
        }
        self._log_event(event)
        return event

    def reserve(self, paths: List[str], exclusive: bool = True) -> Dict[str, Any]:
        """File reservations to prevent conflicts.

        Integrates with ResourceManager to coordinate file access.

        Args:
            paths: List of file paths to reserve
            exclusive: Whether reservation is exclusive (default True)

        Returns:
            Reservation lock dictionary
        """
        lock = {
            "paths": paths,
            "agent": self.agent_id,
            "exclusive": exclusive,
        }

        success = self.resource_manager.reserve(self.agent_id, paths, exclusive)

        event = {
            "type": "file_reserved" if success else "reservation_failed",
            "agent": self.agent_id,
            "paths": paths,
            "exclusive": exclusive,
            "success": success,
        }
        self._log_event(event)

        if success:
            return lock
        return {"error": "Reservation conflict", "paths": paths}

    def release(self) -> None:
        """Release all reservations for this agent."""
        self.resource_manager.release(self.agent_id)
        event = {
            "type": "reservations_released",
            "agent": self.agent_id,
        }
        self._log_event(event)

    def inbox(self, agent: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get messages for a specific agent.

        Args:
            agent: Agent role to filter by (default: current agent)

        Returns:
            List of message events addressed to the agent
        """
        target = agent or self.agent_id
        messages = []

        if not os.path.exists(self._log_file):
            return messages

        with open(self._log_file, "r") as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    if event.get("to") == target or event.get("type") == "message_broadcast":
                        messages.append(event)
                except json.JSONDecodeError:
                    continue

        return messages

    def get_events(self, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all events, optionally filtered by type.

        Args:
            event_type: Filter by event type (e.g., "message_sent")

        Returns:
            List of matching events
        """
        events = []

        if not os.path.exists(self._log_file):
            return events

        with open(self._log_file, "r") as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    if event_type is None or event.get("type") == event_type:
                        events.append(event)
                except json.JSONDecodeError:
                    continue

        return events