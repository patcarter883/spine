"""Swarm Mail actor-model communication system."""

import json
import os
from datetime import datetime
from typing import Any, Optional, List, Dict, Iterator

# Message type constants
MESSAGE_SENT = "message_sent"
MESSAGE_BROADCAST = "message_broadcast"


class _DefaultResourceManager:
    """Default ResourceManager for SwarmMail when none provided."""

    def __init__(self, path: Optional[str] = None):
        self._reservations: dict[str, dict] = {}
        self._path = path

    def reserve(self, agent_id: str, paths: list[str], exclusive: bool = True, ttl_seconds: Optional[int] = None) -> bool:
        for path in paths:
            for reserved_agent, reservation in self._reservations.items():
                if agent_id != reserved_agent and reservation.get("exclusive"):
                    if self._paths_overlap(path, reservation.get("paths", [])):
                        return False
        self._reservations[agent_id] = {
            "paths": paths,
            "exclusive": exclusive,
            "reserved_at": datetime.now().isoformat(),
            "ttl_seconds": ttl_seconds,
        }
        return True

    def acknowledge(self, message_id: str, agent_id: str) -> bool:
        if not hasattr(self, "_acknowledgments"):
            self._acknowledgments = {}
        if message_id not in self._acknowledgments:
            self._acknowledgments[message_id] = set()
        self._acknowledgments[message_id].add(agent_id)
        return True

    def get_acknowledgments(self, message_id: str):
        return self._acknowledgments.get(message_id, set()) if hasattr(self, "_acknowledgments") else set()

    def is_acknowledged_by(self, message_id: str, agent_id: str) -> bool:
        return agent_id in self.get_acknowledgments(message_id)

    def release(self, agent_id: str) -> None:
        self._reservations.pop(agent_id, None)

    def _paths_overlap(self, path1: str, paths2: list[str]) -> bool:
        for p2 in paths2:
            if p2 in path1 or path1 in p2:
                return True
        return False


# Export ResourceManager for backwards compatibility
ResourceManager = _DefaultResourceManager


class SwarmMail:
    """Actor-model coordination with durable state.

    Provides message passing between agents with persistent event logging
    to .spine/events/swarm.log as JSONL.
    """

    def __init__(self, agent_id: str, event_path: str = ".spine/events", resource_manager: Optional[Any] = None, learning_manager: Optional[Any] = None):
        self.agent_id = agent_id
        self.event_path = event_path
        self.resource_manager = resource_manager or ResourceManager()
        self.learning_manager = learning_manager
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
        event["_id"] = f"{event.get('type', 'unknown')}_{event.get('timestamp', '')}"
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

    def inbox(self, agent: Optional[str] = None, include_acknowledged: bool = False) -> List[Dict[str, Any]]:
        """Get messages for a specific agent.
        
        Args:
            agent: Agent role to filter by (default: current agent)
            include_acknowledged: Whether to include already-acknowledged messages
            
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
                        # Exclude acknowledged messages unless requested
                        msg_id = event.get("_id")
                        if msg_id and not include_acknowledged and self.is_acknowledged(msg_id):
                            continue
                        messages.append(event)
                except json.JSONDecodeError:
                    continue
        
        return messages

    def get_events(self, event_type: Optional[str] = None, **filters) -> List[Dict[str, Any]]:
        """Get all events, optionally filtered by type and other fields.

        Args:
            event_type: Filter by event type (e.g., "message_sent")
            **filters: Additional field filters (e.g., to_agent="planner", subject="test")

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
                    if event_type is not None and event.get("type") != event_type:
                        continue
                    # Apply additional filters
                    if filters:
                        match = True
                        for key, value in filters.items():
                            field_value = event.get("to") if key == "to_agent" else event.get("from") if key == "from_agent" else event.get(key)
                            if field_value != value:
                                match = False
                                break
                        if not match:
                            continue
                    events.append(event)
                except json.JSONDecodeError:
                    continue

        return events
    
    def acknowledge(self, message_id: str) -> Dict[str, Any]:
        """Acknowledge receipt of a message.
        
        Args:
            message_id: The _id of the message to acknowledge
            
        Returns:
            The acknowledgment event
        """
        event = {
            "type": "message_acknowledged",
            "message_id": message_id,
            "by": self.agent_id,
        }
        self._log_event(event)
        return event
    
    def is_acknowledged(self, message_id: str) -> bool:
        """Check if a message has been acknowledged by this agent."""
        ack_events = self.get_events(event_type="message_acknowledged")
        for ev in ack_events:
            if ev.get("message_id") == message_id and ev.get("by") == self.agent_id:
                return True
        return False
    
    def get_acknowledgments(self, message_id: str) -> set:
        """Get all agents who acknowledged a message."""
        acks = set()
        ack_events = self.get_events(event_type="message_acknowledged")
        for ev in ack_events:
            if ev.get("message_id") == message_id:
                acks.add(ev.get("by"))
        return acks
    
    def replay_from(self, position: int = 0, **kwargs) -> Iterator[Dict[str, Any]]:
        """Replay events from a given position.
        
        Args:
            position: Line number to start from (0-indexed)
            **kwargs: Additional filters (event_type, etc.)
            
        Yields:
            Event dictionaries
        """
        if not os.path.exists(self._log_file):
            return
            
        with open(self._log_file, "r") as f:
            for i, line in enumerate(f):
                if i < position:
                    continue
                try:
                    event = json.loads(line.strip())
                    event_type = kwargs.get("event_type")
                    if event_type is None or event.get("type") == event_type:
                        yield event
                except json.JSONDecodeError:
                    continue
    
    def replay_since(self, timestamp: str, **kwargs) -> Iterator[Dict[str, Any]]:
        """Replay events after a given timestamp.
        
        Args:
            timestamp: ISO timestamp to start from
            **kwargs: Additional filters
            
        Yields:
            Event dictionaries
        """
        if not os.path.exists(self._log_file):
            return
            
        with open(self._log_file, "r") as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    if event.get("timestamp", "") >= timestamp:
                        yield event
                except json.JSONDecodeError:
                    continue
    
    def get_log_position(self) -> int:
        """Get current log position (line count)."""
        if not os.path.exists(self._log_file):
            return 0
        with open(self._log_file, "r") as f:
            return sum(1 for _ in f)
    
    def get_event_by_id(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific event by its _id."""
        for event in self.get_events():
            if event.get("_id") == message_id:
                return event
        return None
    
    def query_events(self, types: Optional[List[str]] = None, limit: Optional[int] = None, **filters) -> List[Dict[str, Any]]:
        """Query events with multiple filters."""
        events = self.get_events()
        
        results = []
        for event in events:
            if types and event.get("type") not in types:
                continue
            match = True
            for key, value in filters.items():
                if event.get(key) != value:
                    match = False
                    break
            if match:
                results.append(event)
        
        if limit:
            results = results[:limit]
        return results
    
    def set_learning_manager(self, learning_manager: Any) -> None:
        """Set the learning manager for this SwarmMail instance."""
        self.learning_manager = learning_manager
    
    def record_event_to_learning(self, event: Dict[str, Any], success: bool = True, work_item_id: Optional[str] = None) -> Optional[Any]:
        """Record an event to the learning manager.
        
        Args:
            event: The event dictionary to record
            success: Whether the event was successful
            work_item_id: Optional work item ID for grouping
            
        Returns:
            The recorded pattern or None
        """
        if not self.learning_manager:
            return None
        
        return self.learning_manager.record_swarm_event(
            event_type=event.get("type", "unknown"),
            event_data=event,
            success=success,
            work_item_id=work_item_id
        )
    
    def process_unacknowledged_for_learning(self, work_item_id: str) -> List[Any]:
        """Process unacknowledged messages for learning.
        
        Args:
            work_item_id: Work item ID for the patterns
            
        Returns:
            List of recorded patterns
        """
        patterns = []
        for message in self.inbox():
            pattern = self.record_event_to_learning(message, success=True, work_item_id=work_item_id)
            if pattern:
                patterns.append(pattern)
        return patterns