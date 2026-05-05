"""Pattern learning system for SPINE with maturity progression and anti-pattern generation."""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Any


@dataclass
class Pattern:
    """A learned pattern with maturity status."""
    pattern_id: str
    context: str
    solution: str
    status: str = "candidate"
    first_seen: str = ""
    last_confirmed: str = ""
    confirmations: int = 0
    successes: int = 0
    failures: int = 0
    confidence: float = 0.0

    def __post_init__(self):
        if not self.first_seen:
            self.first_seen = datetime.now(timezone.utc).isoformat()
        if not self.last_confirmed:
            self.last_confirmed = self.first_seen

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Pattern":
        return cls(**data)

    def record_success(self) -> None:
        """Record a successful application of this pattern."""
        self.successes += 1
        self.confirmations += 1
        self.last_confirmed = datetime.now(timezone.utc).isoformat()
        self._update_maturity()

    def record_failure(self) -> None:
        """Record a failed application of this pattern."""
        self.failures += 1
        self.confirmations += 1
        self.last_confirmed = datetime.now(timezone.utc).isoformat()
        self._update_maturity()

    def _update_maturity(self) -> None:
        """Update pattern status based on success rate and confirmation count."""
        if self.confirmations < 1:
            return

        success_rate = self.successes / self.confirmations
        self.confidence = success_rate

        if success_rate >= 0.9 and self.confirmations >= 10:
            self.status = "proven"
        elif success_rate >= 0.8 and self.confirmations >= 3:
            self.status = "established"
        elif success_rate >= 0.6:
            self.status = "candidate"
        else:
            self.status = "anti_pattern_candidate"


@dataclass
class AntiPattern:
    """An anti-pattern generated from patterns with >60% failure rate."""
    pattern_id: str
    pattern_context: str
    failure_rate: float
    avoidance: str
    first_seen: str = ""
    confirmed_failures: int = 0

    def __post_init__(self):
        if not self.first_seen:
            self.first_seen = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AntiPattern":
        return cls(**data)


@dataclass
class PatternRecord:
    """Record of a pattern application for tracking."""
    pattern_id: str
    task_id: str
    work_item_id: str
    success: bool
    timestamp: str = ""
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LearningManager:
    """Manages pattern learning with maturity progression.
    
    Enhanced with swarm event integration for automatic pattern recording
    from swarm mail events.
    """

    def __init__(self, knowledge_dir: str = ".spine/knowledge"):
        self.knowledge_dir = knowledge_dir
        self.patterns_path = os.path.join(knowledge_dir, "patterns.json")
        self.anti_patterns_path = os.path.join(knowledge_dir, "anti_patterns.json")
        self.completions_path = os.path.join(knowledge_dir, "completions.jsonl")
        self._ensure_knowledge_dir()
        
        # Swarm event hooks
        self._swarm_event_patterns: dict[str, Pattern] = {}

    def _ensure_knowledge_dir(self) -> None:
        """Ensure knowledge directory exists."""
        os.makedirs(self.knowledge_dir, exist_ok=True)

    def record_completion(
        self,
        pattern: Pattern,
        task_id: str,
        work_item_id: str,
        success: bool,
        context: dict[str, Any] | None = None
    ) -> None:
        """Record a pattern completion and update maturity."""
        record = PatternRecord(
            pattern_id=pattern.pattern_id,
            task_id=task_id,
            work_item_id=work_item_id,
            success=success,
            context=context or {}
        )

        self._append_completion_record(record)

        if success:
            pattern.record_success()
        else:
            pattern.record_failure()

        self._save_pattern(pattern)

        if self._failure_rate(pattern) > 0.6:
            self._generate_anti_pattern(pattern)

    def _append_completion_record(self, record: PatternRecord) -> None:
        """Append completion record to JSONL file."""
        with open(self.completions_path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def _save_pattern(self, pattern: Pattern) -> None:
        """Save pattern to patterns.json."""
        patterns = self._load_patterns()
        patterns[pattern.pattern_id] = pattern.to_dict()
        self._write_json(self.patterns_path, {"patterns": patterns})

    def _generate_anti_pattern(self, pattern: Pattern) -> AntiPattern | None:
        """Generate anti-pattern if failure rate exceeds threshold."""
        failure_rate = self._failure_rate(pattern)

        if failure_rate <= 0.6:
            return None

        anti_pattern = AntiPattern(
            pattern_id=pattern.pattern_id,
            pattern_context=pattern.context,
            failure_rate=failure_rate,
            avoidance=f"Avoid: {pattern.solution}"
        )

        self._save_anti_pattern(anti_pattern)
        return anti_pattern

    def _failure_rate(self, pattern: Pattern) -> float:
        """Calculate failure rate for a pattern."""
        if pattern.confirmations == 0:
            return 0.0
        return pattern.failures / pattern.confirmations

    def _load_patterns(self) -> dict[str, dict[str, Any]]:
        """Load patterns from file."""
        if not os.path.exists(self.patterns_path):
            return {}
        with open(self.patterns_path, "r") as f:
            data = json.load(f)
            return data.get("patterns", {})

    def _save_anti_pattern(self, anti_pattern: AntiPattern) -> None:
        """Save anti-pattern to anti_patterns.json."""
        anti_patterns = self._load_anti_patterns()
        anti_patterns[anti_pattern.pattern_id] = anti_pattern.to_dict()
        self._write_json(self.anti_patterns_path, {"anti_patterns": anti_patterns})

    def _load_anti_patterns(self) -> dict[str, dict[str, Any]]:
        """Load anti-patterns from file."""
        if not os.path.exists(self.anti_patterns_path):
            return {}
        with open(self.anti_patterns_path, "r") as f:
            data = json.load(f)
            return data.get("anti_patterns", {})

    def _write_json(self, path: str, data: dict[str, Any]) -> None:
        """Write data to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """Get a pattern by ID."""
        patterns = self._load_patterns()
        if pattern_id in patterns:
            return Pattern.from_dict(patterns[pattern_id])
        return None

    def get_anti_pattern(self, pattern_id: str) -> Optional[AntiPattern]:
        """Get an anti-pattern by ID."""
        anti_patterns = self._load_anti_patterns()
        if pattern_id in anti_patterns:
            return AntiPattern.from_dict(anti_patterns[pattern_id])
        return None

    def get_patterns_by_status(self, status: str) -> list[Pattern]:
        """Get all patterns with a given status."""
        patterns = self._load_patterns()
        result = []
        for data in patterns.values():
            if data.get("status") == status:
                result.append(Pattern.from_dict(data))
        return result

    def get_all_patterns(self) -> list[Pattern]:
        """Get all patterns."""
        patterns = self._load_patterns()
        return [Pattern.from_dict(data) for data in patterns.values()]

    def get_all_anti_patterns(self) -> list[AntiPattern]:
        """Get all anti-patterns."""
        anti_patterns = self._load_anti_patterns()
        return [AntiPattern.from_dict(data) for data in anti_patterns.values()]

    # --- Swarm Event Integration ---

    def record_swarm_event(
        self,
        event_type: str,
        event_data: dict[str, Any],
        success: bool = True,
        work_item_id: str = "",
    ) -> Optional[Pattern]:
        """Record a swarm event as a pattern.
        
        Args:
            event_type: Type of swarm event (e.g., 'message_sent', 'task_completed')
            event_data: Event data dictionary
            success: Whether the event was successful
            work_item_id: Optional work item identifier
            
        Returns:
            The recorded Pattern or None
        """
        pattern_id = f"swarm_{event_type}_{work_item_id or 'default'}"
        pattern = self.get_pattern(pattern_id)
        
        if pattern is None:
            pattern = Pattern(
                pattern_id=pattern_id,
                context=f"Swarm event: {event_type}",
                solution=str(event_data.get("solution", event_data.get("action", "executed")))
            )
        
        # Record the completion
        self.record_completion(
            pattern=pattern,
            task_id=event_data.get("task_id", event_type),
            work_item_id=work_item_id or "swarm_event",
            success=success,
            context=event_data
        )
        
        return pattern

    def get_swarm_patterns(self) -> list[Pattern]:
        """Get all patterns originating from swarm events."""
        patterns = self._load_patterns()
        return [
            Pattern.from_dict(data)
            for pid, data in patterns.items()
            if pid.startswith("swarm_")
        ]

    def sync_to_hivemind(self, hivemind: Any) -> int:
        """Sync proven swarm patterns to Hivemind.
        
        Args:
            hivemind: Hivemind instance to sync to
            
        Returns:
            Number of patterns synced
        """
        proven = self.get_patterns_by_status("proven")
        swarm_patterns = [p for p in proven if p.pattern_id.startswith("swarm_")]
        synced = 0
        
        for pattern in swarm_patterns:
            hivemind.add_memory(
                content=f"Proven swarm pattern: {pattern.context} -> {pattern.solution}",
                context="swarm_pattern_proven",
                tags=["pattern", "swarm", "proven"],
                metadata={
                    "pattern_id": pattern.pattern_id,
                    "confidence": pattern.confidence,
                    "successes": pattern.successes,
                    "failures": pattern.failures
                }
            )
            synced += 1
        
        return synced