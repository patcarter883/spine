"""Persistence layer for SPINE human handoff protocol."""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Any, Callable
from enum import Enum
from pathlib import Path


LAYER_STRUCTURE = {
    "layer_1": {"name": "Durable Truth", "paths": ["spec/requirements.md", "spec/architecture.md"]},
    "layer_2": {"name": "Working Memory", "paths": []},
    "layer_3": {"name": "Judgment Cache", "paths": ["knowledge/constraints.md", "knowledge/patterns.json"]},
    "layer_4": {"name": "Execution State", "paths": []},
    "layer_5": {"name": "Communication Bus", "paths": []},
}


class ResumeAction(str, Enum):
    """Available actions for human handoff."""
    RESUME = "resume"
    INSPECT = "inspect"
    ADJUST = "adjust"
    CANCEL = "cancel"


@dataclass
class ResumeMarker:
    """Marker for human handoff with resume capabilities."""
    resume_version: str = "1.0"
    work_item_id: str = ""
    checkpoint_ref: str = ""
    saved_at: str = ""
    reason: str = ""
    next_action: str = ""
    human_instructions: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.saved_at:
            self.saved_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResumeMarker":
        return cls(
            resume_version=data.get("resume_version", "1.0"),
            work_item_id=data.get("work_item_id", ""),
            checkpoint_ref=data.get("checkpoint_ref", ""),
            saved_at=data.get("saved_at", ""),
            reason=data.get("reason", ""),
            next_action=data.get("next_action", ""),
            human_instructions=data.get("human_instructions", {}),
        )

    def save(self, path: str = ".spine/resume.json") -> str:
        """Save resume marker to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str = ".spine/resume.json") -> Optional["ResumeMarker"]:
        """Load resume marker from JSON file."""
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def create_handoff(
        self,
        work_item_id: str,
        checkpoint_ref: str,
        reason: str,
        phase: str,
        active_subphases: list[str],
        pending_gates: list[str],
    ) -> "ResumeMarker":
        """Create a handoff marker with full context."""
        self.work_item_id = work_item_id
        self.checkpoint_ref = checkpoint_ref
        self.reason = reason
        self.next_action = "resume_execution"
        self.human_instructions = {
            "message": f"Work paused due to {reason}. Ready to resume {phase} phase.",
            "swarm_state": {
                "active_subphases": active_subphases,
                "pending_gates": pending_gates,
            },
            "options": [
                {"label": "Resume execution", "action": ResumeAction.RESUME.value},
                {"label": "Review current state", "action": ResumeAction.INSPECT.value},
                {"label": "Adjust plan", "action": ResumeAction.ADJUST.value},
                {"label": "Cancel work", "action": ResumeAction.CANCEL.value},
            ],
        }
        return self


def create_resume_marker(
    work_item_id: str,
    checkpoint_ref: str,
    reason: str,
    phase: str,
    active_subphases: list[str],
    pending_gates: list[str],
    path: str = ".spine/resume.json",
) -> ResumeMarker:
    """Factory function to create and save a resume marker."""
    marker = ResumeMarker()
    marker.create_handoff(
        work_item_id=work_item_id,
        checkpoint_ref=checkpoint_ref,
        reason=reason,
        phase=phase,
        active_subphases=active_subphases,
        pending_gates=pending_gates,
    )
    marker.save(path)
    return marker


@dataclass
class Checkpoint:
    """Represents a phase checkpoint with full state."""
    checkpoint_version: str = "1.0"
    checkpoint_id: str = ""
    created_at: str = ""
    work_item_id: str = ""
    phase_name: str = ""
    phase_entered_at: str = ""
    phase_progress: float = 0.0
    state: dict[str, Any] = field(default_factory=dict)
    dag: dict[str, Any] = field(default_factory=dict)
    context_vars: dict[str, Any] = field(default_factory=dict)
    plan_ref: str = ""
    providers: dict[str, Any] = field(default_factory=dict)
    swarm_state: dict[str, Any] = field(default_factory=dict)
    file_reservations: dict[str, list[str]] = field(default_factory=dict)
    checksum: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.checkpoint_id:
            self.checkpoint_id = f"ckpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        return cls(
            checkpoint_version=data.get("checkpoint_version", "1.0"),
            checkpoint_id=data.get("checkpoint_id", ""),
            created_at=data.get("created_at", ""),
            work_item_id=data.get("work_item_id", ""),
            phase_name=data.get("phase_name", ""),
            phase_entered_at=data.get("phase_entered_at", ""),
            phase_progress=data.get("phase_progress", 0.0),
            state=data.get("state", {}),
            dag=data.get("dag", {}),
            context_vars=data.get("context_vars", {}),
            plan_ref=data.get("plan_ref", ""),
            providers=data.get("providers", {}),
            swarm_state=data.get("swarm_state", {}),
            file_reservations=data.get("file_reservations", {}),
            checksum=data.get("checksum", ""),
        )


@dataclass
class Trigger:
    """A checkpoint trigger condition."""
    trigger_type: str
    action: Callable[..., None]
    percent: Optional[int] = None
    batch_size: Optional[int] = None
    signal_name: Optional[str] = None
    minutes: Optional[int] = None


class CheckpointPolicy:
    """Policy manager for checkpoint triggers."""

    def __init__(self, save_checkpoint_func: Callable[..., None]):
        self.triggers: list[Trigger] = []
        self.save_checkpoint = save_checkpoint_func
        self._setup_triggers()

    def _setup_triggers(self) -> None:
        self.triggers = [
            Trigger(trigger_type="phase_complete", action=self.save_checkpoint),
            Trigger(trigger_type="progress_threshold", action=self.save_checkpoint, percent=25),
            Trigger(trigger_type="progress_threshold", action=self.save_checkpoint, percent=50),
            Trigger(trigger_type="progress_threshold", action=self.save_checkpoint, percent=75),
            Trigger(trigger_type="task_batch_complete", action=self.save_checkpoint, batch_size=5),
            Trigger(trigger_type="signal", action=self.save_checkpoint, signal_name="SIGINT"),
            Trigger(trigger_type="interval", action=self.save_checkpoint, minutes=10),
            Trigger(trigger_type="swarm_gate_complete", action=self.save_checkpoint),
        ]

    def should_trigger(self, trigger_type: str, **kwargs) -> bool:
        """Check if any trigger matches the given type and conditions."""
        for trigger in self.triggers:
            if trigger.trigger_type == trigger_type:
                if trigger_type == "progress_threshold":
                    percent = trigger.percent
                    if percent is None:
                        continue
                    return kwargs.get("progress", 0) >= percent
                return True
        return False

    def get_triggers_by_type(self, trigger_type: str) -> list[Trigger]:
        """Get all triggers of a specific type."""
        return [t for t in self.triggers if t.trigger_type == trigger_type]


@dataclass
class Context:
    """Execution context that can be restored from checkpoint."""
    work_item: str = ""
    phase: str = ""
    variables: dict[str, Any] = field(default_factory=dict)
    dag: dict[str, Any] = field(default_factory=dict)
    swarm_state: dict[str, Any] = field(default_factory=dict)
    file_reservations: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def fresh(cls) -> "Context":
        """Create a fresh context for new work."""
        return cls()


@dataclass
class ExecutionPlan:
    """Optimal resume plan built from checkpoint."""
    tasks: list[str] = field(default_factory=list)
    in_flight_recovery: list[dict[str, Any]] = field(default_factory=list)
    verification_needed: bool = False
    file_reservations: dict[str, list[str]] = field(default_factory=dict)
    pending_gates: list[str] = field(default_factory=list)
    excluded_tasks: list[str] = field(default_factory=list)


class RecoveryStrategy:
    """Builds optimal resume plans from checkpoints with swarm state."""

    def resume(self, checkpoint: Checkpoint) -> ExecutionPlan:
        """Build optimal resume plan including swarm state."""
        in_flight = self._get_in_flight_tasks(checkpoint)

        completed_ids = self._get_completed_task_ids(checkpoint)
        remaining_dag = self._rebuild_dag_excluding(
            checkpoint.dag,
            exclude=completed_ids
        )

        swarm_info = checkpoint.swarm_state or {}

        return ExecutionPlan(
            tasks=remaining_dag,
            in_flight_recovery=in_flight,
            verification_needed=self._needs_verification(checkpoint),
            file_reservations=swarm_info.get("file_reservations", {}),
            pending_gates=swarm_info.get("pending_gates", [])
        )

    def _get_in_flight_tasks(self, checkpoint: Checkpoint) -> list[dict[str, Any]]:
        """Identify tasks that were running at checkpoint time."""
        in_flight = []
        for task_id, task_data in checkpoint.dag.get("results", {}).items():
            if task_data.get("status") == "running":
                in_flight.append({
                    "task_id": task_id,
                    "status": "running",
                    "started_at": task_data.get("started_at")
                })
        return in_flight

    def _get_completed_task_ids(self, checkpoint: Checkpoint) -> list[str]:
        """Get list of completed task IDs to exclude from execution plan."""
        completed = checkpoint.state.get("completed_tasks", [])
        results = checkpoint.dag.get("results", {})
        for task_id, task_data in results.items():
            if task_data.get("status") == "success":
                if task_id not in completed:
                    completed.append(task_id)
        return completed

    def _rebuild_dag_excluding(
        self,
        dag: dict[str, Any],
        exclude: list[str]
    ) -> list[str]:
        """Rebuild DAG excluding completed tasks, preserving dependency order."""
        execution_order = dag.get("execution_plan", [])
        dependencies = dag.get("dependencies", {})

        filtered = [t for t in execution_order if t not in exclude]

        ordered = []
        visited = set()

        def visit(task_id: str) -> None:
            if task_id in visited or task_id in exclude:
                return
            visited.add(task_id)
            for dep in dependencies.get(task_id, []):
                visit(dep)
            ordered.append(task_id)

        for task in filtered:
            visit(task)

        return ordered

    def _needs_verification(self, checkpoint: Checkpoint) -> bool:
        """Check if verification is needed based on checkpoint state."""
        failed = checkpoint.state.get("failed_tasks", [])
        return len(failed) > 0 or checkpoint.phase_progress < 1.0


class ContinuityManager:
    """Manages session continuity and state restoration."""

    def __init__(self, state_dir: str = ".spine/state"):
        self.state_dir = state_dir
        self.checkpoints_dir = os.path.join(state_dir, "checkpoints")
        self.recovery = RecoveryStrategy()
        self.layer_structure = LAYER_STRUCTURE

    def get_layer_1_paths(self) -> list[str]:
        """Get Layer 1 (Durable Truth) file paths."""
        return self.layer_structure["layer_1"]["paths"]

    def get_layer_3_paths(self) -> list[str]:
        """Get Layer 3 (Judgment Cache) file paths."""
        return self.layer_structure["layer_3"]["paths"]

    def read_layer_1_durable_truth(self) -> dict[str, str]:
        """Read Layer 1 (Durable Truth): spec/requirements.md and spec/architecture.md."""
        content = {}
        for path in self.get_layer_1_paths():
            if os.path.exists(path):
                with open(path, "r") as f:
                    content[path] = f.read()
        return content

    def read_layer_3_judgment_cache(self) -> dict[str, Any]:
        """Read Layer 3 (Judgment Cache): knowledge/constraints.md and knowledge/patterns.json."""
        content = {}
        for path in self.get_layer_3_paths():
            if os.path.exists(path):
                if path.endswith(".json"):
                    with open(path, "r") as f:
                        content[path] = json.load(f)
                else:
                    with open(path, "r") as f:
                        content[path] = f.read()
        return content

    def restore_session(self, work_item_id: str = None) -> Context:
        """Restore state from checkpoint with swarm state."""
        if work_item_id:
            return self._restore_work_item(work_item_id)

        marker = self._check_resume_marker()
        if marker:
            return self._restore_checkpoint(marker.checkpoint_ref)

        latest = self._find_most_recent_work()
        if latest:
            return self._prompt_resume(latest)

        return Context.fresh()

    def _restore_checkpoint(self, checkpoint_path: str) -> Context:
        """Load checkpoint and rebuild state including swarm state."""
        checkpoint = self._load_json(checkpoint_path)

        context = Context()
        context.work_item = checkpoint.get("work_item_id", "")
        context.phase = checkpoint.get("phase_name", "")
        context.variables = checkpoint.get("context_vars", {})
        context.dag = self._rebuild_dag(checkpoint.get("dag", {}))

        if "swarm_state" in checkpoint:
            context.swarm_state = checkpoint["swarm_state"]
            context.file_reservations = checkpoint.get("file_reservations", {})

        return context

    def _restore_work_item(self, work_item_id: str) -> Context:
        """Find and restore checkpoint for a specific work item."""
        checkpoints = self._find_checkpoints_for_work(work_item_id)
        if not checkpoints:
            return Context.fresh()
        latest_checkpoint = sorted(checkpoints)[-1]
        return self._restore_checkpoint(latest_checkpoint)

    def _check_resume_marker(self) -> Optional[ResumeMarker]:
        """Check for auto-resume marker."""
        marker_path = os.path.join(self.state_dir, "resume.json")
        return ResumeMarker.load(marker_path)

    def _find_most_recent_work(self) -> Optional[str]:
        """Find the most recent work item ID."""
        current_work_path = os.path.join(self.state_dir, "current_work.json")
        if os.path.exists(current_work_path):
            data = self._load_json(current_work_path)
            return data.get("work_item_id")
        return None

    def _prompt_resume(self, work_item_id: str) -> Context:
        """Prompt user to resume most recent work."""
        return Context.fresh()

    def _find_checkpoints_for_work(self, work_item_id: str) -> list[str]:
        """Find all checkpoints for a work item."""
        if not os.path.exists(self.checkpoints_dir):
            return []
        checkpoints = []
        for f in os.listdir(self.checkpoints_dir):
            if f.endswith(".json"):
                path = os.path.join(self.checkpoints_dir, f)
                data = self._load_json(path)
                if data.get("work_item_id") == work_item_id:
                    checkpoints.append(path)
        return checkpoints

    def _rebuild_dag(self, dag_data: dict[str, Any]) -> dict[str, Any]:
        """Rebuild DAG from checkpoint data."""
        return dag_data

    def _load_json(self, path: str) -> dict[str, Any]:
        """Load JSON file."""
        if not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            return json.load(f)

    def build_resume_plan(self, checkpoint_path: str) -> ExecutionPlan:
        """Build optimal resume plan from checkpoint."""
        checkpoint = self._load_checkpoint(checkpoint_path)
        return self.recovery.resume(checkpoint)

    def _load_checkpoint(self, path: str) -> Checkpoint:
        """Load checkpoint from file."""
        data = self._load_json(path)
        return Checkpoint.from_dict(data)
