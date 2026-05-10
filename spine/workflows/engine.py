"""Base workflow engine for SPINE lifecycles.

Orchestrates lifecycle execution using the Ralph Loop hierarchy.
Manages phase/subphase/task transitions, integrates with
SpineStateMachine, and handles checkpoint persistence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any, Dict, List, TYPE_CHECKING

from ..models.types import (
    NodeStatus,
    ProjectNode,
    PhaseNode,
    SubPhaseNode,
    TaskNode,
    HierarchyProgress,
    FeatureSlice,
)
from ..core.hierarchy import (
    RalphLoopEngine,
    HierarchyValidator,
)
from ..core.constants import PhaseName
from ..providers.agents import AgentProvider

if TYPE_CHECKING:
    from ..core.state_machine import SpineStateMachine
    from ..git.worktree_manager import WorktreeManager
    from ..swarm.gates import SwarmGate


# ── Workflow Phase Enum ────────────────────────────────────────────

class WorkflowPhase(str, Enum):
    """Standard phases in a workflow lifecycle."""
    INIT = "init"
    SPEC = "spec"
    DESIGN = "design"
    PLAN = "plan"
    IMPLEMENT = "implement"
    REVIEW = "review"
    VERIFY = "verify"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


# WorkflowPhase → NodeStatus mapping
_PHASE_TO_STATUS: Dict[str, NodeStatus] = {
    "init": NodeStatus.PENDING,
    "spec": NodeStatus.PENDING,
    "design": NodeStatus.PENDING,
    "plan": NodeStatus.PENDING,
    "implement": NodeStatus.RUNNING,
    "review": NodeStatus.RUNNING,
    "verify": NodeStatus.RUNNING,
    "complete": NodeStatus.SUCCESS,
    "failed": NodeStatus.FAILED,
    "cancelled": NodeStatus.CANCELLED,
}


# ── Workflow Context ───────────────────────────────────────────────

@dataclass
class WorkflowContext:
    """Execution context carried through the lifecycle.

    Attrs:
        requirement: Original requirement text.
        spec: Formal specification document.
        design: Architecture design document.
        plan: Detailed execution plan dict.
        variables: Arbitrary key/value context.
        hierarchy: Built hierarchy tree (set after plan phase).
        gate_results: Results from each gate evaluation.
        started_at: ISO timestamp when workflow started.
        completed_at: ISO timestamp when workflow finished.
    """

    requirement: str = ""
    spec: Optional[str] = None
    design: Optional[str] = None
    plan: Optional[Dict[str, Any]] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    hierarchy: Optional[ProjectNode] = None
    gate_results: Dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context for checkpoint persistence."""
        return {
            "requirement": self.requirement,
            "spec": self.spec,
            "design": self.design,
            "plan": self.plan,
            "variables": self.variables,
            "gate_results": self.gate_results,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowContext":
        """Restore context from checkpoint data."""
        return cls(
            requirement=data.get("requirement", ""),
            spec=data.get("spec"),
            design=data.get("design"),
            plan=data.get("plan"),
            variables=data.get("variables", {}),
            gate_results=data.get("gate_results", {}),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at", ""),
        )


# ── Workflow Result ────────────────────────────────────────────────

@dataclass
class WorkflowResult:
    """Result of a completed workflow execution.

    Attrs:
        success: Whether the workflow completed successfully.
        phase_results: Per-phase result dicts.
        hierarchy: The built project hierarchy.
        progress: Final aggregated progress.
        errors: List of error messages.
    """

    success: bool = False
    phase_results: Dict[str, Any] = field(default_factory=dict)
    hierarchy: Optional[ProjectNode] = None
    progress: Optional[HierarchyProgress] = None
    errors: List[str] = field(default_factory=list)

    @property
    def percent_complete(self) -> float:
        """Percent of tasks completed."""
        return self.progress.percent_complete if self.progress else 0.0


# ── Phase Handler Protocol ─────────────────────────────────────────

class PhaseHandler:
    """Protocol for a workflow phase implementation.

    Subclass this to implement phase-specific behavior.
    """

    name: str = ""  # Override in subclass

    def execute(
        self,
        engine: "WorkflowEngine",
    ) -> None:
        """Execute this phase. Override in subclass."""
        raise NotImplementedError


# ── Workflow Engine ────────────────────────────────────────────────

class WorkflowEngine:
    """Base engine orchestrating lifecycle using Ralph Loop hierarchy.

    Integrates:
    - RalphLoopEngine for hierarchical state
    - SpineStateMachine for phase transition & persistence
    - WorktreeManager for parallel execution
    - Swarm gates for verification

    Usage:
        engine = WorkflowEngine(state_machine=sm)
        engine.create_project("my-project", "Build a web app")
        engine.set_phases(["spec", "design", "plan", "implement", "review", "verify"])
        # ... phase execution ...
        result = engine.get_result()
    """

    # Phases this engine runs through (override in subclasses)
    DEFAULT_PHASES: List[str] = [
        WorkflowPhase.PLAN,
        WorkflowPhase.IMPLEMENT,
        WorkflowPhase.VERIFY,
    ]

    def __init__(
        self,
        state_machine: Optional["SpineStateMachine"] = None,
        worktree_manager: Optional["WorktreeManager"] = None,
        gates: Optional[List["SwarmGate"]] = None,
        agent_provider: Optional[Any] = None,
    ):
        """Initialize the workflow engine.

        Args:
            state_machine: Optional SpineStateMachine for persistence.
            worktree_manager: Optional WorktreeManager for parallel execution.
            gates: Optional list of swarm gates for verification.
            agent_provider: Optional AgentProvider for delegating implementation
                work to external coding agents (OpenCode, Codex, Claude Code).
        """
        self.hierarchy_engine = RalphLoopEngine()
        self.transition_manager = self.hierarchy_engine.transition_manager
        self.progress_aggregator = self.hierarchy_engine.progress_aggregator
        self.validator = HierarchyValidator()

        self._state_machine = state_machine
        self._worktree_manager = worktree_manager
        self._gates = gates or []
        self._agent_provider = agent_provider
        self._context = WorkflowContext()
        self._project: Optional[ProjectNode] = None
        self._phases: List[str] = list(self.DEFAULT_PHASES)
        self._current_phase: Optional[str] = None
        self._errors: List[str] = []
        self._checkpoint_dir: str = ".spine/workflow_checkpoints"

        self.auto_complete_parents: bool = True

        if state_machine:
            self.hierarchy_engine.attach_state_machine(state_machine)

    # ── Properties ────────────────────────────────────────────────

    @property
    def project(self) -> Optional[ProjectNode]:
        """The project hierarchy."""
        return self._project

    @property
    def context(self) -> WorkflowContext:
        """Current workflow context."""
        return self._context

    @property
    def current_phase(self) -> Optional[str]:
        """Currently executing phase name."""
        return self._current_phase

    @property
    def phases(self) -> List[str]:
        """Configured lifecycle phases."""
        return self._phases

    @property
    def errors(self) -> List[str]:
        """Accumulated error messages."""
        return self._errors

    # ── Lifecycle Management ──────────────────────────────────────

    def set_phases(self, phases: List[str]) -> None:
        """Configure the lifecycle phases.

        Args:
            phases: Ordered list of phase names from WorkflowPhase.
        """
        self._phases = phases

    def create_project(
        self,
        project_id: str,
        requirement: str,
        name: Optional[str] = None,
    ) -> ProjectNode:
        """Create a new project with hierarchy.

        Args:
            project_id: Unique project identifier.
            requirement: The work requirement.
            name: Human-readable project name (defaults to project_id).

        Returns:
            The created ProjectNode.
        """
        from datetime import datetime, timezone

        proj_name = name or project_id
        self._project = self.hierarchy_engine.create_project(project_id, proj_name)
        self._context.requirement = requirement
        self._context.started_at = datetime.now(timezone.utc).isoformat()
        self._context.hierarchy = self._project
        self._errors = []

        self.hierarchy_engine.transition_node(self._project, NodeStatus.RUNNING)
        return self._project

    def create_phase_node(
        self,
        phase_id: str,
        phase_name: str,
    ) -> PhaseNode:
        """Create a phase node under the project.

        Args:
            phase_id: Unique phase identifier.
            phase_name: Human-readable phase name.

        Returns:
            The created PhaseNode.
        """
        if self._project is None:
            raise ValueError("No project created. Call create_project() first.")
        return self.hierarchy_engine.create_phase(
            phase_id, phase_name, parent_project=self._project,
        )

    def create_subphase_node(
        self,
        sp_id: str,
        sp_name: str,
        parent_phase: PhaseNode,
        parallel: bool = True,
    ) -> SubPhaseNode:
        """Create a subphase node under a phase.

        Args:
            sp_id: Unique subphase identifier.
            sp_name: Human-readable subphase name.
            parent_phase: The parent phase node.
            parallel: Whether tasks can run in parallel.

        Returns:
            The created SubPhaseNode.
        """
        return self.hierarchy_engine.create_subphase(
            sp_id, sp_name, parent_phase=parent_phase, parallel=parallel,
        )

    def create_task_node(
        self,
        task_id: str,
        task_name: str,
        parent_subphase: SubPhaseNode,
    ) -> TaskNode:
        """Create a task node under a subphase.

        Args:
            task_id: Unique task identifier.
            task_name: Human-readable task name.
            parent_subphase: The parent subphase node.

        Returns:
            The created TaskNode.
        """
        return self.hierarchy_engine.create_task(
            task_id, task_name, parent_subphase=parent_subphase,
        )

    # ── Phase Transitions ─────────────────────────────────────────

    def start_phase(self, phase_name: str) -> Optional[PhaseNode]:
        """Start execution of a named phase.

        Finds or creates the phase node and transitions it to RUNNING.

        Args:
            phase_name: Name of the phase to start.

        Returns:
            The PhaseNode or None if project has no such phase.
        """
        if self._project is None:
            raise ValueError("No project created.")
        self._current_phase = phase_name
        ph = self.hierarchy_engine.find_node(self._project, phase_name)
        if ph is not None:
            self.hierarchy_engine.transition_node(ph, NodeStatus.RUNNING)
        return ph

    def complete_phase(self, phase_name: str) -> Optional[PhaseNode]:
        """Mark a phase as successfully completed.

        Args:
            phase_name: Name of the phase to complete.

        Returns:
            The PhaseNode or None.
        """
        if self._project is None:
            return None
        ph = self.hierarchy_engine.find_node(self._project, phase_name)
        if ph is not None:
            self.hierarchy_engine.transition_node(ph, NodeStatus.SUCCESS)
        return ph

    def fail_phase(self, phase_name: str, error: str = "") -> Optional[PhaseNode]:
        """Mark a phase as failed.

        Args:
            phase_name: Name of the phase that failed.
            error: Error message.

        Returns:
            The PhaseNode or None.
        """
        self._errors.append(f"{phase_name}: {error}" if error else phase_name)
        if self._project is None:
            return None
        ph = self.hierarchy_engine.find_node(self._project, phase_name)
        if ph is not None:
            self.hierarchy_engine.transition_node(ph, NodeStatus.FAILED)
        return ph

    # ── Node Transitions ──────────────────────────────────────────

    def transition_node(self, node: Any, target: NodeStatus) -> None:
        """Transition any node to a new status.

        Args:
            node: The node to transition.
            target: Target NodeStatus.
        """
        self.hierarchy_engine.transition_node(node, target)

    def auto_complete(self, node: Any) -> None:
        """Auto-complete a parent node when all children are done.

        Only if auto_complete_parents is enabled AND all children
        have SUCCESS status.

        Args:
            node: The parent node (PhaseNode, SubPhaseNode).
        """
        if not self.auto_complete_parents:
            return

        children: List[Any] = []
        if isinstance(node, PhaseNode):
            children = node.subphases
        elif isinstance(node, SubPhaseNode):
            children = node.tasks
        else:
            return

        if not children:
            return

        all_success = all(
            c.status == NodeStatus.SUCCESS for c in children
        )
        if all_success and node.status != NodeStatus.SUCCESS:
            self.hierarchy_engine.transition_node(node, NodeStatus.SUCCESS)

    def check_and_auto_complete_subphases(self, phase: PhaseNode) -> None:
        """After completing tasks, check and auto-complete parent subphases.

        Args:
            phase: The phase whose subphases should be checked.
        """
        for sp in phase.subphases:
            self.auto_complete(sp)

    # ── Progress ──────────────────────────────────────────────────

    def get_progress(self) -> HierarchyProgress:
        """Get aggregated project progress.

        Returns:
            HierarchyProgress for the current project.
        """
        if self._project is None:
            return HierarchyProgress()
        return self.hierarchy_engine.get_project_progress(self._project)

    def get_phase_progress(self, phase_id: str) -> Optional[HierarchyProgress]:
        """Get progress for a specific phase.

        Args:
            phase_id: The phase node ID.

        Returns:
            HierarchyProgress or None if not found.
        """
        if self._project is None:
            return None
        ph = self.hierarchy_engine.find_node(self._project, phase_id)
        if ph is None or not isinstance(ph, PhaseNode):
            return None
        return self.progress_aggregator.aggregate_phase(ph)

    # ── Task Management ───────────────────────────────────────────

    def mark_task_running(self, task_id: str) -> Optional[TaskNode]:
        """Mark a task as running.

        Args:
            task_id: The task node ID.

        Returns:
            The TaskNode or None.
        """
        task = self._find_task(task_id)
        if task:
            self.hierarchy_engine.transition_node(task, NodeStatus.RUNNING)
        return task

    def mark_task_success(self, task_id: str, result: str = "") -> Optional[TaskNode]:
        """Mark a task as completed.

        Args:
            task_id: The task node ID.
            result: Optional result string.

        Returns:
            The TaskNode or None.
        """
        task = self._find_task(task_id)
        if task:
            task.progress = 100.0
            task.result = result or f"Completed: {task.name}"
            self.hierarchy_engine.transition_node(task, NodeStatus.SUCCESS)
        return task

    def mark_task_failed(self, task_id: str, error: str = "") -> Optional[TaskNode]:
        """Mark a task as failed.

        Args:
            task_id: The task node ID.
            error: Error description.

        Returns:
            The TaskNode or None.
        """
        task = self._find_task(task_id)
        if task:
            task.error = error
            self.hierarchy_engine.transition_node(task, NodeStatus.FAILED)
        return task

    def mark_task_blocked(self, task_id: str) -> Optional[TaskNode]:
        """Mark a task as blocked.

        Args:
            task_id: The task node ID.

        Returns:
            The TaskNode or None.
        """
        task = self._find_task(task_id)
        if task:
            self.hierarchy_engine.transition_node(task, NodeStatus.BLOCKED)
        return task

    def _find_task(self, task_id: str) -> Optional[TaskNode]:
        """Find a task node by ID.

        Args:
            task_id: The task node ID.

        Returns:
            The TaskNode or None.
        """
        if self._project is None:
            return None
        node = self.hierarchy_engine.find_node(self._project, task_id)
        if node is not None and isinstance(node, TaskNode):
            return node
        return None

    # ── Gates ─────────────────────────────────────────────────────

    def run_gate(self, gate: "SwarmGate", state: Optional[Dict] = None) -> Dict[str, Any]:
        """Run a single swarm gate.

        Args:
            gate: The gate to evaluate.
            state: Optional state dict for the gate.

        Returns:
            Gate evaluation result dict.
        """
        from ..models.types import SpineState

        if state is None:
            state = SpineState(
                phase=PhaseName.VERIFICATION,
                previous_phase=None,
                requirement=self._context.requirement,
                plan=self._context.plan,
                tasks={},
                completed_tasks=[],
                failed_tasks=[],
                swarm_state={},
                hive_cells={},
                swarm_events=[],
                variables=self._context.variables,
                errors=list(self._errors),
                providers={},
                agent_provider=None,
                critic_gate_result=None,
                error_state=None,
                error_history=[],
            )

        result = gate.evaluate(state)
        self._context.gate_results[gate.name] = result
        return result

    def run_all_gates(self) -> Dict[str, Any]:
        """Run all configured swarm gates.

        Returns:
            Dict mapping gate name to result.
        """
        results = {}
        for gate in self._gates:
            results[gate.name] = self.run_gate(gate)
        return results

    # ── Checkpoint Persistence ────────────────────────────────────

    def checkpoint(self, phase_name: str = "", phase_progress: float = 0.0) -> Optional[str]:
        """Create and save a checkpoint.

        Args:
            phase_name: Current phase name.
            phase_progress: Progress within phase (0.0-1.0).

        Returns:
            Path to saved checkpoint or None.
        """
        if self._state_machine is None:
            return self._save_checkpoint_local(phase_name, phase_progress)

        try:
            return self._state_machine.checkpoint(
                work_item_id=self._project.id if self._project else "unknown",
                phase_name=phase_name,
                phase_progress=phase_progress,
                state={
                    "requirement": self._context.requirement,
                    "context": self._context.to_dict(),
                    "errors": self._errors,
                },
                dag={
                    "phases": self._phases,
                    "current_phase": self._current_phase,
                },
                context_vars=self._context.variables,
                swarm_state={
                    "active_subphases": [],
                    "pending_gates": [],
                    "file_reservations": {},
                },
                auto_commit=False,
            )
        except Exception as e:
            self._errors.append(f"Checkpoint failed: {e}")
            return None

    def _save_checkpoint_local(
        self, phase_name: str = "", phase_progress: float = 0.0
    ) -> Optional[str]:
        """Save checkpoint locally (no state machine).

        Args:
            phase_name: Current phase name.
            phase_progress: Progress within phase (0.0-1.0).

        Returns:
            Path to saved checkpoint or None.
        """
        import json
        from datetime import datetime, timezone

        os.makedirs(self._checkpoint_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self._checkpoint_dir, f"ckpt_{ts}.json")

        data = {
            "phase_name": phase_name,
            "phase_progress": phase_progress,
            "context": self._context.to_dict(),
            "errors": self._errors,
            "current_phase": self._current_phase,
            "phases": self._phases,
            "project_id": self._project.id if self._project else None,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    # ── Result ────────────────────────────────────────────────────

    def get_result(self) -> WorkflowResult:
        """Build the final workflow result.

        Returns:
            A WorkflowResult summarizing execution.
        """
        success = self._project is not None and self._project.status == NodeStatus.SUCCESS
        progress = self.get_progress() if self._project else None

        return WorkflowResult(
            success=success,
            phase_results={
                "current_phase": self._current_phase,
                "phases_executed": self._phases,
            },
            hierarchy=self._project,
            progress=progress,
            errors=list(self._errors),
        )

    # ── Tree Validation ───────────────────────────────────────────

    def validate_hierarchy(self) -> bool:
        """Validate the project hierarchy.

        Returns:
            True if valid.
        """
        if self._project is None:
            return False
        result = self.validator.validate(self._project)
        if not result.is_valid:
            self._errors.extend(result.errors)
        return result.is_valid

    def collect_all_nodes(self) -> List[Any]:
        """Collect all nodes in the hierarchy.

        Returns:
            Flat list of all nodes.
        """
        if self._project is None:
            return []
        return self.hierarchy_engine.collect_all_nodes(self._project)

    # ── Serialization ─────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialize engine state for persistence."""
        return {
            "context": self._context.to_dict(),
            "phases": self._phases,
            "current_phase": self._current_phase,
            "errors": self._errors,
            "project_id": self._project.id if self._project else None,
            "auto_complete_parents": self.auto_complete_parents,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        state_machine: Optional["SpineStateMachine"] = None,
        worktree_manager: Optional["WorktreeManager"] = None,
    ) -> "WorkflowEngine":
        """Restore engine state from dict.

        Args:
            data: Serialized engine state.
            state_machine: Optional state machine.
            worktree_manager: Optional worktree manager.

        Returns:
            Restored WorkflowEngine.
        """
        engine = cls(
            state_machine=state_machine,
            worktree_manager=worktree_manager,
        )
        engine._context = WorkflowContext.from_dict(data.get("context", {}))
        engine._phases = data.get("phases", list(cls.DEFAULT_PHASES))
        engine._current_phase = data.get("current_phase")
        engine._errors = data.get("errors", [])
        engine.auto_complete_parents = data.get("auto_complete_parents", True)

        # Note: hierarchy tree is rebuilt by re-executing phases
        return engine


__all__ = [
    "WorkflowPhase",
    "WorkflowContext",
    "WorkflowResult",
    "PhaseHandler",
    "WorkflowEngine",
]