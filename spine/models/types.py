"""SPINE data model types."""

from dataclasses import dataclass, field
from enum import Enum
from typing import TypedDict, Literal, Optional, Any, Callable, Dict, List

from .enums import PhaseName, StateStatus, SubPhaseStatus


@dataclass
class Task:
    """A unit of work in a phase."""
    id: str
    description: str
    status: StateStatus = StateStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None


@dataclass
class SubPhase:
    """A parallelizable unit within a phase with swarm patterns."""
    name: str
    weight: float = 1.0
    priority: int = 0
    dependencies: list[str] = field(default_factory=list)
    parallel: bool = True
    agent_role: str = ""
    tasks: list[Task] = field(default_factory=list)
    swarm_gates: list[str] = field(default_factory=list)
    # State tracking for wave-based execution
    status: SubPhaseStatus = field(default_factory=lambda: SubPhaseStatus.PENDING)
    retries: int = 0
    max_retries: int = 3
    blocked_by: Optional[str] = None  # Name of subphase that blocked this one
    error: Optional[str] = None  # Error message from failure
    error_count: int = 0  # Track total error occurrences
    last_error: Optional[str] = None  # Most recent error message

    def fail(self, error: str, blocked_by: Optional[str] = None) -> None:
        """Mark subphase as failed with error info."""
        self.status = SubPhaseStatus.FAILED
        self.error = error
        self.last_error = error
        self.error_count += 1
        self.blocked_by = blocked_by

    def block(self, blocked_by: str) -> None:
        """Mark subphase as blocked by another subphase."""
        self.status = SubPhaseStatus.BLOCKED
        self.blocked_by = blocked_by

    def mark_reworking(self) -> None:
        """Mark subphase as being retried."""
        self.status = SubPhaseStatus.REWORKING
        self.error = None

    def mark_success(self, result: Any = None) -> None:
        """Mark subphase as successful."""
        self.status = SubPhaseStatus.SUCCESS
        self.error = None
        self.last_error = None

    def has_exceeded_error_threshold(self, max_errors: int = 3) -> bool:
        """Check if error count strictly exceeds threshold."""
        return self.error_count > max_errors


@dataclass
class Phase:
    """A phase containing potentially parallel sub-phases."""
    name: PhaseName
    description: str = ""
    subphases: list[SubPhase] = field(default_factory=list)
    swarm_agents: list[str] = field(default_factory=list)
    entry_conditions: list[Callable[[Dict[str, Any]], bool]] = field(default_factory=list)
    exit_criteria: list[Callable[[Dict[str, Any]], bool]] = field(default_factory=list)
    timeout_seconds: int = 3600  # 1 hour default
    # DAG hooks for pre/post execution
    pre_execute_hooks: list[Callable[[Dict[str, Any]], Dict[str, Any]]] = field(default_factory=list)
    post_execute_hooks: list[Callable[[Dict[str, Any]], Dict[str, Any]]] = field(default_factory=list)
    error_state: Optional[str] = None  # Current error state tracking
    error_transitions: Dict[str, str] = field(default_factory=dict)  # Error state transitions


@dataclass
class PhaseResult:
    """Result of phase execution with sub-phase results."""
    subphase_results: dict[str, Any]
    gate_results: dict[str, Any] = field(default_factory=dict)
    subphase_statuses: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_waves(cls, wave_results: list, gates: Optional[dict] = None):
        """Create PhaseResult from wave execution results.
        
        Includes subphase results and their statuses.
        """
        results = {}
        statuses = {}
        for wr in wave_results:
            results[wr.subphase_name] = wr.result
            statuses[wr.subphase_name] = wr.status.value if hasattr(wr.status, 'value') else str(wr.status)
        return cls(subphase_results=results, gate_results=gates or {}, subphase_statuses=statuses)


@dataclass
class SubPhaseResult:
    """Result of a single sub-phase execution."""
    subphase_name: str
    result: Any
    status: SubPhaseStatus = SubPhaseStatus.SUCCESS

    @classmethod
    def failed(cls, subphase_name: str, error: str) -> "SubPhaseResult":
        """Create a failed result."""
        return cls(subphase_name=subphase_name, result=None, status=SubPhaseStatus.FAILED)

    @classmethod
    def blocked(cls, subphase_name: str, blocked_by: str) -> "SubPhaseResult":
        """Create a blocked result."""
        return cls(subphase_name=subphase_name, result=None, status=SubPhaseStatus.BLOCKED)


class SpineState(TypedDict):
    """The central state for SPINE workflow."""
    # Core workflow state
    phase: str
    previous_phase: Optional[str]
    
    # Work item context
    requirement: str
    plan: Optional[dict[str, Any]]
    
    # Task tracking
    tasks: dict[str, Task]  # task_id -> Task
    completed_tasks: list[str]
    failed_tasks: list[str]
    
    # Swarm state (from swarm-tools pattern)
    swarm_state: dict[str, Any]
    hive_cells: dict[str, Any]  # Durable task records
    swarm_events: list[dict[str, Any]]  # Agent communication log
    
    # Execution context
    variables: dict[str, Any]
    errors: list[str]
    
    # Provider state
    providers: dict[str, Any]
    agent_provider: Optional[dict[str, Any]]
    
    # Critic gate state
    critic_gate_result: Optional[Literal["APPROVED", "NEEDS_REVISION", "REJECTED"]]
    
    # Error state tracking
    error_state: Optional[str]  # Current error state (HUMAN_REVIEW, TRANSIENT, FATAL)
    error_history: list[dict[str, Any]]  # Track error occurrences
    
    # Deep Agents integration state
    pending_messages: list[dict[str, Any]]  # Message queue for mid-run injection
    model_call_count: int  # Step counter for DA agents


# ── Ralph Loop Hierarchy Types ─────────────────────────────────────


class HierarchyLevel(Enum):
    """Levels in the Ralph Loop hierarchy."""
    PROJECT = "project"
    PHASE = "phase"
    SUBPHASE = "subphase"
    TASK = "task"


class NodeStatus(Enum):
    """Status values for hierarchy nodes."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    REWORKING = "reworking"
    CANCELLED = "cancelled"


@dataclass
class HierarchyNode:
    """Base node for the Ralph Loop hierarchical automation tree.
    
    Supports nested automation, progress tracking, and state transitions.
    All levels (Project, Phase, Subphase, Task) inherit from this base.
    """
    id: str
    name: str
    level: HierarchyLevel = HierarchyLevel.TASK
    status: NodeStatus = NodeStatus.PENDING
    progress: float = 0.0  # 0.0 to 100.0
    children: List['HierarchyNode'] = field(default_factory=list)
    parent: Optional['HierarchyNode'] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        """Check if node is in a terminal state."""
        return self.status in (NodeStatus.SUCCESS, NodeStatus.CANCELLED)

    def is_active(self) -> bool:
        """Check if node is actively processing."""
        return self.status in (NodeStatus.RUNNING, NodeStatus.REWORKING)

    def is_blocked_or_failed(self) -> bool:
        """Check if node is blocked or failed."""
        return self.status in (NodeStatus.BLOCKED, NodeStatus.FAILED)


@dataclass
class ProjectNode:
    """Ralph Loop project node — top-level container.
    
    Holds phases as its children. Aggregates progress from all phases.
    """
    id: str
    name: str
    level: HierarchyLevel = HierarchyLevel.PROJECT
    status: NodeStatus = NodeStatus.PENDING
    progress: float = 0.0
    phases: List['PhaseNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhaseNode:
    """Ralph Loop phase node — groups related subphases.
    
    Maps to the existing Phase model for integration.
    """
    id: str
    name: str
    parent_id: str = ""
    level: HierarchyLevel = HierarchyLevel.PHASE
    status: NodeStatus = NodeStatus.PENDING
    progress: float = 0.0
    subphases: List['SubPhaseNode'] = field(default_factory=list)
    phase_model: Optional[Phase] = None  # Reference to original Phase model
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubPhaseNode:
    """Ralph Loop subphase node — holds tasks.
    
    Maps to the existing SubPhase model for integration.
    Supports parallel execution of tasks.
    """
    id: str
    name: str
    parent_id: str
    level: HierarchyLevel = HierarchyLevel.SUBPHASE
    status: NodeStatus = NodeStatus.PENDING
    progress: float = 0.0
    parallel: bool = True
    tasks: List['TaskNode'] = field(default_factory=list)
    subphase_model: Optional[SubPhase] = None  # Reference to original SubPhase model
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskNode:
    """Ralph Loop task node — atomic unit of work.
    
    Maps to the existing Task model for integration.
    """
    id: str
    name: str
    parent_id: str
    level: HierarchyLevel = HierarchyLevel.TASK
    status: NodeStatus = NodeStatus.PENDING
    progress: float = 0.0
    result: Optional[str] = None
    error: Optional[str] = None
    task_model: Optional[Task] = None  # Reference to original Task model
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HierarchyProgress:
    """Aggregated progress across a hierarchy subtree."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    blocked_tasks: int = 0
    running_tasks: int = 0
    pending_tasks: int = 0

    @property
    def in_progress_tasks(self) -> int:
        """Tasks currently being executed (running + reworking)."""
        return self.running_tasks

    @property
    def percent_complete(self) -> float:
        """Percentage of tasks completed."""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100.0


@dataclass
class ValidationResult:
    """Result of a hierarchy validation check."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class FeatureSlice:
    """A unit of work at feature granularity for agent delegation.

    The planner produces FeatureSlices -- not micro-tasks.  Each slice
    represents an independent work unit that a coding agent (OpenCode, Codex,
    Claude Code) can execute autonomously.  The agent owns the internal
    decomposition (which files to touch, in what order).

    Attributes:
        id: Unique identifier (e.g. "auth-jwt-middleware").
        description: What to build, at feature granularity.
        scope: Modules/directories the agent should work within.
        depends_on: IDs of slices that must complete before this one.
        agent_role: Which swarm role handles this (coder, test_engineer, reviewer).
        acceptance: What "done" looks like -- gate criteria for verification.
    """

    id: str
    description: str
    scope: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    agent_role: str = "coder"
    acceptance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "scope": self.scope,
            "depends_on": self.depends_on,
            "agent_role": self.agent_role,
            "acceptance": self.acceptance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureSlice":
        # The DA agent sometimes returns description as a structured dict
        # (e.g. {'output': '...', 'exit_code': 0}).  Coerce to string so that
        # downstream code (spec writer, UI) never encounters a raw dict.
        raw_desc = data["description"]
        if isinstance(raw_desc, dict):
            desc = (
                raw_desc.get("output")
                or raw_desc.get("result")
                or raw_desc.get("name")
                or str(raw_desc)
            )
            # Trim overly long agent outputs to a readable summary
            if len(desc) > 300:
                desc = desc[:300].rsplit(".", 1)[0] + "."
        else:
            desc = str(raw_desc)
        return cls(
            id=data["id"],
            description=desc,
            scope=data.get("scope", []),
            depends_on=data.get("depends_on", []),
            agent_role=data.get("agent_role", "coder"),
            acceptance=data.get("acceptance", []),
        )


__all__ = [
    "Task",
    "SubPhase",
    "Phase",
    "PhaseResult",
    "SubPhaseResult",
    "SpineState",
    "HierarchyLevel",
    "NodeStatus",
    "HierarchyNode",
    "ProjectNode",
    "PhaseNode",
    "SubPhaseNode",
    "TaskNode",
    "HierarchyProgress",
    "ValidationResult",
    "FeatureSlice",
]