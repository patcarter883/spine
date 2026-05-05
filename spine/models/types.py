"""SPINE data model types."""

from dataclasses import dataclass, field
from typing import TypedDict, Literal, Optional, Any, Callable, Dict

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
    
    # Critic gate state
    critic_gate_result: Optional[Literal["APPROVED", "NEEDS_REVISION", "REJECTED"]]
    
    # Error state tracking
    error_state: Optional[str]  # Current error state (HUMAN_REVIEW, TRANSIENT, FATAL)
    error_history: list[dict[str, Any]]  # Track error occurrences


__all__ = [
    "Task",
    "SubPhase", 
    "Phase",
    "PhaseResult",
    "SubPhaseResult",
    "SpineState",
]