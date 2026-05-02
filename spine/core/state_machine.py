"""Core state machine implementation using LangGraph."""

from typing import TypedDict, Literal, Optional, Any
from dataclasses import dataclass, field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .constants import PhaseName, StateStatus


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


@dataclass
class Phase:
    """A phase containing potentially parallel sub-phases."""
    name: PhaseName
    description: str = ""
    subphases: list[SubPhase] = field(default_factory=list)
    swarm_agents: list[str] = field(default_factory=list)
    entry_conditions: list[str] = field(default_factory=list)
    exit_criteria: list[str] = field(default_factory=list)
    timeout_seconds: int = 3600  # 1 hour default


@dataclass
class PhaseResult:
    """Result of phase execution with sub-phase results."""
    subphase_results: dict[str, Any]
    gate_results: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_waves(cls, wave_results: list, gates: Optional[dict] = None):
        """Create PhaseResult from wave execution results."""
        results = {}
        for wr in wave_results:
            results[wr.subphase_name] = wr.result
        return cls(subphase_results=results, gate_results=gates or {})


@dataclass
class SubPhaseResult:
    """Result of a single sub-phase execution."""
    subphase_name: str
    result: Any


class SwarmDAGExecutor:
    """Executes a phase with potential parallel sub-phases using swarm agents."""

    def execute_phase(self, phase: Phase, context: dict[str, Any]) -> PhaseResult:
        """Execute a phase with parallel sub-phase wave execution."""
        if not phase.subphases:
            return PhaseResult(subphase_results={})

        subphase_deps = self.build_subphase_deps(phase.subphases)
        wave_results: list[SubPhaseResult] = []
        remaining = set(sp.name for sp in phase.subphases)
        completed: set[str] = set()

        while remaining:
            ready = self.find_ready_subphases(subphase_deps, remaining, completed)
            if not ready:
                break

            wave_result = self.execute_subphase_wave(ready, context)
            wave_results.extend(wave_result)

            for r in wave_result:
                completed.add(r.subphase_name)
                remaining.discard(r.subphase_name)

        gate_results = self.run_swarm_gates(phase.swarm_agents, context)
        return PhaseResult.from_waves(wave_results, gate_results)

    def build_subphase_deps(self, subphases: list[SubPhase]) -> dict[str, set[str]]:
        """Build dependency map for subphases."""
        return {sp.name: set(sp.dependencies) for sp in subphases}

    def find_ready_subphases(self, deps: dict[str, set[str]], remaining: set, completed: set) -> list[str]:
        """Find subphases with no unmet dependencies."""
        return [name for name in remaining if deps[name] <= completed]

    def execute_subphase_wave(self, subphase_names: list[str], context: dict) -> list[SubPhaseResult]:
        """Execute multiple subphases concurrently."""
        results = []
        for name in subphase_names:
            result = self.execute_dag(name, context)
            results.append(SubPhaseResult(subphase_name=name, result=result))
        return results

    def execute_dag(self, dag_or_name: str, context: dict) -> Any:
        """Execute a DAG (placeholder for real execution)."""
        return {"status": "completed", "dag": dag_or_name, "context_keys": list(context.keys())}

    def run_swarm_gates(self, gates: list[str], context: dict) -> dict[str, Any]:
        """Run swarm-specific gates."""
        return {g: {"status": "passed"} for g in gates}

    def run_critic_gate(self, plan: dict[str, Any], context: dict[str, Any]) -> Literal["APPROVED", "NEEDS_REVISION", "REJECTED"]:
        """Execute critic gate review. Returns gate result."""
        if not plan or not plan.get("tasks"):
            return "REJECTED"
        if len(plan.get("tasks", [])) == 0:
            return "REJECTED"
        return "APPROVED"

    def topological_order(self, subphases: list[SubPhase]) -> list[str]:
        """Return subphase names in topological order respecting dependencies."""
        deps = self.build_subphase_deps(subphases)
        result = []
        visited = set()
        temp_mark = set()

        def visit(name: str):
            if name in temp_mark:
                raise ValueError(f"Cycle detected in subphase dependencies: {name}")
            if name in visited:
                return
            temp_mark.add(name)
            for dep in deps.get(name, set()):
                if dep in [sp.name for sp in subphases]:
                    visit(dep)
            temp_mark.discard(name)
            visited.add(name)
            result.append(name)

        for sp in subphases:
            if sp.name not in visited:
                visit(sp.name)

        return result

    def compute_waves(self, subphases: list[SubPhase]) -> list[list[str]]:
        """Group subphases into waves based on dependencies."""
        deps = self.build_subphase_deps(subphases)
        waves = []
        completed: set[str] = set()
        remaining = {sp.name for sp in subphases}

        while remaining:
            ready = [name for name in remaining if deps[name] <= completed]
            if not ready:
                break
            waves.append(ready)
            for name in ready:
                completed.add(name)
                remaining.discard(name)

        return waves


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


def init_phase(state: SpineState) -> SpineState:
    """Initialize the workflow from user requirement."""
    state["phase"] = PhaseName.INIT
    state["plan"] = None
    state["tasks"] = {}
    state["completed_tasks"] = []
    state["failed_tasks"] = []
    state["swarm_state"] = {
        "active_subphases": [],
        "file_reservations": {},
        "pending_gates": []
    }
    state["hive_cells"] = {}
    state["swarm_events"] = []
    state["errors"] = []
    state["critic_gate_result"] = None
    
    # Transition to PLANNING
    state["previous_phase"] = PhaseName.INIT
    state["phase"] = PhaseName.PLANNING
    return state


def planning_phase(state: SpineState) -> SpineState:
    """Execute the PLANNING phase with parallel sub-phases."""
    state["phase"] = PhaseName.PLANNING
    
    # Define sub-phases based on design
    # Wave 1: ANALYZE, TECH_RESEARCH, RISK_ASSESSMENT (parallel)
    # Wave 2: SYNTHESIZE (depends on Wave 1)
    
    subphases = [
        SubPhase(
            name="ANALYZE",
            priority=1,
            parallel=True,
            agent_role="explorer",
            tasks=[Task(id="parse_requirement", description="Parse and analyze requirement")]
        ),
        SubPhase(
            name="TECH_RESEARCH", 
            priority=1,
            parallel=True,
            agent_role="sme",
            tasks=[Task(id="research_stack", description="Research technology stack")]
        ),
        SubPhase(
            name="RISK_ASSESSMENT",
            priority=1,
            parallel=True,
            agent_role="analyst",
            tasks=[Task(id="assess_risks", description="Identify risks and constraints")]
        ),
        SubPhase(
            name="SYNTHESIZE",
            priority=2,
            dependencies=["ANALYZE", "TECH_RESEARCH", "RISK_ASSESSMENT"],
            agent_role="planner",
            swarm_gates=["critic"],
            tasks=[
                Task(id="draft_plan", description="Create execution plan"),
                Task(id="critic_review", description="Critic gate review")
            ]
        ),
    ]
    
    state["swarm_state"]["active_subphases"] = [sp.name for sp in subphases]
    
    # For prototype: simulate completion and create a basic plan
    state["plan"] = {
        "requirement": state["requirement"],
        "phases": ["PLANNING", "EXECUTION", "VERIFICATION"],
        "tasks": [
            {"id": "setup", "description": "Setup environment"},
            {"id": "implement", "description": "Implement core features"},
        ],
        "created_at": "2024-01-01T00:00:00Z"
    }
    
    # Mark all planning tasks complete for prototype
    state["completed_tasks"].extend(["analyze_requirement", "research_stack", "assess_risks", "draft_plan"])
    
    # Critic gate validation per STATEMACHINE.md §7.1
    executor = SwarmDAGExecutor()
    plan = state["plan"] or {}
    critic_result = executor.run_critic_gate(plan, state.get("variables", {}))
    state["critic_gate_result"] = critic_result
    
    if critic_result != "APPROVED":
        state["errors"].append(f"Critic gate {critic_result}: Plan requires revision")
        state["previous_phase"] = PhaseName.PLANNING
        return state
    
    # Transition to EXECUTION
    state["previous_phase"] = PhaseName.PLANNING
    state["phase"] = PhaseName.EXECUTION
    return state


def execution_phase(state: SpineState) -> SpineState:
    """Execute the EXECUTION phase with parallel sub-phases."""
    state["phase"] = PhaseName.EXECUTION
    
    state["swarm_state"]["active_subphases"] = ["BACKEND", "FRONTEND"]
    
    # Simulating execution completion for prototype
    state["completed_tasks"].extend(["backend_impl", "backend_tests", "frontend_impl", "frontend_tests"])
    
    # Transition to VERIFICATION
    state["previous_phase"] = PhaseName.EXECUTION
    state["phase"] = PhaseName.VERIFICATION
    return state


def verification_phase(state: SpineState) -> SpineState:
    """Execute the VERIFICATION phase."""
    state["phase"] = PhaseName.VERIFICATION
    
    # Quality gates
    state["tasks"]["syntax_check"] = Task(
        id="syntax_check",
        description="Verify syntax correctness",
        status=StateStatus.SUCCESS
    )
    state["tasks"]["lint_check"] = Task(
        id="lint_check", 
        description="Run linter checks",
        status=StateStatus.SUCCESS
    )
    state["tasks"]["drift_check"] = Task(
        id="drift_check",
        description="Verify plan drift",
        status=StateStatus.SUCCESS
    )
    
    state["completed_tasks"].extend(["syntax_check", "lint_check", "drift_check"])
    
    # Transition to COMPLETE
    state["previous_phase"] = PhaseName.VERIFICATION
    state["phase"] = PhaseName.COMPLETE
    return state


def should_continue(state: SpineState) -> Literal["planning", "execution", "verification", "__end__"]:
    """Determine next phase based on current state."""
    phase = state.get("phase")
    
    if phase == PhaseName.INIT:
        return "planning"
    elif phase == PhaseName.PLANNING:
        # Check critic gate before EXECUTION
        critic_result = state.get("critic_gate_result")
        if critic_result == "APPROVED":
            return "execution"
        else:
            return "planning"  # Return to PLANNING for revision
    elif phase == PhaseName.EXECUTION:
        return "verification"
    elif phase == PhaseName.VERIFICATION:
        return "__end__"
    else:
        return "__end__"


def create_spine_workflow(checkpoint_path: str = ".spine/spine.db"):
    """Create the SPINE workflow with LangGraph StateGraph."""
    import os
    import sqlite3
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Create checkpointer directly with sqlite connection
    # check_same_thread=False required for LangGraph's threading model
    conn = sqlite3.connect(checkpoint_path, check_same_thread=False)
    memory = SqliteSaver(conn)

    # Build the state graph
    workflow = StateGraph(SpineState)
    
    # Add nodes (phases)
    workflow.add_node("init", init_phase)
    workflow.add_node("planning", planning_phase)
    workflow.add_node("execution", execution_phase)
    workflow.add_node("verification", verification_phase)
    
    # Add edges
    workflow.add_edge("init", "planning")
    workflow.add_conditional_edges(
        "planning",
        should_continue,
        {"planning": "planning", "execution": "execution", "__end__": END}
    )
    workflow.add_edge("execution", "verification")
    workflow.add_edge("verification", END)
    
    # Set entry point
    workflow.set_entry_point("init")
    
    return workflow.compile(checkpointer=memory)


class SpineStateMachine:
    """High-level interface for SPINE workflows."""
    
    def __init__(self, checkpoint_path: str = ".spine/spine.db"):
        import os
        import sqlite3
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        conn = sqlite3.connect(checkpoint_path, check_same_thread=False)
        self._checkpointer = SqliteSaver(conn)
        self.checkpoint_path = checkpoint_path
        self.app = self._create_compiled_workflow()
    
    def _create_compiled_workflow(self):
        """Create the compiled workflow."""
        workflow = StateGraph(SpineState)
        workflow.add_node("init", init_phase)
        workflow.add_node("planning", planning_phase)
        workflow.add_node("execution", execution_phase)
        workflow.add_node("verification", verification_phase)
        workflow.add_edge("init", "planning")
        workflow.add_conditional_edges(
            "planning",
            should_continue,
            {"planning": "planning", "execution": "execution", "__end__": END}
        )
        workflow.add_edge("execution", "verification")
        workflow.add_edge("verification", END)
        workflow.set_entry_point("init")
        
        return workflow.compile(checkpointer=self._checkpointer)
    
    def run(self, requirement: str, thread_id: str = "default") -> SpineState:
        """Execute the full SPINE workflow."""
        initial_state = SpineState(
            phase=PhaseName.INIT,
            previous_phase=None,
            requirement=requirement,
            plan=None,
            tasks={},
            completed_tasks=[],
            failed_tasks=[],
            swarm_state={},
            hive_cells={},
            swarm_events=[],
            variables={},
            errors=[],
            providers={},
            critic_gate_result=None
        )
        
        result = self.app.invoke(
            initial_state,
            {"configurable": {"thread_id": thread_id}}
        )
        return result
    
    def resume(self, thread_id: str = "default") -> Optional[SpineState]:
        """Resume a previous workflow."""
        state = self.app.get_state({"configurable": {"thread_id": thread_id}})
        if state and "values" in state:
            return state["values"]
        return None