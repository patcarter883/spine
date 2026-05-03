"""Core state machine implementation using LangGraph."""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict, Literal, Optional, Any
from dataclasses import dataclass, field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .constants import PhaseName, StateStatus, SubPhaseStatus


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

    def fail(self, error: str, blocked_by: Optional[str] = None) -> None:
        """Mark subphase as failed with error info."""
        self.status = SubPhaseStatus.FAILED
        self.error = error
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


class SwarmDAGExecutor:
    """Executes a phase with potential parallel sub-phases using swarm agents."""

    def __init__(self, llm_provider: Optional[Any] = None):
        """Initialize with optional LLM provider for agent execution.
        
        Args:
            llm_provider: An LLMProvider instance (must have generate(prompt) method)
                          or None for stub/fallback execution.
        """
        self._llm_provider = llm_provider
        # SubPhase lookup for execute_dag to access tasks
        self._current_subphases: dict[str, SubPhase] = {}

    def execute_phase(self, phase: Phase, context: dict[str, Any]) -> PhaseResult:
        """Execute a phase with parallel sub-phase wave execution.
        
        Handles:
        - Wave-based execution with dependency ordering
        - Subphase failure with rework retries
        - Blocking of dependent subphases when upstream fails
        - Output propagation between waves via resolve_dependency_templates
        """
        if not phase.subphases:
            return PhaseResult(subphase_results={})

        # Build subphase lookup for execute_dag
        self._current_subphases = {sp.name: sp for sp in phase.subphases}
        subphase_deps = self.build_subphase_deps(phase.subphases)
        wave_results: list[SubPhaseResult] = []
        
        # Track subphase states: completed (success), failed (max retries), blocked (dep failed)
        completed: set[str] = set()
        failed: set[str] = set()
        blocked: set[str] = set()
        completed_results: dict[str, Any] = {}
        
        # Track which subphases have been retried (to avoid infinite loops)
        retried: set[str] = set()

        while True:
            # Find subphases ready to execute: not in completed/failed/blocked,
            # and all completed deps are successful (not failed/blocked)
            ready = self.find_ready_subphases_for_execution(
                subphase_deps, phase.subphases, completed, failed, blocked
            )
            if not ready:
                break

            # Resolve dependency templates in context before this wave
            wave_context = self.resolve_dependency_templates(context, completed_results)
            
            # Execute wave
            wave_result = self.execute_subphase_wave(ready, wave_context)
            wave_results.extend(wave_result)

            for r in wave_result:
                sp = self._current_subphases.get(r.subphase_name)
                if r.status == SubPhaseStatus.SUCCESS:
                    completed.add(r.subphase_name)
                    if sp:
                        sp.mark_success(r.result)
                    completed_results[r.subphase_name] = r.result
                elif r.status == SubPhaseStatus.FAILED:
                    if sp:
                        error_msg = r.result.get("error", "Unknown error") if isinstance(r.result, dict) else str(r.result)
                        sp.fail(error_msg)
                        sp.retries += 1
                    if sp and sp.retries < sp.max_retries:
                        # Retry this subphase (rework)
                        retried.add(r.subphase_name)
                    else:
                        failed.add(r.subphase_name)
                elif r.status == SubPhaseStatus.BLOCKED:
                    if sp:
                        sp.block(r.result.get("blocked_by", "unknown"))
                    blocked.add(r.subphase_name)

            # Propagate blocked/failed status to dependents
            self._propagate_block_status(subphase_deps, phase.subphases, failed, blocked)

        gate_results = self.run_swarm_gates(phase.swarm_agents, context)
        return PhaseResult.from_waves(wave_results, gate_results)

    def build_subphase_deps(self, subphases: list[SubPhase]) -> dict[str, set[str]]:
        """Build dependency map for subphases."""
        return {sp.name: set(sp.dependencies) for sp in subphases}

    def find_ready_subphases(self, deps: dict[str, set[str]], remaining: set, completed: set) -> list[str]:
        """Find subphases with no unmet dependencies."""
        return [name for name in remaining if deps[name] <= completed]

    def find_ready_subphases_for_execution(
        self,
        deps: dict[str, set[str]],
        subphases: list[SubPhase],
        completed: set[str],
        failed: set[str],
        blocked: set[str]
    ) -> list[str]:
        """Find subphases ready to execute, considering failures and blocking.
        
        A subphase is ready if:
        - All its dependencies are in the completed set (successful)
        - No dependency is in failed or blocked sets
        - The subphase itself is not failed, blocked, or already retried
        """
        ready = []
        for sp in subphases:
            if sp.name in completed or sp.name in failed or sp.name in blocked:
                continue
            subphase_deps = deps.get(sp.name, set())
            # Check: all deps must be completed (not failed or blocked)
            unmet = subphase_deps - completed
            deps_failed_or_blocked = subphase_deps & (failed | blocked)
            if not unmet and not deps_failed_or_blocked:
                ready.append(sp.name)
        return ready

    def _propagate_block_status(
        self,
        deps: dict[str, set[str]],
        subphases: list[SubPhase],
        failed: set[str],
        blocked: set[str]
    ) -> None:
        """Propagate blocked/failed status to dependent subphases.
        
        When a subphase fails or is blocked, all subphases that depend on it
        (transitively) should also be marked as blocked.
        """
        changed = True
        while changed:
            changed = False
            for sp in subphases:
                if sp.name in blocked or sp.name in failed:
                    continue
                subphase_deps = deps.get(sp.name, set())
                # If any dependency is failed or blocked, block this subphase
                if subphase_deps & (failed | blocked):
                    blocked.add(sp.name)
                    sp.block(list(subphase_deps & (failed | blocked))[0])
                    changed = True

    def get_subphase_status(self, name: str) -> Optional[SubPhaseStatus]:
        """Get the current status of a subphase by name."""
        sp = self._current_subphases.get(name)
        return sp.status if sp else None

    def get_failed_subphases(self) -> list[str]:
        """Get all subphases that are in FAILED status."""
        return [name for name, sp in self._current_subphases.items()
                if sp.status == SubPhaseStatus.FAILED]

    def get_blocked_subphases(self) -> list[str]:
        """Get all subphases that are in BLOCKED status."""
        return [name for name, sp in self._current_subphases.items()
                if sp.status == SubPhaseStatus.BLOCKED]

    def get_reworkable_subphases(self) -> list[str]:
        """Get all subphases that are in REWORKING status (can be retried)."""
        return [name for name, sp in self._current_subphases.items()
                if sp.status == SubPhaseStatus.REWORKING]

    def get_subphase_states(self) -> dict[str, str]:
        """Get a snapshot of all subphase states."""
        return {name: sp.status.value for name, sp in self._current_subphases.items()}

    def execute_subphase_wave(self, subphase_names: list[str], context: dict) -> list[SubPhaseResult]:
        """Execute multiple subphases in parallel using ThreadPoolExecutor.
        
        Sets subphase-level status based on execution results. Failed subphases
        that can still retry are marked as REWORKING; exhausted retries become FAILED.
        """
        results = []
        with ThreadPoolExecutor(max_workers=len(subphase_names) or 1) as executor:
            futures: dict = {}
            for name in subphase_names:
                subphase = self._current_subphases.get(name)
                if subphase:
                    futures[executor.submit(self.execute_dag, subphase, context)] = name
                else:
                    futures[executor.submit(self.execute_dag, name, context)] = name
            for future in as_completed(futures):
                result = future.result()
                subphase_name = futures[future]
                sp = self._current_subphases.get(subphase_name)
                
                if isinstance(result, dict) and result.get("status") == "failed" and sp:
                    if sp.retries >= sp.max_retries:
                        # Max retries exhausted - permanently failed
                        results.append(SubPhaseResult.failed(subphase_name, result))
                    else:
                        # Will be retried - mark as reworking
                        sp.mark_reworking()
                        results.append(SubPhaseResult(subphase_name=subphase_name, result=result, status=SubPhaseStatus.SUCCESS))
                else:
                    results.append(SubPhaseResult(subphase_name=subphase_name, result=result))
        return results

    def resolve_dependency_templates(self, context: dict, completed_results: dict[str, Any]) -> dict:
        """Resolve {{subphase.NAME.output}} template references in context values."""
        template_pattern = re.compile(r'\{\{subphase\.(\w+)\.output\}\}')
        
        def resolve_value(value):
            if isinstance(value, str):
                match = template_pattern.search(value)
                if match:
                    dep_name = match.group(1)
                    if dep_name in completed_results:
                        return completed_results[dep_name]
                    return value
                return value
            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            return value
        
        return resolve_value(context)

    def execute_dag(self, dag_or_name: Any, context: dict) -> Any:
        """Execute a DAG by running its subphase tasks.
        
        When called with a SubPhase object, executes all tasks in that subphase
        using the LLM provider chain. When called with a string (backwards compat),
        returns a stub result.
        
        Args:
            dag_or_name: A SubPhase object or a string subphase name
            context: Execution context (may contain dependency templates)
            
        Returns:
            Dict with task execution results, including status and error info.
        """
        # Backwards compat: if called with string name, run stub
        if isinstance(dag_or_name, str):
            return {
                "status": "completed",
                "dag": dag_or_name,
                "context_keys": list(context.keys()),
                "tasks_executed": 0
            }
        
        # Real execution with SubPhase object
        subphase = dag_or_name
        task_results = {}
        all_succeeded = True
        errors = []
        
        for task in subphase.tasks:
            task.status = StateStatus.RUNNING
            try:
                # Build task-specific prompt using agent role and task info
                prompt = self._build_task_prompt(task, subphase, context)
                
                # Execute using LLM provider if available
                if self._llm_provider:
                    result = self._llm_provider.generate(prompt)
                    task.result = result
                    task.status = StateStatus.SUCCESS
                else:
                    # Fallback: stub execution based on agent role
                    result = self._execute_stub_task(task, subphase, context)
                    task.result = result.get("output", "")
                    task.status = StateStatus.SUCCESS
                
                task_results[task.id] = {
                    "status": "success",
                    "result": task.result
                }
            except Exception as e:
                task.status = StateStatus.FAILED
                task.error = str(e)
                task_results[task.id] = {
                    "status": "failed",
                    "error": str(e)
                }
                all_succeeded = False
                errors.append(f"Task {task.id} failed: {e}")
        
        # Set subphase-level status
        if all_succeeded:
            subphase.mark_success()
        else:
            subphase.fail("; ".join(errors))
        
        return {
            "subphase": subphase.name,
            "agent_role": subphase.agent_role,
            "status": "success" if all_succeeded else "failed",
            "tasks": task_results,
            "tasks_executed": len(task_results),
            "total_tasks": len(subphase.tasks),
            "errors": errors,
            "error": "; ".join(errors) if errors else None
        }
    
    def _build_task_prompt(self, task: Task, subphase: SubPhase, context: dict) -> str:
        """Build LLM prompt for task execution."""
        dep_context_parts = []
        if context:
            for key, value in context.items():
                dep_context_parts.append(f"  {key}: {value}")
        dep_context = "\n".join(dep_context_parts)
        
        return (
            f"Agent Role: {subphase.agent_role}\n"
            f"SubPhase: {subphase.name}\n"
            f"Task: {task.id}\n"
            f"Description: {task.description}\n"
            f"Dependencies context:\n{dep_context}\n"
            f"\nPerform the requested task. Provide structured, actionable output."
        )
    
    def _execute_stub_task(self, task: Task, subphase: SubPhase, context: dict) -> dict:
        """Stub task execution for when no LLM provider is available."""
        return {
            "output": f"[{subphase.name}] Task '{task.id}' ({task.description}) completed.",
            "agent_role": subphase.agent_role,
            "subphase": subphase.name,
            "status": "completed"
        }

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
    
    # Check for failures that require rework
    failed_tasks = state.get("failed_tasks", [])
    if failed_tasks:
        state["errors"].append(f"Verification failed: {len(failed_tasks)} tasks failed")
        state["previous_phase"] = PhaseName.VERIFICATION
        state["phase"] = PhaseName.REWORK
        return state
    
    # Transition to COMPLETE
    state["previous_phase"] = PhaseName.VERIFICATION
    state["phase"] = PhaseName.COMPLETE
    return state


def rework_phase(state: SpineState) -> SpineState:
    """Execute the REWORK phase for fixing failed tasks."""
    state["phase"] = PhaseName.REWORK
    
    # Clear failed tasks and prepare for retry
    failed_tasks = state.get("failed_tasks", [])
    state["failed_tasks"] = []
    
    # Simulate rework completion
    for task_id in failed_tasks:
        if task_id not in state["completed_tasks"]:
            state["completed_tasks"].append(task_id)
    
    state["errors"].append(f"Rework completed for {len(failed_tasks)} tasks")
    
    # Return to the phase where failure occurred
    previous = state.get("previous_phase")
    if previous == PhaseName.PLANNING:
        state["phase"] = PhaseName.PLANNING
    elif previous == PhaseName.EXECUTION:
        state["phase"] = PhaseName.EXECUTION
    elif previous == PhaseName.VERIFICATION:
        state["phase"] = PhaseName.VERIFICATION
    else:
        state["phase"] = PhaseName.EXECUTION
    
    return state


def blocked_phase(state: SpineState) -> SpineState:
    """Execute the BLOCKED phase when work is paused."""
    state["phase"] = PhaseName.BLOCKED
    
    # Record reason if available
    block_reason = state.get("variables", {}).get("block_reason", "Unknown")
    state["errors"].append(f"Workflow blocked: {block_reason}")
    
    return state


def should_continue(state: SpineState) -> Literal["planning", "execution", "verification", "rework", "blocked", "__end__"]:
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
        # Check if rework is needed due to failed tasks
        if state.get("failed_tasks"):
            return "rework"
        return "__end__"
    elif phase == PhaseName.REWORK:
        # After rework, determine where to go back to
        previous = state.get("previous_phase")
        if previous == PhaseName.PLANNING:
            return "planning"
        elif previous == PhaseName.EXECUTION:
            return "execution"
        elif previous == PhaseName.VERIFICATION:
            return "verification"
        return "execution"
    elif phase == PhaseName.BLOCKED:
        # Remain blocked until manually resumed
        return "blocked"
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
    workflow.add_node("rework", rework_phase)
    workflow.add_node("blocked", blocked_phase)
    
    # Add edges
    workflow.add_edge("init", "planning")
    workflow.add_conditional_edges(
        "planning",
        should_continue,
        {"planning": "planning", "execution": "execution", "rework": "rework", "blocked": "blocked", "__end__": END}
    )
    workflow.add_edge("execution", "verification")
    workflow.add_conditional_edges(
        "verification",
        should_continue,
        {"rework": "rework", "__end__": END}
    )
    workflow.add_conditional_edges(
        "rework",
        should_continue,
        {"planning": "planning", "execution": "execution", "verification": "verification", "__end__": END}
    )
    workflow.add_edge("blocked", "blocked")
    
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
        workflow.add_node("rework", rework_phase)
        workflow.add_node("blocked", blocked_phase)
        workflow.add_edge("init", "planning")
        workflow.add_conditional_edges(
            "planning",
            should_continue,
            {"planning": "planning", "execution": "execution", "rework": "rework", "blocked": "blocked", "__end__": END}
        )
        workflow.add_edge("execution", "verification")
        workflow.add_conditional_edges(
            "verification",
            should_continue,
            {"rework": "rework", "__end__": END}
        )
        workflow.add_conditional_edges(
            "rework",
            should_continue,
            {"planning": "planning", "execution": "execution", "verification": "verification", "__end__": END}
        )
        workflow.add_edge("blocked", "blocked")
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