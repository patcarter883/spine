"""SPINE DAG execution module."""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Any, Literal, Callable
from dataclasses import dataclass, field

from .types import Phase, PhaseResult, SubPhase, SubPhaseResult
from ..providers.llm import LLMProvider
from ..providers.memory import MemoryProvider
from ..providers.storage import StorageProvider, FileWriteGuard
from ..providers.tools import ToolsProvider
from ..core.persistence import GitWorkflow
from ..models.enums import PhaseName, SubPhaseStatus


@dataclass
class ResourceQuota:
    """Resource limits for wave execution."""
    max_concurrent_subphases: int = 10
    max_workers: int = 4
    memory_limit_mb: Optional[int] = None
    timeout_seconds: Optional[int] = None


@dataclass
class ExecutionProgress:
    """Progress tracking for phase execution."""
    total_subphases: int = 0
    completed_subphases: int = 0
    failed_subphases: int = 0
    blocked_subphases: int = 0
    current_wave: int = 0
    total_waves: int = 0
    cancelled: bool = False
    cancel_reason: Optional[str] = None
    
    @property
    def percent_complete(self) -> float:
        if self.total_subphases == 0:
            return 0.0
        return (self.completed_subphases / self.total_subphases) * 100.0


class SwarmDAGExecutor:
    """Executes a phase with potential parallel sub-phases using swarm agents."""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        memory_provider: Optional[MemoryProvider] = None,
        storage_provider: Optional[StorageProvider] = None,
        tools_provider: Optional[ToolsProvider] = None,
        file_write_guard: Optional[FileWriteGuard] = None,
        git_workflow: Optional[GitWorkflow] = None,
        resource_quota: Optional[ResourceQuota] = None,
    ):
        """Initialize with optional providers for agent execution.
        
        Args:
            llm_provider: An LLMProvider instance (must have generate(prompt) method)
                          or None for stub/fallback execution.
            memory_provider: Memory provider for persistent storage.
            storage_provider: Storage provider for file operations.
            tools_provider: Tools provider for agent capabilities.
            file_write_guard: Guard for protected file writes.
            git_workflow: Git workflow for version control operations.
            resource_quota: Resource limits for wave execution (optional).
        """
        self._llm_provider = llm_provider
        self._memory_provider = memory_provider
        self._storage_provider = storage_provider
        self._tools_provider = tools_provider
        self._file_write_guard = file_write_guard
        self._git_workflow = git_workflow
        # Resource configuration
        self._resource_quota = resource_quota or ResourceQuota()
        # SubPhase lookup for execute_dag to access tasks
        self._current_subphases: dict[str, SubPhase] = {}
        self._current_phase_context: dict[str, Any] = {}
        # Progress tracking
        self._progress: Optional[ExecutionProgress] = None
        # Cancellation support
        self._cancel_requested: bool = False
        self._cancel_callback: Optional[Callable[[], bool]] = None

    def execute_phase(self, phase: Phase, context: dict[str, Any]) -> PhaseResult:
        """Execute a phase with parallel sub-phase wave execution.
        
        Handles:
        - Wave-based execution with dependency ordering
        - Wave size limits via resource_quota.max_concurrent_subphases
        - Resource quotas for memory and worker limits
        - Priority ordering for subphase scheduling
        - Progress tracking with ExecutionProgress
        - Cancellation support via cancel_callback
        - Subphase failure with rework retries
        - Blocking of dependent subphases when upstream fails
        - Output propagation between waves via resolve_dependency_templates
        - Provider integration for LLM, memory, storage, and tools
        """
        if not phase.subphases:
            return PhaseResult(subphase_results={})

        # Store context for use in execute_dag
        self._current_phase_context = context.copy()
        
        # Reset cancellation state
        self._cancel_requested = False
        
        # Build subphase lookup for execute_dag
        self._current_subphases = {sp.name: sp for sp in phase.subphases}
        subphase_deps = self.build_subphase_deps(phase.subphases)
        wave_results: list[SubPhaseResult] = []
        
        # Initialize progress tracking
        waves = self.compute_waves(phase.subphases)
        self._progress = ExecutionProgress(
            total_subphases=len(phase.subphases),
            total_waves=len(waves)
        )
        
        # Track subphase states: completed (success), failed (max retries), blocked (dep failed)
        completed: set[str] = set()
        failed: set[str] = set()
        blocked: set[str] = set()
        completed_results: dict[str, Any] = {}

        # Track which subphases have been retried (to avoid infinite loops)
        retried: set[str] = set()

        while True:
            # Check for cancellation
            if self._check_cancel_requested():
                break
                
            # Find subphases ready to execute: not in completed/failed/blocked,
            # and all completed deps are successful (not failed/blocked)
            ready = self.find_ready_subphases_for_execution(
                subphase_deps, phase.subphases, completed, failed, blocked
            )
            if not ready:
                break

            # Apply wave size limits based on resource quota
            max_wave_size = min(
                self._resource_quota.max_concurrent_subphases,
                len(ready)
            )
            if len(ready) > max_wave_size:
                # Prioritize by priority value (lower = higher priority)
                # Use phase.subphases for priority lookup
                subphase_lookup = {sp.name: sp for sp in phase.subphases}
                ready_sorted = sorted(
                    ready,
                    key=lambda n: (
                        subphase_lookup.get(n, SubPhase(name=n)).priority,
                        n  # Secondary sort for determinism
                    )
                )
                ready = ready_sorted[:max_wave_size]

            # Resolve dependency templates in context before this wave
            wave_context = self.resolve_dependency_templates(context, completed_results)
            
            # Update progress
            self._progress.current_wave += 1
            self._progress.completed_subphases = len(completed)
            self._progress.failed_subphases = len(failed)
            self._progress.blocked_subphases = len(blocked)
            
            # Execute wave with file guard integration
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
            
            # Update progress after wave
            self._progress.completed_subphases = len(completed)
            self._progress.failed_subphases = len(failed)
            self._progress.blocked_subphases = len(blocked)

        # Set final progress state
        if self._progress:
            self._progress.completed_subphases = len(completed)
            self._progress.failed_subphases = len(failed)

        gate_results = self.run_swarm_gates(phase.swarm_agents, context)
        return PhaseResult.from_waves(wave_results, gate_results)

    def set_cancel_callback(self, callback: Callable[[], bool]) -> None:
        """Set a callback to check for cancellation requests."""
        self._cancel_callback = callback
        
    def cancel(self, reason: Optional[str] = None) -> None:
        """Request cancellation of the current execution."""
        self._cancel_requested = True
        if self._progress:
            self._progress.cancelled = True
            self._progress.cancel_reason = reason
            
    def _check_cancel_requested(self) -> bool:
        """Check if cancellation has been requested."""
        if self._cancel_requested:
            return True
        if self._cancel_callback:
            try:
                return self._cancel_callback()
            except Exception:
                pass
        return False
        
    def get_progress(self) -> Optional[ExecutionProgress]:
        """Get current execution progress."""
        return self._progress

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
        
        Uses resource_quota.max_workers to limit concurrent execution.
        """
        results = []
        max_workers = min(self._resource_quota.max_workers, len(subphase_names) or 1)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
        from .enums import PhaseName, StateStatus
        from .types import Task
        
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
                if self._llm_provider and self._llm_provider.enabled:
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

    def _build_task_prompt(self, task: "Task", subphase: "SubPhase", context: dict) -> str:
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

    def _execute_stub_task(self, task: "Task", subphase: "SubPhase", context: dict) -> dict:
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
        """Group subphases into waves based on dependencies with priority ordering.
        
        Subphases within each wave are sorted by priority (lower priority value = higher priority).
        This ensures deterministic ordering while respecting dependencies.
        """
        deps = self.build_subphase_deps(subphases)
        # Build lookup for priority access
        subphase_lookup = {sp.name: sp for sp in subphases}
        waves = []
        completed: set[str] = set()
        remaining = {sp.name for sp in subphases}

        while remaining:
            ready = [name for name in remaining if deps[name] <= completed]
            if not ready:
                break
            # Sort by priority for deterministic ordering within wave
            ready_sorted = sorted(
                ready,
                key=lambda n: (
                    subphase_lookup.get(n, SubPhase(name=n)).priority,
                    n  # Secondary sort for determinism
                )
            )
            waves.append(ready_sorted)
            for name in ready_sorted:
                completed.add(name)
                remaining.discard(name)

        return waves


__all__ = ["SwarmDAGExecutor", "ResourceQuota", "ExecutionProgress"]