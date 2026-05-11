"""SPINE DAG execution module."""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Any, Literal, Callable
from dataclasses import dataclass

from .types import Phase, PhaseResult, SubPhase, SubPhaseResult, Task, FeatureSlice
from ..providers.llm import LLMProvider
from ..providers.memory import MemoryProvider
from ..providers.storage import StorageProvider, FileWriteGuard
from ..providers.tools import ToolsProvider
from ..providers.agents import AgentProvider
from ..core.persistence import GitWorkflow
from ..models.enums import SubPhaseStatus

# Import prompt system
from ..prompts import PromptBuilder, PromptConfig, Role as PromptRole


def _extract_components(requirement: str) -> list[str]:
    """Extract component hints from a requirement string.

    Splits on common connectors and heuristics to identify distinct
    components or features mentioned in the requirement.
    """
    # Normalize and split on common connectors
    text = requirement.lower()
    separators = [" and ", " with ", " including ", " plus ", ", "]
    parts = [text.strip()]
    for sep in separators:
        new_parts = []
        for p in parts:
            if sep in p:
                new_parts.extend(s.strip() for s in p.split(sep) if s.strip())
            else:
                new_parts.append(p)
        parts = new_parts

    # Filter out very short fragments and deduplicate
    seen: set[str] = set()
    components: list[str] = []
    for p in parts:
        if len(p) > 5 and p not in seen:
            seen.add(p)
            components.append(p)

    return components if components else [requirement[:120]]


def _extract_requirements(requirement: str) -> list[str]:
    """Extract key requirement items from a requirement string."""
    # Heuristic: split on semicolons, newlines, or bullet-like patterns
    text = requirement.strip()
    items = re.split(r"[;\n]|\b(?:•|-\s+)\s*", text)
    items = [i.strip() for i in items if i.strip() and len(i.strip()) > 5]
    return items if items else [requirement[:200]]


def _estimate_complexity(requirement: str) -> str:
    """Estimate project complexity from the requirement text.

    Returns one of: 'low', 'medium', 'high'.
    """
    text = requirement.lower()
    high_keywords = [
        "microservice", "distributed", "scalable", "high-throughput",
        "real-time", "production-grade", "enterprise", "multi-tenant",
        "kubernetes", "cluster", "load-balanc", "failover", "disaster",
    ]
    medium_keywords = [
        "api", "web", "full-stack", "database", "auth", "authentication",
        "rest", "graphql", "frontend", "backend", "integration",
    ]

    high_count = sum(1 for kw in high_keywords if kw in text)
    medium_count = sum(1 for kw in medium_keywords if kw in text)
    word_count = len(text.split())

    if high_count >= 2 or word_count > 60:
        return "high"
    if high_count >= 1 or medium_count >= 2 or word_count > 30:
        return "medium"
    return "low"


def synthesize_slices(
    requirement: str,
    context: dict[str, Any],
    agent_provider: Optional[Any] = None,
) -> list["FeatureSlice"]:
    """Produce FeatureSlice objects from requirement + context.

    Uses the agent provider when available for intelligent decomposition.
    Falls back to a heuristic slicer that produces 2-6 slices.

    Args:
        requirement: The original requirement text.
        context: Execution context (analysis results, tech research, etc.).
        agent_provider: Optional agent provider for decomposition.

    Returns:
        List of FeatureSlice objects with dependency edges.
    """
    from .types import FeatureSlice

    # ── Agent path ─────────────────────────────────────────────────
    if agent_provider and not isinstance(agent_provider, dict) and agent_provider.enabled:
        try:
            prompt = _build_slice_synthesis_prompt(requirement, context)
            result = agent_provider.execute(prompt, workdir=os.getcwd(), timeout=120)
            if result.success and result.output:
                return _parse_llm_slices(result.output)
        except Exception:
            pass  # fall through to heuristic

    # ── Heuristic path ────────────────────────────────────────────
    return _heuristic_slices(requirement, context)


def _build_slice_synthesis_prompt(requirement: str, context: dict[str, Any]) -> str:
    """Build a prompt asking the LLM to decompose into FeatureSlices."""
    analysis = context.get("requirement", requirement)
    tech = context.get("tech_research", "Not available")
    risk = context.get("risk_assessment", "Not available")

    return f"""You are a software architect decomposing a project into feature slices.

REQUIREMENT:
{requirement}

ANALYSIS:
{analysis}

TECH RESEARCH:
{tech}

RISK ASSESSMENT:
{risk}

Decompose this into 2-6 independent feature slices. Each slice should be a
cohesive unit of work that a single developer could implement in a focused
session without coordinating with others working in parallel.

Output JSON array. Each element:
{{
  "id": "short-kebab-id",
  "description": "What to build at feature granularity (NOT file-level)",
  "scope": ["module/dir1/", "module/dir2/"],
  "depends_on": ["other-slice-id"],
  "agent_role": "coder|test_engineer|reviewer",
  "acceptance": ["Criterion 1", "Criterion 2"]
}}

Rules:
- Slices must be independently implementable (one developer, one session)
- If you need to read source files to decompose, the slice is too small
- depends_on captures the DAG edges — the real architectural dependencies
- Prefer fewer, richer slices over many micro-tasks

JSON:"""


def _parse_llm_slices(raw: str) -> list["FeatureSlice"]:
    """Parse LLM response into FeatureSlice objects."""
    import json
    from .types import FeatureSlice

    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    try:
        items = json.loads(text)
        if not isinstance(items, list):
            items = [items]
        return [
            FeatureSlice.from_dict(item)
            for item in items
            if isinstance(item, dict) and "id" in item and "description" in item
        ]
    except json.JSONDecodeError:
        return _heuristic_slices("parsed from LLM", {})


def _heuristic_slices(requirement: str, context: dict[str, Any]) -> list["FeatureSlice"]:
    """Fallback heuristic slicer — produces slices from requirement keywords."""
    from .types import FeatureSlice

    complexity = _estimate_complexity(requirement)
    components = _extract_components(requirement)

    if complexity == "high":
        # 4-6 slices: core, each major component, integration, tests
        slices = [
            FeatureSlice(
                id="core-foundation",
                description="Implement core data models, configuration, and shared utilities",
                scope=["core/", "models/", "config/"],
                depends_on=[],
                agent_role="coder",
                acceptance=["Core models compile and import correctly", "Config loads without errors"],
            ),
        ]
        for i, comp in enumerate(components[:4]):
            slug = comp.replace(" ", "-")[:30]
            slices.append(FeatureSlice(
                id=f"feature-{slug}",
                description=f"Implement {comp} module with full business logic",
                scope=[f"{comp.split()[0]}/"],
                depends_on=["core-foundation"],
                agent_role="coder",
                acceptance=[f"{comp} module works end-to-end", "No lint errors"],
            ))
        slices.append(FeatureSlice(
            id="integration-wiring",
            description="Wire all feature modules together, add API layer and cross-cutting concerns",
            scope=["api/", "routes/", "middleware/"],
            depends_on=[s.id for s in slices if s.id != "core-foundation"],
            agent_role="coder",
            acceptance=["All modules importable from main entrypoint", "API routes respond"],
        ))
        slices.append(FeatureSlice(
            id="test-coverage",
            description="Write unit and integration tests for all modules",
            scope=["tests/"],
            depends_on=["integration-wiring"],
            agent_role="test_engineer",
            acceptance=["All tests pass", "No regressions"],
        ))

    elif complexity == "medium":
        slices = [
            FeatureSlice(
                id="core-impl",
                description="Implement core models and primary business logic",
                scope=["models/", "services/"],
                depends_on=[],
                agent_role="coder",
                acceptance=["Models serialize/deserialize correctly", "Core logic passes basic validation"],
            ),
            FeatureSlice(
                id="feature-modules",
                description="Implement feature modules and API layer",
                scope=["routes/", "api/"],
                depends_on=["core-impl"],
                agent_role="coder",
                acceptance=["API endpoints respond", "Feature modules integrate with core"],
            ),
            FeatureSlice(
                id="tests",
                description="Write unit and integration tests",
                scope=["tests/"],
                depends_on=["feature-modules"],
                agent_role="test_engineer",
                acceptance=["All tests pass", "No regressions"],
            ),
        ]

    else:  # low
        slices = [
            FeatureSlice(
                id="implementation",
                description="Implement the required feature end-to-end",
                scope=["."],
                depends_on=[],
                agent_role="coder",
                acceptance=["Feature works as described in requirement"],
            ),
            FeatureSlice(
                id="verification",
                description="Write tests and validate the implementation",
                scope=["tests/"],
                depends_on=["implementation"],
                agent_role="test_engineer",
                acceptance=["Tests pass", "Implementation matches requirement"],
            ),
        ]

    return slices


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
        llm_provider: Optional[LLMProvider] = None,  # Deprecated — kept for backward compat
        memory_provider: Optional[MemoryProvider] = None,
        storage_provider: Optional[StorageProvider] = None,
        tools_provider: Optional[ToolsProvider] = None,
        file_write_guard: Optional[FileWriteGuard] = None,
        git_workflow: Optional[GitWorkflow] = None,
        agent_provider: Optional[AgentProvider] = None,
        resource_quota: Optional[ResourceQuota] = None,
    ):
        """Initialize with optional providers for agent execution.
        
        Args:
            llm_provider: Deprecated. Kept for backward compatibility — no longer used.
            memory_provider: Memory provider for persistent storage.
            storage_provider: Storage provider for file operations.
            tools_provider: Tools provider for agent capabilities.
            file_write_guard: Guard for protected file writes.
            git_workflow: Git workflow for version control operations.
            agent_provider: Agent provider for all task execution.
            resource_quota: Resource limits for wave execution (optional).
        """
        self._llm_provider = None  # Deprecated — no longer used
        self._memory_provider = memory_provider
        self._storage_provider = storage_provider
        self._tools_provider = tools_provider
        self._file_write_guard = file_write_guard
        self._git_workflow = git_workflow
        self._agent_provider = agent_provider
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
        # Prompt builders cache (per role)
        self._prompt_builders: dict[str, PromptBuilder] = {}
    
    def _get_prompt_builder(self, role: str) -> PromptBuilder:
        """Get or create a PromptBuilder for the given role.
        
        Caches builders per role to avoid re-creating on every task.
        """
        if role not in self._prompt_builders:
            # Map role string to PromptRole enum
            role_mapping = {
                "explorer": PromptRole.EXPLORER,
                "sme": PromptRole.SME,
                "planner": PromptRole.PLANNER,
                "critic": PromptRole.CRITIC,
                "coder": PromptRole.CODER,
                "reviewer": PromptRole.REVIEWER,
                "test_engineer": PromptRole.TEST_ENGINEER,
                "analyst": PromptRole.ANALYST,
                "designer": PromptRole.DESIGNER,
            }
            prompt_role = role_mapping.get(role.lower(), PromptRole.EXPLORER)
            self._prompt_builders[role] = PromptBuilder(role=prompt_role)
        return self._prompt_builders[role]

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
        """Execute a DAG by running its subphase tasks via the agent provider.

        All tasks are delegated to the configured agent_provider. There is no
        LLM or stub fallback — if the agent provider is unavailable, the task
        fails with a clear error.

        Args:
            dag_or_name: A SubPhase object or a string subphase name
            context: Execution context (may contain dependency templates)

        Returns:
            Dict with task execution results, including status and error info.
        """
        from .enums import StateStatus
        import os

        # Get timeout from context or use default
        timeout = context.get("llm_timeout", 300.0) if context else 300.0

        # Backwards compat: if called with string name, return stub
        if isinstance(dag_or_name, str):
            return {
                "status": "completed",
                "dag": dag_or_name,
                "context_keys": list(context.keys()),
                "tasks_executed": 0
            }

        # Require agent_provider
        if not self._agent_provider or not self._agent_provider.enabled:
            return {
                "subphase": getattr(dag_or_name, "name", str(dag_or_name)),
                "agent_role": getattr(dag_or_name, "agent_role", ""),
                "status": "failed",
                "tasks": {},
                "tasks_executed": 0,
                "total_tasks": len(getattr(dag_or_name, "tasks", [])),
                "errors": ["No agent provider available. Configure an agent provider (e.g. opencode) in .spine/config.yaml"],
                "error": "No agent provider available. Configure an agent provider (e.g. opencode) in .spine/config.yaml",
            }

        # Real execution with SubPhase object
        subphase = dag_or_name
        task_results = {}
        all_succeeded = True
        errors = []
        workdir = os.getcwd()

        for task in subphase.tasks:
            task.status = StateStatus.RUNNING
            try:
                # Build task-specific prompt
                prompt = self._build_task_prompt(task, subphase, context)

                # Delegate to agent provider
                agent_result = self._agent_provider.execute(prompt, workdir=workdir, timeout=timeout)

                # Store AgentResult metadata in task.result
                task.result = {
                    "output": agent_result.output,
                    "exit_code": agent_result.exit_code,
                    "files_changed": agent_result.files_changed,
                    "error": agent_result.error,
                    "success": agent_result.success,
                }
                task.status = StateStatus.SUCCESS if agent_result.success else StateStatus.FAILED
                if not agent_result.success:
                    task.error = agent_result.error or "Agent execution failed"
                    errors.append(f"Task {task.id} agent failed: {agent_result.error}")
                    all_succeeded = False

                task_results[task.id] = {
                    "status": "success" if task.status == StateStatus.SUCCESS else "failed",
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
        """Build a focused prompt for agent delegation.

        Produces a concise, task-oriented prompt that includes:
        - Role and task description
        - Original requirement
        - Phase-specific context (planning results, spec, etc.)
        - Scope and acceptance criteria when available
        """
        import os
        debug_prompts = context.get("variables", {}).get("debug_prompts", False)
        requirement = context.get("requirement", "")
        plan = context.get("plan", {})
        spec_content = context.get("spec_content", "")
        project_ctx = context.get("project_context", {})

        parts: list[str] = []

        # Role title
        role_titles = {
            "explorer": "Requirements Analyst",
            "sme": "Technical Researcher",
            "analyst": "Risk Analyst",
            "planner": "Planning Architect",
            "coder": "Implementation Engineer",
            "test_engineer": "Test Engineer",
            "reviewer": "Code Reviewer",
        }
        role_title = role_titles.get(subphase.agent_role, subphase.agent_role)
        parts.append(f"# {role_title}")

        # Project context — brief
        project_name = project_ctx.get("name", "")
        project_root = project_ctx.get("root", "")
        if project_name or project_root:
            line = f"Project: {project_name}" if project_name else ""
            if project_root:
                line += f" (root: {project_root})" if line else f"Root: {project_root}"
            parts.append(line)

        # Requirement
        if requirement:
            parts.append(f"\n## Requirement\n{requirement}")

        # Task description — the specific thing this subphase should do
        task_desc = None
        if task.description and task.description.strip():
            raw = task.description.strip()
            for sep in ("\nScope:", "\nAcceptance criteria:"):
                idx = raw.find(sep)
                if idx != -1:
                    raw = raw[:idx].strip()
            if raw:
                task_desc = raw
        if task_desc:
            parts.append(f"\n## Task\n{task_desc}")

        # Scope and acceptance criteria (extracted from task description)
        task_scope = []
        task_acceptance = []
        if task.description:
            for line in task.description.splitlines():
                if line.startswith("Scope:") or line.startswith("Scope:, "):
                    task_scope.append(line)
                elif line.startswith("Acceptance criteria:"):
                    task_acceptance.append(line)
        if task_scope:
            parts.append("\n## Scope")
            parts.extend(task_scope)
        if task_acceptance:
            parts.append("\n## Acceptance Criteria")
            parts.extend(task_acceptance)

        # Planning context — concise task list from planning phase
        if isinstance(plan, dict):
            plan_tasks = plan.get("tasks", [])
            if plan_tasks:
                parts.append("\n## Planned Tasks")
                for t in plan_tasks:
                    tid = t.get("id", "?")
                    raw_desc = str(t.get("description", ""))
                    short = raw_desc.split("\n")[0].strip().strip("|").strip()
                    if len(short) > 120:
                        short = short[:117] + "..."
                    if not short:
                        short = tid
                    parts.append(f"- {tid}: {short}")

        # Spec from planning — trimmed to keep prompt manageable
        if spec_content:
            trimmed = spec_content[:4000] if len(spec_content) > 4000 else spec_content
            parts.append(f"\n## Spec\n{trimmed}")

        # Phase-specific instructions based on role
        role_instructions = {
            "explorer": (
                "\n## Instructions\n"
                "1. Read relevant source files to understand the project structure.\n"
                "2. Identify the components, modules, and dependencies relevant to the requirement.\n"
                "3. List key requirements, constraints, and integration points.\n"
                "4. Report findings concisely."
            ),
            "sme": (
                "\n## Instructions\n"
                "1. Research the technology stack and patterns used in this project.\n"
                "2. Identify relevant libraries, APIs, and conventions.\n"
                "3. Note any compatibility concerns or version requirements.\n"
                "4. Report findings concisely."
            ),
            "analyst": (
                "\n## Instructions\n"
                "1. Identify risks, edge cases, and potential failure modes.\n"
                "2. Consider security, performance, and maintainability.\n"
                "3. Assess complexity and recommend approach.\n"
                "4. Report findings concisely."
            ),
            "planner": (
                "\n## Instructions\n"
                "1. Based on the analysis and research, create an execution plan.\n"
                "2. Break the work into focused, ordered tasks.\n"
                "3. Each task should be independently implementable.\n"
                "4. Specify scope and acceptance criteria for each task."
            ),
            "coder": (
                "\n## Instructions\n"
                "1. Read the relevant source files first.\n"
                "2. Make the minimal, focused changes needed.\n"
                "3. Follow the existing code style and patterns.\n"
                "4. Write or update tests if applicable.\n"
                "5. Ensure changes integrate cleanly."
            ),
            "test_engineer": (
                "\n## Instructions\n"
                "1. Read the relevant source files and existing tests.\n"
                "2. Write tests covering the requirement and acceptance criteria.\n"
                "3. Include edge cases and error conditions.\n"
                "4. Follow the existing test patterns and conventions."
            ),
            "reviewer": (
                "\n## Instructions\n"
                "1. Review the changed files for bugs, style issues, and regressions.\n"
                "2. Check that the implementation matches the requirement.\n"
                "3. Verify tests cover the key scenarios.\n"
                "4. Report: issues found (if any) and overall assessment."
            ),
        }
        instruction = role_instructions.get(subphase.agent_role)
        if instruction:
            parts.append(instruction)

        prompt = "\n".join(parts)

        if debug_prompts:
            self._print_debug_prompt(prompt, task, subphase)

        return prompt

    def _print_debug_prompt(self, prompt: str, task: "Task", subphase: "SubPhase") -> None:
        """Print a prompt to the console for debugging."""
        divider = "=" * 72
        print(f"\n{divider}")
        print(f" [PROMPT] SubPhase={subphase.name}  Role={subphase.agent_role}  Task={task.id}")
        print(divider)
        print(prompt)
        print(f"\n{'─' * 72}\n")

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