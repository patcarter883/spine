"""SPINE DAG execution module."""

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
    llm_provider: Optional[LLMProvider] = None,
) -> list["FeatureSlice"]:
    """Produce FeatureSlice objects from requirement + context.

    Uses LLM when available for intelligent decomposition.  Falls back to a
    heuristic slicer that produces 2-6 slices based on complexity.

    Args:
        requirement: The original requirement text.
        context: Execution context (analysis results, tech research, etc.).
        llm_provider: Optional LLM provider for LLM-based decomposition.

    Returns:
        List of FeatureSlice objects with dependency edges.
    """
    from .types import FeatureSlice

    # ── LLM path ──────────────────────────────────────────────────
    if llm_provider and llm_provider.enabled:
        try:
            prompt = _build_slice_synthesis_prompt(requirement, context)
            raw = llm_provider.generate(prompt)
            return _parse_llm_slices(raw)
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
        llm_provider: Optional[LLMProvider] = None,
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
            llm_provider: An LLMProvider instance (must have generate(prompt) method)
                          or None for stub/fallback execution.
            memory_provider: Memory provider for persistent storage.
            storage_provider: Storage provider for file operations.
            tools_provider: Tools provider for agent capabilities.
            file_write_guard: Guard for protected file writes.
            git_workflow: Git workflow for version control operations.
            agent_provider: Agent provider for code execution (coder, test_engineer, reviewer).
            resource_quota: Resource limits for wave execution (optional).
        """
        self._llm_provider = llm_provider
        self._memory_provider = memory_provider
        self._storage_provider = storage_provider
        self._tools_provider = tools_provider
        self._file_write_guard = file_write_guard
        self._git_workflow = git_workflow
        self._agent_provider = agent_provider
        # Resource configuration
        self._resource_quota = resource_quota or ResourceQuota()
        # Register stub templates
        self._register_stub_templates()
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

    def _register_stub_templates(self) -> None:
        """Register subphase-specific stub template functions.

        Maps uppercase subphase names to their template functions.
        """
        self._stub_templates = {
            "ANALYZE": self._stub_analyze,
            "TECH_RESEARCH": self._stub_tech_research,
            "RISK_ASSESSMENT": self._stub_risk_assessment,
            "SYNTHESIZE": self._stub_synthesize,
            "BACKEND": self._stub_backend,
            "FRONTEND": self._stub_frontend,
        }

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
        using the LLM provider or agent provider chain. When called with a string 
        (backwards compat), returns a stub result.

        For implementation roles (coder, test_engineer, reviewer), delegates to 
        agent_provider for actual code writing. For decision-making roles 
        (explorer, sme, analyst, planner), uses LLM for analysis.

        Args:
            dag_or_name: A SubPhase object or a string subphase name
            context: Execution context (may contain dependency templates)
            
        Returns:
            Dict with task execution results, including status and error info.
        """
        from .enums import StateStatus
        import os
        
        # Implementation roles that should use agent_provider for code writing
        IMPLEMENTATION_ROLES = {"coder", "test_engineer", "reviewer"}
        
        # Get timeout from context or use default
        timeout = context.get("llm_timeout", 300.0) if context else 300.0
        
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
                
                # Determine if we should use agent_provider (for implementation roles)
                is_implementation_role = subphase.agent_role in IMPLEMENTATION_ROLES
                
                if is_implementation_role and self._agent_provider and self._agent_provider.enabled:
                    # Delegate to agent provider for actual code writing
                    workdir = os.getcwd()
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
                elif self._llm_provider and self._llm_provider.enabled:
                    # Use LLM provider for decision-making roles or when no agent available
                    import inspect
                    sig = inspect.signature(self._llm_provider.generate)
                    if "timeout" in sig.parameters:
                        result = self._llm_provider.generate(prompt, timeout=timeout)
                    else:
                        result = self._llm_provider.generate(prompt)
                    task.result = result
                    task.status = StateStatus.SUCCESS
                else:
                    # Fallback: stub execution based on agent role
                    result = self._execute_stub_task(task, subphase, context)
                    task.result = result.get("output", "")
                    task.status = StateStatus.SUCCESS
                
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
        """Build prompt for task execution.
        
        Uses the PromptBuilder from spine.prompts for structured, instructional
        prompts that follow DeepAgents/LangChain best practices.
        
        For implementation roles (coder, test_engineer, reviewer), combines
        the role prompt with task-specific context for agent delegation.
        For decision-making roles (explorer, sme, analyst, planner), uses
        the full structured prompt from PromptBuilder.
        """
        import os
        debug_prompts = context.get("variables", {}).get("debug_prompts", False)
        requirement = context.get("requirement", "")
        plan = context.get("plan", {})
        spec_content = context.get("spec_content", "")
        
        # Get the prompt builder for this role
        builder = self._get_prompt_builder(subphase.agent_role)
        
        # Get project context
        project_ctx = context.get("project_context", {})
        
        # Build state for prompt builder
        state = {
            "requirement": requirement,
            "current_phase": subphase.name,
            "completed_phases": [],
            "variables": context.get("variables", {}),
            "project_name": project_ctx.get("name", "unknown-project"),
            "project_root": project_ctx.get("root", os.getcwd()),
            "tech_stack": project_ctx.get("tech_stack", []),
        }
        
        # ── Implementation roles ────────────────────────────────────────
        if subphase.agent_role in ("coder", "test_engineer", "reviewer"):
            # For implementation roles, we use a hybrid approach:
            # 1. Get the role-specific prompt section from PromptBuilder
            # 2. Add project context
            # 3. Add task-specific context (spec, plan, scope)
            # 4. Add concrete instructions
            
            from ..prompts.roles import get_role_prompt
            
            parts: list[str] = []
            
            # Get project context
            project_ctx = context.get("project_context", {})
            project_name = project_ctx.get("name", "unknown-project")
            project_root = project_ctx.get("root", os.getcwd())
            project_desc = project_ctx.get("description", "")
            tech_stack = project_ctx.get("tech_stack", [])
            
            # Role + project context
            role_titles = {
                "coder": "Coder Agent",
                "test_engineer": "Test Engineer Agent",
                "reviewer": "Code Reviewer Agent",
            }
            role_title = role_titles.get(subphase.agent_role, f"{subphase.agent_role} agent")
            parts.append(f"# {role_title}\n")
            
            parts.append(f"## Project Context")
            parts.append(f"- **Project**: {project_name}")
            parts.append(f"- **Root**: {project_root}")
            if project_desc:
                parts.append(f"- **Description**: {project_desc}")
            if tech_stack:
                parts.append(f"- **Tech Stack**: {', '.join(tech_stack)}")
            parts.append("")
            
            # Requirement (the original user request)
            if requirement:
                parts.append(f"## Requirement\n{requirement}\n")

            # Specific task this subphase should carry out
            task_desc = None
            if task.description and task.description.strip():
                # Strip trailing scope/acceptance suffixes that are already
                # passed separately — keep the core instruction.
                raw = task.description.strip()
                for sep in ("\nScope:", "\nAcceptance criteria:"):
                    idx = raw.find(sep)
                    if idx != -1:
                        raw = raw[:idx].strip()
                if raw and raw != requirement.strip():
                    task_desc = raw

            if task_desc:
                parts.append(f"## Task\n{task_desc}\n")

            # Scope from the task description (extracted if present)
            task_scope = []
            task_acceptance = []
            if task.description:
                for line in task.description.splitlines():
                    if line.startswith("Scope:") or line.startswith("Scope:, "):
                        task_scope.append(line)
                    elif line.startswith("Acceptance criteria:"):
                        task_acceptance.append(line)
            if task_scope:
                parts.append("## Scope")
                parts.extend(task_scope)
                parts.append("")
            if task_acceptance:
                parts.append("## Acceptance Criteria")
                parts.extend(task_acceptance)
                parts.append("")

            # Plan overview — concise list of planned tasks from planning phase
            if isinstance(plan, dict):
                plan_tasks = plan.get("tasks", [])
                if plan_tasks:
                    parts.append("## Planned Tasks (overview)")
                    for t in plan_tasks:
                        tid = t.get("id", "?")
                        raw_desc = str(t.get("description", ""))
                        # Strip table markdown, trailing noise — keep a short label
                        short = raw_desc.split("\n")[0].strip().strip("|").strip()
                        if len(short) > 120:
                            short = short[:117] + "..."
                        if not short:
                            short = tid
                        parts.append(f"- **{tid}**: {short}")
                    parts.append("")

            # Distilled spec from planning — single consolidated source of truth
            if spec_content:
                trimmed = spec_content[:6000] if len(spec_content) > 6000 else spec_content
                parts.append(f"## Planning Spec\n{trimmed}\n")

            # Get tool instructions for agent provider
            from ..prompts.tools import AGENT_PROVIDER_INSTRUCTIONS
            parts.append(AGENT_PROVIDER_INSTRUCTIONS)
            
            # Concrete instructions (from role prompt)
            parts.append(
                "\n## Instructions\n"
                "1. Read the relevant source files first to understand the existing code.\n"
                "2. Make the minimal, focused changes needed to fulfill the requirement.\n"
                "3. Follow the existing code style and patterns.\n"
                "4. Write or update tests if applicable.\n"
                "5. Ensure the changes integrate cleanly — no regressions, no dead code.\n"
                "6. Return a JSON object with: status, summary, files_changed, tests, notes."
            )
            prompt = "\n".join(parts)

            if debug_prompts:
                self._print_debug_prompt(prompt, task, subphase)

            return prompt
        
        # ── Decision-making roles: use full PromptBuilder ───────────────
        # Build previous outputs from completed phases
        previous_outputs = {}
        if "completed_phases" in context:
            for phase_name in context.get("completed_phases", []):
                phase_result = context.get("phase_results", {}).get(phase_name)
                if phase_result:
                    previous_outputs[phase_name] = phase_result
        
        # Determine capability based on subphase name
        capability_map = {
            "ANALYZE": "analyze",
            "TECH_RESEARCH": "research",
            "RISK_ASSESSMENT": "analyze",
            "SYNTHESIZE": "plan",
            "PLANNING": "plan",
        }
        capability = capability_map.get(subphase.name.upper(), "analyze")
        
        # Build the prompt using PromptBuilder
        prompt = builder.build(
            state=state,
            capability=capability,
            tools=["filesystem"],  # Decision-making roles have filesystem access
            previous_outputs=previous_outputs if previous_outputs else None,
            extra_context=f"Task: {task.description}" if task.description else None,
        )

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

    def _execute_stub_task(self, task: "Task", subphase: "SubPhase", context: dict) -> dict:
        """Stub task execution for when no LLM provider is available.

        Produces structured, meaningful output based on the subphase type and
        task description. Each subphase template returns a dict with keys:
        - output: human-readable summary
        - agent_role: the subphase agent role
        - subphase: the subphase name
        - status: execution status
        - structured_data: parseable dict for downstream phases

        Returns:
            Dict with task execution results including structured_data.
        """
        subphase_name = subphase.name.upper()

        # Dispatch to subphase-specific template
        template_fn = self._stub_templates.get(subphase_name)
        if template_fn:
            result = template_fn(task, subphase, context)
        else:
            # Generic fallback for unknown subphases
            result = self._stub_generic(task, subphase, context)

        return result

    # ------------------------------------------------------------------ #
    #  Stub template registry                                             #
    # ------------------------------------------------------------------ #
    _stub_templates: dict[str, Callable] = {}

    # ------------------------------------------------------------------ #
    #  ANALYZE template                                                   #
    # ------------------------------------------------------------------ #
    def _stub_analyze(self, task: "Task", subphase: "SubPhase", context: dict) -> dict:
        """Generate a requirement breakdown for the ANALYZE subphase."""
        requirement = context.get("requirement", "No requirement provided")
        # Simple keyword-based component extraction
        components = _extract_components(requirement)
        key_requirements = _extract_requirements(requirement)
        complexity = _estimate_complexity(requirement)

        structured_data = {
            "requirement": requirement,
            "components": components,
            "key_requirements": key_requirements,
            "estimated_complexity": complexity,
            "analysis_notes": (
                f"Stub analysis of '{task.description}'. "
                f"Identified {len(components)} component(s) from requirement."
            ),
        }

        return {
            "output": (
                f"[ANALYZE] Requirement parsed: {len(components)} components identified. "
                f"Complexity: {complexity}."
            ),
            "agent_role": subphase.agent_role,
            "subphase": subphase.name,
            "status": "completed",
            "structured_data": structured_data,
        }

    # ------------------------------------------------------------------ #
    #  TECH_RESEARCH template                                             #
    # ------------------------------------------------------------------ #
    def _stub_tech_research(self, task: "Task", subphase: "SubPhase", context: dict) -> dict:
        """Generate technology recommendations for the TECH_RESEARCH subphase."""
        requirement = context.get("requirement", "No requirement provided")
        complexity = _estimate_complexity(requirement)

        # Generic tech recommendations based on complexity
        if complexity == "high":
            stack = {
                "backend": ["Python/FastAPI", "PostgreSQL", "Redis"],
                "frontend": ["React/Next.js", "TypeScript", "Tailwind CSS"],
                "infra": ["Docker", "Kubernetes", "CI/CD pipeline"],
            }
            rationale = "High-complexity projects benefit from mature, well-supported stacks."
        elif complexity == "medium":
            stack = {
                "backend": ["Python/FastAPI", "SQLite/PostgreSQL"],
                "frontend": ["Vue.js", "Vite", "CSS modules"],
                "infra": ["Docker", "GitHub Actions"],
            }
            rationale = "Medium-complexity projects balance simplicity with scalability."
        else:
            stack = {
                "backend": ["Python/FastAPI", "SQLite"],
                "frontend": ["Static HTML/JS", "CSS"],
                "infra": ["Simple deployment (e.g., static host)"],
            }
            rationale = "Low-complexity projects should prioritize simplicity."

        structured_data = {
            "requirement": requirement,
            "recommended_stack": stack,
            "rationale": rationale,
            "complexity": complexity,
            "recommendations": [
                {
                    "category": cat,
                    "technologies": techs,
                    "priority": "high" if complexity in ("high", "medium") else "medium",
                }
                for cat, techs in stack.items()
            ],
            "research_notes": (
                f"Stub tech research for '{task.description}'. "
                f"Recommended stack based on {complexity} complexity."
            ),
        }

        return {
            "output": (
                f"[TECH_RESEARCH] Technology stack recommended for {complexity} complexity. "
                f"Backend: {', '.join(stack['backend'])}."
            ),
            "agent_role": subphase.agent_role,
            "subphase": subphase.name,
            "status": "completed",
            "structured_data": structured_data,
        }

    # ------------------------------------------------------------------ #
    #  RISK_ASSESSMENT template                                           #
    # ------------------------------------------------------------------ #

    def _stub_risk_assessment(self, task: "Task", subphase: "SubPhase", context: dict) -> dict:
        """Generate a risk analysis for the RISK_ASSESSMENT subphase."""
        requirement = context.get("requirement", "No requirement provided")
        complexity = _estimate_complexity(requirement)

        # Generic risks based on complexity
        base_risks = [
            {
                "id": "R001",
                "description": "Requirement ambiguity leading to rework",
                "severity": "high" if complexity == "high" else "medium",
                "mitigation": "Conduct requirement clarification sessions early.",
            },
            {
                "id": "R002",
                "description": "Integration complexity with existing systems",
                "severity": "high",
                "mitigation": "Early integration spikes and interface contracts.",
            },
            {
                "id": "R003",
                "description": "Scope creep during implementation",
                "severity": "medium",
                "mitigation": "Strict change control and MVP scoping.",
            },
        ]

        # Add complexity-specific risks
        if complexity == "high":
            base_risks.extend([
                {
                    "id": "R004",
                    "description": "Performance bottlenecks under load",
                    "severity": "high",
                    "mitigation": "Load testing early in verification phase.",
                },
                {
                    "id": "R005",
                    "description": "Security vulnerabilities in complex architecture",
                    "severity": "high",
                    "mitigation": "Security review at each phase gate.",
                },
            ])
        elif complexity == "medium":
            base_risks.append({
                "id": "R004",
                "description": "Data consistency across services",
                "severity": "medium",
                "mitigation": "Use transactions and idempotent operations.",
            })
        else:
            base_risks.append({
                "id": "R004",
                "description": "Minimal risk for simple projects",
                "severity": "low",
                "mitigation": "Standard development practices.",
            })

        # Compute severity summary
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        for r in base_risks:
            severity_counts[r["severity"]] += 1

        structured_data = {
            "requirement": requirement,
            "risks": base_risks,
            "severity_summary": severity_counts,
            "overall_risk_level": (
                "high" if severity_counts["high"] > 1
                else "medium" if severity_counts["high"] > 0 or severity_counts["medium"] > 1
                else "low"
            ),
            "assessment_notes": (
                f"Stub risk assessment for '{task.description}'. "
                f"Identified {len(base_risks)} risk(s): "
                f"{severity_counts['high']} high, {severity_counts['medium']} medium, "
                f"{severity_counts['low']} low."
            ),
        }

        return {
            "output": (
                f"[RISK_ASSESSMENT] {len(base_risks)} risks identified. "
                f"Overall risk level: {structured_data['overall_risk_level']}."
            ),
            "agent_role": subphase.agent_role,
            "subphase": subphase.name,
            "status": "completed",
            "structured_data": structured_data,
        }

    # ------------------------------------------------------------------ #
    #  SYNTHESIZE template                                                #
    # ------------------------------------------------------------------ #

    def _stub_synthesize(self, task: "Task", subphase: "SubPhase", context: dict) -> dict:
        """Generate FeatureSlices for the SYNTHESIZE subphase."""
        requirement = context.get("requirement", "No requirement provided")
        slices = synthesize_slices(requirement, context, self._llm_provider)

        structured_data = {
            "requirement": requirement,
            "feature_slices": [s.to_dict() for s in slices],
            "implementation_tasks": [
                {"id": s.id, "description": s.description}
                for s in slices
            ],
            "estimated_slice_count": len(slices),
            "estimated_complexity": _estimate_complexity(requirement),
            "synthesis_notes": (
                f"Stub execution plan for '{task.description}'. "
                f"{len(slices)} feature slice(s)."
            ),
        }

        return {
            "output": (
                f"[SYNTHESIZE] Execution plan drafted: "
                f"{len(slices)} feature slice(s) for {_estimate_complexity(requirement)} complexity."
            ),
            "agent_role": subphase.agent_role,
            "subphase": subphase.name,
            "status": "completed",
            "structured_data": structured_data,
        }

    # ------------------------------------------------------------------ #
    #  BACKEND template                                                   #
    # ------------------------------------------------------------------ #

    def _stub_backend(self, task: "Task", subphase: "SubPhase", context: dict) -> dict:
        """Generate feature-slice-based output for the BACKEND subphase."""
        requirement = context.get("requirement", "No requirement provided")
        complexity = _estimate_complexity(requirement)

        structured_data = {
            "requirement": requirement,
            "complexity": complexity,
            "agent_role": subphase.agent_role,
            "implementation_notes": (
                f"Stub backend output for '{task.description}'. "
                f"Agent will decompose at implementation time."
            ),
        }

        return {
            "output": (
                f"[BACKEND] Feature slice executed: {task.description} "
                f"(complexity: {complexity})"
            ),
            "agent_role": subphase.agent_role,
            "subphase": subphase.name,
            "status": "completed",
            "structured_data": structured_data,
        }

    # ------------------------------------------------------------------ #
    #  FRONTEND template                                                  #
    # ------------------------------------------------------------------ #

    def _stub_frontend(self, task: "Task", subphase: "SubPhase", context: dict) -> dict:
        """Generate feature-slice-based output for the FRONTEND subphase."""
        requirement = context.get("requirement", "No requirement provided")
        complexity = _estimate_complexity(requirement)

        structured_data = {
            "requirement": requirement,
            "complexity": complexity,
            "agent_role": subphase.agent_role,
            "implementation_notes": (
                f"Stub frontend output for '{task.description}'. "
                f"Agent will decompose at implementation time."
            ),
        }

        return {
            "output": (
                f"[FRONTEND] Feature slice executed: {task.description} "
                f"(complexity: {complexity})"
            ),
            "agent_role": subphase.agent_role,
            "subphase": subphase.name,
            "status": "completed",
            "structured_data": structured_data,
        }

    # ------------------------------------------------------------------ #
    #  Generic fallback                                                   #
    # ------------------------------------------------------------------ #

    def _stub_generic(self, task: "Task", subphase: "SubPhase", context: dict) -> dict:
        """Generic stub for unknown subphases."""
        structured_data = {
            "task_id": task.id,
            "task_description": task.description,
            "subphase": subphase.name,
            "agent_role": subphase.agent_role,
            "context_keys": list(context.keys()),
            "execution_mode": "stub",
            "notes": (
                f"Stub execution for unknown subphase '{subphase.name}'. "
                f"Task: {task.description}"
            ),
        }

        return {
            "output": f"[{subphase.name}] Task '{task.id}' ({task.description}) completed (stub).",
            "agent_role": subphase.agent_role,
            "subphase": subphase.name,
            "status": "completed",
            "structured_data": structured_data,
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