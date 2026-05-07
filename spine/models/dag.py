"""SPINE DAG execution module."""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Any, Literal, Callable
from dataclasses import dataclass

from .types import Phase, PhaseResult, SubPhase, SubPhaseResult, Task
from ..providers.llm import LLMProvider
from ..providers.memory import MemoryProvider
from ..providers.storage import StorageProvider, FileWriteGuard
from ..providers.tools import ToolsProvider
from ..core.persistence import GitWorkflow
from ..models.enums import SubPhaseStatus


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


def _get_synthesis_tasks(phases: list[str]) -> list[tuple[str, list[str]]]:
    """Generate task descriptions and dependencies for synthesis phases."""
    task_map: dict[str, list[tuple[str, list[str]]]] = {
        "SETUP": [
            ("Initialize project structure and dependencies", []),
            ("Configure development environment", []),
        ],
        "CORE_IMPLEMENTATION": [
            ("Implement core data models", ["SETUP"]),
            ("Implement core business logic", ["SETUP"]),
        ],
        "FEATURE_DEVELOPMENT": [
            ("Implement feature module A", ["CORE_IMPLEMENTATION"]),
            ("Implement feature module B", ["CORE_IMPLEMENTATION"]),
            ("Implement feature module C", ["CORE_IMPLEMENTATION"]),
            ("Integrate feature modules", ["CORE_IMPLEMENTATION"]),
        ],
        "INTEGRATION": [
            ("Set up API layer / routes", ["CORE_IMPLEMENTATION"]),
            ("Integrate frontend with backend", ["FEATURE_DEVELOPMENT"]),
        ],
        "VERIFICATION": [
            ("Write unit tests", ["CORE_IMPLEMENTATION"]),
            ("Write integration tests", ["INTEGRATION"]),
            ("Run end-to-end tests", ["INTEGRATION"]),
            ("Performance and security review", ["INTEGRATION"]),
        ],
    }
    tasks: list[tuple[str, list[str]]] = []
    for phase in phases:
        tasks.extend(task_map.get(phase, [(f"Implement {phase}", [])]))
    return tasks


def _generate_backend_file_structure(requirement: str) -> dict[str, list[str]]:
    """Generate a backend file structure based on the requirement."""
    complexity = _estimate_complexity(requirement)

    structure: dict[str, list[str]] = {
        "models": ["models/__init__.py", "models/base.py"],
        "routes": ["routes/__init__.py", "routes/main.py"],
        "services": ["services/__init__.py", "services/core.py"],
        "tests": ["tests/__init__.py", "tests/test_core.py"],
    }

    if complexity in ("high", "medium"):
        structure["models"].extend(["models/schema.py", "models/validators.py"])
        structure["routes"].extend(["routes/auth.py", "routes/api.py"])
        structure["services"].extend(["services/auth.py", "services/utils.py"])
        structure["tests"].extend(["tests/test_auth.py", "tests/test_api.py"])

    if complexity == "high":
        structure["models"].extend(["models/migrations.py", "models/enums.py"])
        structure["services"].extend(["services/cache.py", "services/queue.py"])
        structure["tests"].extend(["tests/test_integration.py", "tests/conftest.py"])

    return structure


def _generate_frontend_file_structure(requirement: str) -> dict[str, list[str]]:
    """Generate a frontend file structure based on the requirement."""
    complexity = _estimate_complexity(requirement)

    structure: dict[str, list[str]] = {
        "components": ["components/__init__.py", "components/Layout.py"],
        "pages": ["pages/__init__.py", "pages/Home.py"],
        "styles": ["styles/__init__.py", "styles/main.css"],
        "tests": ["tests/__init__.py", "tests/test_layout.py"],
    }

    if complexity in ("high", "medium"):
        structure["components"].extend([
            "components/Button.py",
            "components/Form.py",
            "components/Header.py",
        ])
        structure["pages"].extend([
            "pages/Dashboard.py",
            "pages/Settings.py",
        ])
        structure["styles"].extend([
            "styles/theme.py",
            "styles/responsive.css",
        ])
        structure["tests"].extend([
            "tests/test_button.py",
            "tests/test_dashboard.py",
        ])

    if complexity == "high":
        structure["components"].extend([
            "components/Modal.py",
            "components/Table.py",
            "components/Chart.py",
        ])
        structure["pages"].extend([
            "pages/Analytics.py",
            "pages/Profile.py",
        ])
        structure["tests"].extend([
            "tests/test_modal.py",
            "tests/test_analytics.py",
            "tests/conftest.py",
        ])

    return structure


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
        using the LLM provider chain. When called with a string (backwards compat),
        returns a stub result.

        Args:
            dag_or_name: A SubPhase object or a string subphase name
            context: Execution context (may contain dependency templates)
            
        Returns:
            Dict with task execution results, including status and error info.
        """
        from .enums import StateStatus
        
        # Get timeout from context or use default
        timeout = context.get("llm_timeout", 30.0) if context else 30.0
        
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
                
                # Execute using LLM provider if available (with timeout)
                if self._llm_provider and self._llm_provider.enabled:
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
        """Generate a structured execution plan for the SYNTHESIZE subphase."""
        requirement = context.get("requirement", "No requirement provided")
        complexity = _estimate_complexity(requirement)

        if complexity == "high":
            phases = ["SETUP", "CORE_IMPLEMENTATION", "FEATURE_DEVELOPMENT", "INTEGRATION", "VERIFICATION"]
            task_count = 12
        elif complexity == "medium":
            phases = ["SETUP", "IMPLEMENTATION", "INTEGRATION", "VERIFICATION"]
            task_count = 8
        else:
            phases = ["SETUP", "IMPLEMENTATION", "VERIFICATION"]
            task_count = 5

        tasks = [
            {
                "id": f"T{i:03d}",
                "description": desc,
                "depends_on": deps,
                "estimated_effort": "medium",
            }
            for i, (desc, deps) in enumerate(_get_synthesis_tasks(phases), start=1)
        ]

        structured_data = {
            "requirement": requirement,
            "phases": phases,
            "tasks": tasks,
            "estimated_task_count": task_count,
            "estimated_complexity": complexity,
            "synthesis_notes": (
                f"Stub execution plan for '{task.description}'. "
                f"{len(phases)} phases, {task_count} tasks."
            ),
        }

        return {
            "output": (
                f"[SYNTHESIZE] Execution plan drafted: {len(phases)} phases, "
                f"{task_count} tasks for {complexity} complexity."
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
        """Generate implementation outline for the BACKEND subphase."""
        requirement = context.get("requirement", "No requirement provided")
        plan = context.get("plan")
        plan_tasks = []
        if plan and isinstance(plan, dict):
            plan_tasks = plan.get("tasks", [])

        file_structure = _generate_backend_file_structure(requirement)

        structured_data = {
            "requirement": requirement,
            "file_structure": file_structure,
            "implementation_phases": [
                {"phase": "models", "description": "Data models and schemas", "files": file_structure["models"]},
                {"phase": "routes", "description": "API routes and handlers", "files": file_structure["routes"]},
                {"phase": "services", "description": "Business logic services", "files": file_structure["services"]},
                {"phase": "tests", "description": "Backend tests", "files": file_structure["tests"]},
            ],
            "planned_tasks": plan_tasks,
            "implementation_notes": (
                f"Stub backend implementation outline for '{task.description}'. "
                f"Generated {sum(len(v) for v in file_structure.values())} file(s) across "
                f"{len(file_structure)} module(s)."
            ),
        }

        return {
            "output": (
                f"[BACKEND] Implementation outline generated: "
                f"{sum(len(v) for v in file_structure.values())} file(s) across "
                f"{len(file_structure)} module(s)."
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
        """Generate implementation outline for the FRONTEND subphase."""
        requirement = context.get("requirement", "No requirement provided")
        file_structure = _generate_frontend_file_structure(requirement)

        structured_data = {
            "requirement": requirement,
            "file_structure": file_structure,
            "implementation_phases": [
                {"phase": "components", "description": "UI components", "files": file_structure["components"]},
                {"phase": "pages", "description": "Page routes/views", "files": file_structure["pages"]},
                {"phase": "styles", "description": "Styles and theming", "files": file_structure["styles"]},
                {"phase": "tests", "description": "Frontend tests", "files": file_structure["tests"]},
            ],
            "implementation_notes": (
                f"Stub frontend implementation outline for '{task.description}'. "
                f"Generated {sum(len(v) for v in file_structure.values())} file(s) across "
                f"{len(file_structure)} module(s)."
            ),
        }

        return {
            "output": (
                f"[FRONTEND] Implementation outline generated: "
                f"{sum(len(v) for v in file_structure.values())} file(s) across "
                f"{len(file_structure)} module(s)."
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