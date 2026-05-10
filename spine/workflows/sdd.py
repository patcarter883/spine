"""Spec-Driven Development (SDD) Workflow Lifecycle.

Full lifecycle for greenfield projects:
  1. SPEC     — Gather requirements, write formal spec
  2. DESIGN   — Design architecture, define interfaces
  3. PLAN     — Create detailed plan as SubPhaseNodes with TaskNodes
  4. IMPLEMENT— Execute tasks in parallel worktrees
  5. REVIEW   — Run swarm gates (critic, reviewer, quality)
  6. VERIFY   — Test execution, validation, completion

Integrates:
- RalphLoopEngine for hierarchical lifecycle management
- WorktreeManager for parallel execution during IMPLEMENT
- Swarm gates (CriticGate, QualityGate, CompletionGate) for REVIEW
- SpineStateMachine for checkpoint persistence
"""

from __future__ import annotations

from typing import Optional, List, TYPE_CHECKING

from .engine import (
    WorkflowEngine,
    WorkflowPhase,
    WorkflowResult,
)
from ..models.types import (
    NodeStatus,
    PhaseNode,
    SubPhaseNode,
)

if TYPE_CHECKING:
    from ..core.state_machine import SpineStateMachine
    from ..git.worktree_manager import WorktreeManager
    from ..swarm.gates import SwarmGate


class SDDWorkflow(WorkflowEngine):
    """Full Spec-Driven Development lifecycle.

    Orchestrates 6 phases for greenfield development using the
    Ralph Loop hierarchy pattern.

    Usage:
        sdd = SDDWorkflow(state_machine=sm, worktree_manager=wtm)
        sdd.create_project("my-proj", "Build a web app")
        result = sdd.execute()
    """

    # Override: full 6-phase lifecycle
    DEFAULT_PHASES: List[str] = [
        WorkflowPhase.SPEC,
        WorkflowPhase.DESIGN,
        WorkflowPhase.PLAN,
        WorkflowPhase.IMPLEMENT,
        WorkflowPhase.REVIEW,
        WorkflowPhase.VERIFY,
    ]

    def __init__(
        self,
        state_machine: Optional["SpineStateMachine"] = None,
        worktree_manager: Optional["WorktreeManager"] = None,
        gates: Optional[List["SwarmGate"]] = None,
    ):
        """Initialize the SDD workflow.

        Args:
            state_machine: Optional SpineStateMachine for persistence.
            worktree_manager: Optional WorktreeManager for parallel execution.
            gates: Optional list of swarm gates for verification.
        """
        super().__init__(
            state_machine=state_machine,
            worktree_manager=worktree_manager,
            gates=gates,
        )
        self.set_phases(list(self.DEFAULT_PHASES))

    # ── Full Execution ────────────────────────────────────────────

    def execute(self) -> WorkflowResult:
        """Execute the full SDD lifecycle.

        Runs all 6 phases sequentially. Each phase builds hierarchy
        nodes and tracks progress.

        Returns:
            A WorkflowResult summarizing execution.
        """
        project = self._project
        if project is None:
            raise ValueError(
                "No project created. Call create_project() before execute()."
            )

        try:
            self._build_spec_phase()
            self._build_design_phase()
            self._build_plan_phase()
            self._build_implement_phase()
            self._build_review_phase()
            self._build_verify_phase()

            self.transition_node(project, NodeStatus.SUCCESS)
            self.validate_hierarchy()

        except Exception as e:
            self._errors.append(str(e))
            self.transition_node(project, NodeStatus.FAILED)

        return self.get_result()

    # ── SPEC Phase ────────────────────────────────────────────────

    def _build_spec_phase(self) -> PhaseNode:
        """Build and execute the SPEC phase.

        SPEC phase tasks:
        - Gather requirements
        - Write formal specification
        - Validate spec completeness
        """
        phase = self.create_phase_node(
            WorkflowPhase.SPEC, "Specification",
        )
        self.start_phase(WorkflowPhase.SPEC)

        # Subphase: Requirements Gathering
        sp_req = self.create_subphase_node(
            "spec-requirements", "Requirements Gathering",
            parent_phase=phase, parallel=False,
        )

        tasks_req = [
            ("spec-gather", "Gather project requirements"),
            ("spec-analyze", "Analyze and document requirements"),
            ("spec-validate", "Validate requirement completeness"),
        ]
        for tid, tname in tasks_req:
            self.create_task_node(tid, tname, parent_subphase=sp_req)

        # Subphase: Formal Spec
        sp_formal = self.create_subphase_node(
            "spec-formal", "Formal Specification",
            parent_phase=phase, parallel=False,
        )

        tasks_formal = [
            ("spec-write", "Write formal specification document"),
            ("spec-review", "Peer review specification"),
            ("spec-finalize", "Finalize and lock specification"),
        ]
        for tid, tname in tasks_formal:
            self.create_task_node(tid, tname, parent_subphase=sp_formal)

        # Execute all tasks (synchronous stub)
        self._execute_subphase_tasks(sp_req)
        self._execute_subphase_tasks(sp_formal)
        self.check_and_auto_complete_subphases(phase)
        self.auto_complete(phase)

        self._context.spec = f"Formal specification for: {self._context.requirement}"
        if self._context.plan is None:
            self._context.plan = {}
        self._context.plan["requirement"] = self._context.requirement
        self._context.plan["spec"] = self._context.spec
        self._context.plan["phases"] = ["SPEC", "DESIGN", "PLAN", "IMPLEMENT", "REVIEW", "VERIFY"]

        return phase

    # ── DESIGN Phase ──────────────────────────────────────────────

    def _build_design_phase(self) -> PhaseNode:
        """Build and execute the DESIGN phase.

        DESIGN phase tasks:
        - Define architecture components
        - Design interfaces and APIs
        - Document design decisions
        """
        phase = self.create_phase_node(
            WorkflowPhase.DESIGN, "Design",
        )
        self.start_phase(WorkflowPhase.DESIGN)

        # Subphase: Architecture
        sp_arch = self.create_subphase_node(
            "design-architecture", "Architecture Design",
            parent_phase=phase, parallel=False,
        )

        tasks_arch = [
            ("design-components", "Define architecture components"),
            ("design-dependencies", "Map component dependencies"),
            ("design-dataflow", "Document data flow"),
        ]
        for tid, tname in tasks_arch:
            self.create_task_node(tid, tname, parent_subphase=sp_arch)

        # Subphase: Interfaces
        sp_interfaces = self.create_subphase_node(
            "design-interfaces", "Interface Design",
            parent_phase=phase, parallel=False,
        )

        tasks_iface = [
            ("design-api", "Design API interfaces"),
            ("design-schema", "Define data schemas"),
            ("design-contracts", "Document interface contracts"),
        ]
        for tid, tname in tasks_iface:
            self.create_task_node(tid, tname, parent_subphase=sp_interfaces)

        # Execute all tasks
        self._execute_subphase_tasks(sp_arch)
        self._execute_subphase_tasks(sp_interfaces)
        self.check_and_auto_complete_subphases(phase)
        self.auto_complete(phase)

        self._context.design = f"Architecture design for: {self._context.requirement}"
        if self._context.plan:
            self._context.plan["architecture"] = self._context.design

        return phase

    # ── PLAN Phase ────────────────────────────────────────────────

    def _build_plan_phase(self) -> PhaseNode:
        """Build and execute the PLAN phase.

        PLAN phase tasks:
        - Task decomposition into SubPhaseNodes with TaskNodes
        - Dependency mapping between tasks
        - Task priority and assignment
        """
        phase = self.create_phase_node(
            WorkflowPhase.PLAN, "Planning",
        )
        self.start_phase(WorkflowPhase.PLAN)

        # Subphase: Decomposition
        sp_decomp = self.create_subphase_node(
            "plan-decomposition", "Task Decomposition",
            parent_phase=phase, parallel=False,
        )

        tasks_decomp = [
            ("plan-breakdown", "Break down work into tasks"),
            ("plan-dependencies", "Map task dependencies"),
            ("plan-priorities", "Assign task priorities"),
        ]
        for tid, tname in tasks_decomp:
            self.create_task_node(tid, tname, parent_subphase=sp_decomp)

        # Subphase: Resource Planning
        sp_resources = self.create_subphase_node(
            "plan-resources", "Resource Planning",
            parent_phase=phase, parallel=False,
        )

        tasks_res = [
            ("plan-estimate", "Estimate task effort"),
            ("plan-assign", "Assign tasks to implementation subphases"),
            ("plan-schedule", "Create execution schedule"),
        ]
        for tid, tname in tasks_res:
            self.create_task_node(tid, tname, parent_subphase=sp_resources)

        # Execute all tasks
        self._execute_subphase_tasks(sp_decomp)
        self._execute_subphase_tasks(sp_resources)
        self.check_and_auto_complete_subphases(phase)
        self.auto_complete(phase)

        # Build the implementation plan with concrete tasks
        if self._context.plan:
            self._context.plan["implementation_tasks"] = [
                {"id": "impl-core", "description": "Implement core logic"},
                {"id": "impl-models", "description": "Implement data models"},
                {"id": "impl-errors", "description": "Implement error handling"},
                {"id": "impl-edge", "description": "Implement edge cases"},
            ]

        return phase

    # ── IMPLEMENT Phase ───────────────────────────────────────────

    def _build_implement_phase(self) -> PhaseNode:
        """Build and execute the IMPLEMENT phase.

        IMPLEMENT phase tasks:
        - Execute tasks in parallel worktrees
        - Coordinate worktree creation and cleanup
        - Track implementation progress
        """
        phase = self.create_phase_node(
            WorkflowPhase.IMPLEMENT, "Implementation",
        )
        self.start_phase(WorkflowPhase.IMPLEMENT)

        # Subphase: Core Implementation (parallel capable)
        sp_core = self.create_subphase_node(
            "implement-core", "Core Implementation",
            parent_phase=phase, parallel=True,
        )

        impl_tasks = self._context.plan.get("implementation_tasks", []) if self._context.plan else []
        if not impl_tasks:
            impl_tasks = [
                {"id": "impl-core", "description": "Implement core logic"},
                {"id": "impl-models", "description": "Implement data models"},
            ]

        for task_def in impl_tasks:
            self.create_task_node(
                task_def["id"], task_def["description"],
                parent_subphase=sp_core,
            )

        # Subphase: Integration
        sp_integration = self.create_subphase_node(
            "implement-integration", "Integration",
            parent_phase=phase, parallel=False,
        )

        tasks_int = [
            ("impl-integrate", "Integrate all components"),
            ("impl-wire", "Wire dependencies and configurations"),
        ]
        for tid, tname in tasks_int:
            self.create_task_node(tid, tname, parent_subphase=sp_integration)

        # Execute tasks - core tasks in worktrees if available
        if self._worktree_manager:
            self._execute_tasks_in_worktrees(sp_core)
        else:
            self._execute_subphase_tasks(sp_core)
        self._execute_subphase_tasks(sp_integration)

        self.check_and_auto_complete_subphases(phase)
        self.auto_complete(phase)

        return phase

    # ── REVIEW Phase ───────────────────────────────────────────────

    def _build_review_phase(self) -> PhaseNode:
        """Build and execute the REVIEW phase.

        REVIEW phase tasks:
        - Run code quality checks
        - Run critic gate review
        - Validate against spec and design
        """
        phase = self.create_phase_node(
            WorkflowPhase.REVIEW, "Review",
        )
        self.start_phase(WorkflowPhase.REVIEW)

        # Subphase: Code Review
        sp_code = self.create_subphase_node(
            "review-code", "Code Review",
            parent_phase=phase, parallel=False,
        )

        tasks_code = [
            ("review-code-check", "Code quality review"),
            ("review-style", "Style and conventions check"),
            ("review-correctness", "Correctness validation"),
        ]
        for tid, tname in tasks_code:
            self.create_task_node(tid, tname, parent_subphase=sp_code)

        # Subphase: Gate Execution
        sp_gates = self.create_subphase_node(
            "review-gates", "Gate Execution",
            parent_phase=phase, parallel=True,
        )

        tasks_gates = [
            ("review-critic", "Run critic gate"),
            ("review-quality", "Run quality gate"),
        ]
        for tid, tname in tasks_gates:
            self.create_task_node(tid, tname, parent_subphase=sp_gates)

        # Execute code review tasks
        self._execute_subphase_tasks(sp_code)

        # Run gates
        if self._gates:
            self.run_all_gates()
        self._execute_subphase_tasks(sp_gates)

        self.check_and_auto_complete_subphases(phase)
        self.auto_complete(phase)

        return phase

    # ── VERIFY Phase ───────────────────────────────────────────────

    def _build_verify_phase(self) -> PhaseNode:
        """Build and execute the VERIFY phase.

        VERIFY phase tasks:
        - Run test suite
        - Validate against specification
        - Verify completion criteria
        """
        phase = self.create_phase_node(
            WorkflowPhase.VERIFY, "Verification",
        )
        self.start_phase(WorkflowPhase.VERIFY)

        # Subphase: Testing
        sp_test = self.create_subphase_node(
            "verify-tests", "Test Execution",
            parent_phase=phase, parallel=False,
        )

        tasks_test = [
            ("verify-unit", "Run unit tests"),
            ("verify-integration", "Run integration tests"),
            ("verify-coverage", "Check test coverage"),
        ]
        for tid, tname in tasks_test:
            self.create_task_node(tid, tname, parent_subphase=sp_test)

        # Subphase: Validation
        sp_validation = self.create_subphase_node(
            "verify-validation", "Final Validation",
            parent_phase=phase, parallel=False,
        )

        tasks_val = [
            ("verify-spec-check", "Validate against specification"),
            ("verify-regression", "Run regression checks"),
            ("verify-signoff", "Sign off completion"),
        ]
        for tid, tname in tasks_val:
            self.create_task_node(tid, tname, parent_subphase=sp_validation)

        # Execute all tasks
        self._execute_subphase_tasks(sp_test)
        self._execute_subphase_tasks(sp_validation)

        self.check_and_auto_complete_subphases(phase)
        self.auto_complete(phase)

        return phase

    # ── Helpers ───────────────────────────────────────────────────

    def _execute_subphase_tasks(self, sp: SubPhaseNode) -> None:
        """Execute all tasks in a subphase (synchronous stub).

        Transitions each task from PENDING → RUNNING → SUCCESS.

        Args:
            sp: The subphase whose tasks to execute.
        """
        self.transition_node(sp, NodeStatus.RUNNING)
        for task in sp.tasks:
            self.transition_node(task, NodeStatus.RUNNING)
            task.progress = 100.0
            task.result = f"Completed: {task.name}"
            self.transition_node(task, NodeStatus.SUCCESS)
        self.auto_complete(sp)

    def _execute_tasks_in_worktrees(self, sp: SubPhaseNode) -> None:
        """Execute tasks using parallel worktrees.

        Creates a worktree per task for isolated execution.

        Args:
            sp: The subphase whose tasks to execute in worktrees.
        """
        self.transition_node(sp, NodeStatus.RUNNING)

        created_worktrees: List[str] = []
        try:
            for task in sp.tasks:
                self.transition_node(task, NodeStatus.RUNNING)
                if self._worktree_manager:
                    try:
                        self._worktree_manager.create_worktree(
                            task_id=task.id,
                            base_branch="main",
                        )
                        created_worktrees.append(task.id)
                        # Task runs in worktree - stub completion
                        task.progress = 100.0
                        task.result = f"Completed in worktree: {task.name}"
                    except Exception as e:
                        task.error = str(e)
                        self.transition_node(task, NodeStatus.FAILED)
                        continue
                self.transition_node(task, NodeStatus.SUCCESS)
        finally:
            # Cleanup worktrees after execution
            for wt_id in created_worktrees:
                if self._worktree_manager:
                    try:
                        self._worktree_manager.cleanup_worktree(wt_id)
                    except Exception:
                        pass  # best-effort cleanup

        self.auto_complete(sp)


__all__ = [
    "SDDWorkflow",
]
