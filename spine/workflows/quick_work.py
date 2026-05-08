"""Quick Work Lifecycle — streamlined 3-phase workflow.

A minimal lifecycle for rapid execution:
  1. PLAN     — Quick planning with lightweight tasking
  2. IMPLEMENT— Execute directly (no parallel worktrees by default)
  3. VERIFY   — Quick validation

Suitable for small, well-understood tasks that don't need
full SPEC/DESIGN phases.
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


class QuickWorkflow(WorkflowEngine):
    """Streamlined Quick Work lifecycle.

    Only 3 phases: PLAN → IMPLEMENT → VERIFY.
    Designed for small tasks with well-understood scope.

    Usage:
        qw = QuickWorkflow(state_machine=sm)
        qw.create_project("fix-bug", "Fix the login validation bug")
        result = qw.execute()
    """

    # Override: minimal 3-phase lifecycle
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
    ):
        """Initialize the Quick Work workflow.

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
        """Execute the full Quick Work lifecycle.

        Returns:
            A WorkflowResult summarizing execution.
        """
        project = self._project
        if project is None:
            raise ValueError(
                "No project created. Call create_project() before execute()."
            )

        try:
            self._build_plan_phase()
            self._build_implement_phase()
            self._build_verify_phase()

            self.transition_node(project, NodeStatus.SUCCESS)
            self.validate_hierarchy()

        except Exception as e:
            self._errors.append(str(e))
            self.transition_node(project, NodeStatus.FAILED)

        return self.get_result()

    # ── PLAN Phase (Quick) ────────────────────────────────────────

    def _build_plan_phase(self) -> PhaseNode:
        """Build and execute the Quick PLAN phase.

        Quick planning:
        - Single subphase with minimal planning tasks
        - Direct task list creation
        """
        phase = self.create_phase_node(
            WorkflowPhase.PLAN, "Quick Plan",
        )
        self.start_phase(WorkflowPhase.PLAN)

        sp = self.create_subphase_node(
            "quick-plan-tasks", "Planning",
            parent_phase=phase, parallel=False,
        )

        tasks = [
            ("quick-assess", "Quickly assess the requirement"),
            ("quick-breakdown", "Break down into minimal tasks"),
            ("quick-plan-done", "Confirm plan completeness"),
        ]
        for tid, tname in tasks:
            self.create_task_node(tid, tname, parent_subphase=sp)

        self._execute_subphase_tasks(sp)
        self.auto_complete(phase)

        # Build quick plan
        self._context.plan = {
            "requirement": self._context.requirement,
            "approach": "quick-work",
            "phases": ["PLAN", "IMPLEMENT", "VERIFY"],
            "implementation_tasks": [
                {"id": "quick-impl", "description": "Implement required changes"},
            ],
        }

        return phase

    # ── IMPLEMENT Phase (Quick) ───────────────────────────────────

    def _build_implement_phase(self) -> PhaseNode:
        """Build and execute the Quick IMPLEMENT phase.

        Single subphase with direct task execution.
        """
        phase = self.create_phase_node(
            WorkflowPhase.IMPLEMENT, "Quick Implementation",
        )
        self.start_phase(WorkflowPhase.IMPLEMENT)

        sp = self.create_subphase_node(
            "quick-impl-tasks", "Implementation",
            parent_phase=phase, parallel=False,
        )

        impl_tasks = self._context.plan.get("implementation_tasks", []) if self._context.plan else []
        if not impl_tasks:
            impl_tasks = [
                {"id": "quick-impl", "description": "Implement required changes"},
            ]

        for task_def in impl_tasks:
            self.create_task_node(
                task_def["id"], task_def["description"],
                parent_subphase=sp,
            )

        self._execute_subphase_tasks(sp)
        self.auto_complete(phase)

        return phase

    # ── VERIFY Phase (Quick) ──────────────────────────────────────

    def _build_verify_phase(self) -> PhaseNode:
        """Build and execute the Quick VERIFY phase.

        Minimal verification:
        - Run tests
        - Quick validation
        """
        phase = self.create_phase_node(
            WorkflowPhase.VERIFY, "Quick Verification",
        )
        self.start_phase(WorkflowPhase.VERIFY)

        sp = self.create_subphase_node(
            "quick-verify-tasks", "Verification",
            parent_phase=phase, parallel=False,
        )

        tasks = [
            ("quick-test", "Run tests"),
            ("quick-check", "Validate completion"),
            ("quick-done", "Confirm result"),
        ]
        for tid, tname in tasks:
            self.create_task_node(tid, tname, parent_subphase=sp)

        # Run gates if configured
        if self._gates:
            self.run_all_gates()

        self._execute_subphase_tasks(sp)
        self.auto_complete(phase)

        return phase

    # ── Helpers ───────────────────────────────────────────────────

    def _execute_subphase_tasks(self, sp: SubPhaseNode) -> None:
        """Execute all tasks in a subphase (synchronous stub).

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


__all__ = [
    "QuickWorkflow",
]
