"""Ralph Loop Hierarchical Automation Framework.

Implements the Ralph Loop pattern for nested automation:
Project → Phase → Subphase → Task

Each level supports:
- Nested automation with state transitions
- Progress tracking and roll-up aggregation
- Integration with existing spine/core/state_machine.py

Key components:
- RalphLoopEngine: Core engine managing hierarchical state transitions
- ProgressAggregator: Aggregates progress from tasks up to project
- TransitionManager: Validates and performs state transitions
- HierarchyValidator: Validates tree structure consistency
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, Set, Callable, TYPE_CHECKING

from ..models.types import (
    HierarchyLevel,
    NodeStatus,
    HierarchyNode,
    ProjectNode,
    PhaseNode,
    SubPhaseNode,
    TaskNode,
    HierarchyProgress,
    ValidationResult,
    Phase,
    SubPhase,
    Task,
)

if TYPE_CHECKING:
    from .state_machine import SpineStateMachine


# ═══════════════════════════════════════════════════════════════════
# TransitionManager
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TransitionRule:
    """A rule defining valid state transitions."""
    source: NodeStatus
    target: NodeStatus
    description: str = ""


class TransitionManager:
    """Manages valid state transitions for hierarchy nodes.
    
    Uses a rule-based approach where each rule defines a valid source→target
    transition. Invalid transitions raise ValueError.
    """

    # Default transition matrix — defines all valid state transitions
    DEFAULT_TRANSITIONS: List[TransitionRule] = [
        # Initiation transitions
        TransitionRule(NodeStatus.PENDING, NodeStatus.RUNNING, "Start work"),
        TransitionRule(NodeStatus.PENDING, NodeStatus.BLOCKED, "Block before starting"),
        TransitionRule(NodeStatus.PENDING, NodeStatus.CANCELLED, "Cancel pending work"),
        TransitionRule(NodeStatus.PENDING, NodeStatus.SUCCESS, "No-op completion"),
        
        # Execution transitions
        TransitionRule(NodeStatus.RUNNING, NodeStatus.SUCCESS, "Work completed"),
        TransitionRule(NodeStatus.RUNNING, NodeStatus.FAILED, "Work failed"),
        TransitionRule(NodeStatus.RUNNING, NodeStatus.BLOCKED, "Blocked during execution"),
        TransitionRule(NodeStatus.RUNNING, NodeStatus.CANCELLED, "Cancel running work"),
        
        # Recovery transitions
        TransitionRule(NodeStatus.FAILED, NodeStatus.REWORKING, "Retry after failure"),
        TransitionRule(NodeStatus.FAILED, NodeStatus.CANCELLED, "Abandon failed work"),
        
        TransitionRule(NodeStatus.REWORKING, NodeStatus.RUNNING, "Retry execution"),
        TransitionRule(NodeStatus.REWORKING, NodeStatus.FAILED, "Retry failed again"),
        TransitionRule(NodeStatus.REWORKING, NodeStatus.CANCELLED, "Cancel rework"),
        
        # Unblock transitions
        TransitionRule(NodeStatus.BLOCKED, NodeStatus.RUNNING, "Resume after unblock"),
        TransitionRule(NodeStatus.BLOCKED, NodeStatus.CANCELLED, "Cancel blocked work"),
        TransitionRule(NodeStatus.BLOCKED, NodeStatus.PENDING, "Reset blocked work"),
        
        # Cancelled can be reset
        TransitionRule(NodeStatus.CANCELLED, NodeStatus.PENDING, "Reset cancelled"),
        
        # Direct success/failure from pending (for simple nodes)
        TransitionRule(NodeStatus.RUNNING, NodeStatus.REWORKING, "Request retry"),
    ]

    def __init__(self, rules: Optional[List[TransitionRule]] = None):
        """Initialize with optional custom rules.
        
        Args:
            rules: Custom transition rules. If None, uses DEFAULT_TRANSITIONS.
        """
        self._rules: Dict[str, Set[str]] = {}
        self._load_rules(rules or self.DEFAULT_TRANSITIONS)

    @property
    def rules(self) -> List[TransitionRule]:
        """Reconstruct rules list from internal map."""
        result = []
        for src, targets in self._rules.items():
            for tgt in targets:
                result.append(TransitionRule(
                    source=NodeStatus(src),
                    target=NodeStatus(tgt),
                ))
        return result

    def _load_rules(self, rules: List[TransitionRule]) -> None:
        """Load transition rules into the lookup map."""
        for rule in rules:
            src = rule.source.value
            tgt = rule.target.value
            if src not in self._rules:
                self._rules[src] = set()
            self._rules[src].add(tgt)

    def register_transition(self, source: NodeStatus, target: NodeStatus) -> None:
        """Register a new custom transition rule."""
        src = source.value
        tgt = target.value
        if src not in self._rules:
            self._rules[src] = set()
        self._rules[src].add(tgt)

    def can_transition(self, node: HierarchyNode, target: NodeStatus) -> bool:
        """Check if a transition from node's current state to target is valid.
        
        Args:
            node: The hierarchy node to check.
            target: Desired target status.
            
        Returns:
            True if the transition is valid.
        """
        src = node.status.value if isinstance(node.status, NodeStatus) else str(node.status)
        tgt = target.value
        allowed = self._rules.get(src, set())
        return tgt in allowed

    def perform_transition(self, node: HierarchyNode, target: NodeStatus) -> bool:
        """Perform a state transition on a node.
        
        Args:
            node: The hierarchy node to transition.
            target: Target status.
            
        Returns:
            True if transition was successful.
            
        Raises:
            ValueError: If the transition is not allowed.
        """
        if not self.can_transition(node, target):
            raise ValueError(
                f"Invalid transition: {node.status.value} → {target.value} "
                f"for node {node.id} ({node.name})"
            )
        node.status = target
        return True


# ═══════════════════════════════════════════════════════════════════
# ProgressAggregator
# ═══════════════════════════════════════════════════════════════════

class ProgressAggregator:
    """Aggregates progress metrics up the hierarchy tree.
    
    Progress rolls up from:
    - Tasks → Subphases → Phases → Project
    - Each level aggregates from its direct children
    """

    @staticmethod
    def aggregate_from_children(children: List[Any]) -> HierarchyProgress:
        """Aggregate progress from a flat list of child nodes.
        
        Works with any hierarchy level: TaskNodes aggregate to SubPhase,
        SubPhaseNodes aggregate to Phase, etc.
        
        Args:
            children: List of child nodes (TaskNodes or SubPhaseNodes).
            
        Returns:
            HierarchyProgress with aggregated stats.
        """
        progress = HierarchyProgress()
        
        for child in children:
            if hasattr(child, 'level') and child.level == HierarchyLevel.TASK:
                # Direct task
                progress.total_tasks += 1
                if child.status == NodeStatus.SUCCESS:
                    progress.completed_tasks += 1
                elif child.status == NodeStatus.FAILED:
                    progress.failed_tasks += 1
                elif child.status == NodeStatus.BLOCKED:
                    progress.blocked_tasks += 1
                elif child.status in (NodeStatus.RUNNING, NodeStatus.REWORKING):
                    progress.running_tasks += 1
                elif child.status == NodeStatus.PENDING:
                    progress.pending_tasks += 1
            elif hasattr(child, 'tasks') and child.tasks:
                # SubPhase with nested tasks
                sub = ProgressAggregator.aggregate_from_children(child.tasks)
                progress.total_tasks += sub.total_tasks
                progress.completed_tasks += sub.completed_tasks
                progress.failed_tasks += sub.failed_tasks
                progress.blocked_tasks += sub.blocked_tasks
                progress.running_tasks += sub.running_tasks
                progress.pending_tasks += sub.pending_tasks
            elif hasattr(child, 'subphases') and child.subphases:
                # Phase with nested subphases
                sub = ProgressAggregator.aggregate_from_children(child.subphases)
                progress.total_tasks += sub.total_tasks
                progress.completed_tasks += sub.completed_tasks
                progress.failed_tasks += sub.failed_tasks
                progress.blocked_tasks += sub.blocked_tasks
                progress.running_tasks += sub.running_tasks
                progress.pending_tasks += sub.pending_tasks

        return progress

    @staticmethod
    def aggregate_phase(phase: PhaseNode) -> HierarchyProgress:
        """Aggregate progress for a single PhaseNode.
        
        Args:
            phase: The PhaseNode to aggregate.
            
        Returns:
            HierarchyProgress for the phase.
        """
        return ProgressAggregator.aggregate_from_children(phase.subphases)

    @staticmethod
    def aggregate_project(project: ProjectNode) -> HierarchyProgress:
        """Aggregate progress for a ProjectNode.
        
        Args:
            project: The ProjectNode to aggregate.
            
        Returns:
            HierarchyProgress for the entire project.
        """
        return ProgressAggregator.aggregate_from_children(project.phases)


# ═══════════════════════════════════════════════════════════════════
# HierarchyValidator
# ═══════════════════════════════════════════════════════════════════

class HierarchyValidator:
    """Validates the structural integrity of a hierarchy tree.
    
    Checks for:
    - Duplicate node IDs
    - Orphaned nodes (missing parent references)
    - Invalid parent_id references
    - Level consistency (correct nesting)
    """

    def validate(self, root: ProjectNode) -> ValidationResult:
        """Validate a project hierarchy tree.
        
        Args:
            root: The root ProjectNode to validate.
            
        Returns:
            ValidationResult with is_valid flag and error list.
        """
        errors: List[str] = []
        warnings: List[str] = []
        seen_ids: Set[str] = set()

        # Collect all node IDs and check for duplicates
        all_ids: List[str] = []
        
        def collect_ids(node: Any, prefix: str = "") -> None:
            node_id = getattr(node, 'id', None)
            if node_id:
                if node_id in seen_ids:
                    errors.append(f"Duplicate node ID: '{node_id}'")
                seen_ids.add(node_id)
                all_ids.append(node_id)
            
            # Recurse into children
            for attr_name in ('phases', 'subphases', 'tasks'):
                children = getattr(node, attr_name, None)
                if children and isinstance(children, list):
                    for child in children:
                        collect_ids(child, f"{prefix}/{node_id}" if node_id else prefix)

        collect_ids(root)

        if not root.id:
            errors.append("Root project node has no ID")

        # Check parent_id references
        def check_parent_refs(node: Any) -> None:
            parent_id = getattr(node, 'parent_id', None)
            if parent_id and parent_id not in seen_ids:
                errors.append(
                    f"Node '{getattr(node, 'id', '?')}' references "
                    f"non-existent parent_id '{parent_id}'"
                )
            
            for attr_name in ('phases', 'subphases', 'tasks'):
                children = getattr(node, attr_name, None)
                if children and isinstance(children, list):
                    for child in children:
                        check_parent_refs(child)

        check_parent_refs(root)

        # Check for orphaned nodes (Phase without parent project, etc.)
        # This catches PhaseNode used as root instead of ProjectNode
        if not isinstance(root, ProjectNode):
            errors.append(
                f"Root node must be a ProjectNode, got {type(root).__name__}"
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )


# ═══════════════════════════════════════════════════════════════════
# RalphLoopEngine
# ═══════════════════════════════════════════════════════════════════

class RalphLoopEngine:
    """Core engine for the Ralph Loop hierarchical automation framework.
    
    Manages the full lifecycle of a hierarchical project:
    - Creates and manages the Project → Phase → Subphase → Task tree
    - Handles state transitions with validation
    - Aggregates progress from leaf tasks up to the project root
    - Integrates with the existing SpineStateMachine for workflow execution
    
    Usage:
        engine = RalphLoopEngine()
        proj = engine.create_project("my-proj", "My Project")
        phase = engine.create_phase("p1", "Planning", parent_project=proj)
        sp = engine.create_subphase("sp1", "Analyze", parent_phase=phase)
        task = engine.create_task("t1", "Parse Requirements", parent_subphase=sp)
        
        engine.transition_node(task, NodeStatus.RUNNING)
        task.progress = 100.0
        engine.transition_node(task, NodeStatus.SUCCESS)
        
        progress = engine.get_project_progress(proj)
    """

    def __init__(
        self,
        transition_manager: Optional[TransitionManager] = None,
        progress_aggregator: Optional[ProgressAggregator] = None,
    ):
        """Initialize the Ralph Loop engine.
        
        Args:
            transition_manager: Custom transition manager.
            progress_aggregator: Custom progress aggregator.
        """
        self.transition_manager = transition_manager or TransitionManager()
        self.progress_aggregator = progress_aggregator or ProgressAggregator()
        self.auto_complete_parents: bool = False
        self._state_machine: Optional[SpineStateMachine] = None

    def attach_state_machine(self, sm: SpineStateMachine) -> None:
        """Attach a SpineStateMachine for integration.
        
        Args:
            sm: The state machine instance to attach.
        """
        self._state_machine = sm

    # ── Tree Construction ──────────────────────────────────────

    def create_project(self, id: str, name: str) -> ProjectNode:
        """Create a new project (root of the hierarchy).
        
        Args:
            id: Unique project identifier.
            name: Human-readable project name.
            
        Returns:
            A new ProjectNode.
        """
        return ProjectNode(id=id, name=name)

    def create_phase(
        self,
        id: str,
        name: str,
        parent_project: ProjectNode,
        phase_model: Optional[Phase] = None,
    ) -> PhaseNode:
        """Create a phase node under a project.
        
        Args:
            id: Unique phase identifier.
            name: Human-readable phase name.
            parent_project: The parent project.
            phase_model: Optional existing Phase model reference.
            
        Returns:
            A new PhaseNode attached to the project.
        """
        phase = PhaseNode(
            id=id,
            name=name,
            parent_id=parent_project.id,
            phase_model=phase_model,
        )
        parent_project.phases.append(phase)
        return phase

    def create_subphase(
        self,
        id: str,
        name: str,
        parent_phase: PhaseNode,
        parallel: bool = True,
        subphase_model: Optional[SubPhase] = None,
    ) -> SubPhaseNode:
        """Create a subphase node under a phase.
        
        Args:
            id: Unique subphase identifier.
            name: Human-readable subphase name.
            parent_phase: The parent phase.
            parallel: Whether tasks run in parallel.
            subphase_model: Optional existing SubPhase model reference.
            
        Returns:
            A new SubPhaseNode attached to the phase.
        """
        sp = SubPhaseNode(
            id=id,
            name=name,
            parent_id=parent_phase.id,
            parallel=parallel,
            subphase_model=subphase_model,
        )
        parent_phase.subphases.append(sp)
        return sp

    def create_task(
        self,
        id: str,
        name: str,
        parent_subphase: SubPhaseNode,
        task_model: Optional[Task] = None,
    ) -> TaskNode:
        """Create a task node under a subphase.
        
        Args:
            id: Unique task identifier.
            name: Human-readable task name.
            parent_subphase: The parent subphase.
            task_model: Optional existing Task model reference.
            
        Returns:
            A new TaskNode attached to the subphase.
        """
        task = TaskNode(
            id=id,
            name=name,
            parent_id=parent_subphase.id,
            task_model=task_model,
        )
        parent_subphase.tasks.append(task)
        return task

    # ── Conversions ────────────────────────────────────────────

    def create_project_from_phases(
        self,
        id: str,
        name: str,
        phases: List[Phase],
    ) -> ProjectNode:
        """Create a full hierarchy tree from existing Phase models.
        
        Args:
            id: Project identifier.
            name: Project name.
            phases: List of Phase models to convert.
            
        Returns:
            A complete ProjectNode with converted hierarchy.
        """
        project = self.create_project(id, name)
        for i, phase in enumerate(phases):
            phase_name = phase.name.value if hasattr(phase.name, 'value') else str(phase.name)
            pn = self.create_phase(
                f"{id}-phase-{i}",
                phase_name.capitalize(),
                parent_project=project,
                phase_model=phase,
            )
            for j, sp in enumerate(phase.subphases):
                sp_name = sp.name
                spn = self.create_subphase(
                    f"{id}-sp-{i}-{j}",
                    sp_name,
                    parent_phase=pn,
                    parallel=sp.parallel,
                    subphase_model=sp,
                )
                for k, task in enumerate(sp.tasks):
                    self.create_task(
                        f"{id}-task-{i}-{j}-{k}",
                        task.description,
                        parent_subphase=spn,
                        task_model=task,
                    )
        return project

    # ── Tree Traversal ─────────────────────────────────────────

    def find_node(self, root: ProjectNode, node_id: str) -> Optional[Any]:
        """Find a node by ID anywhere in the tree.
        
        Args:
            root: The root project node.
            node_id: The ID to search for.
            
        Returns:
            The found node or None.
        """
        if root.id == node_id:
            return root

        for phase in root.phases:
            if phase.id == node_id:
                return phase
            for sp in phase.subphases:
                if sp.id == node_id:
                    return sp
                for task in sp.tasks:
                    if task.id == node_id:
                        return task
        return None

    def collect_all_nodes(self, root: ProjectNode) -> List[Any]:
        """Collect all nodes in the tree in pre-order.
        
        Args:
            root: The root project node.
            
        Returns:
            Flat list of all nodes.
        """
        nodes: List[Any] = [root]
        for phase in root.phases:
            nodes.append(phase)
            for sp in phase.subphases:
                nodes.append(sp)
                nodes.extend(sp.tasks)
        return nodes

    # ── State Transitions ──────────────────────────────────────

    def transition_node(self, node: Any, target: NodeStatus) -> None:
        """Transition a node to a new state.
        
        Works with any node type that has id, name, status attributes.
        
        Args:
            node: The node to transition.
            target: Target status.
            
        Raises:
            ValueError: If transition is invalid.
        """
        # Build a temporary HierarchyNode for validation
        temp = HierarchyNode(
            id=getattr(node, 'id', '?'),
            name=getattr(node, 'name', '?'),
            status=node.status,
        )
        self.transition_manager.perform_transition(temp, target)
        node.status = target

    # ── Progress ───────────────────────────────────────────────

    def get_project_progress(self, project: ProjectNode) -> HierarchyProgress:
        """Get aggregated progress for an entire project.
        
        Args:
            project: The project to assess.
            
        Returns:
            HierarchyProgress with rolled-up stats.
        """
        return self.progress_aggregator.aggregate_project(project)

    # ── Execution ──────────────────────────────────────────────

    def execute_project(
        self,
        project: ProjectNode,
        context: Optional[Dict[str, Any]] = None,
    ) -> ProjectNode:
        """Execute an entire project through all phases.
        
        This is a stub/synchronous execution that transitions all nodes
        through their lifecycle. In production, this would integrate
        with the LLM provider chain and async execution.
        
        Args:
            project: The project to execute.
            context: Optional execution context.
            
        Returns:
            The project with updated states after execution.
        """
        ctx = context or {}

        # Transition project to RUNNING
        self.transition_node(project, NodeStatus.RUNNING)

        for phase in project.phases:
            self.transition_node(phase, NodeStatus.RUNNING)

            for sp in phase.subphases:
                self.transition_node(sp, NodeStatus.RUNNING)

                for task in sp.tasks:
                    self.transition_node(task, NodeStatus.RUNNING)
                    # Stub: mark task as complete
                    task.result = f"Completed: {task.name}"
                    task.progress = 100.0
                    self.transition_node(task, NodeStatus.SUCCESS)

                self.transition_node(sp, NodeStatus.SUCCESS)

            self.transition_node(phase, NodeStatus.SUCCESS)

        self.transition_node(project, NodeStatus.SUCCESS)
        return project


__all__ = [
    "TransitionRule",
    "TransitionManager",
    "ProgressAggregator",
    "HierarchyValidator",
    "RalphLoopEngine",
]