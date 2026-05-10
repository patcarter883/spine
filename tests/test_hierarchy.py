"""Tests for Ralph Loop hierarchical automation framework.

Tests cover:
- HierarchyNode data model and tree operations
- RalphLoopEngine state transitions
- Progress roll-up aggregation
- Nested automation support
- Integration with existing state_machine.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from dataclasses import asdict

from spine.models.enums import PhaseName, StateStatus, SubPhaseStatus
from spine.models.types import (
    Task, SubPhase, Phase, PhaseResult, SubPhaseResult, SpineState,
    HierarchyNode, HierarchyLevel, HierarchyProgress, NodeStatus,
    ProjectNode, PhaseNode, SubPhaseNode, TaskNode,
)
from spine.core.hierarchy import (
    RalphLoopEngine, ProgressAggregator, TransitionManager,
    HierarchyValidator,
)
from spine.core.state_machine import SpineStateMachine
from spine.core.constants import ErrorState


# ── HierarchyNode Tests ────────────────────────────────────────────

class TestHierarchyNode:
    """Tests for the HierarchyNode base type."""

    def test_create_root_node(self):
        node = HierarchyNode(id="root", name="Root")
        assert node.id == "root"
        assert node.name == "Root"
        assert node.status == NodeStatus.PENDING
        assert node.progress == 0.0

    def test_node_with_children(self):
        parent = HierarchyNode(id="p", name="Parent")
        child = HierarchyNode(id="c", name="Child", parent=parent)
        parent.children.append(child)

        assert len(parent.children) == 1
        assert parent.children[0].id == "c"
        assert child.parent is parent

    def test_node_set_progress(self):
        node = HierarchyNode(id="n", name="Test")
        node.progress = 50.0
        assert node.progress == 50.0

    def test_node_transition_state(self):
        node = HierarchyNode(id="n", name="Test")
        assert node.status == NodeStatus.PENDING
        node.status = NodeStatus.RUNNING
        assert node.status == NodeStatus.RUNNING
        node.status = NodeStatus.SUCCESS
        assert node.status == NodeStatus.SUCCESS

    def test_node_metadata_dict(self):
        node = HierarchyNode(
            id="n", name="Test",
            metadata={"key": "value", "tags": ["a", "b"]}
        )
        assert node.metadata["key"] == "value"
        assert node.metadata["tags"] == ["a", "b"]

    def test_hierarchy_level_enum_values(self):
        assert HierarchyLevel.PROJECT.value == "project"
        assert HierarchyLevel.PHASE.value == "phase"
        assert HierarchyLevel.SUBPHASE.value == "subphase"
        assert HierarchyLevel.TASK.value == "task"


class TestSpecializedNodes:
    """Tests for ProjectNode, PhaseNode, SubPhaseNode, TaskNode."""

    def test_project_node_defaults(self):
        proj = ProjectNode(id="proj-1", name="My Project")
        assert proj.id == "proj-1"
        assert proj.name == "My Project"
        assert proj.level == HierarchyLevel.PROJECT
        assert proj.phases == []
        assert proj.status == NodeStatus.PENDING

    def test_phase_node_defaults(self):
        phase = PhaseNode(id="phase-1", name="Planning", parent_id="proj-1")
        assert phase.level == HierarchyLevel.PHASE
        assert phase.parent_id == "proj-1"
        assert phase.subphases == []
        assert phase.status == NodeStatus.PENDING

    def test_subphase_node_defaults(self):
        sp = SubPhaseNode(id="sp-1", name="Analyze", parent_id="phase-1")
        assert sp.level == HierarchyLevel.SUBPHASE
        assert sp.parent_id == "phase-1"
        assert sp.tasks == []
        assert sp.parallel is True
        assert sp.status == NodeStatus.PENDING

    def test_task_node_defaults(self):
        task = TaskNode(id="task-1", name="Parse Input", parent_id="sp-1")
        assert task.level == HierarchyLevel.TASK
        assert task.parent_id == "sp-1"
        assert task.status == NodeStatus.PENDING

    def test_task_node_has_result(self):
        task = TaskNode(id="t1", name="Test", parent_id="sp-1")
        task.result = "completed successfully"
        task.error = None
        assert task.result == "completed successfully"
        assert task.error is None

    def test_phase_node_holds_original_phase_model(self):
        """PhaseNode can carry a reference to the original Phase model."""
        orig = Phase(name=PhaseName.PLANNING, description="Plan phase")
        pn = PhaseNode(id="p1", name="Planning", parent_id="proj-1", phase_model=orig)
        assert pn.phase_model is orig
        assert pn.phase_model.name == PhaseName.PLANNING


# ── HierarchyProgress Tests ────────────────────────────────────────

class TestHierarchyProgress:
    """Tests for progress aggregation across hierarchy levels."""

    def test_progress_defaults(self):
        p = HierarchyProgress()
        assert p.total_tasks == 0
        assert p.completed_tasks == 0
        assert p.failed_tasks == 0
        assert p.blocked_tasks == 0
        assert p.percent_complete == 0.0

    def test_progress_percent_calculation(self):
        p = HierarchyProgress(total_tasks=10, completed_tasks=7)
        assert p.percent_complete == 70.0

    def test_progress_zero_division(self):
        p = HierarchyProgress(total_tasks=0, completed_tasks=0)
        assert p.percent_complete == 0.0

    def test_progress_all_statuses(self):
        p = HierarchyProgress(
            total_tasks=20,
            completed_tasks=15,
            failed_tasks=3,
            blocked_tasks=2,
        )
        assert p.completed_tasks == 15
        assert p.failed_tasks == 3
        assert p.blocked_tasks == 2
        assert p.in_progress_tasks == 0


# ── ProgressAggregator Tests ───────────────────────────────────────

class TestProgressAggregator:
    """Tests for progress roll-up from tasks up to project."""

    def test_aggregate_single_task(self):
        agg = ProgressAggregator()
        task = TaskNode(id="t1", name="Task 1", parent_id="sp1")
        task.status = NodeStatus.SUCCESS
        task.progress = 100.0

        progress = agg.aggregate_from_children([task])
        assert progress.total_tasks == 1
        assert progress.completed_tasks == 1
        assert progress.percent_complete == 100.0

    def test_aggregate_mixed_tasks(self):
        agg = ProgressAggregator()
        tasks = [
            TaskNode(id="t1", name="T1", parent_id="p1", status=NodeStatus.SUCCESS, progress=100.0),
            TaskNode(id="t2", name="T2", parent_id="p1", status=NodeStatus.RUNNING, progress=50.0),
            TaskNode(id="t3", name="T3", parent_id="p1", status=NodeStatus.FAILED, progress=5.0),
            TaskNode(id="t4", name="T4", parent_id="p1", status=NodeStatus.BLOCKED, progress=0.0),
        ]

        progress = agg.aggregate_from_children(tasks)
        assert progress.total_tasks == 4
        assert progress.completed_tasks == 1
        assert progress.failed_tasks == 1
        assert progress.blocked_tasks == 1
        assert progress.percent_complete == 25.0

    def test_aggregate_empty_children(self):
        agg = ProgressAggregator()
        progress = agg.aggregate_from_children([])
        assert progress.total_tasks == 0
        assert progress.completed_tasks == 0
        assert progress.percent_complete == 0.0

    def test_aggregate_subphases_to_phases(self):
        """Progress rolls up from tasks -> subphases -> phases."""
        agg = ProgressAggregator()

        # Subphase 1: 2 tasks, both done
        sp1 = SubPhaseNode(id="sp1", name="SP1", parent_id="p1")
        sp1.tasks = [
            TaskNode(id="t1", name="T1", parent_id="sp1", status=NodeStatus.SUCCESS, progress=100.0),
            TaskNode(id="t2", name="T2", parent_id="sp1", status=NodeStatus.SUCCESS, progress=100.0),
        ]

        # Subphase 2: 2 tasks, 1 done, 1 running
        sp2 = SubPhaseNode(id="sp2", name="SP2", parent_id="p1")
        sp2.tasks = [
            TaskNode(id="t3", name="T3", parent_id="sp2", status=NodeStatus.SUCCESS, progress=100.0),
            TaskNode(id="t4", name="T4", parent_id="sp2", status=NodeStatus.RUNNING, progress=0.0),
        ]

        phase = PhaseNode(id="p1", name="Phase1", parent_id="proj-1")
        phase.subphases = [sp1, sp2]

        progress = agg.aggregate_phase(phase)
        assert progress.total_tasks == 4
        assert progress.completed_tasks == 3
        assert progress.percent_complete == 75.0

    def test_aggregate_project(self):
        """Full roll-up: tasks -> subphases -> phases -> project."""
        agg = ProgressAggregator()

        # Phase 1: fully done (2 tasks)
        p1 = PhaseNode(id="p1", name="P1", parent_id="proj-1")
        sp1 = SubPhaseNode(id="sp1", name="SP1", parent_id="p1")
        sp1.tasks = [
            TaskNode(id="t1", name="T1", parent_id="sp1", status=NodeStatus.SUCCESS, progress=100.0),
            TaskNode(id="t2", name="T2", parent_id="sp1", status=NodeStatus.SUCCESS, progress=100.0),
        ]
        p1.subphases = [sp1]

        # Phase 2: half done (2 tasks)
        p2 = PhaseNode(id="p2", name="P2", parent_id="proj-1")
        sp2 = SubPhaseNode(id="sp2", name="SP2", parent_id="p2")
        sp2.tasks = [
            TaskNode(id="t3", name="T3", parent_id="sp2", status=NodeStatus.SUCCESS, progress=100.0),
            TaskNode(id="t4", name="T4", parent_id="sp2", status=NodeStatus.PENDING, progress=0.0),
        ]
        p2.subphases = [sp2]

        proj = ProjectNode(id="proj-1", name="Project")
        proj.phases = [p1, p2]

        progress = agg.aggregate_project(proj)
        assert progress.total_tasks == 4
        assert progress.completed_tasks == 3
        assert progress.percent_complete == 75.0


# ── TransitionManager Tests ────────────────────────────────────────

class TestTransitionManager:
    """Tests for hierarchical state transitions."""

    def test_valid_transition_pending_to_running(self):
        tm = TransitionManager()
        node = HierarchyNode(id="n", name="Test")
        result = tm.can_transition(node, NodeStatus.RUNNING)
        assert result is True

    def test_valid_transition_running_to_success(self):
        tm = TransitionManager()
        node = HierarchyNode(id="n", name="Test", status=NodeStatus.RUNNING)
        result = tm.can_transition(node, NodeStatus.SUCCESS)
        assert result is True

    def test_invalid_transition_success_to_pending(self):
        tm = TransitionManager()
        node = HierarchyNode(id="n", name="Test", status=NodeStatus.SUCCESS)
        result = tm.can_transition(node, NodeStatus.PENDING)
        assert result is False

    def test_valid_transition_failed_to_reworking(self):
        tm = TransitionManager()
        node = HierarchyNode(id="n", name="Test", status=NodeStatus.FAILED)
        result = tm.can_transition(node, NodeStatus.REWORKING)
        assert result is True

    def test_invalid_transition_running_to_pending(self):
        tm = TransitionManager()
        node = HierarchyNode(id="n", name="Test", status=NodeStatus.RUNNING)
        result = tm.can_transition(node, NodeStatus.PENDING)
        assert result is False

    def test_perform_transition_changes_status(self):
        tm = TransitionManager()
        node = HierarchyNode(id="n", name="Test")
        result = tm.perform_transition(node, NodeStatus.RUNNING)
        assert result is True
        assert node.status == NodeStatus.RUNNING

    def test_perform_invalid_transition_raises(self):
        tm = TransitionManager()
        node = HierarchyNode(id="n", name="Test", status=NodeStatus.SUCCESS)
        with pytest.raises(ValueError, match="Invalid transition"):
            tm.perform_transition(node, NodeStatus.PENDING)

    def test_transition_chain_pending_thru_success(self):
        tm = TransitionManager()
        node = HierarchyNode(id="n", name="Test")

        assert tm.perform_transition(node, NodeStatus.RUNNING)
        assert node.status == NodeStatus.RUNNING

        assert tm.perform_transition(node, NodeStatus.SUCCESS)
        assert node.status == NodeStatus.SUCCESS

    def test_rework_cycle(self):
        """Failed -> Reworking -> Running -> Success cycle."""
        tm = TransitionManager()
        node = HierarchyNode(id="n", name="Test")

        tm.perform_transition(node, NodeStatus.RUNNING)
        tm.perform_transition(node, NodeStatus.FAILED)
        assert node.status == NodeStatus.FAILED

        tm.perform_transition(node, NodeStatus.REWORKING)
        assert node.status == NodeStatus.REWORKING

        tm.perform_transition(node, NodeStatus.RUNNING)
        tm.perform_transition(node, NodeStatus.SUCCESS)
        assert node.status == NodeStatus.SUCCESS

    def test_block_unblock_cycle(self):
        tm = TransitionManager()
        node = HierarchyNode(id="n", name="Test", status=NodeStatus.RUNNING)
        tm.perform_transition(node, NodeStatus.BLOCKED)
        assert node.status == NodeStatus.BLOCKED

        tm.perform_transition(node, NodeStatus.RUNNING)
        assert node.status == NodeStatus.RUNNING

        tm.perform_transition(node, NodeStatus.SUCCESS)
        assert node.status == NodeStatus.SUCCESS

    def test_register_custom_transition(self):
        tm = TransitionManager()
        tm.register_transition(NodeStatus.CANCELLED, NodeStatus.PENDING)
        node = HierarchyNode(id="n", name="Test", status=NodeStatus.CANCELLED)
        assert tm.can_transition(node, NodeStatus.PENDING) is True


# ── RalphLoopEngine Tests ──────────────────────────────────────────

class TestRalphLoopEngine:
    """Tests for the core Ralph Loop automation engine."""

    def test_create_engine(self):
        engine = RalphLoopEngine()
        assert engine is not None
        assert engine.transition_manager is not None
        assert engine.progress_aggregator is not None

    def test_create_project(self):
        engine = RalphLoopEngine()
        proj = engine.create_project("test-proj", "My Test Project")
        assert proj.id == "test-proj"
        assert proj.name == "My Test Project"
        assert proj.level == HierarchyLevel.PROJECT
        assert proj.status == NodeStatus.PENDING

    def test_create_phase(self):
        engine = RalphLoopEngine()
        proj = engine.create_project("proj", "Test")
        phase = engine.create_phase("phase-1", "Planning", parent_project=proj)
        assert phase.id == "phase-1"
        assert phase.parent_id == proj.id
        assert phase in proj.phases

    def test_create_subphase(self):
        engine = RalphLoopEngine()
        proj = engine.create_project("proj", "Test")
        phase = engine.create_phase("p1", "Planning", parent_project=proj)
        sp = engine.create_subphase("sp-1", "Analyze", parent_phase=phase)
        assert sp.id == "sp-1"
        assert sp.parent_id == phase.id
        assert sp in phase.subphases

    def test_create_task(self):
        engine = RalphLoopEngine()
        proj = engine.create_project("proj", "Test")
        phase = engine.create_phase("p1", "Phase", parent_project=proj)
        sp = engine.create_subphase("sp1", "SP", parent_phase=phase)
        task = engine.create_task("t1", "Task 1", parent_subphase=sp)
        assert task.id == "t1"
        assert task.parent_id == sp.id
        assert task in sp.tasks

    def test_full_tree_structure(self):
        """Build a full Project->Phase->Subphase->Task tree."""
        engine = RalphLoopEngine()
        proj = engine.create_project("proj", "Full Test")

        for i in range(3):
            phase = engine.create_phase(f"p{i}", f"Phase {i}", parent_project=proj)
            for j in range(2):
                sp = engine.create_subphase(f"sp-{i}-{j}", f"SP {j}", parent_phase=phase)
                for k in range(2):
                    engine.create_task(f"t-{i}-{j}-{k}", f"Task {k}", parent_subphase=sp)

        # Verify counts
        assert len(proj.phases) == 3
        assert sum(len(p.subphases) for p in proj.phases) == 6
        assert sum(len(sp.tasks) for p in proj.phases for sp in p.subphases) == 12

    def test_transition_node_in_tree(self):
        engine = RalphLoopEngine()
        proj = engine.create_project("proj", "Test")
        phase = engine.create_phase("p1", "Phase", parent_project=proj)
        sp = engine.create_subphase("sp1", "SP", parent_phase=phase)
        task = engine.create_task("t1", "Task", parent_subphase=sp)

        engine.transition_node(task, NodeStatus.RUNNING)
        assert task.status == NodeStatus.RUNNING
        assert task.progress == 0.0

        task.progress = 100.0
        engine.transition_node(task, NodeStatus.SUCCESS)
        assert task.status == NodeStatus.SUCCESS

    def test_engine_reports_project_progress(self):
        """Engine aggregates progress across the full tree."""
        engine = RalphLoopEngine()
        proj = engine.create_project("proj", "Test")
        phase = engine.create_phase("p1", "Phase", parent_project=proj)
        sp = engine.create_subphase("sp1", "SP", parent_phase=phase)

        t1 = engine.create_task("t1", "T1", parent_subphase=sp)
        t2 = engine.create_task("t2", "T2", parent_subphase=sp)

        engine.transition_node(t1, NodeStatus.RUNNING)
        t1.progress = 100.0
        engine.transition_node(t1, NodeStatus.SUCCESS)

        progress = engine.get_project_progress(proj)
        assert progress.total_tasks == 2
        assert progress.completed_tasks == 1
        assert progress.percent_complete == 50.0

    def test_find_node_by_id(self):
        engine = RalphLoopEngine()
        proj = engine.create_project("proj", "Test")
        phase = engine.create_phase("p1", "Phase", parent_project=proj)
        sp = engine.create_subphase("sp1", "SP", parent_phase=phase)
        task = engine.create_task("t1", "Task", parent_subphase=sp)

        found = engine.find_node(proj, "t1")
        assert found is task

        not_found = engine.find_node(proj, "nonexistent")
        assert not_found is None

    def test_validate_tree_no_cycles(self):
        """A single-parent tree with no cycles should validate."""
        engine = RalphLoopEngine()
        proj = engine.create_project("proj", "Test")
        phase = engine.create_phase("p1", "Phase", parent_project=proj)
        engine.create_subphase("sp1", "SP", parent_phase=phase)
        engine.create_task("t1", "Task", parent_subphase=engine.find_node(proj, "sp1"))

        validator = HierarchyValidator()
        result = validator.validate(proj)
        assert result.is_valid is True
        assert result.errors == []

    def test_collect_all_nodes(self):
        engine = RalphLoopEngine()
        proj = engine.create_project("proj", "Test")
        p1 = engine.create_phase("p1", "P1", parent_project=proj)
        sp1 = engine.create_subphase("sp1", "SP1", parent_phase=p1)
        engine.create_task("t1", "T1", parent_subphase=sp1)
        engine.create_task("t2", "T2", parent_subphase=sp1)

        nodes = engine.collect_all_nodes(proj)
        # project + phase + subphase + 2 tasks
        assert len(nodes) == 5
        ids = {n.id for n in nodes}
        assert ids == {"proj", "p1", "sp1", "t1", "t2"}

    def test_execute_simple_tree(self):
        """Execute a simple tree end-to-end with stub execution."""
        engine = RalphLoopEngine()
        proj = engine.create_project("proj", "Simple")
        phase = engine.create_phase("p1", "Phase", parent_project=proj)
        sp = engine.create_subphase("sp1", "SP", parent_phase=phase)
        engine.create_task("t1", "Task 1", parent_subphase=sp)
        engine.create_task("t2", "Task 2", parent_subphase=sp)

        result = engine.execute_project(proj)
        assert result is not None  # returns a ProjectNode with updated state
        assert result.status == NodeStatus.SUCCESS

        progress = engine.get_project_progress(result)
        assert progress.completed_tasks == 2
        assert progress.percent_complete == 100.0


# ── HierarchyValidator Tests ───────────────────────────────────────

class TestHierarchyValidator:
    """Tests for tree validation rules."""

    def test_validator_rejects_duplicate_ids(self):
        """Tree with duplicate node IDs should fail."""
        proj = ProjectNode(id="proj", name="Test")
        p1 = PhaseNode(id="dup", name="P1", parent_id="proj")
        p2 = PhaseNode(id="dup", name="P2", parent_id="proj")
        proj.phases = [p1, p2]

        validator = HierarchyValidator()
        result = validator.validate(proj)
        assert result.is_valid is False
        assert any("duplicate" in e.lower() for e in result.errors)

    def test_validator_rejects_orphan_phase(self):
        """Phase without a parent project should fail if strict."""
        phase = PhaseNode(id="p1", name="Orphan")
        validator = HierarchyValidator()
        result = validator.validate(phase)
        # Orphaned node should raise an issue
        assert not result.is_valid

    def test_validator_checks_parent_references(self):
        """Parent references should point to existing nodes."""
        proj = ProjectNode(id="proj", name="Test")
        phase = PhaseNode(id="p1", name="P1", parent_id="nonexistent")
        proj.phases = [phase]

        validator = HierarchyValidator()
        result = validator.validate(proj)
        assert result.is_valid is False


# ── Integration with State Machine Tests ───────────────────────────

class TestStateMachineIntegration:
    """Tests for integrating RalphLoop with spine state machine."""

    def test_ralph_loop_engine_accepts_state_machine(self):
        """RalphLoop can be created with a reference to the state machine."""
        engine = RalphLoopEngine()
        sm = SpineStateMachine()
        engine.attach_state_machine(sm)
        assert engine._state_machine is sm

    def test_create_hierarchy_from_phase_list(self):
        """Convert existing Phase/SubPhase/Task models into hierarchy."""
        engine = RalphLoopEngine()
        phases = [
            Phase(
                name=PhaseName.PLANNING,
                subphases=[
                    SubPhase(
                        name="ANALYZE",
                        tasks=[Task(id="t1", description="Analyze input")],
                    )
                ],
            ),
        ]

        proj = engine.create_project_from_phases("proj", "From Phases", phases)
        assert proj is not None
        assert len(proj.phases) == 1
        assert proj.phases[0].name == "Planning"
        assert len(proj.phases[0].subphases) == 1
        assert proj.phases[0].subphases[0].name == "ANALYZE"
        assert len(proj.phases[0].subphases[0].tasks) == 1

    def test_state_machine_uses_ralph_loop_for_tracking(self):
        """State machine can delegate hierarchy tracking to RalphLoop."""
        sm = SpineStateMachine()
        engine = RalphLoopEngine()
        engine.attach_state_machine(sm)

        proj = engine.create_project("sm-proj", "SM Project")
        phase = engine.create_phase("p1", "Planning", parent_project=proj)
        sp = engine.create_subphase("sp1", "Analyze", parent_phase=phase)
        task = engine.create_task("t1", "Parse", parent_subphase=sp)

        # Transition task
        engine.transition_node(task, NodeStatus.RUNNING)
        task.progress = 100.0
        engine.transition_node(task, NodeStatus.SUCCESS)

        # Subphase should auto-complete after all tasks done
        # (auto-completion is opt-in via engine config)
        engine.auto_complete_parents = True
        sp_progress = engine.progress_aggregator.aggregate_from_children(sp.tasks)
        if sp_progress.completed_tasks == sp_progress.total_tasks and sp_progress.total_tasks > 0:
            engine.transition_node(sp, NodeStatus.SUCCESS)

        assert task.status == NodeStatus.SUCCESS
        # Progress still tracked even without auto-complete
        proj_progress = engine.get_project_progress(proj)
        assert proj_progress.total_tasks == 1
        assert proj_progress.completed_tasks == 1


# ── NodeStatus Enum Tests ──────────────────────────────────────────

class TestNodeStatus:
    """Tests for NodeStatus enum completeness."""

    def test_all_expected_statuses_exist(self):
        assert hasattr(NodeStatus, "PENDING")
        assert hasattr(NodeStatus, "RUNNING")
        assert hasattr(NodeStatus, "SUCCESS")
        assert hasattr(NodeStatus, "FAILED")
        assert hasattr(NodeStatus, "BLOCKED")
        assert hasattr(NodeStatus, "REWORKING")
        assert hasattr(NodeStatus, "CANCELLED")

    def test_status_values_are_strings(self):
        assert isinstance(NodeStatus.PENDING.value, str)
        assert NodeStatus.PENDING.value == "pending"
        assert NodeStatus.RUNNING.value == "running"
        assert NodeStatus.SUCCESS.value == "success"


# ── Edge Case Tests ────────────────────────────────────────────────

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_project_progress(self):
        engine = RalphLoopEngine()
        proj = engine.create_project("empty", "Empty")
        progress = engine.get_project_progress(proj)
        assert progress.total_tasks == 0
        assert progress.percent_complete == 0.0

    def test_transition_node_not_in_engine_tracking(self):
        """Transitioning a free node not in any tree still works."""
        engine = RalphLoopEngine()
        node = HierarchyNode(id="free", name="Free")
        engine.transition_node(node, NodeStatus.RUNNING)
        assert node.status == NodeStatus.RUNNING

    def test_single_phase_no_subphases_progress(self):
        engine = RalphLoopEngine()
        proj = engine.create_project("proj", "Test")
        engine.create_phase("p1", "Empty Phase", parent_project=proj)
        progress = engine.get_project_progress(proj)
        assert progress.total_tasks == 0

    def test_multiple_levels_independent_transitions(self):
        """Each level can transition independently without forcing children."""
        engine = RalphLoopEngine()
        proj = engine.create_project("proj", "Test")
        phase = engine.create_phase("p1", "Phase", parent_project=proj)
        sp = engine.create_subphase("sp1", "SP", parent_phase=phase)
        engine.create_task("t1", "Task", parent_subphase=sp)

        # Transition parent without transitioning children
        engine.transition_node(proj, NodeStatus.RUNNING)
        assert proj.status == NodeStatus.RUNNING
        assert phase.status == NodeStatus.PENDING  # unchanged
        assert sp.status == NodeStatus.PENDING  # unchanged

    def test_failed_task_affects_subphase_progress(self):
        agg = ProgressAggregator()
        tasks = [
            TaskNode(id="t1", name="T1", parent_id="sp1", status=NodeStatus.SUCCESS, progress=100.0),
            TaskNode(id="t2", name="T2", parent_id="sp1", status=NodeStatus.FAILED, progress=10.0),
        ]
        progress = agg.aggregate_from_children(tasks)
        assert progress.failed_tasks == 1
        assert progress.completed_tasks == 1
        assert progress.total_tasks == 2
        # percent should still reflect only completed vs total
        assert progress.percent_complete == 50.0


# ── Named Transitions Smoke Test ───────────────────────────────────

class TestNamedTransitions:
    """Tests for named/pattern-based transition rules."""

    def test_transition_manager_has_default_rules(self):
        tm = TransitionManager()
        assert len(tm.rules) > 0

    def test_transition_rules_are_symmetric_where_needed(self):
        """Failed<->Reworking, Blocked<->Running should be bidirectional where sensible."""
        tm = TransitionManager()
        # PENDING -> RUNNING is allowed
        node = HierarchyNode(id="n", name="Test", status=NodeStatus.PENDING)
        assert tm.can_transition(node, NodeStatus.RUNNING) is True
        # PENDING -> SUCCESS is allowed (direct completion)
        assert tm.can_transition(node, NodeStatus.SUCCESS) is True
        # But PENDING -> FAILED should NOT be allowed (hasn't run yet)
        assert tm.can_transition(node, NodeStatus.FAILED) is False

    def test_terminal_states_block_most_transitions(self):
        tm = TransitionManager()
        success_node = HierarchyNode(id="n", name="Test", status=NodeStatus.SUCCESS)
        assert tm.can_transition(success_node, NodeStatus.RUNNING) is False
        assert tm.can_transition(success_node, NodeStatus.FAILED) is False

        cancelled_node = HierarchyNode(id="n2", name="Test2", status=NodeStatus.CANCELLED)
        assert tm.can_transition(cancelled_node, NodeStatus.RUNNING) is False
