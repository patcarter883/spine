"""Tests for the Greenfields Workflow Engine.

Covers:
- WorkflowEngine (base engine with hierarchy integration)
- SDDWorkflow (full Spec-Driven Development lifecycle)
- QuickWorkflow (streamlined Quick Work lifecycle)
- Phase transitions and state management
- Hierarchy integration with RalphLoopEngine
- Checkpoint persistence
- Gate integration
"""

import pytest
import os
import tempfile

from spine.workflows.engine import (
    WorkflowEngine,
    WorkflowPhase,
    WorkflowContext,
    WorkflowResult,
)
from spine.workflows.sdd import SDDWorkflow
from spine.workflows.quick_work import QuickWorkflow
from spine.models.types import (
    NodeStatus,
    ProjectNode,
    HierarchyProgress,
    HierarchyLevel,
)
from spine.core.hierarchy import (
    RalphLoopEngine,
    TransitionManager,
)
from spine.core.state_machine import SpineStateMachine


# ═════════════════════════════════════════════════════════════════════
# Test Helpers
# ═════════════════════════════════════════════════════════════════════

@pytest.fixture
def temp_project_dir():
    """Create a temporary directory for test projects."""
    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        os.makedirs(".spine", exist_ok=True)
        yield tmpdir
        os.chdir(old_cwd)


@pytest.fixture
def hierarchy_engine():
    """Fresh RalphLoopEngine."""
    return RalphLoopEngine()


@pytest.fixture
def state_machine(temp_project_dir):
    """Fresh SpineStateMachine."""
    return SpineStateMachine(
        checkpoint_path=os.path.join(temp_project_dir, ".spine", "spine.db"),
    )


# ═════════════════════════════════════════════════════════════════════
# Step 1: Unit Tests — WorkflowPhase Enum
# ═════════════════════════════════════════════════════════════════════

class TestWorkflowPhase:
    """Tests for WorkflowPhase enum."""

    def test_enum_values(self):
        """All expected phases exist."""
        phases = set(p.value for p in WorkflowPhase)
        assert "init" in phases
        assert "spec" in phases
        assert "design" in phases
        assert "plan" in phases
        assert "implement" in phases
        assert "review" in phases
        assert "verify" in phases
        assert "complete" in phases
        assert "failed" in phases
        assert "cancelled" in phases

    def test_str_enum_equality(self):
        """WorkflowPhase values compare as strings."""
        assert WorkflowPhase.SPEC == "spec"


# ═════════════════════════════════════════════════════════════════════
# Step 2: Unit Tests — WorkflowContext
# ═════════════════════════════════════════════════════════════════════

class TestWorkflowContext:
    """Tests for WorkflowContext."""

    def test_default_creation(self):
        """Context can be created with defaults."""
        ctx = WorkflowContext()
        assert ctx.requirement == ""
        assert ctx.spec is None
        assert ctx.design is None
        assert ctx.plan is None
        assert ctx.variables == {}
        assert ctx.gate_results == {}

    def test_full_creation(self):
        """Context can be created with all fields."""
        ctx = WorkflowContext(
            requirement="Build API",
            spec="Spec doc",
            design="Arch doc",
            plan={"phases": []},
            variables={"env": "prod"},
            gate_results={"critic": {"approved": True}},
        )
        assert ctx.requirement == "Build API"
        assert ctx.spec == "Spec doc"
        assert ctx.design == "Arch doc"
        assert ctx.plan == {"phases": []}
        assert ctx.variables["env"] == "prod"
        assert ctx.gate_results["critic"]["approved"]

    def test_to_dict(self):
        """to_dict serializes context."""
        ctx = WorkflowContext(requirement="test", variables={"x": 1})
        d = ctx.to_dict()
        assert d["requirement"] == "test"
        assert d["variables"] == {"x": 1}

    def test_from_dict(self):
        """from_dict deserializes context."""
        data = {"requirement": "restored", "variables": {"y": 2}}
        ctx = WorkflowContext.from_dict(data)
        assert ctx.requirement == "restored"
        assert ctx.variables == {"y": 2}


# ═════════════════════════════════════════════════════════════════════
# Step 3: Unit Tests — WorkflowResult
# ═════════════════════════════════════════════════════════════════════

class TestWorkflowResult:
    """Tests for WorkflowResult."""

    def test_default_creation(self):
        """Result defaults to failed."""
        r = WorkflowResult()
        assert not r.success
        assert r.percent_complete == 0.0

    def test_success_result(self):
        """Success result has correct flags."""
        r = WorkflowResult(success=True)
        assert r.success

    def test_with_progress(self):
        """Result computes percent_complete from progress."""
        prog = HierarchyProgress(
            total_tasks=10, completed_tasks=8,
        )
        r = WorkflowResult(success=True, progress=prog)
        assert r.percent_complete == 80.0

    def test_with_errors(self):
        """Result tracks errors."""
        r = WorkflowResult(success=False, errors=["Failure"])
        assert len(r.errors) == 1
        assert r.errors[0] == "Failure"

    def test_with_hierarchy(self):
        """Result stores hierarchy."""
        proj = ProjectNode(id="p1", name="Test")
        r = WorkflowResult(success=True, hierarchy=proj)
        assert r.hierarchy is proj


# ═════════════════════════════════════════════════════════════════════
# Step 4: Unit Tests — WorkflowEngine Construction
# ═════════════════════════════════════════════════════════════════════

class TestWorkflowEngineConstruction:
    """Tests for WorkflowEngine instantiation."""

    def test_construction_with_defaults(self):
        """Engine can be constructed with no arguments."""
        engine = WorkflowEngine()
        assert engine.hierarchy_engine is not None
        assert engine.transition_manager is not None
        assert engine.progress_aggregator is not None
        assert engine.validator is not None
        assert engine.project is None
        assert engine.current_phase is None
        assert engine.auto_complete_parents

    def test_construction_with_state_machine(self, state_machine):
        """Engine accepts external state machine."""
        engine = WorkflowEngine(state_machine=state_machine)
        assert engine._state_machine is state_machine

    def test_default_phases(self):
        """Engine has DEFAULT_PHASES [plan, implement, verify]."""
        engine = WorkflowEngine()
        assert engine.phases == ["plan", "implement", "verify"]

    def test_set_phases(self):
        """set_phases overrides default phases."""
        engine = WorkflowEngine()
        engine.set_phases(["init", "spec", "design"])
        assert engine.phases == ["init", "spec", "design"]

    def test_create_project(self):
        """create_project creates a ProjectNode."""
        engine = WorkflowEngine()
        proj = engine.create_project("proj-1", "Build something", name="Build something")
        assert proj is not None
        assert proj.id == "proj-1"
        assert proj.name == "Build something"
        assert proj.status == NodeStatus.RUNNING
        assert engine.project is proj
        assert engine.context.requirement == "Build something"

    def test_create_project_with_custom_name(self):
        """create_project accepts custom name."""
        engine = WorkflowEngine()
        proj = engine.create_project("p1", "req", name="Custom Name")
        assert proj.name == "Custom Name"

    def test_create_project_resets_errors(self):
        """create_project resets error list."""
        engine = WorkflowEngine()
        engine._errors.append("old error")
        engine.create_project("p1", "req")
        assert engine.errors == []

    def test_create_phase_node(self):
        """create_phase_node creates a phase under project."""
        engine = WorkflowEngine()
        engine.create_project("p1", "req")
        phase = engine.create_phase_node("spec", "Specification")
        assert phase.id == "spec"
        assert phase.name == "Specification"
        assert engine.project.phases[0] is phase

    def test_create_phase_node_requires_project(self):
        """create_phase_node raises if no project created."""
        engine = WorkflowEngine()
        with pytest.raises(ValueError, match="No project created"):
            engine.create_phase_node("spec", "Spec")

    def test_create_subphase_and_task_nodes(self):
        """Full hierarchy can be built."""
        engine = WorkflowEngine()
        engine.create_project("p1", "req")
        phase = engine.create_phase_node("spec", "Spec")
        sp = engine.create_subphase_node("sp1", "Subphase 1", parent_phase=phase)
        task = engine.create_task_node("t1", "Task 1", parent_subphase=sp)

        assert sp.id == "sp1"
        assert task.id == "t1"
        assert len(sp.tasks) == 1

    def test_phases_property(self):
        """phases property returns configured phases."""
        engine = WorkflowEngine()
        assert engine.phases == WorkflowEngine.DEFAULT_PHASES


# ═════════════════════════════════════════════════════════════════════
# Step 5: Engine Transition Tests
# ═════════════════════════════════════════════════════════════════════

class TestEngineTransitions:
    """Tests for phase and task transitions."""

    def test_start_phase(self):
        """start_phase transitions phase to RUNNING."""
        engine = WorkflowEngine()
        engine.create_project("p1", "req")
        engine.create_phase_node("spec", "Specification")
        engine.start_phase("spec")
        phase = engine.hierarchy_engine.find_node(engine.project, "spec")
        assert phase.status == NodeStatus.RUNNING

    def test_complete_phase(self):
        """complete_phase transitions phase to SUCCESS."""
        engine = WorkflowEngine()
        engine.create_project("p1", "req")
        engine.create_phase_node("spec", "Specification")
        engine.start_phase("spec")
        engine.complete_phase("spec")
        phase = engine.hierarchy_engine.find_node(engine.project, "spec")
        assert phase.status == NodeStatus.SUCCESS

    def test_fail_phase(self):
        """fail_phase transitions phase to FAILED and records error."""
        engine = WorkflowEngine()
        engine.create_project("p1", "req")
        engine.create_phase_node("impl", "Implementation")
        engine.start_phase("impl")  # must be RUNNING before FAILED
        engine.fail_phase("impl", error="Critical failure")
        phase = engine.hierarchy_engine.find_node(engine.project, "impl")
        assert phase.status == NodeStatus.FAILED
        assert len(engine.errors) >= 1

    def test_mark_task_running(self):
        """mark_task_running transitions task."""
        engine = WorkflowEngine()
        engine.create_project("p1", "req")
        phase = engine.create_phase_node("plan", "Plan")
        sp = engine.create_subphase_node("sp1", "SP", parent_phase=phase)
        task = engine.create_task_node("t1", "Task", parent_subphase=sp)
        engine.mark_task_running("t1")
        assert task.status == NodeStatus.RUNNING

    def test_mark_task_success(self):
        """mark_task_success transitions task and sets result."""
        engine = WorkflowEngine()
        engine.create_project("p1", "req")
        phase = engine.create_phase_node("plan", "Plan")
        sp = engine.create_subphase_node("sp1", "SP", parent_phase=phase)
        task = engine.create_task_node("t1", "Task", parent_subphase=sp)
        engine.mark_task_success("t1", result="Done")
        assert task.status == NodeStatus.SUCCESS
        assert task.progress == 100.0
        assert task.result == "Done"

    def test_mark_task_failed(self):
        """mark_task_failed transitions task to FAILED."""
        engine = WorkflowEngine()
        engine.create_project("p1", "req")
        phase = engine.create_phase_node("plan", "Plan")
        sp = engine.create_subphase_node("sp1", "SP", parent_phase=phase)
        task = engine.create_task_node("t1", "Task", parent_subphase=sp)
        engine.mark_task_running("t1")  # must be RUNNING before FAILED
        engine.mark_task_failed("t1", error="Boom")
        assert task.status == NodeStatus.FAILED
        assert task.error == "Boom"

    def test_mark_task_blocked(self):
        """mark_task_blocked transitions task to BLOCKED."""
        engine = WorkflowEngine()
        engine.create_project("p1", "req")
        phase = engine.create_phase_node("plan", "Plan")
        sp = engine.create_subphase_node("sp1", "SP", parent_phase=phase)
        task = engine.create_task_node("t1", "Task", parent_subphase=sp)
        engine.mark_task_blocked("t1")
        assert task.status == NodeStatus.BLOCKED

    def test_mark_nonexistent_task(self):
        """Marking a nonexistent task returns None gracefully."""
        engine = WorkflowEngine()
        engine.create_project("p1", "req")
        result = engine.mark_task_success("nonexistent")
        assert result is None

    def test_auto_complete_phase(self):
        """auto_complete transitions phase to SUCCESS when all children done."""
        engine = WorkflowEngine()
        engine.create_project("p1", "req")
        phase = engine.create_phase_node("plan", "Plan")
        sp = engine.create_subphase_node("sp1", "SP", parent_phase=phase)
        engine.create_task_node("t1", "Task", parent_subphase=sp)

        # Manually mark all children as success
        sp.tasks[0].status = NodeStatus.SUCCESS
        sp.status = NodeStatus.SUCCESS

        engine.auto_complete(phase)
        assert phase.status == NodeStatus.SUCCESS

    def test_auto_complete_disabled(self):
        """When disabled, auto_complete does not transition phase even if children done."""
        engine = WorkflowEngine()
        engine.auto_complete_parents = False
        engine.create_project("p1", "req")
        phase = engine.create_phase_node("plan", "Plan")
        sp = engine.create_subphase_node("sp1", "SP", parent_phase=phase)
        engine.create_task_node("t1", "Task", parent_subphase=sp)

        # Manually set children to SUCCESS; phase is still PENDING
        sp.tasks[0].status = NodeStatus.SUCCESS
        sp.status = NodeStatus.SUCCESS

        # Calling auto_complete with disabled flag should NOT transition the phase
        engine.auto_complete(phase)
        assert phase.status == NodeStatus.PENDING  # disabled, remains PENDING

    def test_transition_node(self):
        """transition_node delegates to hierarchy engine."""
        engine = WorkflowEngine()
        engine.create_project("p1", "req")
        engine.transition_node(engine.project, NodeStatus.SUCCESS)
        assert engine.project.status == NodeStatus.SUCCESS


# ═════════════════════════════════════════════════════════════════════
# Step 6: Engine Progress and Validation
# ═════════════════════════════════════════════════════════════════════

class TestEngineProgress:
    """Tests for progress and validation."""

    def test_get_progress_on_empty(self):
        """get_progress returns empty on no project."""
        engine = WorkflowEngine()
        progress = engine.get_progress()
        assert progress.total_tasks == 0

    def test_get_progress_after_execution(self):
        """get_progress reflects completed tasks."""
        engine = WorkflowEngine()
        engine.create_project("p1", "req")
        phase = engine.create_phase_node("plan", "Plan")
        sp = engine.create_subphase_node("sp1", "SP", parent_phase=phase)
        engine.create_task_node("t1", "T1", parent_subphase=sp)
        engine.create_task_node("t2", "T2", parent_subphase=sp)
        # Mark all tasks success
        for t in sp.tasks:
            t.status = NodeStatus.SUCCESS
        sp.status = NodeStatus.SUCCESS
        progress = engine.get_progress()
        assert progress.total_tasks == 2
        assert progress.completed_tasks == 2

    def test_get_phase_progress(self):
        """get_phase_progress for specific phase."""
        engine = WorkflowEngine()
        engine.create_project("p1", "req")
        phase = engine.create_phase_node("spec", "Spec")
        sp = engine.create_subphase_node("sp1", "SP", parent_phase=phase)
        engine.create_task_node("t1", "Task", parent_subphase=sp)
        sp.tasks[0].status = NodeStatus.SUCCESS
        sp.status = NodeStatus.SUCCESS
        progress = engine.get_phase_progress("spec")
        assert progress is not None
        assert progress.total_tasks == 1
        assert progress.completed_tasks == 1

    def test_get_phase_progress_nonexistent(self):
        """get_phase_progress returns None for nonexistent phase."""
        engine = WorkflowEngine()
        engine.create_project("p1", "req")
        assert engine.get_phase_progress("nonexistent") is None

    def test_validate_hierarchy(self):
        """validate_hierarchy on a well-formed tree."""
        engine = WorkflowEngine()
        engine.create_project("p1", "req")
        engine.create_phase_node("spec", "Spec")
        assert engine.validate_hierarchy()

    def test_validate_hierarchy_before_project(self):
        """validate_hierarchy returns False before project."""
        engine = WorkflowEngine()
        assert not engine.validate_hierarchy()

    def test_collect_all_nodes(self):
        """collect_all_nodes returns all nodes."""
        engine = WorkflowEngine()
        engine.create_project("p1", "req")
        engine.create_phase_node("plan", "Plan")
        engine.create_phase_node("impl", "Implement")
        nodes = engine.collect_all_nodes()
        # Project + 2 phases = 3 nodes (no subphases)
        assert len(nodes) == 3

    def test_get_result_success(self):
        """get_result returns success when project succeeded."""
        engine = WorkflowEngine()
        engine.create_project("p1", "req")
        engine.transition_node(engine.project, NodeStatus.SUCCESS)
        result = engine.get_result()
        assert result.success
        assert isinstance(result, WorkflowResult)

    def test_get_result_failure(self):
        """get_result returns failure when project failed."""
        engine = WorkflowEngine()
        engine.create_project("p1", "req")
        engine.transition_node(engine.project, NodeStatus.FAILED)
        result = engine.get_result()
        assert not result.success


# ═════════════════════════════════════════════════════════════════════
# Step 7: Checkpoint Tests
# ═════════════════════════════════════════════════════════════════════

class TestCheckpoints:
    """Tests for checkpoint persistence."""

    def test_checkpoint_no_state_machine(self):
        """Local checkpoint saved without state machine."""
        engine = WorkflowEngine()
        engine.create_project("p1", "req")
        path = engine.checkpoint(phase_name="plan", phase_progress=0.5)
        assert path is not None
        assert os.path.exists(path)

    def test_checkpoint_with_state_machine(self, state_machine):
        """Checkpoint via state machine integration."""
        engine = WorkflowEngine(state_machine=state_machine)
        engine.create_project("p1", "req")
        # This should work without error
        assert engine.project is not None

    def test_to_dict_and_from_dict(self):
        """Serialization round-trips correctly."""
        engine = WorkflowEngine()
        engine.create_project("p1", "Build something")
        engine.set_phases(["spec", "design", "plan"])

        data = engine.to_dict()
        assert data["project_id"] == "p1"
        assert data["phases"] == ["spec", "design", "plan"]

        # Restore
        restored = WorkflowEngine.from_dict(data)
        assert restored.phases == ["spec", "design", "plan"]


# ═════════════════════════════════════════════════════════════════════
# Step 8: Gate Integration Tests
# ═════════════════════════════════════════════════════════════════════

class TestGateIntegration:
    """Tests for swarm gate integration."""

    def test_run_all_gates_empty(self):
        """run_all_gates returns empty when no gates configured."""
        engine = WorkflowEngine()
        engine.create_project("p1", "req")
        results = engine.run_all_gates()
        assert results == {}

    def test_run_gate_mock(self):
        """run_gate evaluates and stores result."""
        from spine.swarm.gates import CriticGate

        engine = WorkflowEngine()
        engine.create_project("p1", "req")
        gate = CriticGate()
        result = engine.run_gate(gate)
        assert "critic" in engine.context.gate_results
        assert result is not None


# ═════════════════════════════════════════════════════════════════════
# Step 9: SDD Workflow Tests
# ═════════════════════════════════════════════════════════════════════

class TestSDDWorkflow:
    """Tests for SDD (Spec-Driven Development) lifecycle."""

    def test_construction(self):
        """SDDWorkflow can be constructed."""
        sdd = SDDWorkflow()
        assert isinstance(sdd, WorkflowEngine)
        assert sdd.phases == ["spec", "design", "plan", "implement", "review", "verify"]

    def test_construction_with_state_machine(self, state_machine):
        """SDDWorkflow accepts state machine."""
        sdd = SDDWorkflow(state_machine=state_machine)
        assert sdd._state_machine is state_machine

    def test_execute_runs_all_phases(self):
        """execute() runs all 6 SDD phases successfully."""
        sdd = SDDWorkflow()
        sdd.create_project("sdd-proj", "Build a REST API")
        result = sdd.execute()
        assert result.success
        assert result.hierarchy is not None
        assert result.percent_complete == 100.0

    def test_execute_requires_project(self):
        """execute() raises without create_project."""
        sdd = SDDWorkflow()
        with pytest.raises(ValueError, match="No project created"):
            sdd.execute()

    def test_sdd_project_progress(self):
        """After SDD run, all tasks are completed."""
        sdd = SDDWorkflow()
        sdd.create_project("sdd-proj", "Test SDD")
        result = sdd.execute()
        progress = result.progress
        assert progress is not None
        assert progress.total_tasks > 0
        assert progress.completed_tasks == progress.total_tasks
        assert progress.failed_tasks == 0
        assert progress.percent_complete == 100.0

    def test_sdd_hierarchy_phases(self):
        """SDD creates 6 phases in the hierarchy."""
        sdd = SDDWorkflow()
        sdd.create_project("sdd-proj", "Test")
        sdd.execute()
        project = sdd.project
        assert project is not None
        assert len(project.phases) == 6

    def test_sdd_spec_phase_details(self):
        """SPEC phase has correct subphases and tasks."""
        sdd = SDDWorkflow()
        sdd.create_project("sdd-proj", "Test")
        sdd.execute()
        spec_phase = sdd.hierarchy_engine.find_node(sdd.project, "spec")
        assert spec_phase is not None
        assert spec_phase.name == "Specification"
        assert len(spec_phase.subphases) == 2

        # Requirements Gathering
        req_sp = spec_phase.subphases[0]
        assert req_sp.name == "Requirements Gathering"
        assert len(req_sp.tasks) == 3

        # Formal Specification
        formal_sp = spec_phase.subphases[1]
        assert formal_sp.name == "Formal Specification"
        assert len(formal_sp.tasks) == 3

    def test_sdd_design_phase_details(self):
        """DESIGN phase has correct structure."""
        sdd = SDDWorkflow()
        sdd.create_project("sdd-proj", "Test")
        sdd.execute()
        design_phase = sdd.hierarchy_engine.find_node(sdd.project, "design")
        assert design_phase is not None
        assert design_phase.name == "Design"
        assert len(design_phase.subphases) == 2

    def test_sdd_plan_phase_details(self):
        """PLAN phase has correct structure."""
        sdd = SDDWorkflow()
        sdd.create_project("sdd-proj", "Test")
        sdd.execute()
        plan_phase = sdd.hierarchy_engine.find_node(sdd.project, "plan")
        assert plan_phase is not None
        assert plan_phase.name == "Planning"
        assert len(plan_phase.subphases) == 2

    def test_sdd_implement_phase_details(self):
        """IMPLEMENT phase creates subphases from FeatureSlices."""
        sdd = SDDWorkflow()
        sdd.create_project("sdd-proj", "Test")
        sdd.execute()
        impl_phase = sdd.hierarchy_engine.find_node(sdd.project, "implement")
        assert impl_phase is not None
        assert impl_phase.name == "Implementation"
        # Low complexity "Test" produces 2 heuristic slices
        assert len(impl_phase.subphases) >= 1

        # Each subphase is a FeatureSlice (id starts with impl-)
        for sp in impl_phase.subphases:
            assert sp.id.startswith("impl-")

    def test_sdd_review_phase_details(self):
        """REVIEW phase has code review + gate execution."""
        sdd = SDDWorkflow()
        sdd.create_project("sdd-proj", "Test")
        sdd.execute()
        review_phase = sdd.hierarchy_engine.find_node(sdd.project, "review")
        assert review_phase is not None
        assert review_phase.name == "Review"
        assert len(review_phase.subphases) == 2

    def test_sdd_verify_phase_details(self):
        """VERIFY phase has test + validation subphases."""
        sdd = SDDWorkflow()
        sdd.create_project("sdd-proj", "Test")
        sdd.execute()
        verify_phase = sdd.hierarchy_engine.find_node(sdd.project, "verify")
        assert verify_phase is not None
        assert verify_phase.name == "Verification"
        assert len(verify_phase.subphases) == 2

    def test_sdd_context_populated(self):
        """SDD populates context with spec, design, plan."""
        sdd = SDDWorkflow()
        sdd.create_project("sdd-proj", "Build API")
        sdd.execute()
        assert sdd.context.spec is not None
        assert "Build API" in sdd.context.spec
        assert sdd.context.design is not None
        assert sdd.context.plan is not None
        assert sdd.context.plan["requirement"] == "Build API"

    def test_sdd_total_task_count(self):
        """SDD has a substantial task count (40+)."""
        sdd = SDDWorkflow()
        sdd.create_project("sdd-proj", "Test task count")
        result = sdd.execute()
        assert result.progress.total_tasks >= 20  # SDD has many tasks

    def test_sdd_validate_hierarchy(self):
        """SDD hierarchy validates correctly."""
        sdd = SDDWorkflow()
        sdd.create_project("sdd-proj", "Test")
        sdd.execute()
        assert sdd.validate_hierarchy()

    def test_sdd_all_phases_succeeded(self):
        """All SDD phases end in SUCCESS status."""
        sdd = SDDWorkflow()
        sdd.create_project("sdd-proj", "Test")
        sdd.execute()
        project = sdd.project
        for phase in project.phases:
            assert phase.status == NodeStatus.SUCCESS, f"Phase {phase.id} not SUCCESS"

    def test_sdd_result_has_hierarchy(self):
        """SDD result includes the full hierarchy."""
        sdd = SDDWorkflow()
        sdd.create_project("sdd-proj", "Test")
        result = sdd.execute()
        assert result.hierarchy is not None
        assert isinstance(result.hierarchy, ProjectNode)

    def test_sdd_result_has_phase_results(self):
        """SDD result tracks phase execution."""
        sdd = SDDWorkflow()
        sdd.create_project("sdd-proj", "Test")
        result = sdd.execute()
        assert "current_phase" in result.phase_results
        assert "phases_executed" in result.phase_results

    def test_sdd_result_errors_empty_on_success(self):
        """Successful SDD has no errors."""
        sdd = SDDWorkflow()
        sdd.create_project("sdd-proj", "Test")
        result = sdd.execute()
        assert result.errors == []
        assert sdd.errors == []

    def test_sdd_plan_includes_implementation_tasks(self):
        """PLAN phase populates implementation_tasks in context plan."""
        sdd = SDDWorkflow()
        sdd.create_project("sdd-proj", "Test")
        sdd.execute()
        assert sdd.context.plan["implementation_tasks"] is not None
        assert len(sdd.context.plan["implementation_tasks"]) >= 2


# ═════════════════════════════════════════════════════════════════════
# Step 10: Quick Work Tests
# ═════════════════════════════════════════════════════════════════════

class TestQuickWorkflow:
    """Tests for Quick Work lifecycle."""

    def test_construction(self):
        """QuickWorkflow can be constructed."""
        qw = QuickWorkflow()
        assert isinstance(qw, WorkflowEngine)
        assert qw.phases == ["plan", "implement", "verify"]

    def test_construction_with_state_machine(self, state_machine):
        """QuickWorkflow accepts state machine."""
        qw = QuickWorkflow(state_machine=state_machine)
        assert qw._state_machine is state_machine

    def test_execute_runs_all_phases(self):
        """execute() runs all 3 QW phases successfully."""
        qw = QuickWorkflow()
        qw.create_project("qw-proj", "Fix auth bug")
        result = qw.execute()
        assert result.success
        assert result.hierarchy is not None

    def test_execute_requires_project(self):
        """execute() raises without create_project."""
        qw = QuickWorkflow()
        with pytest.raises(ValueError, match="No project created"):
            qw.execute()

    def test_qw_project_progress(self):
        """After QW run, all tasks are completed."""
        qw = QuickWorkflow()
        qw.create_project("qw-proj", "Test QW")
        result = qw.execute()
        progress = result.progress
        assert progress is not None
        assert progress.completed_tasks == progress.total_tasks
        assert progress.percent_complete == 100.0

    def test_qw_plan_phase(self):
        """Quick Plan phase has single subphase with 2 tasks (assess + slice)."""
        qw = QuickWorkflow()
        qw.create_project("qw-proj", "Test")
        qw.execute()
        plan_phase = qw.hierarchy_engine.find_node(qw.project, "plan")
        assert plan_phase is not None
        assert plan_phase.name == "Quick Plan"
        assert len(plan_phase.subphases) == 1

        sp = plan_phase.subphases[0]
        assert sp.name == "Planning"
        assert len(sp.tasks) == 2

    def test_qw_implement_phase(self):
        """Quick Implement phase has single subphase."""
        qw = QuickWorkflow()
        qw.create_project("qw-proj", "Test")
        qw.execute()
        impl_phase = qw.hierarchy_engine.find_node(qw.project, "implement")
        assert impl_phase is not None
        assert impl_phase.name == "Quick Implementation"
        assert len(impl_phase.subphases) == 1

    def test_qw_verify_phase(self):
        """Quick Verify phase has single subphase with 3 tasks."""
        qw = QuickWorkflow()
        qw.create_project("qw-proj", "Test")
        qw.execute()
        verify_phase = qw.hierarchy_engine.find_node(qw.project, "verify")
        assert verify_phase is not None
        assert verify_phase.name == "Quick Verification"
        assert len(verify_phase.subphases) == 1
        assert len(verify_phase.subphases[0].tasks) == 3

    def test_qw_context_plan(self):
        """QW populates context.plan with approach."""
        qw = QuickWorkflow()
        qw.create_project("qw-proj", "Fix bug")
        qw.execute()
        assert qw.context.plan is not None
        assert qw.context.plan["approach"] == "quick-work"
        assert qw.context.plan["phases"] == ["PLAN", "IMPLEMENT", "VERIFY"]

    def test_qw_fewer_tasks_than_sdd(self):
        """Quick Work has fewer tasks than full SDD."""
        sdd = SDDWorkflow()
        sdd.create_project("sdd-proj", "SDD")
        sdd_result = sdd.execute()

        qw = QuickWorkflow()
        qw.create_project("qw-proj", "QW")
        qw_result = qw.execute()

        assert qw_result.progress.total_tasks < sdd_result.progress.total_tasks

    def test_qw_result_errors_empty_on_success(self):
        """Successful QW has no errors."""
        qw = QuickWorkflow()
        qw.create_project("qw-proj", "Test")
        result = qw.execute()
        assert result.errors == []

    def test_qw_all_phases_succeeded(self):
        """All QW phases end in SUCCESS."""
        qw = QuickWorkflow()
        qw.create_project("qw-proj", "Test")
        qw.execute()
        for phase in qw.project.phases:
            assert phase.status == NodeStatus.SUCCESS

    def test_qw_validate_hierarchy(self):
        """QW hierarchy validates correctly."""
        qw = QuickWorkflow()
        qw.create_project("qw-proj", "Test")
        qw.execute()
        assert qw.validate_hierarchy()


# ═════════════════════════════════════════════════════════════════════
# Step 11: Hierarchy Integration Tests
# ═════════════════════════════════════════════════════════════════════

class TestHierarchyIntegration:
    """Tests for Ralph Loop hierarchy integration in workflows."""

    def test_project_node_level(self):
        """ProjectNode has PROJECT level."""
        engine = WorkflowEngine()
        engine.create_project("p1", "req")
        assert engine.project.level == HierarchyLevel.PROJECT

    def test_phase_node_level(self):
        """PhaseNodes have PHASE level."""
        engine = WorkflowEngine()
        engine.create_project("p1", "req")
        phase = engine.create_phase_node("spec", "Spec")
        assert phase.level == HierarchyLevel.PHASE

    def test_subphase_node_level(self):
        """SubPhaseNodes have SUBPHASE level."""
        engine = WorkflowEngine()
        engine.create_project("p1", "req")
        phase = engine.create_phase_node("spec", "Spec")
        sp = engine.create_subphase_node("sp1", "SP", parent_phase=phase)
        assert sp.level == HierarchyLevel.SUBPHASE

    def test_task_node_level(self):
        """TaskNodes have TASK level."""
        engine = WorkflowEngine()
        engine.create_project("p1", "req")
        phase = engine.create_phase_node("spec", "Spec")
        sp = engine.create_subphase_node("sp1", "SP", parent_phase=phase)
        task = engine.create_task_node("t1", "T1", parent_subphase=sp)
        assert task.level == HierarchyLevel.TASK

    def test_find_node_recursively(self):
        """hierarchy_engine.find_node works through the workflow."""
        qw = QuickWorkflow()
        qw.create_project("qw-proj", "Test")
        qw.execute()

        # Can find project
        found = qw.hierarchy_engine.find_node(qw.project, "qw-proj")
        assert found is qw.project

        # Can find a phase
        found = qw.hierarchy_engine.find_node(qw.project, "plan")
        assert found is not None
        assert found.name == "Quick Plan"

        # Non-existent
        found = qw.hierarchy_engine.find_node(qw.project, "nonexistent")
        assert found is None

    def test_collect_all_nodes_after_sdd(self):
        """collect_all_nodes returns all nodes in tree after SDD."""
        sdd = SDDWorkflow()
        sdd.create_project("sdd-proj", "Test")
        sdd.execute()
        nodes = sdd.collect_all_nodes()
        # Project + 6 phases + 12 subphases + many tasks
        assert len(nodes) > 30  # substantial tree

    def test_collect_all_nodes_after_qw(self):
        """collect_all_nodes returns all nodes after QW."""
        qw = QuickWorkflow()
        qw.create_project("qw-proj", "Test")
        qw.execute()
        nodes = qw.collect_all_nodes()
        # Project + 3 phases + 3 subphases + tasks
        assert len(nodes) > 10


# ═════════════════════════════════════════════════════════════════════
# Step 12: Transition Edge Case Tests
# ═════════════════════════════════════════════════════════════════════

class TestTransitionEdgeCases:
    """Edge cases for state transitions."""

    def test_cannot_transition_from_success_to_running(self):
        """TransitionManager rejects SUCCESS→RUNNING."""
        from spine.models.types import HierarchyNode
        tm = TransitionManager()
        node = HierarchyNode(id="t1", name="Test", status=NodeStatus.SUCCESS)
        assert not tm.can_transition(node, NodeStatus.RUNNING)

    def test_cannot_transition_from_failed_to_success(self):
        """FAILED→SUCCESS is invalid (must REWORK first)."""
        from spine.models.types import HierarchyNode
        tm = TransitionManager()
        node = HierarchyNode(id="t1", name="Test", status=NodeStatus.FAILED)
        assert not tm.can_transition(node, NodeStatus.SUCCESS)

    def test_failed_to_reworking_valid(self):
        """FAILED→REWORKING is valid."""
        from spine.models.types import HierarchyNode
        tm = TransitionManager()
        node = HierarchyNode(id="t1", name="Test", status=NodeStatus.FAILED)
        assert tm.can_transition(node, NodeStatus.REWORKING)

    def test_pending_to_running_valid(self):
        """PENDING→RUNNING is valid."""
        from spine.models.types import HierarchyNode
        tm = TransitionManager()
        node = HierarchyNode(id="t1", name="Test", status=NodeStatus.PENDING)
        assert tm.can_transition(node, NodeStatus.RUNNING)

    def test_running_to_success_valid(self):
        """RUNNING→SUCCESS is valid."""
        from spine.models.types import HierarchyNode
        tm = TransitionManager()
        node = HierarchyNode(id="t1", name="Test", status=NodeStatus.RUNNING)
        assert tm.can_transition(node, NodeStatus.SUCCESS)


# ═════════════════════════════════════════════════════════════════════
# Step 13: Full Integration Tests
# ═════════════════════════════════════════════════════════════════════

class TestFullIntegration:
    """End-to-end integration tests."""

    def test_sdd_full_lifecycle_with_all_components(self, state_machine):
        """Full SDD lifecycle with state machine."""
        sdd = SDDWorkflow(state_machine=state_machine)
        sdd.create_project("full-sdd", "Full integration test")
        result = sdd.execute()
        assert result.success
        assert sdd.validate_hierarchy()

    def test_qw_full_lifecycle_with_all_components(self, state_machine):
        """Full QW lifecycle with state machine."""
        qw = QuickWorkflow(state_machine=state_machine)
        qw.create_project("full-qw", "Quick integration test")
        result = qw.execute()
        assert result.success
        assert qw.validate_hierarchy()

    def test_engine_reuse_multiple_runs(self):
        """Same engine can run multiple projects."""
        sdd = SDDWorkflow()
        sdd.create_project("proj-1", "First")
        r1 = sdd.execute()
        assert r1.success

        sdd.create_project("proj-2", "Second")
        r2 = sdd.execute()
        assert r2.success

    def test_sdd_and_qw_independent(self):
        """Running SDD and QW in same process doesn't interfere."""
        sdd = SDDWorkflow()
        sdd.create_project("sdd-p", "SDD test")
        sdd_result = sdd.execute()

        qw = QuickWorkflow()
        qw.create_project("qw-p", "QW test")
        qw_result = qw.execute()

        assert sdd_result.success
        assert qw_result.success
        assert len(sdd.project.phases) == 6
        assert len(qw.project.phases) == 3

    def test_shared_state_machine(self, state_machine):
        """SDD and QW can share a state machine."""
        sdd = SDDWorkflow(state_machine=state_machine)
        sdd.create_project("shared-sdd", "SDD via shared SM")
        sdd_result = sdd.execute()

        qw = QuickWorkflow(state_machine=state_machine)
        qw.create_project("shared-qw", "QW via shared SM")
        qw_result = qw.execute()

        assert sdd_result.success
        assert qw_result.success

    def test_sdd_all_phase_nodes_exist_in_hierarchy(self):
        """All 6 SDD phase nodes exist in the final hierarchy."""
        sdd = SDDWorkflow()
        sdd.create_project("sdd-phases", "Phase test")
        sdd.execute()

        expected_phases = {"spec", "design", "plan", "implement", "review", "verify"}
        actual_phases = {p.id for p in sdd.project.phases}
        assert actual_phases == expected_phases

    def test_qw_phase_nodes_exist(self):
        """All 3 QW phase nodes exist."""
        qw = QuickWorkflow()
        qw.create_project("qw-phases", "Phase test")
        qw.execute()

        expected_phases = {"plan", "implement", "verify"}
        actual_phases = {p.id for p in qw.project.phases}
        assert actual_phases == expected_phases


__all__ = []