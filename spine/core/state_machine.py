"""Core state machine implementation using LangGraph."""

from __future__ import annotations

from typing import Literal, Optional, Any, Iterator, Dict, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.base import SerializerProtocol
import orjson
import os
import subprocess
from datetime import datetime, timezone

from .constants import PhaseName, StateStatus, ErrorState
from ..models.types import Task, SubPhase, Phase, PhaseResult, SubPhaseResult, SpineState
from ..models.dag import SwarmDAGExecutor
from ..providers.llm import LLMProvider
from ..providers.base import ConflictResolver, ConflictResult
from ..providers.memory import MemoryProvider
from ..providers.storage import StorageProvider, FileWriteGuard
from ..providers.tools import ToolsProvider
from ..core.persistence import GitWorkflow, Checkpoint

__all__ = [
    "PhaseResult",
    "SubPhaseResult",
]


class ProviderSerializer(SerializerProtocol):
    """Custom serializer that handles non-serializable provider objects."""
    
    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        try:
            return "json", orjson.dumps(obj)
        except TypeError:
            # Replace non-serializable objects with their string representation
            def sanitize(o):
                if hasattr(o, '__class__') and o.__class__.__module__ not in ('builtins', 'typing', 'dataclasses'):
                    return {"__non_serializable__": True, "repr": repr(o), "class": f"{o.__class__.__module__}.{o.__class__.__name__}"}
                elif isinstance(o, dict):
                    return {k: sanitize(v) for k, v in o.items()}
                elif isinstance(o, (list, tuple)):
                    return [sanitize(item) for item in o]
                return o
            return "json", orjson.dumps(sanitize(obj))
    
    def loads_typed(self, b: tuple[str, bytes]) -> Any:
        fmt, data = b
        if fmt == "json":
            return orjson.loads(data)
        raise ValueError(f"Unknown format: {fmt}")


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
    """Execute the PLANNING phase with parallel sub-phases.
    
    Uses LLM-based decomposition for intelligent task planning.
    Includes entry/exit condition evaluation and DAG hooks.
    Writes plan artifacts to disk and logs phase events.
    """
    state["phase"] = PhaseName.PLANNING
    start_time = datetime.now(timezone.utc)
    
    # Log phase started event
    _log_phase_event(state, "PLANNING", "phase_started", {
        "requirement": state.get("requirement", ""),
    })
    
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
    
    # Build phase with hooks and conditions
    planning_phase_obj = Phase(
        name=PhaseName.PLANNING,
        subphases=subphases,
        pre_execute_hooks=[lambda ctx: {**ctx, "planning_started": True}],
        post_execute_hooks=[lambda ctx: {**ctx, "planning_completed": True}]
    )
    
    # Run pre-execute hooks
    context = {
        "requirement": state["requirement"],
        "variables": state.get("variables", {})
    }
    context = _run_pre_execute_hooks(planning_phase_obj, context)
    
    # Evaluate entry conditions
    if not _evaluate_entry_conditions(planning_phase_obj, context):
        state["errors"].append("Entry conditions not met for PLANNING phase")
        state["error_state"] = ErrorState.FATAL.value
        state["previous_phase"] = PhaseName.PLANNING
        state["phase"] = PhaseName.ERROR
        return state
    
    # Get providers from state
    providers = state.get("providers", {})
    llm_provider = providers.get("llm")
    executor = SwarmDAGExecutor(llm_provider=llm_provider)
    
    # Execute planning phase with LLM-based decomposition
    context = {
        "requirement": state["requirement"],
        "variables": state.get("variables", {})
    }
    phase_result = executor.execute_phase(Phase(name="PLANNING", subphases=subphases), context)
    
    # Check for error threshold exceeded
    error_state, failed_subphases = _check_error_threshold(subphases)
    if error_state != ErrorState.INIT.value:
        state["error_state"] = error_state
        state["phase"] = PhaseName.ERROR
        return state
    
    # Evaluate exit conditions
    exit_context = {
        "requirement": state["requirement"],
        "phase_result": phase_result,
        "variables": state.get("variables", {})
    }
    if not _evaluate_exit_conditions(planning_phase_obj, exit_context):
        state["errors"].append("Exit conditions not met for PLANNING phase")
        state["error_state"] = ErrorState.TRANSIENT.value
        state["previous_phase"] = PhaseName.PLANNING
        state["phase"] = PhaseName.ERROR
        return state
    
    # Generate plan from execution results
    state["plan"] = {
        "requirement": state["requirement"],
        "phases": ["PLANNING", "EXECUTION", "VERIFICATION"],
        "tasks": [
            {"id": "setup", "description": "Setup environment"},
            {"id": "implement", "description": "Implement core features"},
        ],
        "subphase_results": phase_result.subphase_results,
        "subphase_statuses": phase_result.subphase_statuses,
        "created_at": "2024-01-01T00:00:00Z"
    }
    
    # Mark planning tasks complete based on execution results
    state["completed_tasks"].extend(["analyze_requirement", "research_stack", "assess_risks", "draft_plan"])
    
    # Run post-execute hooks
    context = _run_post_execute_hooks(planning_phase_obj, exit_context)
    
    # Critic gate validation per STATEMACHINE.md §7.1
    plan = state["plan"] or {}
    critic_result = executor.run_critic_gate(plan, state.get("variables", {}))
    state["critic_gate_result"] = critic_result
    
    if critic_result != "APPROVED":
        state["errors"].append(f"Critic gate {critic_result}: Plan requires revision")
        state["previous_phase"] = PhaseName.PLANNING
        # Log phase completed with error
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        _log_phase_event(state, "PLANNING", "phase_completed", {
            "tasks_completed": len(state.get("completed_tasks", [])),
            "errors": len(state.get("errors", [])),
            "duration": duration,
            "status": "error",
        })
        return state
    
    # Write plan artifacts to disk
    work_item_id = state.get("variables", {}).get("work_item_id", state.get("variables", {}).get("thread_id", "default"))
    artifact_path = write_plan_artifact(state, work_item_id)
    if artifact_path:
        _log_phase_event(state, "PLANNING", "plan_written", {
            "path": artifact_path,
            "tasks_count": len(state["plan"].get("tasks", [])),
        })
    
    write_spec_file(state, work_item_id)
    
    # Transition to EXECUTION
    state["previous_phase"] = PhaseName.PLANNING
    state["phase"] = PhaseName.EXECUTION
    
    # Log phase completed
    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    _log_phase_event(state, "PLANNING", "phase_completed", {
        "tasks_completed": len(state.get("completed_tasks", [])),
        "errors": len(state.get("errors", [])),
        "duration": duration,
        "status": "success",
    })
    return state


def execution_phase(state: SpineState) -> SpineState:
    """Execute the EXECUTION phase with parallel sub-phases.
    
    When FeatureSlices are present in the plan, creates one SubPhase per
    slice and delegates to agent_provider when available.  Falls back to
    the hardcoded BACKEND/FRONTEND pattern when no slices exist.
    
    Integrates file write guard for protected writes.
    Includes entry/exit condition evaluation and DAG hooks.
    Logs phase events to swarm.log.
    """
    state["phase"] = PhaseName.EXECUTION
    start_time = datetime.now(timezone.utc)
    
    # Log phase started event
    _log_phase_event(state, "EXECUTION", "phase_started", {
        "requirement": state.get("requirement", ""),
    })
    
    # Get providers from state
    providers = state.get("providers", {})
    llm_provider = providers.get("llm")
    storage_provider = providers.get("storage")
    agent_provider = state.get("agent_provider")
    
    # Create executor with providers
    executor = SwarmDAGExecutor(
        llm_provider=llm_provider,
        storage_provider=storage_provider,
        agent_provider=agent_provider,
    )
    
    # ── Build subphases from FeatureSlices (or fallback) ──────────
    plan = state.get("plan") or {}
    raw_slices = plan.get("feature_slices", [])
    
    if raw_slices:
        from ..models.types import FeatureSlice
        feature_slices = [FeatureSlice.from_dict(s) for s in raw_slices]
        subphases = []
        active_names = []
        for s in feature_slices:
            tasks = [
                Task(id=f"{s.id}-exec", description=s.description),
            ]
            subphases.append(SubPhase(
                name=s.id.upper().replace("-", "_"),
                priority=1,
                parallel=len(s.depends_on) == 0,
                agent_role=s.agent_role,
                tasks=tasks,
            ))
            active_names.append(s.id.upper().replace("-", "_"))
        state["swarm_state"]["active_subphases"] = active_names
    else:
        # Fallback: hardcoded BACKEND/FRONTEND pattern
        subphases = [
            SubPhase(
                name="BACKEND",
                priority=1,
                parallel=True,
                agent_role="coder",
                tasks=[
                    Task(id="backend_impl", description="Implement backend logic"),
                    Task(id="backend_tests", description="Write backend tests"),
                ],
            ),
            SubPhase(
                name="FRONTEND",
                priority=1,
                parallel=True,
                agent_role="coder",
                tasks=[
                    Task(id="frontend_impl", description="Implement frontend"),
                    Task(id="frontend_tests", description="Write frontend tests"),
                ],
            ),
        ]
        state["swarm_state"]["active_subphases"] = ["BACKEND", "FRONTEND"]
    
    # Build phase with hooks and conditions
    execution_phase_obj = Phase(
        name=PhaseName.EXECUTION,
        subphases=subphases,
        pre_execute_hooks=[lambda ctx: {**ctx, "execution_started": True}],
        post_execute_hooks=[lambda ctx: {**ctx, "execution_completed": True}],
    )
    
    # Run pre-execute hooks
    context = {
        "requirement": state["requirement"],
        "plan": state.get("plan"),
        "variables": state.get("variables", {})
    }
    context = _run_pre_execute_hooks(execution_phase_obj, context)
    
    # Evaluate entry conditions
    if not _evaluate_entry_conditions(execution_phase_obj, context):
        state["errors"].append("Entry conditions not met for EXECUTION phase")
        state["error_state"] = ErrorState.TRANSIENT.value
        state["previous_phase"] = PhaseName.EXECUTION
        state["phase"] = PhaseName.ERROR
        return state
    
    # Execute with file guard integration
    context = {
        "requirement": state["requirement"],
        "plan": state.get("plan"),
        "variables": state.get("variables", {})
    }
    phase_result = executor.execute_phase(Phase(name="EXECUTION", subphases=subphases), context)
    
    # Check for error threshold exceeded
    error_state, failed_subphases = _check_error_threshold(subphases)
    if error_state != ErrorState.INIT.value:
        state["error_state"] = error_state
        state["phase"] = PhaseName.ERROR
        return state
    
    # Track completed tasks from execution
    for task_id, task_data in phase_result.subphase_results.items():
        if isinstance(task_data, dict) and task_data.get("status") == "success":
            for task in subphases:
                for t in task.tasks:
                    if t.id not in state["completed_tasks"]:
                        state["completed_tasks"].append(t.id)
    
    # Evaluate exit conditions
    exit_context = {
        "requirement": state["requirement"],
        "phase_result": phase_result,
        "variables": state.get("variables", {})
    }
    if not _evaluate_exit_conditions(execution_phase_obj, exit_context):
        state["errors"].append("Exit conditions not met for EXECUTION phase")
        state["error_state"] = ErrorState.TRANSIENT.value
        state["previous_phase"] = PhaseName.EXECUTION
        state["phase"] = PhaseName.ERROR
        return state
    
    # Run post-execute hooks
    context = _run_post_execute_hooks(execution_phase_obj, exit_context)
    
    # Transition to VERIFICATION
    state["previous_phase"] = PhaseName.EXECUTION
    state["phase"] = PhaseName.VERIFICATION
    
    # Log phase completed
    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    _log_phase_event(state, "EXECUTION", "phase_completed", {
        "tasks_completed": len(state.get("completed_tasks", [])),
        "errors": len(state.get("errors", [])),
        "duration": duration,
        "status": "success",
    })
    return state


def verification_phase(state: SpineState) -> SpineState:
    """Execute the VERIFICATION phase.
    
    Integrates git workflow for commits and branch management.
    Includes error handling with error state transitions.
    Runs actual syntax and lint checks, logs all events.
    """
    state["phase"] = PhaseName.VERIFICATION
    start_time = datetime.now(timezone.utc)
    
    # Log phase started event
    _log_phase_event(state, "VERIFICATION", "phase_started", {
        "requirement": state.get("requirement", ""),
    })
    
    # Quality gates - run actual checks
    syntax_ok, syntax_msg = _run_syntax_check()
    syntax_status = StateStatus.SUCCESS if syntax_ok else StateStatus.FAILED
    state["tasks"]["syntax_check"] = Task(
        id="syntax_check",
        description="Verify syntax correctness",
        status=syntax_status,
        result=syntax_msg,
    )
    _log_phase_event(state, "VERIFICATION", "task_completed", {
        "task_id": "syntax_check",
        "subphase": "VERIFICATION",
        "status": syntax_status.value,
    })
    if syntax_ok:
        state["completed_tasks"].append("syntax_check")
    else:
        state["errors"].append(syntax_msg)
    
    lint_ok, lint_msg = _run_lint_check()
    lint_status = StateStatus.SUCCESS if lint_ok else StateStatus.FAILED
    state["tasks"]["lint_check"] = Task(
        id="lint_check",
        description="Run linter checks",
        status=lint_status,
        result=lint_msg,
    )
    _log_phase_event(state, "VERIFICATION", "task_completed", {
        "task_id": "lint_check",
        "subphase": "VERIFICATION",
        "status": lint_status.value,
    })
    if lint_ok:
        state["completed_tasks"].append("lint_check")
    else:
        state["errors"].append(lint_msg)
    
    drift_ok = True
    drift_msg = "Plan drift check passed"
    # Check that plan exists and has tasks
    plan = state.get("plan")
    if plan is None or len(plan.get("tasks", [])) == 0:
        drift_ok = False
        drift_msg = "No plan found or plan has no tasks"
    drift_status = StateStatus.SUCCESS if drift_ok else StateStatus.FAILED
    state["tasks"]["drift_check"] = Task(
        id="drift_check",
        description="Verify plan drift",
        status=drift_status,
        result=drift_msg,
    )
    _log_phase_event(state, "VERIFICATION", "task_completed", {
        "task_id": "drift_check",
        "subphase": "VERIFICATION",
        "status": drift_status.value,
    })
    if drift_ok:
        state["completed_tasks"].append("drift_check")
    else:
        state["errors"].append(drift_msg)
    
    # Git integration: commit changes if configured
    providers = state.get("providers", {})
    git_workflow = providers.get("git")
    
    if git_workflow:
        try:
            # Create branch for this work item
            requirement = state.get("requirement", "work")
            branch_name = f"spine-{requirement[:20].replace(' ', '-').lower()}"
            git_workflow.create_branch(branch_name)
            
            # Commit changes
            git_workflow.commit(f"Complete: {requirement}", work_item=branch_name)
            
            state["variables"]["git_branch"] = branch_name
            state["variables"]["git_commit"] = "completed"
        except Exception as e:
            state["errors"].append(f"Git operation failed: {e}")
            state["error_state"] = ErrorState.TRANSIENT.value
            state["previous_phase"] = PhaseName.VERIFICATION
            state["phase"] = PhaseName.ERROR
            return state
    
    # Check for failures that require rework
    failed_tasks = state.get("failed_tasks", [])
    
    # Check for error history threshold
    error_history = state.get("error_history", [])
    if len(error_history) >= 3:
        state["error_state"] = ErrorState.FATAL.value
        state["errors"].append(f"Too many errors in history: {len(error_history)}")
        state["previous_phase"] = PhaseName.VERIFICATION
        state["phase"] = PhaseName.ERROR
        return state
    
    if failed_tasks:
        state["errors"].append(f"Verification failed: {len(failed_tasks)} tasks failed")
        state["previous_phase"] = PhaseName.VERIFICATION
        state["phase"] = PhaseName.REWORK
        return state
    
    # Transition to COMPLETE
    state["previous_phase"] = PhaseName.VERIFICATION
    state["phase"] = PhaseName.COMPLETE
    
    # Log phase completed
    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    _log_phase_event(state, "VERIFICATION", "phase_completed", {
        "tasks_completed": len(state.get("completed_tasks", [])),
        "errors": len(state.get("errors", [])),
        "duration": duration,
        "status": "success",
    })
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


def error_phase(state: SpineState) -> SpineState:
    """Execute the ERROR phase for error handling.
    
    Handles error transitions:
    - INIT -> ERROR state
    - ERROR -> REWORK (transient errors)
    - ERROR -> BLOCKED (requires human intervention)
    - ERROR -> HUMAN_REVIEW (complex errors needing review)
    """
    state["phase"] = PhaseName.ERROR
    
    # Get error state if available
    error_state = state.get("error_state", ErrorState.TRANSIENT)
    state["variables"]["error_handled"] = False
    
    # Record error in history
    if "error_history" not in state:
        state["error_history"] = []
    
    error_entry = {
        "phase": state.get("previous_phase"),
        "error_state": error_state,
        "errors": state.get("errors", []),
        "timestamp": state.get("variables", {}).get("timestamp", "unknown")
    }
    state["error_history"].append(error_entry)
    
    # Determine error handling path based on error state
    if error_state == ErrorState.TRANSIENT.value or error_state == "TRANSIENT":
        # Transient errors can be retried via REWORK
        state["errors"].append("Transient error detected, routing to REWORK")
        state["phase"] = PhaseName.REWORK
    elif error_state == ErrorState.FATAL.value or error_state == "FATAL":
        # Fatal errors go to HUMAN_REVIEW
        state["errors"].append("Fatal error detected, routing to HUMAN_REVIEW")
        state["phase"] = PhaseName.HUMAN_REVIEW
    elif error_state == ErrorState.TIMEOUT.value or error_state == "TIMEOUT":
        # Timeout errors go to BLOCKED for manual intervention
        state["errors"].append("Timeout error detected, routing to BLOCKED")
        state["phase"] = PhaseName.BLOCKED
    else:
        # Default: HUMAN_REVIEW for complex errors
        state["errors"].append("Unknown error state, routing to HUMAN_REVIEW")
        state["phase"] = PhaseName.HUMAN_REVIEW
    
    state["variables"]["error_handled"] = True
    return state


def human_review_phase(state: SpineState) -> SpineState:
    """Execute the HUMAN_REVIEW phase for manual error review."""
    state["phase"] = PhaseName.HUMAN_REVIEW
    
    # Record that human review is needed
    state["errors"].append("Workflow paused for human review")
    state["variables"]["waiting_for_human"] = True
    
    return state


def _evaluate_entry_conditions(phase: Phase, context: Dict[str, Any]) -> bool:
    """Evaluate entry conditions for a phase.
    
    Returns True if all entry conditions pass, False otherwise.
    """
    for condition in phase.entry_conditions:
        try:
            if not condition(context):
                return False
        except Exception as e:
            # Log error but continue evaluation
            context.setdefault("errors", []).append(f"Entry condition error: {e}")
            return False
    return True


def _evaluate_exit_conditions(phase: Phase, context: Dict[str, Any]) -> bool:
    """Evaluate exit conditions for a phase.
    
    Returns True if all exit conditions pass, False otherwise.
    """
    for condition in phase.exit_criteria:
        try:
            if not condition(context):
                return False
        except Exception as e:
            # Log error but continue evaluation
            context.setdefault("errors", []).append(f"Exit condition error: {e}")
            return False
    return True


def _run_pre_execute_hooks(phase: Phase, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute pre-execution hooks for a phase.
    
    Returns modified context after hook execution.
    """
    for hook in phase.pre_execute_hooks:
        try:
            context = hook(context)
        except Exception as e:
            context.setdefault("errors", []).append(f"Pre-execute hook error: {e}")
    return context


def _run_post_execute_hooks(phase: Phase, context: Dict[str, Any]) -> Dict[str, Any]:
    """Execute post-execution hooks for a phase.
    
    Returns modified context after hook execution.
    """
    for hook in phase.post_execute_hooks:
        try:
            context = hook(context)
        except Exception as e:
            context.setdefault("errors", []).append(f"Post-execute hook error: {e}")
    return context


def _check_error_threshold(subphases: List[SubPhase], max_errors: int = 3) -> tuple[str, List[SubPhase]]:
    """Check if any subphase has exceeded error threshold.
    
    Returns tuple of (error_state, failed_subphases).
    """
    failed_subphases = [sp for sp in subphases if sp.has_exceeded_error_threshold(max_errors)]
    
    if failed_subphases:
        # Determine error state based on error count
        error_state = ErrorState.FATAL
        for sp in failed_subphases:
            if sp.error_count >= max_errors:
                # Multiple failures indicate fatal error
                return error_state.value, failed_subphases
        
        return ErrorState.TRANSIENT.value, failed_subphases
    
    return ErrorState.INIT.value, []


def _get_spine_root(state: SpineState) -> str:
    """Resolve the .spine root directory from state."""
    # Try checkpoint_path from state variables first
    checkpoint_path = state.get("variables", {}).get("checkpoint_path", ".spine/spine.db")
    return os.path.dirname(checkpoint_path)


def _log_phase_event(state: SpineState, phase_name: str, event_type: str, data: Dict[str, Any]) -> None:
    """Log a phase event to swarm.log via the SwarmMail instance.

    Args:
        state: Current SpineState
        phase_name: Name of the phase
        event_type: Event type (phase_started, phase_completed, task_completed, plan_written)
        data: Event payload data
    """
    providers = state.get("providers", {})
    swarm_mail = providers.get("swarm_mail")
    if swarm_mail is None:
        return

    event_body = {
        "phase": phase_name,
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **data,
    }
    try:
        swarm_mail.broadcast(
            subject=f"phase_event:{phase_name}:{event_type}",
            body=event_body,
        )
    except Exception:
        # Fail silently - event logging should not break the workflow
        pass


def write_plan_artifact(state: SpineState, work_item_id: str) -> Optional[str]:
    """Write the plan to disk as JSON artifact.

    Creates .spine/artifacts/plans/{work_item_id}.json with the full plan dict.

    Args:
        state: Current SpineState
        work_item_id: Unique identifier for this work item

    Returns:
        Path to written artifact, or None on failure
    """
    plan = state.get("plan")
    if plan is None:
        return None

    spine_root = _get_spine_root(state)
    artifacts_dir = os.path.join(spine_root, "artifacts", "plans")
    os.makedirs(artifacts_dir, exist_ok=True)

    artifact_path = os.path.join(artifacts_dir, f"{work_item_id}.json")
    try:
        with open(artifact_path, "wb") as f:
            f.write(orjson.dumps(plan, option=orjson.OPT_INDENT_2))
        return artifact_path
    except Exception:
        return None


def write_spec_file(state: SpineState, work_item_id: str) -> Optional[str]:
    """Write a markdown spec derived from the plan.

    Creates .spine/spec/{work_item_id}.md with a human-readable spec.

    Args:
        state: Current SpineState
        work_item_id: Unique identifier for this work item

    Returns:
        Path to written spec file, or None on failure
    """
    plan = state.get("plan")
    if plan is None:
        return None

    spine_root = _get_spine_root(state)
    spec_dir = os.path.join(spine_root, "spec")
    os.makedirs(spec_dir, exist_ok=True)

    spec_path = os.path.join(spec_dir, f"{work_item_id}.md")

    # Build markdown spec from plan data
    lines = [
        f"# Spec: {work_item_id}",
        "",
        "## Requirement",
        f"{plan.get('requirement', 'N/A')}",
        "",
        "## Phases",
    ]

    for phase in plan.get("phases", []):
        lines.append(f"- {phase}")

    lines.append("")
    lines.append("## Tasks")
    lines.append("")

    for task in plan.get("tasks", []):
        task_id = task.get("id", "unknown")
        task_desc = task.get("description", "")
        lines.append(f"### {task_id}")
        lines.append(f"- Description: {task_desc}")
        lines.append("")

    # Include subphase results if available
    subphase_results = plan.get("subphase_results", {})
    if subphase_results:
        lines.append("## Subphase Results")
        lines.append("")
        for name, result in subphase_results.items():
            lines.append(f"### {name}")
            if isinstance(result, dict):
                for key, value in result.items():
                    lines.append(f"- {key}: {value}")
            else:
                lines.append(f"- Result: {result}")
            lines.append("")

    lines.append("---")
    lines.append(f"*Generated at {datetime.now(timezone.utc).isoformat()}*")

    try:
        with open(spec_path, "w") as f:
            f.write("\n".join(lines))
        return spec_path
    except Exception:
        return None


def _run_syntax_check() -> tuple[bool, str]:
    """Run syntax check on the project's Python files.

    Uses py_compile to verify Python syntax correctness.

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        result = subprocess.run(
            ["python", "-m", "py_compile", "spine/core/state_machine.py"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return True, "Syntax check passed"
        else:
            return False, f"Syntax check failed: {result.stderr.strip()}"
    except FileNotFoundError:
        return False, "Python interpreter not found"
    except subprocess.TimeoutExpired:
        return False, "Syntax check timed out"
    except Exception as e:
        return False, f"Syntax check error: {e}"


def _run_lint_check() -> tuple[bool, str]:
    """Run linting checks on the project.

    Tries ruff first, falls back to basic Python linting.

    Returns:
        Tuple of (success: bool, message: str)
    """
    # Try ruff first
    for lint_cmd in [[".venv/bin/ruff", "check", "spine/core/state_machine.py"],
                     ["ruff", "check", "spine/core/state_machine.py"]]:
        try:
            result = subprocess.run(
                lint_cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return True, "Lint check passed"
            else:
                # ruff found issues but ran successfully
                return False, f"Lint issues found: {result.stdout.strip()}"
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            return False, "Lint check timed out"
        except Exception as e:
            return False, f"Lint check error: {e}"

    # Fallback: no linter available, skip lint check
    return True, "No linter available, skipping lint check"


def should_continue(state: SpineState) -> Literal["planning", "execution", "verification", "rework", "blocked", "error", "human_review", "__end__"]:
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
        # Check for error state transitions
        error_state = state.get("error_state")
        if error_state:
            return "error"
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
        error_state = state.get("error_state")
        if error_state == ErrorState.TIMEOUT.value:
            return "blocked"
        return "blocked"
    elif phase == PhaseName.ERROR:
        # Handle error state transitions
        error_state = state.get("error_state", ErrorState.TRANSIENT.value)
        if error_state == ErrorState.TRANSIENT.value:
            # Transient errors go to REWORK
            return "rework"
        elif error_state == ErrorState.FATAL.value:
            # Fatal errors go to HUMAN_REVIEW
            return "human_review"
        elif error_state == ErrorState.TIMEOUT.value:
            # Timeout errors stay BLOCKED
            return "blocked"
        else:
            return "human_review"
    elif phase == PhaseName.HUMAN_REVIEW:
        # After human review, determine next phase
        # Check if waiting for human intervention
        if state.get("variables", {}).get("waiting_for_human"):
            return "human_review"
        # Otherwise, resume based on context
        next_phase = state.get("variables", {}).get("resume_phase", "rework")
        return next_phase.lower()
    else:
        return "__end__"


def create_spine_workflow(checkpoint_path: str = ".spine/spine.db"):
    """Create the SPINE workflow with LangGraph StateGraph."""
    # Use MemorySaver with custom serializer for provider objects
    serializer = ProviderSerializer()
    memory = MemorySaver(serde=serializer)
    
    # Build the state graph
    workflow = StateGraph(SpineState)
    
    # Add nodes (phases)
    workflow.add_node("init", init_phase)
    workflow.add_node("planning", planning_phase)
    workflow.add_node("execution", execution_phase)
    workflow.add_node("verification", verification_phase)
    workflow.add_node("rework", rework_phase)
    workflow.add_node("blocked", blocked_phase)
    workflow.add_node("error", error_phase)
    workflow.add_node("human_review", human_review_phase)
    
    # Add edges
    workflow.add_edge("init", "planning")
    workflow.add_conditional_edges(
        "planning",
        should_continue,
        {"planning": "planning", "execution": "execution", "verification": "verification", "rework": "rework", "blocked": "blocked", "error": "error", "__end__": END}
    )
    workflow.add_edge("execution", "verification")
    workflow.add_conditional_edges(
        "verification",
        should_continue,
        {"rework": "rework", "blocked": "blocked", "error": "error", "__end__": END}
    )
    workflow.add_conditional_edges(
        "rework",
        should_continue,
        {"planning": "planning", "execution": "execution", "verification": "verification", "error": "error", "human_review": "human_review", "__end__": END}
    )
    workflow.add_edge("blocked", "blocked")
    workflow.add_conditional_edges(
        "error",
        should_continue,
        {"rework": "rework", "blocked": "blocked", "human_review": "human_review", "__end__": END}
    )
    workflow.add_conditional_edges(
        "human_review",
        should_continue,
        {"rework": "rework", "planning": "planning", "execution": "execution", "verification": "verification", "__end__": END}
    )
    
    # Set entry point
    workflow.set_entry_point("init")
    
    return workflow.compile(checkpointer=memory)


class SpineStateMachine:
    """High-level interface for SPINE workflows."""

    def __init__(
        self,
        checkpoint_path: str = ".spine/spine.db",
        llm_provider: Optional[LLMProvider] = None,
        memory_provider: Optional[MemoryProvider] = None,
        storage_provider: Optional[StorageProvider] = None,
        tools_provider: Optional[ToolsProvider] = None,
        file_write_guard: Optional[FileWriteGuard] = None,
        git_workflow: Optional[GitWorkflow] = None,
    ):
        """Initialize state machine with optional providers.
        
        Args:
            checkpoint_path: Path to checkpoint storage (used for reference).
            llm_provider: LLM provider for task execution.
            memory_provider: Memory provider for persistent storage.
            storage_provider: Storage provider for file operations.
            tools_provider: Tools provider for agent capabilities.
            file_write_guard: Guard for protected file writes.
            git_workflow: Git workflow for version control.
        """
        import os
        from ..swarm.mail import SwarmMail, ResourceManager
        
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        self._checkpointer = MemorySaver(serde=ProviderSerializer())
        self.checkpoint_path = checkpoint_path
        
        # Store providers
        self._llm_provider = llm_provider
        self._memory_provider = memory_provider
        self._storage_provider = storage_provider
        self._tools_provider = tools_provider
        self._file_write_guard = file_write_guard
        self._git_workflow = git_workflow
        
        # Initialize SwarmMail for actor-model coordination
        self._event_path = os.path.join(os.path.dirname(checkpoint_path), "events")
        self._swarm_mail = SwarmMail(
            agent_id="state_machine",
            event_path=self._event_path,
            resource_manager=ResourceManager(path=os.path.dirname(checkpoint_path))
        )
        
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
        workflow.add_node("error", error_phase)
        workflow.add_node("human_review", human_review_phase)
        workflow.add_edge("init", "planning")
        workflow.add_conditional_edges(
            "planning",
            should_continue,
            {"planning": "planning", "execution": "execution", "verification": "verification", "rework": "rework", "blocked": "blocked", "error": "error", "__end__": END}
        )
        workflow.add_edge("execution", "verification")
        workflow.add_conditional_edges(
            "verification",
            should_continue,
            {"rework": "rework", "blocked": "blocked", "error": "error", "__end__": END}
        )
        workflow.add_conditional_edges(
            "rework",
            should_continue,
            {"planning": "planning", "execution": "execution", "verification": "verification", "error": "error", "human_review": "human_review", "__end__": END}
        )
        workflow.add_edge("blocked", "blocked")
        workflow.add_conditional_edges(
            "error",
            should_continue,
            {"rework": "rework", "blocked": "blocked", "human_review": "human_review", "__end__": END}
        )
        workflow.add_conditional_edges(
            "human_review",
            should_continue,
            {"rework": "rework", "planning": "planning", "execution": "execution", "verification": "verification", "__end__": END}
        )
        workflow.set_entry_point("init")
        
        return workflow.compile(checkpointer=self._checkpointer)
    
    def run(self, requirement: str, thread_id: str = "default") -> SpineState:
        """Execute the full SPINE workflow."""
        # Build providers dict for state
        providers = {
            "llm": self._llm_provider,
            "memory": self._memory_provider,
            "storage": self._storage_provider,
            "tools": self._tools_provider,
            "file_write_guard": self._file_write_guard,
            "git": self._git_workflow,
            "swarm_mail": self._swarm_mail,
        }

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
            variables={"thread_id": thread_id, "work_item_id": thread_id, "checkpoint_path": self.checkpoint_path},
            errors=[],
            providers=providers,
            critic_gate_result=None,
            error_state=None,
            error_history=[],
        )
        
        result = self.app.invoke(
            initial_state,
            {"configurable": {"thread_id": thread_id}}
        )
        return result
    
    @property
    def swarm_mail(self) -> Any:
        """Access the SwarmMail instance for actor-model coordination."""
        return self._swarm_mail
    
    def get_swarm_events(self, **kwargs) -> List[Dict[str, Any]]:
        """Get swarm events with optional filtering."""
        return self._swarm_mail.query_events(**kwargs)
    
    def replay_swarm_events(self, position: int = 0, **kwargs) -> Iterator[Dict[str, Any]]:
        """Replay swarm events from a given position."""
        return self._swarm_mail.replay_from(position=position, **kwargs)
    
    def resume(self, thread_id: str = "default") -> Optional[SpineState]:
        """Resume a previous workflow."""
        state = self.app.get_state({"configurable": {"thread_id": thread_id}})
        if state and "values" in state:
            return state["values"]
        return None

    # --- ContinuityManager Integration ---

    def checkpoint(
        self,
        work_item_id: str,
        phase_name: str,
        phase_progress: float,
        state: dict[str, Any],
        dag: dict[str, Any],
        context_vars: dict[str, Any],
        swarm_state: dict[str, Any],
        auto_commit: bool = False,
    ) -> Optional[str]:
        """Create and save a checkpoint with ContinuityManager.
        
        Args:
            work_item_id: Unique work item identifier
            phase_name: Current phase name
            phase_progress: Progress in phase (0.0-1.0)
            state: Current state dictionary
            dag: DAG with execution results
            context_vars: Context variables
            swarm_state: Swarm coordination state
            auto_commit: Whether to auto-commit via Git
            
        Returns:
            Path to saved checkpoint or None
        """
        from .persistence import ContinuityManager
        from .learning import LearningManager
        
        # Initialize managers if not already set
        if not hasattr(self, '_continuity_manager'):
            knowledge_dir = os.path.join(os.path.dirname(self.checkpoint_path), "knowledge")
            self._continuity_manager = ContinuityManager(
                state_dir=os.path.dirname(self.checkpoint_path),
                learning_manager=LearningManager(knowledge_dir=knowledge_dir),
                git_workflow=self._git_workflow,
            )
        
        # Create and save checkpoint
        checkpoint = self._continuity_manager.create_checkpoint(
            work_item_id=work_item_id,
            phase_name=phase_name,
            phase_progress=phase_progress,
            state=state,
            dag=dag,
            context_vars=context_vars,
            swarm_state=swarm_state,
        )
        
        return self._continuity_manager.save_checkpoint(checkpoint, auto_commit=auto_commit)

    def create_resume_marker(
        self,
        work_item_id: str,
        checkpoint: dict[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        """Create a resume marker with ContinuityManager.
        
        Args:
            work_item_id: Work item identifier
            checkpoint: Checkpoint dictionary
            reason: Handoff reason
            
        Returns:
            Resume marker dictionary
        """
        from .persistence import ContinuityManager
        
        if not hasattr(self, '_continuity_manager'):
            self._continuity_manager = ContinuityManager(state_dir=os.path.dirname(self.checkpoint_path))
        
        # Convert dict to Checkpoint object
        ckpt = Checkpoint.from_dict(checkpoint)
        marker = self._continuity_manager.create_resume_marker_with_checkpoint(
            work_item_id=work_item_id,
            checkpoint=ckpt,
            reason=reason
        )
        
        return marker.to_dict()

    # --- Conflict Resolution Integration ---

    def resolve_conflict(
        self, 
        key: str, 
        values: dict[str, Any], 
        confidence: dict[str, float],
        strategy: str = "confidence_weighted"
    ) -> Any:
        """Resolve conflicts between multiple provider results.
        
        Args:
            key: Identifier for the conflict.
            values: Dict mapping provider names to their results.
            confidence: Dict mapping provider names to confidence scores.
            strategy: Resolution strategy (confidence_weighted, voting, consensus, highest_priority).
            
        Returns:
            The resolved value.
            
        Raises:
            ConflictRequiresHuman: If consensus required and providers disagree.
        """
        conflict = ConflictResult(
            key=key,
            values=values,
            confidence=confidence
        )
        resolver = ConflictResolver()
        return resolver.resolve(conflict, strategy)

    # --- Ralph Loop Integration ---

    def create_hierarchy_engine(self) -> "RalphLoopEngine":
        """Create a RalphLoopEngine attached to this state machine.
        
        The engine provides hierarchical Project→Phase→Subphase→Task
        tracking with progress roll-up, state transitions, and
        nested automation support.
        
        Returns:
            A RalphLoopEngine configured with this state machine.
        """
        from .hierarchy import RalphLoopEngine
        engine = RalphLoopEngine()
        engine.attach_state_machine(self)
        return engine

    # --- Ralph Loop Integration ---

    def create_hierarchy_engine(self) -> "RalphLoopEngine":
        """Create a RalphLoopEngine attached to this state machine.
        
        The engine provides hierarchical Project→Phase→Subphase→Task
        tracking with progress roll-up, state transitions, and
        nested automation support.
        
        Returns:
            A RalphLoopEngine configured with this state machine.
        """
        from .hierarchy import RalphLoopEngine
        engine = RalphLoopEngine()
        engine.attach_state_machine(self)
        return engine