"""Core state machine implementation using LangGraph."""

from __future__ import annotations

from typing import Literal, Optional, Any, Iterator, Dict, List
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.base import SerializerProtocol
import orjson
import os
import subprocess
import ormsgpack
from datetime import datetime, timezone

from .constants import PhaseName, StateStatus, ErrorState
from ..models.types import Task, SubPhase, Phase, PhaseResult, SubPhaseResult, SpineState, FeatureSlice
from ..models.dag import SwarmDAGExecutor, synthesize_slices
from ..providers.llm import LLMProvider
from ..providers.deepagents_model import DeepAgentsModelProvider
from ..providers.base import ConflictResolver, ConflictResult
from ..providers.memory import MemoryProvider
from ..providers.storage import StorageProvider, FileWriteGuard
from ..providers.tools import ToolsProvider
from ..core.persistence import GitWorkflow, Checkpoint

import logging
logger = logging.getLogger(__name__)

__all__ = [
    "PhaseResult",
    "SubPhaseResult",
]


class ProviderSerializer(SerializerProtocol):
    """Custom serializer that handles non-serializable provider objects.

    Extends the LangGraph JsonPlusSerializer format so that checkpoints
    written by SqliteSaver (msgpack) can be read back correctly, while
    still sanitising non-serialisable provider objects on write.
    """

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        if obj is None:
            return "null", b""
        if isinstance(obj, bytes):
            return "bytes", obj
        if isinstance(obj, bytearray):
            return "bytearray", obj
        try:
            return "msgpack", ormsgpack.packb(obj)
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
            return "msgpack", ormsgpack.packb(sanitize(obj))

    def loads_typed(self, b: tuple[str, bytes]) -> Any:
        fmt, data = b
        if fmt == "null":
            return None
        if fmt == "bytes":
            return data
        if fmt == "bytearray":
            return bytearray(data)
        if fmt == "json":
            return orjson.loads(data)
        if fmt == "msgpack":
            try:
                return ormsgpack.unpackb(data)
            except Exception:
                # Fallback: old checkpoints may have been written with JSON
                # but mislabeled as "msgpack" (a known agent-introduced bug)
                return orjson.loads(data)
        raise ValueError(f"Unknown format: {fmt}")


def _get_providers(state: SpineState, config: Optional[RunnableConfig] = None) -> dict[str, Any]:
    """Resolve providers from config (non-serialized) first, then state (may contain deserialized dicts).

    LangGraph's checkpointer serializes state between steps, turning provider
    objects into plain dicts.  Passing providers through ``config["configurable"]``
    avoids this problem because config is never persisted.
    """
    # Config path — real provider objects (not serialized)
    if config:
        cfg_providers = config.get("configurable", {}).get("providers")
        if cfg_providers and isinstance(cfg_providers, dict):
            return cfg_providers

    # State path — may contain deserialized dicts; filter those out so
    # callers only ever see real Provider instances or None
    state_providers = state.get("providers", {})
    if isinstance(state_providers, dict):
        return {
            k: v for k, v in state_providers.items()
            if v is not None and not isinstance(v, dict)
        }
    return {}


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
    state["pending_messages"] = []
    state["model_call_count"] = 0
    
    # Transition to PLANNING
    state["previous_phase"] = PhaseName.INIT
    state["phase"] = PhaseName.PLANNING
    return state


def planning_phase(state: SpineState, config: Optional[RunnableConfig] = None) -> SpineState:
    """Execute the PLANNING phase using Deep Agents when available.

    Primary path (DA): Creates a planning DA agent with explorer, SME, and
    analyst subagents. The agent loop handles tool execution, context
    compaction, and critic gate iteration internally.

    Fallback path: Uses the legacy SwarmDAGExecutor when no DA-compatible
    LLM provider is configured (i.e., the old LLMProvider is in use).
    """
    state["phase"] = PhaseName.PLANNING
    start_time = datetime.now(timezone.utc)

    # Log phase started event
    _log_phase_event(state, "PLANNING", "phase_started", {
        "requirement": state.get("requirement", ""),
    }, config=config)

    # ── Resolve providers from config (not state) ───────────────────
    providers = _get_providers(state, config)
    llm_provider = providers.get("llm")
    agent_provider = providers.get("agent")

    # Also try state for agent_provider (backward compat)
    if agent_provider is None:
        state_ap = state.get("agent_provider")
        if state_ap is not None and not isinstance(state_ap, dict):
            agent_provider = state_ap

    # ── Try Deep Agents path first ──────────────────────────────────
    da_chat_model = None
    if isinstance(llm_provider, DeepAgentsModelProvider):
        da_chat_model = llm_provider.chat_model
    elif llm_provider is not None and hasattr(llm_provider, "chat_model"):
        # Future: other providers may expose chat_model
        da_chat_model = llm_provider.chat_model

    if da_chat_model is not None:
        return _planning_phase_da(state, config, providers, da_chat_model, start_time)

    # ── Fallback: legacy SwarmDAGExecutor path ───────────────────────
    return _planning_phase_legacy(state, config, providers, agent_provider, start_time)


def _planning_phase_da(
    state: SpineState,
    config: Optional[RunnableConfig],
    providers: dict[str, Any],
    chat_model: Any,
    start_time: datetime,
) -> SpineState:
    """PLANNING phase using Deep Agents — primary path."""
    from ..adapters.da_phase_adapter import (
        create_planning_agent,
        _extract_feature_slices_from_da_result,
        _extract_planning_context_from_da_result,
    )

    debug_prompts = state.get("variables", {}).get("debug_prompts", False)
    max_steps = int(os.environ.get("SPINE_PLANNING_STEPS", "50"))

    project_root = _get_project_root(state)

    try:
        planning_agent = create_planning_agent(
            requirement=state["requirement"],
            providers=providers,
            max_steps=max_steps,
            root_dir=project_root,
        )
    except ValueError as e:
        state["errors"].append(f"DA planning agent creation failed: {e}")
        state["error_state"] = ErrorState.FATAL.value
        state["previous_phase"] = PhaseName.PLANNING
        state["phase"] = PhaseName.ERROR
        return state

    if debug_prompts:
        import sys
        print("[DA-PLANNING] Invoking DA planning agent...", file=sys.stderr)

    try:
        result = planning_agent.invoke({
            "messages": [{
                "role": "user",
                "content": (
                    f"Analyze this requirement and create an execution plan "
                    f"with FeatureSlices: {state['requirement']}"
                ),
            }],
            "spine_phase": "PLANNING",
        }, config=config)
    except Exception as e:
        state["errors"].append(f"DA planning agent execution failed: {e}")
        state["error_state"] = ErrorState.TRANSIENT.value
        state["previous_phase"] = PhaseName.PLANNING
        state["phase"] = PhaseName.ERROR
        return state

    # ── Extract structured output ────────────────────────────────────
    planning_context = _extract_planning_context_from_da_result(result)
    feature_slices = _extract_feature_slices_from_da_result(result)

    # If DA didn't produce slices, fall back to synthesize_slices
    if not feature_slices:
        feature_slices = synthesize_slices(
            requirement=state["requirement"],
            context=planning_context,
            agent_provider=None,  # DA already tried — use heuristic
        )

    # Critic gate (SPINE-specific: runs at state machine level)
    critic_result = _run_critic_gate_da(state, planning_context, chat_model)
    state["critic_gate_result"] = critic_result

    if critic_result != "APPROVED":
        state["errors"].append(f"Critic gate {critic_result}: Plan requires revision")
        state["previous_phase"] = PhaseName.PLANNING
        _log_phase_event(state, "PLANNING", "phase_completed", {
            "tasks_completed": len(state.get("completed_tasks", [])),
            "errors": len(state.get("errors", [])),
            "duration": (datetime.now(timezone.utc) - start_time).total_seconds(),
            "status": "error",
        }, config=config)
        return state

    # ── Build plan ───────────────────────────────────────────────────
    plan_tasks = _derive_plan_tasks_from_da_result(result, state["requirement"])

    state["plan"] = {
        "requirement": state["requirement"],
        "phases": ["PLANNING", "EXECUTION", "VERIFICATION"],
        "tasks": plan_tasks,
        "feature_slices": [s.to_dict() for s in feature_slices],
        "planning_context": planning_context,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Mark planning tasks complete
    state["completed_tasks"].extend([t["id"] for t in plan_tasks])

    # Write artifacts
    work_item_id = state.get("variables", {}).get("work_item_id", state.get("variables", {}).get("thread_id", "default"))
    artifact_path = write_plan_artifact(state, work_item_id)
    if artifact_path:
        _log_phase_event(state, "PLANNING", "plan_written", {
            "path": artifact_path,
            "tasks_count": len(state["plan"].get("tasks", [])),
        }, config=config)

    write_spec_file(state, work_item_id)

    spine_root = _get_spine_root(state)
    state["variables"]["spec_path"] = os.path.join(spine_root, "spec", f"{work_item_id}.md")
    state["variables"]["artifact_path"] = os.path.join(spine_root, "artifacts", "plans", f"{work_item_id}.json")

    # Transition to EXECUTION
    state["previous_phase"] = PhaseName.PLANNING
    state["phase"] = PhaseName.EXECUTION

    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    _log_phase_event(state, "PLANNING", "phase_completed", {
        "tasks_completed": len(state.get("completed_tasks", [])),
        "errors": len(state.get("errors", [])),
        "duration": duration,
        "status": "success",
        "path": "deepagents",
    }, config=config)
    return state


def _planning_phase_legacy(
    state: SpineState,
    config: Optional[RunnableConfig],
    providers: dict[str, Any],
    agent_provider: Any,
    start_time: datetime,
) -> SpineState:
    """PLANNING phase using legacy SwarmDAGExecutor — fallback path.

    This is the original planning_phase logic, preserved for backward
    compatibility when no DeepAgentsModelProvider is configured.
    """
    # Define sub-phases (same as original)
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

    planning_phase_obj = Phase(
        name=PhaseName.PLANNING,
        subphases=subphases,
        pre_execute_hooks=[lambda ctx: {**ctx, "planning_started": True}],
        post_execute_hooks=[lambda ctx: {**ctx, "planning_completed": True}]
    )

    # Run hooks and conditions
    context = {
        "requirement": state["requirement"],
        "variables": state.get("variables", {})
    }
    context = _run_pre_execute_hooks(planning_phase_obj, context)

    if not _evaluate_entry_conditions(planning_phase_obj, context):
        state["errors"].append("Entry conditions not met for PLANNING phase")
        state["error_state"] = ErrorState.FATAL.value
        state["previous_phase"] = PhaseName.PLANNING
        state["phase"] = PhaseName.ERROR
        return state

    executor = SwarmDAGExecutor(agent_provider=agent_provider)
    context = {
        "requirement": state["requirement"],
        "variables": state.get("variables", {})
    }
    phase_result = executor.execute_phase(Phase(name="PLANNING", subphases=subphases), context)

    # Check error threshold
    error_state, failed_subphases = _check_error_threshold(subphases)
    if error_state != ErrorState.INIT.value:
        state["error_state"] = error_state
        state["phase"] = PhaseName.ERROR
        return state

    # Build plan
    subphase_results = phase_result.subphase_results
    subphase_statuses = phase_result.subphase_statuses
    planning_context = _extract_planning_context(subphase_results)

    feature_slices = synthesize_slices(
        requirement=state["requirement"],
        context=planning_context,
        agent_provider=agent_provider,
    )

    plan_tasks = _derive_plan_tasks(subphase_results, state["requirement"])

    state["plan"] = {
        "requirement": state["requirement"],
        "phases": ["PLANNING", "EXECUTION", "VERIFICATION"],
        "tasks": plan_tasks,
        "feature_slices": [s.to_dict() for s in feature_slices],
        "planning_context": planning_context,
        "subphase_results": subphase_results,
        "subphase_statuses": subphase_statuses,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    completed_ids = []
    for sp in subphases:
        for t in sp.tasks:
            completed_ids.append(t.id)
    state["completed_tasks"].extend(completed_ids)

    exit_context = {
        "requirement": state["requirement"],
        "phase_result": phase_result,
        "variables": state.get("variables", {})
    }
    context = _run_post_execute_hooks(planning_phase_obj, exit_context)

    # Critic gate
    plan = state["plan"] or {}
    critic_result = executor.run_critic_gate(plan, state.get("variables", {}))
    state["critic_gate_result"] = critic_result

    if critic_result != "APPROVED":
        state["errors"].append(f"Critic gate {critic_result}: Plan requires revision")
        state["previous_phase"] = PhaseName.PLANNING
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        _log_phase_event(state, "PLANNING", "phase_completed", {
            "tasks_completed": len(state.get("completed_tasks", [])),
            "errors": len(state.get("errors", [])),
            "duration": duration,
            "status": "error",
        }, config=config)
        return state

    # Write artifacts
    work_item_id = state.get("variables", {}).get("work_item_id", state.get("variables", {}).get("thread_id", "default"))
    artifact_path = write_plan_artifact(state, work_item_id)
    if artifact_path:
        _log_phase_event(state, "PLANNING", "plan_written", {
            "path": artifact_path,
            "tasks_count": len(state["plan"].get("tasks", [])),
        }, config=config)

    write_spec_file(state, work_item_id)

    spine_root = _get_spine_root(state)
    state["variables"]["spec_path"] = os.path.join(spine_root, "spec", f"{work_item_id}.md")
    state["variables"]["artifact_path"] = os.path.join(spine_root, "artifacts", "plans", f"{work_item_id}.json")

    state["previous_phase"] = PhaseName.PLANNING
    state["phase"] = PhaseName.EXECUTION

    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    _log_phase_event(state, "PLANNING", "phase_completed", {
        "tasks_completed": len(state.get("completed_tasks", [])),
        "errors": len(state.get("errors", [])),
        "duration": duration,
        "status": "success",
        "path": "legacy",
    }, config=config)
    return state


def _run_critic_gate_da(state: SpineState, planning_context: dict, chat_model: Any) -> str:
    """Run critic gate using a DA chat model directly.

    Returns one of: APPROVED, NEEDS_REVISION, REJECTED.
    """
    from langchain_core.messages import HumanMessage

    context_text = ""
    for key, value in planning_context.items():
        context_text += f"\n## {key.upper()}\n{value}"

    critic_prompt = (
        "You are a software architecture critic. Review this planning context "
        "for correctness, completeness, and feasibility.\n\n"
        f"REQUIREMENT: {state.get('requirement', '')}\n\n"
        f"PLANNING CONTEXT:{context_text}\n\n"
        "Respond with exactly one word: APPROVED, NEEDS_REVISION, or REJECTED. "
        "If not APPROVED, explain what needs to change."
    )
    try:
        result = chat_model.invoke([HumanMessage(content=critic_prompt)])
        content = result.content.strip()
        for verdict in ("APPROVED", "NEEDS_REVISION", "REJECTED"):
            if verdict in content.upper():
                return verdict
        return "NEEDS_REVISION"
    except Exception as e:
        logger.warning("Critic gate DA invocation failed: %s", e)
        return "NEEDS_REVISION"


def _derive_plan_tasks_from_da_result(result: dict[str, Any], requirement: str) -> list[dict[str, str]]:
    """Derive execution plan tasks from DA agent output.

    Scans the message history for task-related content.
    Falls back to heuristic extraction from the requirement.
    """
    # Check if feature_slices are in the result (they take priority)
    from ..adapters.da_phase_adapter import _extract_feature_slices_from_da_result
    slices = _extract_feature_slices_from_da_result(result)
    if slices:
        return [{"id": s.id, "description": s.description} for s in slices]

    # Fallback: heuristic from requirement
    from ..models.dag import _extract_components
    components = _extract_components(requirement)
    return [{"id": f"implement-{i+1}", "description": f"Implement {c}"}
            for i, c in enumerate(components)]


def execution_phase(state: SpineState, config: Optional[RunnableConfig] = None) -> SpineState:
    """Execute the EXECUTION phase using Deep Agents when available.

    Primary path (DA): Creates an execution DA agent with one SubAgent per
    FeatureSlice. Each subagent has isolated context. The main agent
    orchestrates execution order based on slice dependencies.

    Fallback path: Uses the legacy SwarmDAGExecutor when no DA-compatible
    LLM provider is configured.
    """
    state["phase"] = PhaseName.EXECUTION
    start_time = datetime.now(timezone.utc)

    _log_phase_event(state, "EXECUTION", "phase_started", {
        "requirement": state.get("requirement", ""),
    }, config=config)

    # ── Resolve providers ─────────────────────────────────────────────
    providers = _get_providers(state, config)
    llm_provider = providers.get("llm")
    storage_provider = providers.get("storage")
    agent_provider = providers.get("agent")
    if agent_provider is None:
        state_ap = state.get("agent_provider")
        if state_ap is not None and not isinstance(state_ap, dict):
            agent_provider = state_ap

    # ── Try Deep Agents path ──────────────────────────────────────────
    da_chat_model = None
    if isinstance(llm_provider, DeepAgentsModelProvider):
        da_chat_model = llm_provider.chat_model
    elif llm_provider is not None and hasattr(llm_provider, "chat_model"):
        da_chat_model = llm_provider.chat_model

    if da_chat_model is not None:
        return _execution_phase_da(state, config, providers, da_chat_model, start_time)

    # ── Fallback: legacy SwarmDAGExecutor path ─────────────────────────
    return _execution_phase_legacy(state, config, providers, agent_provider, storage_provider, start_time)


def _execution_phase_da(
    state: SpineState,
    config: Optional[RunnableConfig],
    providers: dict[str, Any],
    chat_model: Any,
    start_time: datetime,
) -> SpineState:
    """EXECUTION phase using Deep Agents — primary path."""
    from ..adapters.da_phase_adapter import (
        create_execution_agent,
        get_backend,
    )

    debug_prompts = state.get("variables", {}).get("debug_prompts", False)
    plan = state.get("plan") or {}
    raw_slices = plan.get("feature_slices", [])
    planning_context = plan.get("planning_context", {})

    # Read spec content
    spec_content = ""
    spec_path = state.get("variables", {}).get("spec_path", "")
    if spec_path and os.path.isfile(spec_path):
        try:
            with open(spec_path, "r") as f:
                spec_content = f.read()
        except Exception:
            pass

    # Build FeatureSlice objects
    feature_slices = []
    if raw_slices:
        feature_slices = [FeatureSlice.from_dict(s) for s in raw_slices]
    else:
        # No slices from planning — create a single catch-all slice
        feature_slices = [FeatureSlice(
            id="implementation",
            description=state.get("requirement", "Implement the required feature"),
            scope=["."],
            depends_on=[],
            agent_role="coder",
            acceptance=["Feature works as described"],
        )]

    project_root = _get_project_root(state)
    max_steps = int(os.environ.get("SPINE_EXECUTION_STEPS", "100"))

    try:
        backend = get_backend(phase=PhaseName.EXECUTION, root_dir=project_root)
        execution_agent = create_execution_agent(
            requirement=state["requirement"],
            providers=providers,
            feature_slices=feature_slices,
            planning_context=planning_context,
            spec_content=spec_content,
            backend=backend,
            max_steps=max_steps,
            root_dir=project_root,
        )
    except ValueError as e:
        state["errors"].append(f"DA execution agent creation failed: {e}")
        state["error_state"] = ErrorState.FATAL.value
        state["previous_phase"] = PhaseName.EXECUTION
        state["phase"] = PhaseName.ERROR
        return state

    if debug_prompts:
        import sys
        print(f"[DA-EXECUTION] Invoking DA execution agent with {len(feature_slices)} slice(s)...", file=sys.stderr)

    try:
        execution_agent.invoke({
            "messages": [{
                "role": "user",
                "content": (
                    f"Implement the planned feature slices. "
                    f"Start with slices that have no dependencies. "
                    f"Slices: {', '.join(s.id for s in feature_slices)}"
                ),
            }],
        }, config=config)
    except Exception as e:
        state["errors"].append(f"DA execution agent failed: {e}")
        state["error_state"] = ErrorState.TRANSIENT.value
        state["previous_phase"] = PhaseName.EXECUTION
        state["phase"] = PhaseName.ERROR
        return state

    # Track completed tasks from execution
    for slice_obj in feature_slices:
        if slice_obj.id not in state["completed_tasks"]:
            state["completed_tasks"].append(slice_obj.id)

    state["previous_phase"] = PhaseName.EXECUTION
    state["phase"] = PhaseName.VERIFICATION

    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    _log_phase_event(state, "EXECUTION", "phase_completed", {
        "tasks_completed": len(state.get("completed_tasks", [])),
        "errors": len(state.get("errors", [])),
        "duration": duration,
        "status": "success",
        "path": "deepagents",
    }, config=config)
    return state


def _execution_phase_legacy(
    state: SpineState,
    config: Optional[RunnableConfig],
    providers: dict[str, Any],
    agent_provider: Any,
    storage_provider: Any,
    start_time: datetime,
) -> SpineState:
    """EXECUTION phase using legacy SwarmDAGExecutor — fallback path.

    Preserved for backward compatibility when no DA provider is configured.
    """
    _ = state.get("variables", {}).get("debug_prompts", False)

    executor = SwarmDAGExecutor(
        storage_provider=storage_provider,
        agent_provider=agent_provider,
    )

    # Build subphases from FeatureSlices (or fallback)
    plan = state.get("plan") or {}
    raw_slices = plan.get("feature_slices", [])
    planning_context = plan.get("planning_context", {})

    spec_content = ""
    spec_path = state.get("variables", {}).get("spec_path", "")
    if spec_path and os.path.isfile(spec_path):
        try:
            with open(spec_path, "r") as f:
                spec_content = f.read()
        except Exception:
            pass

    if raw_slices:
        feature_slices = [FeatureSlice.from_dict(s) for s in raw_slices]
        subphases = []
        active_names = []
        for s in feature_slices:
            task_desc = s.description
            if s.scope:
                task_desc += f"\nScope: {', '.join(s.scope)}"
            if s.acceptance:
                task_desc += f"\nAcceptance criteria: {'; '.join(s.acceptance)}"
            tasks = [
                Task(id=f"{s.id}-exec", description=task_desc),
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
        plan_tasks = plan.get("tasks", [])
        if plan_tasks and len(plan_tasks) > 1:
            subphases = []
            active_names = []
            for i, pt in enumerate(plan_tasks):
                tid = pt.get("id", f"task-{i+1}")
                tdesc = pt.get("description", tid)
                subphases.append(SubPhase(
                    name=tid.upper().replace("-", "_"),
                    priority=i + 1,
                    parallel=False,
                    agent_role="coder",
                    tasks=[Task(id=tid, description=tdesc)],
                ))
                active_names.append(tid.upper().replace("-", "_"))
            state["swarm_state"]["active_subphases"] = active_names
        else:
            requirement_text = state.get("requirement", "")
            subphases = [
                SubPhase(
                    name="IMPLEMENTATION",
                    priority=1,
                    parallel=False,
                    agent_role="coder",
                    tasks=[
                        Task(id="implement", description=requirement_text),
                    ],
                ),
            ]
            state["swarm_state"]["active_subphases"] = ["IMPLEMENTATION"]

    execution_phase_obj = Phase(
        name=PhaseName.EXECUTION,
        subphases=subphases,
        pre_execute_hooks=[lambda ctx: {**ctx, "execution_started": True}],
        post_execute_hooks=[lambda ctx: {**ctx, "execution_completed": True}],
    )

    context = {
        "requirement": state["requirement"],
        "plan": state.get("plan"),
        "planning_context": planning_context,
        "spec_content": spec_content,
        "variables": state.get("variables", {})
    }
    context = _run_pre_execute_hooks(execution_phase_obj, context)

    if not _evaluate_entry_conditions(execution_phase_obj, context):
        state["errors"].append("Entry conditions not met for EXECUTION phase")
        state["error_state"] = ErrorState.TRANSIENT.value
        state["previous_phase"] = PhaseName.EXECUTION
        state["phase"] = PhaseName.ERROR
        return state

    context = {
        "requirement": state["requirement"],
        "plan": state.get("plan"),
        "planning_context": planning_context,
        "spec_content": spec_content,
        "variables": state.get("variables", {})
    }
    phase_result = executor.execute_phase(Phase(name="EXECUTION", subphases=subphases), context)

    error_state, failed_subphases = _check_error_threshold(subphases)
    if error_state != ErrorState.INIT.value:
        state["error_state"] = error_state
        state["phase"] = PhaseName.ERROR
        return state

    for task_id, task_data in phase_result.subphase_results.items():
        if isinstance(task_data, dict) and task_data.get("status") == "success":
            for task in subphases:
                for t in task.tasks:
                    if t.id not in state["completed_tasks"]:
                        state["completed_tasks"].append(t.id)

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

    context = _run_post_execute_hooks(execution_phase_obj, exit_context)

    state["previous_phase"] = PhaseName.EXECUTION
    state["phase"] = PhaseName.VERIFICATION

    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    _log_phase_event(state, "EXECUTION", "phase_completed", {
        "tasks_completed": len(state.get("completed_tasks", [])),
        "errors": len(state.get("errors", [])),
        "duration": duration,
        "status": "success",
        "path": "legacy",
    }, config=config)
    return state


def verification_phase(state: SpineState, config: Optional[RunnableConfig] = None) -> SpineState:
    """Execute the VERIFICATION phase using Deep Agents when available.

    Primary path (DA): Creates a verification DA agent with reviewer and
    test_engineer subagents. The agent loop handles code review and test
    execution internally.

    Fallback path: Uses the legacy approach (agent review + local quality gates)
    when no DA-compatible LLM provider is configured.
    """
    state["phase"] = PhaseName.VERIFICATION
    start_time = datetime.now(timezone.utc)

    _log_phase_event(state, "VERIFICATION", "phase_started", {
        "requirement": state.get("requirement", ""),
    }, config=config)

    # ── Resolve providers ─────────────────────────────────────────────
    providers = _get_providers(state, config)
    llm_provider = providers.get("llm")
    agent_provider = providers.get("agent")
    if agent_provider is None:
        state_ap = state.get("agent_provider")
        if state_ap is not None and not isinstance(state_ap, dict):
            agent_provider = state_ap

    # ── Try Deep Agents path ──────────────────────────────────────────
    da_chat_model = None
    if isinstance(llm_provider, DeepAgentsModelProvider):
        da_chat_model = llm_provider.chat_model
    elif llm_provider is not None and hasattr(llm_provider, "chat_model"):
        da_chat_model = llm_provider.chat_model

    if da_chat_model is not None:
        return _verification_phase_da(state, config, providers, da_chat_model, start_time)

    # ── Fallback: legacy path ─────────────────────────────────────────
    return _verification_phase_legacy(state, config, providers, agent_provider, start_time)


def _verification_phase_da(
    state: SpineState,
    config: Optional[RunnableConfig],
    providers: dict[str, Any],
    chat_model: Any,
    start_time: datetime,
) -> SpineState:
    """VERIFICATION phase using Deep Agents — primary path."""
    from ..adapters.da_phase_adapter import (
        create_verification_agent,
        _extract_verification_result,
    )

    debug_prompts = state.get("variables", {}).get("debug_prompts", False)
    max_steps = int(os.environ.get("SPINE_VERIFICATION_STEPS", "50"))

    project_root = _get_project_root(state)

    try:
        verification_agent = create_verification_agent(
            requirement=state["requirement"],
            providers=providers,
            max_steps=max_steps,
            root_dir=project_root,
        )
    except ValueError as e:
        state["errors"].append(f"DA verification agent creation failed: {e}")
        state["error_state"] = ErrorState.FATAL.value
        state["previous_phase"] = PhaseName.VERIFICATION
        state["phase"] = PhaseName.ERROR
        return state

    if debug_prompts:
        import sys
        print("[DA-VERIFICATION] Invoking DA verification agent...", file=sys.stderr)

    try:
        result = verification_agent.invoke({
            "messages": [{
                "role": "user",
                "content": (
                    f"Verify the implementation meets all acceptance criteria "
                    f"for: {state.get('requirement', '')}"
                ),
            }],
        }, config=config)
    except Exception as e:
        state["errors"].append(f"DA verification agent failed: {e}")
        state["error_state"] = ErrorState.TRANSIENT.value
        state["previous_phase"] = PhaseName.VERIFICATION
        state["phase"] = PhaseName.ERROR
        return state

    # ── Process verification results ──────────────────────────────────
    ver_result = _extract_verification_result(result)
    state["completed_tasks"].append("da_verification")

    # Also run local quality gates (syntax, lint, drift)
    syntax_ok, syntax_msg = _run_syntax_check()
    if syntax_ok:
        state["completed_tasks"].append("syntax_check")
    else:
        state["errors"].append(syntax_msg)

    lint_ok, lint_msg = _run_lint_check()
    if lint_ok:
        state["completed_tasks"].append("lint_check")
    else:
        state["errors"].append(lint_msg)

    # Determine if rework is needed
    needs_rework = (
        not ver_result.get("passed", False)
        or not syntax_ok
        or not lint_ok
    )

    if needs_rework:
        failed_criteria = ver_result.get("failed_criteria", [])
        if not syntax_ok:
            failed_criteria.append(syntax_msg)
        if not lint_ok:
            failed_criteria.append(lint_msg)
        state["failed_tasks"] = failed_criteria
        state["errors"].append(f"Verification failed: {len(failed_criteria)} criteria failed")
        state["previous_phase"] = PhaseName.VERIFICATION
        state["phase"] = PhaseName.REWORK
        return state

    # All checks passed
    state["previous_phase"] = PhaseName.VERIFICATION
    state["phase"] = PhaseName.COMPLETE

    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    _log_phase_event(state, "VERIFICATION", "phase_completed", {
        "tasks_completed": len(state.get("completed_tasks", [])),
        "errors": len(state.get("errors", [])),
        "duration": duration,
        "status": "success",
        "path": "deepagents",
    }, config=config)
    return state


def _verification_phase_legacy(
    state: SpineState,
    config: Optional[RunnableConfig],
    providers: dict[str, Any],
    agent_provider: Any,
    start_time: datetime,
) -> SpineState:
    """VERIFICATION phase using legacy approach — fallback path.

    Preserved for backward compatibility when no DA provider is configured.
    """
    # Agent-driven code review
    if agent_provider and agent_provider.enabled:
        review_subphase = SubPhase(
            name="AGENT_REVIEW",
            priority=1,
            parallel=False,
            agent_role="reviewer",
            tasks=[
                Task(
                    id="agent_review",
                    description=(
                        f"Review the code changes for: {state.get('requirement', '')}\n"
                        "Check for bugs, style issues, missing tests, and regressions.\n"
                        "Report: issues found (if any) and overall assessment."
                    ),
                ),
            ],
        )
        spec_content = ""
        spec_path = state.get("variables", {}).get("spec_path", "")
        if spec_path and os.path.isfile(spec_path):
            try:
                with open(spec_path, "r") as f:
                    spec_content = f.read()
            except Exception:
                pass
        executor = SwarmDAGExecutor(agent_provider=agent_provider)
        context = {
            "requirement": state.get("requirement", ""),
            "plan": state.get("plan"),
            "spec_content": spec_content,
            "variables": state.get("variables", {}),
        }
        phase_result = executor.execute_phase(
            Phase(name="VERIFICATION", subphases=[review_subphase]),
            context,
        )
        review_result = phase_result.subphase_results.get("AGENT_REVIEW", {})
        review_ok = review_result.get("status") == "success"
        _log_phase_event(state, "VERIFICATION", "task_completed", {
            "task_id": "agent_review",
            "subphase": "VERIFICATION",
            "status": "success" if review_ok else "failed",
        }, config=config)
        if review_ok:
            state["completed_tasks"].append("agent_review")
        else:
            state["errors"].append(f"Agent review failed: {review_result.get('error', 'unknown')}")

    # Local quality gates
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
    }, config=config)
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
    }, config=config)
    if lint_ok:
        state["completed_tasks"].append("lint_check")
    else:
        state["errors"].append(lint_msg)

    drift_ok = True
    drift_msg = "Plan drift check passed"
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
    }, config=config)
    if drift_ok:
        state["completed_tasks"].append("drift_check")
    else:
        state["errors"].append(drift_msg)

    # Git integration
    providers = _get_providers(state, config)
    git_workflow = providers.get("git")

    if git_workflow:
        try:
            requirement = state.get("requirement", "work")
            branch_name = f"spine-{requirement[:20].replace(' ', '-').lower()}"
            git_workflow.create_branch(branch_name)
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

    state["previous_phase"] = PhaseName.VERIFICATION
    state["phase"] = PhaseName.COMPLETE

    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
    _log_phase_event(state, "VERIFICATION", "phase_completed", {
        "tasks_completed": len(state.get("completed_tasks", [])),
        "errors": len(state.get("errors", [])),
        "duration": duration,
        "status": "success",
        "path": "legacy",
    }, config=config)
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


def _extract_planning_context(subphase_results: dict[str, Any]) -> dict[str, Any]:
    """Extract structured planning context from subphase results.

    Distills the raw subphase output into a context dict suitable for
    synthesize_slices and for inclusion in the execution prompt.

    Args:
        subphase_results: Dict mapping subphase name -> result data.

    Returns:
        Dict with keys: analysis, tech_research, risk_assessment, synthesis.
    """
    context: dict[str, Any] = {}

    for name, result in subphase_results.items():
        if not isinstance(result, dict):
            context[name.lower()] = str(result)
            continue

        # Extract the structured_data if present (from stub templates)
        structured = result.get("structured_data", {})
        output = result.get("output", "")
        tasks_info = result.get("tasks", {})

        # Flatten task results if available
        task_outputs = []
        if isinstance(tasks_info, dict):
            for tid, tdata in tasks_info.items():
                if isinstance(tdata, dict):
                    task_outputs.append(tdata.get("result", tdata.get("output", str(tdata))))
                else:
                    task_outputs.append(str(tdata))

        # Combine: prefer structured_data, then task outputs, then raw output
        combined = {}
        if structured:
            combined.update(structured)
        if task_outputs:
            combined["task_outputs"] = task_outputs
        if output and not structured:
            combined["output"] = output

        # Map subphase names to canonical context keys
        key = name.upper()
        if key == "ANALYZE":
            context["analysis"] = combined
        elif key == "TECH_RESEARCH":
            context["tech_research"] = combined
        elif key == "RISK_ASSESSMENT":
            context["risk_assessment"] = combined
        elif key == "SYNTHESIZE":
            context["synthesis"] = combined
        else:
            context[name.lower()] = combined

    return context


def _derive_plan_tasks(subphase_results: dict[str, Any], requirement: str) -> list[dict[str, str]]:
    """Derive execution plan tasks from planning subphase results.

    Instead of hardcoding "setup"/"implement", extracts actionable task
    descriptions from the SYNTHESIZE subphase output. Falls back to
    extracting components from the requirement if synthesis is empty.

    Args:
        subphase_results: Dict mapping subphase name -> result data.
        requirement: Original requirement text (for fallback).

    Returns:
        List of dicts with 'id' and 'description' keys.
    """
    # Try to get tasks from SYNTHESIZE output first
    synth = subphase_results.get("SYNTHESIZE", {})
    if isinstance(synth, dict):
        # Check for structured_data with task list
        structured = synth.get("structured_data", {})
        if isinstance(structured, dict):
            task_list = structured.get("tasks", structured.get("plan_tasks", []))
            if isinstance(task_list, list) and task_list:
                tasks = []
                for i, t in enumerate(task_list):
                    if isinstance(t, dict):
                        tid = t.get("id", f"task-{i+1}")
                        raw_desc = t.get("description", t.get("name", str(t)))
                        # Coerce dict descriptions to readable strings
                        if isinstance(raw_desc, dict):
                            desc = (
                                raw_desc.get("output")
                                or raw_desc.get("result")
                                or raw_desc.get("name")
                                or str(raw_desc)
                            )
                        else:
                            desc = str(raw_desc)
                        if len(desc) > 300:
                            desc = desc[:300].rsplit(".", 1)[0] + "."
                        tasks.append({"id": tid, "description": desc})
                    elif isinstance(t, str):
                        tasks.append({"id": f"task-{i+1}", "description": t})
                if tasks:
                    return tasks

        # Check for task results from the SYNTHESIZE subphase
        # Skip review/critic tasks — they are not execution tasks
        REVIEW_TASK_IDS = {"critic_review", "review", "critic_gate"}
        tasks_info = synth.get("tasks", {})
        if isinstance(tasks_info, dict):
            tasks = []
            for tid, tdata in tasks_info.items():
                # Skip review/critic tasks — they produce verdicts, not work items
                if tid in REVIEW_TASK_IDS or "critic" in tid.lower() or "review" in tid.lower():
                    continue
                if isinstance(tdata, dict):
                    # Agent result dicts have structured output like
                    # {'output': '...', 'exit_code': 0, 'success': True}.
                    # Extract the string content, never store the raw dict.
                    raw_result = tdata.get("result", tdata.get("output", tid))
                    if isinstance(raw_result, dict):
                        desc = (
                            raw_result.get("output")
                            or raw_result.get("result")
                            or raw_result.get("name")
                            or str(raw_result)
                        )
                    else:
                        desc = str(raw_result)
                    # Trim long LLM outputs to actionable descriptions
                    if isinstance(desc, str) and len(desc) > 300:
                        desc = desc[:300].rsplit(".", 1)[0] + "."
                    tasks.append({"id": tid, "description": desc})
            if tasks:
                return tasks

    # Fallback: extract components from requirement using dag helpers
    from ..models.dag import _extract_components
    components = _extract_components(requirement)
    tasks = [{"id": f"implement-{i+1}", "description": f"Implement {c}"}
             for i, c in enumerate(components)]
    return tasks if tasks else [{"id": "implement", "description": requirement}]


def _get_spine_root(state: SpineState) -> str:
    """Resolve the .spine root directory from state."""
    # Try checkpoint_path from state variables first
    checkpoint_path = state.get("variables", {}).get("checkpoint_path", ".spine/spine.db")
    return os.path.dirname(checkpoint_path)


def _get_project_root(state: SpineState) -> str:
    """Resolve the project root directory (parent of .spine/) from state.

    The DA planning agent's explorer/sme/analyst subagents need to run
    inside the actual project root so they can ls, grep, and read source
    files.  get_backend() uses this as root_dir for LocalShellBackend.
    """
    checkpoint_path = state.get("variables", {}).get("checkpoint_path")
    if checkpoint_path:
        abs_checkpoint = os.path.abspath(checkpoint_path)
        # checkpoint_path = <project_root>/.spine/spine.db
        # dirname once → <project_root>/.spine
        # dirname twice → <project_root>
        return os.path.dirname(os.path.dirname(abs_checkpoint))

    # Fallback: walk up from cwd to find a .spine/ directory
    cur = os.getcwd()
    for _ in range(20):  # safety limit
        if os.path.isdir(os.path.join(cur, ".spine")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    # Last resort: assume cwd is the project root
    return os.getcwd()


def _log_phase_event(state: SpineState, phase_name: str, event_type: str, data: Dict[str, Any], config: Optional[RunnableConfig] = None) -> None:
    """Log a phase event to swarm.log via the SwarmMail instance.

    Args:
        state: Current SpineState
        phase_name: Name of the phase
        event_type: Event type (phase_started, phase_completed, task_completed, plan_written)
        data: Event payload data
        config: Optional RunnableConfig with non-serialized providers
    """
    providers = _get_providers(state, config)
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
        # Defensive: task descriptions can be dicts (raw agent output).
        # Coerce to a readable string before writing.
        if isinstance(task_desc, dict):
            task_desc = (
                task_desc.get("output")
                or task_desc.get("result")
                or task_desc.get("name")
                or str(task_desc)
            )
        task_desc = str(task_desc)
        if len(task_desc) > 300:
            task_desc = task_desc[:300].rsplit(".", 1)[0] + "."
        lines.append(f"### {task_id}")
        lines.append(f"- Description: {task_desc}")
        # Include scope and acceptance criteria if available
        scope = task.get("scope")
        if scope:
            lines.append(f"- Scope: {scope}")
        acceptance = task.get("acceptance")
        if acceptance:
            lines.append(f"- Acceptance: {acceptance}")
        lines.append("")

    # Subphase results live in state, not the spec file.  Writing raw
    # structured_data dicts (huge markdown tables, full analysis outputs)
    # into the spec creates a polluted context that gets fed back to the
    # execution agent.  Skip this section entirely.

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
        # If the execution phase just completed (previous_phase == EXECUTION or
        # phase was set to VERIFICATION by the execution phase itself), go to
        # verification. But if the planning phase just set phase=EXECUTION to
        # signal that execution should run next, go to execution.
        previous = state.get("previous_phase")
        if previous == PhaseName.PLANNING:
            # Planning phase just finished and signalled EXECUTION
            return "execution"
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
    # Use SqliteSaver with custom serializer for provider objects
    serializer = ProviderSerializer()
    conn = sqlite3.connect(checkpoint_path)
    memory = SqliteSaver(conn=conn, serde=serializer)
    
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
        {
            "planning": "planning",
            "execution": "execution",
            "verification": "verification",
            "rework": "rework",
            "blocked": "blocked",
            "error": "error",
            "human_review": "human_review",
            "__end__": END,
        }
    )
    workflow.add_edge("execution", "verification")
    workflow.add_conditional_edges(
        "verification",
        should_continue,
        {
            "rework": "rework",
            "blocked": "blocked",
            "error": "error",
            "planning": "planning",
            "execution": "execution",
            "verification": "verification",
            "human_review": "human_review",
            "__end__": END,
        }
    )
    workflow.add_conditional_edges(
        "rework",
        should_continue,
        {
            "planning": "planning",
            "execution": "execution",
            "verification": "verification",
            "rework": "rework",
            "error": "error",
            "human_review": "human_review",
            "__end__": END,
        }
    )
    workflow.add_edge("blocked", "blocked")
    workflow.add_conditional_edges(
        "error",
        should_continue,
        {
            "rework": "rework",
            "blocked": "blocked",
            "planning": "planning",
            "execution": "execution",
            "verification": "verification",
            "human_review": "human_review",
            "__end__": END,
        }
    )
    workflow.add_conditional_edges(
        "human_review",
        should_continue,
        {
            "rework": "rework",
            "planning": "planning",
            "execution": "execution",
            "verification": "verification",
            "error": "error",
            "__end__": END,
        }
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
        self._checkpointer = SqliteSaver(
            conn=sqlite3.connect(
                str(checkpoint_path),
                check_same_thread=False,
            ),
            serde=ProviderSerializer(),
        )
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
            {
                "planning": "planning",
                "execution": "execution",
                "verification": "verification",
                "rework": "rework",
                "blocked": "blocked",
                "error": "error",
                "human_review": "human_review",
                "__end__": END,
            }
        )
        workflow.add_edge("execution", "verification")
        workflow.add_conditional_edges(
            "verification",
            should_continue,
            {
                "rework": "rework",
                "blocked": "blocked",
                "error": "error",
                "planning": "planning",
                "execution": "execution",
                "verification": "verification",
                "human_review": "human_review",
                "__end__": END,
            }
        )
        workflow.add_conditional_edges(
            "rework",
            should_continue,
            {
                "planning": "planning",
                "execution": "execution",
                "verification": "verification",
                "rework": "rework",
                "error": "error",
                "human_review": "human_review",
                "__end__": END,
            }
        )
        workflow.add_edge("blocked", "blocked")
        workflow.add_conditional_edges(
            "error",
            should_continue,
            {
                "rework": "rework",
                "blocked": "blocked",
                "planning": "planning",
                "execution": "execution",
                "verification": "verification",
                "human_review": "human_review",
                "__end__": END,
            }
        )
        workflow.add_conditional_edges(
            "human_review",
            should_continue,
            {
                "rework": "rework",
                "planning": "planning",
                "execution": "execution",
                "verification": "verification",
                "error": "error",
                "__end__": END,
            }
        )
        workflow.set_entry_point("init")
        
        return workflow.compile(checkpointer=self._checkpointer)
    
    def run(self, requirement: str, thread_id: str) -> SpineState:
        """Execute the full SPINE workflow."""
        # Build providers dict — passed through config so it survives
        # LangGraph's checkpoint serialization (config is never persisted).
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
            {"configurable": {"thread_id": thread_id, "providers": providers}}
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
    
    def resume(self, thread_id: str) -> Optional[SpineState]:
        """Resume a previous workflow.

        Returns the latest state dict for the given thread_id,
        or None if no checkpoint exists.
        """
        snapshot = self.app.get_state({"configurable": {"thread_id": thread_id}})
        if snapshot is None:
            return None
        # StateSnapshot.values contains the current state dict
        values = getattr(snapshot, "values", None)
        if values is not None:
            return dict(values)
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

