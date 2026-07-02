"""CRITIC phase as a LangGraph subgraph.

The subgraph has two internal nodes:
1. ``structural_check`` — fast, no-LLM structural check.
2. ``agent_check`` — deep LLM-based quality review.

Additionally, for the PLAN phase, a ``plan_validation`` node validates
plan.json structure (feature_slices, dependencies, cycles).

Parameterized by ``reviewed_phase`` so the same subgraph builder can be
used for critic_specify, critic_plan, critic_tasks.
"""

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from spine.config import SpineConfig
from spine.models.enums import PhaseName, ReviewStatus
from spine.workflow.subgraph_state import CriticSubgraphState
from spine.workflow.critic_review import structural_critic_check, agent_critic_check
from spine.agents.plan_do import run_plan_node
from spine.agents.prompt_format import Tag, hostage_layout, xml_blocks

logger = logging.getLogger(__name__)


async def _critic_directive_node(
    state: CriticSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """No-tool planning step before the critic's agent_check.

    Gives the planner the reviewed phase and a short task framing; the
    do node then runs the full critic agent with the directive
    prepended to its prompt.
    """
    work_id = state.get("work_id", "unknown")
    reviewed_phase = state.get("reviewed_phase", "unknown")
    description = state.get("description", "")
    task = hostage_layout(
        xml_blocks((Tag.OBJECTIVE, description)),
        (
            f"Plan a code review of the {reviewed_phase!r} phase output. The "
            "do node will inspect the structured artifact "
            "(specification.json / plan.json / etc.) and return a verdict "
            "(PASSED / NEEDS_REVISION / NEEDS_REVIEW). Identify what to focus on."
        ),
    )
    directive = await run_plan_node(
        state=dict(state),
        config=config,
        phase_path=PhaseName.CRITIC.value,
        task_description=task,
        role_hint=f"critic for the {reviewed_phase!r} phase",
        workspace_root=state.get("workspace_root", "."),
    )
    logger.info(
        "[%s] CRITIC plan-directive: approach=%r",
        work_id, directive.approach[:80],
    )
    return {"critic_directive": directive.model_dump()}


async def _structural_check_node(
    state: CriticSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Run the structural critic check within the subgraph."""
    reviewed_phase = state.get("reviewed_phase", "unknown")
    # structural_critic_check expects a dict-like state with "artifacts" key
    pseudo_state = {"artifacts": state.get("artifacts", {})}
    result = structural_critic_check(pseudo_state, reviewed_phase)
    return {
        "structural_result": result,
        "phase_status": "success"
        if result["status"] == ReviewStatus.PASSED.value
        else result["status"],
    }


async def _plan_validation_node(
    state: CriticSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Validate plan.json structure when reviewing the PLAN phase.

    Checks plan.json for:
    - Non-empty feature_slices array
    - Every slice has acceptance_criteria and target_files
    - All dependency IDs reference existing slice IDs
    - No dependency cycles (simple DFS detection)

    Returns phase_status=NeedsRevision on failure, NeedsReview on error.
    """
    reviewed_phase = state.get("reviewed_phase", "")
    if PhaseName(reviewed_phase) != PhaseName.PLAN:
        # Not reviewing PLAN phase — skip validation
        return {"phase_status": "passed"}

    workspace_root = state.get("workspace_root", ".")
    work_id = state.get("work_id", "unknown")

    # Load plan.json from disk
    plan_path = Path(workspace_root) / ".spine" / "artifacts" / work_id / "plan" / "plan.json"
    if not plan_path.exists():
        # No plan.json — not a structured plan, skip validation
        return {"phase_status": "passed"}

    try:
        plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"[{work_id}] Could not load plan.json for validation: {e}")
        return {
            "phase_status": ReviewStatus.NEEDS_REVISION.value,
            "validation_result": {
                "status": ReviewStatus.NEEDS_REVISION.value,
                "tier": "structural",
                "reason": f"Could not load plan.json: {e}",
                "suggestions": ["Ensure plan.json is valid JSON"],
            },
        }

    # Validate feature_slices structure
    feature_slices = plan_data.get("feature_slices")
    if not isinstance(feature_slices, list) or len(feature_slices) == 0:
        return {
            "phase_status": ReviewStatus.NEEDS_REVISION.value,
            "validation_result": {
                "status": ReviewStatus.NEEDS_REVISION.value,
                "tier": "structural",
                "reason": "plan.json must contain a non-empty 'feature_slices' array",
                "suggestions": ["Add at least one feature slice with id, title, target_files, acceptance_criteria, and dependencies"],
            },
        }

    # Build ID set and validate each slice
    slice_ids: set[str] = set()
    errors: list[str] = []
    for i, s in enumerate(feature_slices):
        if not isinstance(s, dict):
            errors.append(f"Slice at index {i} is not a valid object")
            continue

        sid = s.get("id", "")
        if sid:
            slice_ids.add(sid)
        else:
            errors.append(f"Slice at index {i} is missing 'id'")

        if not s.get("acceptance_criteria"):
            errors.append(f"Slice '{sid or i}' is missing 'acceptance_criteria'")
        if not s.get("target_files"):
            errors.append(f"Slice '{sid or i}' is missing 'target_files'")

    if errors:
        return {
            "phase_status": ReviewStatus.NEEDS_REVISION.value,
            "validation_result": {
                "status": ReviewStatus.NEEDS_REVISION.value,
                "tier": "structural",
                "reason": "; ".join(errors),
                "suggestions": ["Ensure every slice has: id, target_files, acceptance_criteria"],
            },
        }

    # Dependency integrity: all referenced IDs must exist
    dep_errors: list[str] = []
    dep_graph: dict[str, list[str]] = {}
    for s in feature_slices:
        sid = s.get("id", "")
        deps = s.get("dependencies") or []
        dep_graph[sid] = deps
        for dep_id in deps:
            if dep_id not in slice_ids:
                dep_errors.append(f"Slice '{sid}' depends on unknown slice '{dep_id}'")

    if dep_errors:
        return {
            "phase_status": ReviewStatus.NEEDS_REVISION.value,
            "validation_result": {
                "status": ReviewStatus.NEEDS_REVISION.value,
                "tier": "structural",
                "reason": "Dependency integrity check failed: " + "; ".join(dep_errors),
                "suggestions": ["Ensure all dependency IDs reference existing slice IDs", "Check for typos in dependency references"],
            },
        }

    # Cycle detection via DFS
    def has_cycle(graph: dict[str, list[str]]) -> bool:
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {node: WHITE for node in graph}

        def dfs(node: str) -> bool:
            color[node] = GRAY
            for neighbor in graph.get(node, []):
                if neighbor not in color:
                    continue
                if color[neighbor] == GRAY:
                    return True
                if color[neighbor] == WHITE and dfs(neighbor):
                    return True
            color[node] = BLACK
            return False

        return any(color[n] == WHITE and dfs(n) for n in color)

    if has_cycle(dep_graph):
        return {
            "phase_status": ReviewStatus.NEEDS_REVISION.value,
            "validation_result": {
                "status": ReviewStatus.NEEDS_REVISION.value,
                "tier": "structural",
                "reason": "Dependency cycle detected among feature slices",
                "suggestions": ["Remove circular dependencies between slices", "Reorder slices into a proper DAG"],
            },
        }

    # Reference-symbol gate (deterministic, no LLM): every slice
    # reference_symbols entry must resolve — in the codebase index, in a
    # sibling slice's `provides`, or as an external-library name. Dangling
    # entries get a precise revision message ("did you mean ...?"); a dangling
    # symbol whose owner a scope_exclusions bullet protects escalates as a
    # spec_contradiction the plan cannot rework its way out of (trace
    # 019f2077: UIApi.get_llm_providers, 4 wasted critic rounds). Fully
    # defensive — a gate crash must never take the critic down.
    try:
        from spine.workflow.plan_reference_gate import check_reference_symbols

        prior_lcr = state.get("last_critic_review") or {}
        prior_gate = (
            prior_lcr.get("reference_gate")
            if prior_lcr.get("phase") == PhaseName.PLAN.value
            else None
        )
        gate = check_reference_symbols(
            plan_data,
            state.get("specification_json"),
            prior_gate,
            db_path=SpineConfig.load().checkpoint_path,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[{work_id}] reference-symbol gate skipped: {e}")
        gate = None
    if gate:
        return {
            "phase_status": gate["status"],
            "validation_result": gate,
            "reference_gate_result": gate,
        }

    return {"phase_status": "passed", "reference_gate_result": {}}


async def _agent_check_node(
    state: CriticSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Run the agent critic check within the subgraph."""
    reviewed_phase = state.get("reviewed_phase", "unknown")
    work_id = state.get("work_id", "unknown")
    logger.info("[%s] Critic agent check starting for phase '%s'", work_id, reviewed_phase)
    # agent_critic_check expects a dict-like state
    pseudo_state = {
        "artifacts": state.get("artifacts", {}),
        "description": state.get("description", ""),
        "workspace_root": state.get("workspace_root", "."),
        "work_id": state.get("work_id", "unknown"),
        "work_type": state.get("work_type", ""),
        "feedback": state.get("feedback", []),
        "retry_count": {reviewed_phase: state.get("retry_count", 0)},
        "max_retries": SpineConfig.load().max_critic_retries,
        "specification_json": state.get("specification_json"),
        "plan_json": state.get("plan_json"),
        # Forward the directive from the plan-before-do split so
        # agent_critic_check can prepend it to the critic's prompt.
        "critic_directive": state.get("critic_directive"),
        # The critic's own prior verdict — agent_critic_check renders it into
        # the REWORK prompt so round N confirms round N-1's asks were met
        # instead of shifting the goalposts. Without this key the anti-churn
        # prompt never fired in production (trace 019f2077: four rounds, four
        # unrelated objection sets).
        "last_critic_review": state.get("last_critic_review"),
    }
    result = await agent_critic_check(pseudo_state, reviewed_phase, config)
    logger.info(
        "[%s] Critic agent check complete: status=%s",
        work_id, result.get("status"),
    )

    # Deterministic plan-validation is a HARD FLOOR: the LLM's verdict may not
    # upgrade a structural failure (dependency cycle, dangling dep, missing
    # target_files/acceptance_criteria, dangling reference_symbols) into a
    # PASS. The validation reason is folded into the agent result so the
    # rework prompt actually sees it. A NEEDS_REVIEW validation verdict (the
    # reference-symbol gate's spec_contradiction) escalates as-is — its
    # blocker_category rides along so the result mapper routes the review to
    # SPECIFY. Deterministic verdicts deliberately bypass the LLM
    # corroboration pass: they are checkable facts, not agent opinions.
    validation = state.get("validation_result") or {}
    if validation.get("status") and validation["status"] != ReviewStatus.PASSED.value:
        effective_status = (
            ReviewStatus.NEEDS_REVIEW.value
            if validation["status"] == ReviewStatus.NEEDS_REVIEW.value
            else ReviewStatus.NEEDS_REVISION.value
        )
        if result.get("status") != effective_status:
            logger.info(
                "[%s] Critic agent voted %s but deterministic validation failed "
                "(%s) — forcing %s",
                work_id, result.get("status"),
                validation.get("reason", ""), effective_status,
            )
        merged = dict(result)
        merged["status"] = effective_status
        if validation.get("blocker_category"):
            merged["blocker_category"] = validation["blocker_category"]
            merged["cited_exclusions"] = validation.get("cited_exclusions") or []
        val_reason = validation.get("reason")
        if val_reason:
            existing = merged.get("reason") or ""
            merged["reason"] = (
                f"Plan validation failed: {val_reason}. {existing}".strip()
            )
        existing_sugg = list(merged.get("suggestions") or [])
        merged["suggestions"] = list(validation.get("suggestions") or []) + existing_sugg
        return {
            "agent_result": merged,
            "phase_status": effective_status,
        }

    return {
        "agent_result": result,
        "phase_status": result["status"],
    }


def _critic_subgraph_router(state: CriticSubgraphState) -> str:
    """Route based on structural check result.
    
    Structural check passes → proceed to agent check.
    Structural check fails → still proceed to agent check (which will also fail).
    This ensures both tiers produce feedback for the rework phase to address.
    """
    structural = state.get("structural_result", {})
    status = structural.get("status", ReviewStatus.NEEDS_REVISION.value)
    if status == ReviewStatus.PASSED.value:
        return "passed"
    if status == ReviewStatus.NEEDS_REVIEW.value:
        return "needs_review"
    return "needs_revision"


def _plan_validation_router(state: CriticSubgraphState) -> str:
    """Route after plan validation node."""
    phase_status = state.get("phase_status", "passed")
    if phase_status == "passed":
        return "passed"
    if phase_status == ReviewStatus.NEEDS_REVIEW.value:
        return "needs_review"
    return "needs_revision"


def build_critic_subgraph(reviewed_phase: str) -> Any:
    """Build a CRITIC phase subgraph for a specific reviewed phase.

    The critic performs two-tier review:
    1. Structural check (fast, no-LLM) - checks artifacts exist and have content
    2. Agent check (LLM-based) - quality review
    
    If structural fails, we still run the agent check so both tiers produce
    feedback. The agent will see the structural failure in the state and can
    provide additional context.
    """
    builder = StateGraph(CriticSubgraphState)

    builder.add_node("structural_check", _structural_check_node)
    # No-tool planning step inserted before agent_check (plan→do split).
    builder.add_node("plan_directive", _critic_directive_node)
    builder.add_node("agent_check", _agent_check_node)

    if PhaseName(reviewed_phase) == PhaseName.PLAN:
        builder.add_node("plan_validation", _plan_validation_node)

    builder.add_edge(START, "structural_check")
    # Route to plan_validation (PLAN reviews) or plan_directive (others).
    # plan_validation is its own router target so it can also funnel into
    # plan_directive before the agent runs.
    builder.add_conditional_edges(
        "structural_check",
        _critic_subgraph_router,
        {
            "passed": "plan_directive" if PhaseName(reviewed_phase) != PhaseName.PLAN else "plan_validation",
            "needs_revision": "plan_directive" if PhaseName(reviewed_phase) != PhaseName.PLAN else "plan_validation",
            "needs_review": "plan_directive" if PhaseName(reviewed_phase) != PhaseName.PLAN else "plan_validation",
        },
    )

    if PhaseName(reviewed_phase) == PhaseName.PLAN:
        builder.add_conditional_edges(
            "plan_validation",
            _plan_validation_router,
            {
                "passed": "plan_directive",
                "needs_revision": "plan_directive",
                "needs_review": "plan_directive",
            },
        )

    builder.add_edge("plan_directive", "agent_check")
    builder.add_edge("agent_check", END)

    return builder
