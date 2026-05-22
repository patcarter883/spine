"""SPINE CRITIC phase — reviews the output of the previous phase.

The critic is wired into the workflow graph as a conditional node.
It performs two-tier review (structural + agent) and routes:
- PASSED → next phase
- NEEDS_REVISION → rework previous phase
- NEEDS_REVIEW → flag for human review

This module provides the node function for the LangGraph graph.
The actual review logic is in ``spine.workflow.critic_review``.

Phase node functions are async to avoid event-loop binding errors when
subagents inherit the parent checkpointer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from typing import Optional

from langchain_core.runnables import RunnableConfig

from spine.models.enums import PhaseName, ReviewStatus
from spine.models.state import WorkflowState
from spine.workflow.critic_review import (
    _get_reviewed_phase,
    structural_critic_check,
    agent_critic_check,
)
from spine.agents.artifacts import materialize_phase_artifacts
from spine.workflow.registry import get_registry

logger = logging.getLogger(__name__)


def _validate_plan_structure(
    state: WorkflowState, workspace_root: str, work_id: str
) -> dict[str, Any] | None:
    """Validate structured plan format when reviewing the PLAN phase.

    Checks plan.json for:
    - Non-empty ``feature_slices`` array.
    - Every slice has ``acceptance_criteria`` and ``target_files``.
    - All dependency IDs reference existing slice IDs.
    - No dependency cycles (simple DFS detection).

    Reads plan.json from the artifacts dict first, falling back to reading
    from disk at ``.spine/artifacts/{work_id}/plan/plan.json``.

    Args:
        state: The current workflow state.
        workspace_root: Absolute workspace root path.
        work_id: The current work item ID.

    Returns:
        ``None`` if validation passes or plan.json is absent (not a structured
        plan). Otherwise a review dict with ``status``, ``tier``, ``reason``,
        and ``suggestions`` keys.
    """
    # Locate plan.json — first in artifacts, then on disk
    plan_data = _load_plan_json(state, workspace_root, work_id)
    if plan_data is None:
        # No plan.json present — not a structured plan, skip validation
        return None

    feature_slices = plan_data.get("feature_slices")
    if not isinstance(feature_slices, list) or len(feature_slices) == 0:
        return {
            "status": ReviewStatus.NEEDS_REVISION.value,
            "tier": "structural",
            "reason": ("plan.json must contain a non-empty 'feature_slices' array"),
            "suggestions": [
                "Add at least one feature slice with id, title, target_files, "
                "acceptance_criteria, and dependencies",
            ],
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
            "status": ReviewStatus.NEEDS_REVISION.value,
            "tier": "structural",
            "reason": "; ".join(errors),
            "suggestions": [
                "Ensure every slice has: id, target_files, acceptance_criteria",
            ],
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
            "status": ReviewStatus.NEEDS_REVISION.value,
            "tier": "structural",
            "reason": "Dependency integrity check failed: " + "; ".join(dep_errors),
            "suggestions": [
                "Ensure all dependency IDs reference existing slice IDs",
                "Check for typos in dependency references",
            ],
        }

    # Cycle detection via DFS
    if _has_cycle(dep_graph):
        return {
            "status": ReviewStatus.NEEDS_REVISION.value,
            "tier": "structural",
            "reason": "Dependency cycle detected among feature slices",
            "suggestions": [
                "Remove circular dependencies between slices",
                "Reorder slices into a proper DAG",
            ],
        }

    return None


def _load_plan_json(
    state: WorkflowState, workspace_root: str, work_id: str
) -> dict[str, Any] | None:
    """Load plan.json from artifacts dict or disk, returning parsed JSON or None."""
    # Try artifacts dict first
    artifacts = state.get("artifacts", {})
    plan_artifacts = artifacts.get(PhaseName.PLAN.value, {})
    plan_json_str = plan_artifacts.get("plan.json") if isinstance(plan_artifacts, dict) else None
    if plan_json_str and isinstance(plan_json_str, str):
        try:
            return json.loads(plan_json_str)
        except (json.JSONDecodeError, ValueError):
            pass

    # Fall back to disk
    plan_path = Path(workspace_root) / ".spine" / "artifacts" / work_id / "plan" / "plan.json"
    if plan_path.exists():
        try:
            return json.loads(plan_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    return None


def _has_cycle(graph: dict[str, list[str]]) -> bool:
    """Detect cycles in a directed graph using DFS.

    Args:
        graph: Adjacency list mapping node → list of dependencies.

    Returns:
        ``True`` if a cycle exists, ``False`` otherwise.
    """
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


async def call_critic(
    state: WorkflowState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """Execute the CRITIC phase node.

    Runs two-tier review and updates state with feedback and retry counts.
    The routing decision (passed/needs_revision/needs_review) is handled
    by the conditional edge in the composer, using ``critic_router()``.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        Partial state update with critic feedback and retry count increment.
    """
    work_id = state.get("work_id", "unknown")
    reviewed_phase = _get_reviewed_phase(state)
    workspace_root = state.get("workspace_root", ".")

    logger.info(f"[{work_id}] CRITIC reviewing {reviewed_phase}")

    # ── Tier 1: Structural check ──
    structural_result = structural_critic_check(state, reviewed_phase)

    if structural_result["status"] != "passed":
        logger.info(f"[{work_id}] Structural check failed for {reviewed_phase}")
        # Increment retry count for the reviewed phase
        retry_count = state.get("retry_count", {})
        current = retry_count.get(reviewed_phase, 0)
        phase_artifacts = {"review.md": structural_result["reason"]}
        materialize_phase_artifacts(
            PhaseName.CRITIC.value, phase_artifacts, workspace_root, work_id=work_id
        )
        return {
            "artifacts": {PhaseName.CRITIC.value: phase_artifacts},
            "feedback": [structural_result],
            "retry_count": {reviewed_phase: current + 1},
            "current_phase": PhaseName.CRITIC.value,
            "status": "running",
            "prompt_request": None,
        }

    # ── Tier 1.5: PLAN-specific structural validation ──
    if reviewed_phase == PhaseName.PLAN.value:
        plan_validation = _validate_plan_structure(state, workspace_root, work_id)
        if plan_validation is not None:
            logger.info(
                f"[{work_id}] Plan structure validation failed: {plan_validation['reason']}"
            )
            retry_count = state.get("retry_count", {})
            current = retry_count.get(reviewed_phase, 0)
            phase_artifacts = {"review.md": plan_validation["reason"]}
            materialize_phase_artifacts(
                PhaseName.CRITIC.value,
                phase_artifacts,
                workspace_root,
                work_id=work_id,
            )
            return {
                "artifacts": {PhaseName.CRITIC.value: phase_artifacts},
                "feedback": [plan_validation],
                "retry_count": {reviewed_phase: current + 1},
                "current_phase": PhaseName.CRITIC.value,
                "status": "running",
                "prompt_request": None,
            }

    # ── Tier 2: Agent check ──
    agent_result = await agent_critic_check(state, reviewed_phase, config=config)
    logger.info(f"[{work_id}] Agent check for {reviewed_phase}: {agent_result['status']}")

    # Increment retry count if needs revision
    retry_count = state.get("retry_count", {})
    current = retry_count.get(reviewed_phase, 0)
    new_retry = current + 1 if agent_result["status"] != "passed" else current

    # Materialize this phase's artifacts to disk immediately
    phase_artifacts = {"review.md": agent_result["reason"]}
    materialize_phase_artifacts(
        PhaseName.CRITIC.value, phase_artifacts, workspace_root, work_id=work_id
    )

    return {
        "artifacts": {PhaseName.CRITIC.value: phase_artifacts},
        "feedback": [agent_result],
        "retry_count": {reviewed_phase: new_retry},
        "current_phase": PhaseName.CRITIC.value,
        "status": "running",
        "prompt_request": None,
    }


def _build_critic_agent(state: WorkflowState, config: Optional[RunnableConfig] = None) -> Any:
    """Build a critic Deep Agent. Used by the registry as build_agent_fn."""
    from spine.critic.agent import build_critic_agent

    return build_critic_agent(state, config)


# ── Self-register on import ──
_registry = get_registry()
_registry.register(
    name=PhaseName.CRITIC.value,
    call_fn=call_critic,
    build_agent_fn=_build_critic_agent,
    description="Review previous phase output (structural + agent tiers)",
)
