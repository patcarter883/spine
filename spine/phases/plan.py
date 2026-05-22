"""SPINE PLAN phase — define the technical architecture.

This phase takes a specification and produces a technical plan document.
Context engineering: specification is on disk (not inlined). SpineContext
passed at invoke time.

After the agent completes, this phase reads the structured ``plan.json``
(written by the plan agent's structured tool) and computes execution waves
via :func:`spine.workflow.slice_scheduler.compute_execution_waves`.  If
validation fails (cycles, missing dependencies) the phase returns a
``needs_review`` status with actionable feedback instead of propagating
the error.

Phase node functions are async to avoid event-loop binding errors when
subagents inherit the parent checkpointer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from typing import Optional

from langchain_core.runnables import RunnableConfig

from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.plan_agent import build_plan_agent
from spine.agents.helpers import extract_response
from spine.agents.retry import ainvoke_with_retry
from spine.agents.context import build_context
from spine.agents.artifacts import (
    materialize_artifacts,
    materialize_phase_artifacts,
    _artifact_path,
)
from spine.workflow.registry import get_registry

logger = logging.getLogger(__name__)


def _read_plan_json(workspace_root: str, work_id: str) -> dict[str, Any] | None:
    """Read and parse plan.json from the plan artifact directory.

    Args:
        workspace_root: Absolute path to the project workspace.
        work_id: The current work item ID.

    Returns:
        Parsed plan.json dict, or None if the file doesn't exist or is
        invalid.
    """
    plan_json_path = Path(workspace_root) / ".spine" / "artifacts" / work_id / "plan" / "plan.json"
    if not plan_json_path.exists():
        logger.debug("[%s] plan.json not found at %s", work_id, plan_json_path)
        return None

    try:
        content = plan_json_path.read_text(encoding="utf-8")
        data = json.loads(content)
        logger.info("[%s] Read plan.json (%d chars)", work_id, len(content))
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("[%s] Failed to read plan.json: %s", work_id, exc)
        return None


def _compute_waves(plan_data: dict[str, Any], work_id: str) -> tuple[list[list[dict]], str | None]:
    """Compute execution waves from structured plan data.

    Wraps :func:`~spine.workflow.slice_scheduler.compute_execution_waves`
    with error handling for validation failures (cycles, missing deps).

    Args:
        plan_data: Parsed plan.json content containing slices with
            dependency information.
        work_id: Work item ID for logging.

    Returns:
        Tuple of (waves, error_message). On success, error_message is
        None. On validation failure, waves is empty and error_message
        describes the problem.
    """
    try:
        from spine.workflow.slice_scheduler import (
            FeatureSlice,
            compute_execution_waves,
        )
    except ImportError:
        logger.debug(
            "[%s] slice_scheduler not available — skipping wave computation",
            work_id,
        )
        return [], None

    # Extract feature_slices array from plan.json data.
    raw_slices = plan_data.get("feature_slices")
    if not isinstance(raw_slices, list) or not raw_slices:
        logger.debug(
            "[%s] plan.json has no feature_slices — skipping wave computation",
            work_id,
        )
        return [], None

    try:
        # Convert plan.json slice dicts to FeatureSlice dataclass objects.
        # plan.json field names match the FeatureSlice dataclass exactly:
        #   id, title, target_files, execution_requirements, dependencies,
        #   acceptance_criteria, complexity.
        # FeatureSlice.from_dict() handles unknown keys gracefully.
        scheduler_slices = [FeatureSlice.from_dict(sd) for sd in raw_slices]

        waves = compute_execution_waves(scheduler_slices)

        # Convert FeatureSlice objects to plain dicts for state storage.
        wave_dicts: list[list[dict]] = [[asdict(s) for s in wave] for wave in waves]

        logger.info(
            "[%s] Computed %d execution wave(s) with %d total slices",
            work_id,
            len(wave_dicts),
            sum(len(w) for w in wave_dicts),
        )
        return wave_dicts, None
    except (ValueError, KeyError, TypeError) as exc:
        error_msg = f"Execution wave computation failed: {exc}"
        logger.warning("[%s] %s", work_id, error_msg)
        return [], error_msg


async def call_plan(
    state: WorkflowState, config: Optional[RunnableConfig] = None
) -> dict[str, Any]:
    """Execute the PLAN phase.

    Delegates to the plan Deep Agent, which designs the technical architecture
    based on the specification. If reworking, includes prior feedback.

    The original work description is NOT passed to PLAN — the specification
    artifact from SPECIFY already captures and expands on it.  The only
    additional input beyond prior artifacts is review feedback.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.

    Returns:
        Partial state update with plan artifacts.
    """
    work_id = state.get("work_id", "unknown")
    work_type = state.get("work_type", "")
    retry_count = state.get("retry_count", {}).get(PhaseName.PLAN.value, 0)
    feedback = state.get("feedback", [])
    workspace_root = state.get("workspace_root", ".")

    logger.info(f"[{work_id}] PLAN phase starting (retry={retry_count})")

    try:
        agent = build_plan_agent(state, config)

        # Materialize prior artifacts to disk
        materialize_artifacts(state, workspace_root, work_id=work_id)

        # Build prompt — specification is on disk at work_id-scoped path.
        # The original description is intentionally NOT re-included — the
        # specification file already captures and expands on it.
        plan_dir = f".spine/artifacts/{work_id}/plan"
        context_seed = f"globalThis.context = {{work_id: '{work_id}', phase: 'plan', plan_dir: '{plan_dir}'}};\n\n"

        rework_prefix = ""
        if retry_count > 0:
            rework_prefix = "⚠ **REWORK PASS**: Your primary objective is to revise the prior plan. Address all points from the critic feedback.\n\n"

        spec_path = _artifact_path(work_id, PhaseName.SPECIFY.value)

        # Check if spec exists to formulate the exact dynamic spec instruction
        has_spec = False
        artifacts = state.get("artifacts", {}) or {}
        if artifacts.get(PhaseName.SPECIFY.value):
            has_spec = True

        spec_instruction = (
            f"The full specification is available on disk at `{spec_path}/specification.md` "
            "and will also be loaded by `read_prior_artifacts` under "
            "`artifacts.specify['specification.md']`. Use it as the source of truth "
            "for ALL researcher dispatches — every subagent must receive the relevant "
            "spec section so it can find the matching codebase files, patterns, and "
            "conventions. Do NOT dispatch researchers with just the work description.\n\n"
            if has_spec
            else "No prior specification exists (quick workflow). Work directly from the description returned by `read_prior_artifacts`. Do NOT dispatch researcher subagents unless the work description explicitly requires codebase exploration.\n\n"
        )

        prompt = (
            context_seed
            + rework_prefix
            + "Create a detailed technical plan based on the specification.\n\n"
            + spec_instruction
        )
        if retry_count > 0 and feedback:
            feedback_text = "\n".join(
                f"- [{f.get('tier', 'unknown')}] {f.get('reason', '')}"
                for f in feedback
                if isinstance(f, dict)
            )
            prompt += f"## Previous Review Feedback\n{feedback_text}\n"

        ctx = build_context(state, PhaseName.PLAN)

        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.PLAN.value,
            work_id=work_id,
            work_type=work_type,
            context=ctx,
        )

        plan_content = extract_response(result)

        # Materialize this phase's artifacts to disk immediately.
        # plan.md is the narrative artifact produced by the agent; plan.json
        # (if written by the structured plan tool) is already on disk.
        phase_artifacts: dict[str, str] = {"plan.md": plan_content}

        # Read plan.json from disk — written by the structured plan tool
        # (add-structured-plan-tool dependency).  Contains structured slice
        # data with dependency information for wave computation.
        plan_json_data = _read_plan_json(workspace_root, work_id)
        if plan_json_data is not None:
            # Include plan.json content in artifacts dict for state tracking.
            # Re-serialize so the artifact value is a string consistent with
            # the phase_artifacts convention ({filename: content_str}).
            phase_artifacts["plan.json"] = json.dumps(plan_json_data, indent=2)

        materialize_phase_artifacts(
            PhaseName.PLAN.value, phase_artifacts, workspace_root, work_id=work_id
        )

        # Compute execution waves from the structured plan data.  Waves
        # are topologically sorted groups of independent slices that
        # IMPLEMENT can dispatch concurrently within each wave.
        execution_waves: list[list[dict]] = []
        wave_error: str | None = None

        if plan_json_data is not None:
            execution_waves, wave_error = _compute_waves(plan_json_data, work_id)

        # Validation errors (cycles, missing deps) result in needs_review
        # so a human can fix before implementation starts.
        if wave_error is not None:
            logger.warning(
                "[%s] PLAN phase returning needs_review due to wave validation error: %s",
                work_id,
                wave_error,
            )
            return {
                "artifacts": {PhaseName.PLAN.value: phase_artifacts},
                "current_phase": PhaseName.PLAN.value,
                "status": "needs_review",
                "prompt_request": {
                    "message": (
                        f"PLAN phase produced a plan but execution wave "
                        f"validation failed.\n\n{wave_error}\n\n"
                        f"Please review the plan's slice dependencies "
                        f"and fix any cycles or missing references."
                    ),
                    "phase": PhaseName.PLAN.value,
                },
                "execution_waves": [],
            }

        return {
            "artifacts": {PhaseName.PLAN.value: phase_artifacts},
            "current_phase": PhaseName.PLAN.value,
            "status": "running",
            "prompt_request": None,
            "execution_waves": execution_waves,
        }

    except Exception as e:
        logger.error(f"[{work_id}] PLAN phase failed: {e}", exc_info=True)
        return {
            "artifacts": {PhaseName.PLAN.value: {}},
            "current_phase": PhaseName.PLAN.value,
            "status": "running",
            "prompt_request": {
                "message": f"PLAN phase failed: {e}",
                "phase": PhaseName.PLAN.value,
            },
        }


_registry = get_registry()
_registry.register(
    name=PhaseName.PLAN.value,
    call_fn=call_plan,
    build_agent_fn=build_plan_agent,
    description="Define the technical architecture and plan",
)
