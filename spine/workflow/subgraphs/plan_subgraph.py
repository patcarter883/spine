"""PLAN phase as a LangGraph subgraph.

The subgraph has two internal nodes:
1. ``run_agent`` — builds and invokes the plan Deep Agent.
2. ``save_artifacts`` — scans disk for artifacts, computes execution waves.

The plan agent produces both ``plan.md`` (narrative) and ``plan.json``
(structured with feature_slices). After the agent completes, the subgraph
reads ``plan.json`` and computes execution waves via the slice scheduler
so the downstream IMPLEMENT phase can use wave-based dispatch.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from spine.models.enums import PhaseName
from spine.workflow.subgraph_state import PlanSubgraphState
from spine.agents.plan_agent import build_plan_agent
from spine.agents.helpers import extract_response
from spine.agents.retry import ainvoke_with_retry
from spine.agents.context import build_context
from spine.agents.artifacts import (
    materialize_artifacts,
    materialize_phase_artifacts,
    scan_artifact_dir,
    _artifact_path,
)

logger = logging.getLogger(__name__)
_MAX_ARTIFACT_STATE_CHARS = 500


async def _run_plan_agent(
    state: PlanSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Run the plan Deep Agent within the subgraph.

    For quick workflows (no specification), the agent works from the
    work description via ``read_prior_artifacts``. For spec workflows,
    it reads the specification from disk.

    The agent is instructed to use ``write_structured_plan`` to produce
    both ``plan.md`` and ``plan.json`` with structured feature_slices.
    """
    work_id = state.get("work_id", "unknown")
    work_type = state.get("work_type", "")
    workspace_root = state.get("workspace_root", ".")

    logger.info(f"[{work_id}] PLAN subgraph: run_agent starting")

    try:
        agent = build_plan_agent(dict(state), config)
        materialize_artifacts(dict(state), workspace_root, work_id=work_id)

        has_spec = state.get("has_spec", False)
        spec_path = state.get("spec_path", "")

        plan_path = _artifact_path(work_id, PhaseName.PLAN.value)

        if has_spec and spec_path:
            spec_instruction = (
                "The full specification is available on disk at "
                f"`{spec_path}/specification.md` — read it carefully with "
                "`read_file` — your plan must implement exactly what the spec describes.\n\n"
            )
        else:
            spec_instruction = (
                "No prior specification exists (quick workflow). "
                "Work directly from the description returned by `read_prior_artifacts`.\n\n"
            )

        prompt = (
            "Create a detailed technical plan with structured feature slices.\n\n"
            + spec_instruction
            + "After completing your research, call `write_structured_plan` to produce "
            "both `plan.md` (narrative) and `plan.json` (structured JSON with feature_slices). "
            "The structured plan is REQUIRED — the downstream implementation phase needs the "
            "feature_slices array in `plan.json` to dispatch slice-implementer subagents.\n\n"
            f"Write the plan artifacts to `{plan_path}/`."
        )

        ctx = build_context(dict(state), PhaseName.PLAN)

        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.PLAN.value,
            work_id=work_id,
            work_type=work_type,
            context=ctx,
        )

        # ── Read plan.json from disk (written by write_structured_plan) ──
        plan_json_path = (
            Path(workspace_root) / ".spine" / "artifacts" / work_id / "plan" / "plan.json"
        )
        plan_json_str: str | None = None
        execution_waves: list[list[dict]] = []

        if plan_json_path.exists():
            try:
                raw = plan_json_path.read_text(encoding="utf-8")
                plan_data = json.loads(raw)
                plan_json_str = raw
                logger.info("[%s] Read plan.json (%d chars)", work_id, len(raw))

                # Compute execution waves from structured plan data
                wave_error: str | None = None
                execution_waves, wave_error = _compute_waves(plan_data, work_id)

                if wave_error is not None:
                    logger.warning(
                        "[%s] PLAN subgraph: wave computation error: %s",
                        work_id,
                        wave_error,
                    )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("[%s] Failed to read plan.json: %s", work_id, exc)

        return {
            "messages": result.get("messages", []),
            "agent_response": extract_response(result),
            "plan_json": plan_json_str,
            "execution_waves": execution_waves,
        }

    except Exception as e:
        logger.error(f"[{work_id}] PLAN subgraph agent failed: {e}", exc_info=True)
        return {
            "messages": [],
            "agent_response": f"Agent error: {e}",
            "phase_status": "error",
        }


def _compute_waves(
    plan_data: dict[str, Any],
    work_id: str,
) -> tuple[list[list[dict]], str | None]:
    """Compute execution waves from structured plan data.

    Args:
        plan_data: Parsed plan.json content.
        work_id: Work item ID for logging.

    Returns:
        ``(waves, error_message)``. On success, error_message is None.
    """
    try:
        from dataclasses import asdict

        from spine.workflow.slice_scheduler import FeatureSlice, compute_execution_waves
    except ImportError:
        logger.debug("[%s] slice_scheduler not available", work_id)
        return [], None

    raw_slices = plan_data.get("feature_slices")
    if not isinstance(raw_slices, list) or not raw_slices:
        logger.debug("[%s] plan.json has no feature_slices", work_id)
        return [], None

    try:
        scheduler_slices = [FeatureSlice.from_dict(sd) for sd in raw_slices]
        waves = compute_execution_waves(scheduler_slices)
        wave_dicts: list[list[dict]] = [[asdict(s) for s in wave] for wave in waves]
        logger.info(
            "[%s] Computed %d execution wave(s) with %d total slices",
            work_id,
            len(wave_dicts),
            sum(len(w) for w in wave_dicts),
        )
        return wave_dicts, None
    except (ValueError, KeyError, TypeError) as exc:
        return [], str(exc)


async def _save_plan_artifacts(
    state: PlanSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Save artifacts from the plan agent to disk and state.

    Reads ``plan.json`` from disk (written by the structured plan tool)
    and includes it alongside ``plan.md`` in the artifacts output.
    Computes execution waves from the structured plan data.
    """
    workspace_root = state.get("workspace_root", ".")
    work_id = state.get("work_id", "unknown")
    agent_response = state.get("agent_response", "")
    existing_phase_status = state.get("phase_status", "")
    plan_json_str = state.get("plan_json")
    execution_waves = state.get("execution_waves", [])

    if existing_phase_status in ("error", "needs_review"):
        return {
            "artifacts_output": {},
            "phase_status": existing_phase_status,
        }

    disk_artifacts = scan_artifact_dir(
        workspace_root,
        work_id,
        PhaseName.PLAN.value,
        max_preview_chars=_MAX_ARTIFACT_STATE_CHARS,
    )

    if not disk_artifacts and agent_response.strip():
        # No artifacts on disk — materialize plan.md from agent response.
        # plan.json was already written by write_structured_plan if the
        # agent used it, so it may still be on disk.
        phase_artifacts: dict[str, str] = {"plan.md": agent_response}

        if plan_json_str:
            phase_artifacts["plan.json"] = plan_json_str

        materialize_phase_artifacts(
            PhaseName.PLAN.value,
            phase_artifacts,
            workspace_root,
            work_id=work_id,
        )
        disk_artifacts = {
            "plan.md": agent_response[:_MAX_ARTIFACT_STATE_CHARS],
        }
        if plan_json_str:
            disk_artifacts["plan.json"] = plan_json_str[:_MAX_ARTIFACT_STATE_CHARS]

    # Merge plan.json into disk_artifacts if it exists on disk but wasn't
    # picked up by scan_artifact_dir (e.g. binary/JSON file filtering).
    if isinstance(disk_artifacts, dict) and "plan.json" not in disk_artifacts:
        plan_json_path = (
            Path(workspace_root) / ".spine" / "artifacts" / work_id / "plan" / "plan.json"
        )
        if plan_json_path.exists() and plan_json_str:
            disk_artifacts["plan.json"] = plan_json_str[:_MAX_ARTIFACT_STATE_CHARS]

    return {
        "artifacts_output": disk_artifacts,
        "phase_status": "success" if disk_artifacts else "needs_review",
        "execution_waves": execution_waves,
    }


def build_plan_subgraph() -> Any:
    """Build the PLAN phase subgraph."""
    builder = StateGraph(PlanSubgraphState)
    builder.add_node("run_agent", _run_plan_agent)
    builder.add_node("save_artifacts", _save_plan_artifacts)
    builder.add_edge(START, "run_agent")
    builder.add_edge("run_agent", "save_artifacts")
    builder.add_edge("save_artifacts", END)
    return builder
