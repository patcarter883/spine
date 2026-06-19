"""SPINE project-level integration verification pipeline.

Confirms that each roadmap phase's requirements are truly satisfied by the
combined codebase — beyond what per-member VERIFY phases check.  The
aggregator's exact-string coverage is the cheap first pass; this layer
adds LLM judgment for cross-member integration gaps.

Graph topology::

    START → phase_verify_router (conditional)
      ↓  [Send("phase_verifier", {phase, ...}) × N — parallel]
    phase_verifier_node × N  (one per roadmap phase)
      ↓  [fan-in — all phases complete]
    synthesize_verify
      ↓
    save_verify_result
      ↓
    END

Persisted result: ``.spine/project/{id}/project_verification.json``
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from operator import add as _op_add
from pathlib import Path
from typing import Annotated, Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict

from spine.agents.project_verify_agent import build_project_verify_agent
from spine.agents.project_verify_tools import load_phase_verification_result
from spine.agents.context import SpineContext
from spine.agents.retry import ainvoke_with_retry
from spine.config import SpineConfig
from spine.models.enums import PhaseName
from spine.persistence.project_store import ProjectStore
from spine.project.aggregator import aggregate_project_coverage

logger = logging.getLogger(__name__)


# ── State ────────────────────────────────────────────────────────────────


class ProjectVerifyState(TypedDict, total=False):
    """State for the project verification graph."""

    project_id: str
    workspace_root: str
    project_path: str        # ProjectStore base path (from SpineConfig.project_path)
    spec_dict: dict          # ProjectSpec as dict
    coverage: dict           # aggregate_project_coverage output
    member_states: dict      # work_id → checkpoint state

    # Per-phase context (populated by the Send payload)
    phase: dict              # RoadmapPhase as dict
    coverage_phase: dict     # Aggregator coverage for this phase

    # Accumulated by parallel phase_verifier nodes (reducer: list append)
    phase_results: Annotated[list[dict], _op_add]

    # Set by synthesize_verify
    overall_verdict: str
    run_at: str


# ── Nodes ────────────────────────────────────────────────────────────────


def _phase_verify_router(
    state: ProjectVerifyState,
) -> list[Send] | Literal["synthesize_verify"]:
    """Fan-out to per-phase verifier nodes via Send API."""
    spec_dict = state.get("spec_dict", {})
    phases = spec_dict.get("roadmap", {}).get("phases", [])
    coverage = state.get("coverage", {})

    if not phases:
        logger.warning("[project-verify] No roadmap phases to verify — skipping.")
        return "synthesize_verify"

    # Build per-phase coverage index
    coverage_by_phase = {p["id"]: p for p in coverage.get("phases", [])}

    sends: list[Send] = []
    for phase in phases:
        phase_id = phase.get("id", "")
        sends.append(
            Send(
                "phase_verifier",
                {
                    "project_id": state.get("project_id"),
                    "workspace_root": state.get("workspace_root"),
                    "project_path": state.get("project_path"),
                    "spec_dict": spec_dict,
                    "member_states": state.get("member_states", {}),
                    "phase": phase,
                    "coverage_phase": coverage_by_phase.get(phase_id, {}),
                },
            )
        )
    logger.info("[project-verify] Dispatching %d phase verifier(s)", len(sends))
    return sends


async def _phase_verifier_node(
    state: ProjectVerifyState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Run an integration verifier agent for one roadmap phase."""
    project_id = state.get("project_id", "unknown")
    workspace_root = state.get("workspace_root", ".")
    phase = state.get("phase", {})
    phase_id = phase.get("id", "unknown")
    coverage_phase = state.get("coverage_phase", {})
    member_states = state.get("member_states", {})

    logger.info("[project-verify] Phase '%s' verifier starting", phase_id)

    try:
        agent = build_project_verify_agent(
            project_id=project_id,
            phase_dict=phase,
            coverage_phase=coverage_phase,
            member_states=member_states,
            workspace_root=workspace_root,
            config=config,
        )

        ctx = SpineContext(
            work_id=f"project-verify-{project_id}-{phase_id}",
            phase=PhaseName.PROJECT_VERIFY.value,
            workspace_root=workspace_root,
        )

        prompt = (
            f"Verify roadmap phase '{phase_id}' — {phase.get('title', '')}.\n\n"
            "Call ``read_phase_evidence`` first, then review all evidence and call "
            "``write_phase_verification`` with your verdict."
        )

        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.PROJECT_VERIFY.value,
            work_id=f"project-verify-{project_id}-{phase_id}",
            work_type="task",
            context=ctx,
        )

        # Read the written result file
        phase_result = load_phase_verification_result(project_id, phase_id, workspace_root)
        if phase_result is None:
            logger.warning(
                "[project-verify] Phase '%s' agent did not write a result — marking FAILED",
                phase_id,
            )
            phase_result = {
                "phase_id": phase_id,
                "verdict": "FAILED",
                "summary": "Agent did not produce a structured verification result.",
                "requirement_results": [],
                "integration_gaps": ["No verification result written by agent."],
                "recommendations": ["Re-run project verify after investigating the agent log."],
            }

    except Exception as exc:
        logger.error("[project-verify] Phase '%s' verifier failed: %s", phase_id, exc, exc_info=True)
        phase_result = {
            "phase_id": phase_id,
            "verdict": "FAILED",
            "summary": f"Verifier agent raised an error: {exc}",
            "requirement_results": [],
            "integration_gaps": [str(exc)],
            "recommendations": [],
        }

    logger.info(
        "[project-verify] Phase '%s' verdict: %s",
        phase_id, phase_result.get("verdict"),
    )
    return {"phase_results": [phase_result]}


async def _synthesize_verify_node(
    state: ProjectVerifyState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Roll up per-phase verdicts into a project-level verdict."""
    phase_results: list[dict] = state.get("phase_results", [])
    run_at = datetime.now(tz=timezone.utc).isoformat()

    if not phase_results:
        return {
            "overall_verdict": "N/A",
            "run_at": run_at,
        }

    verdicts = [r.get("verdict", "FAILED") for r in phase_results]
    if all(v == "VERIFIED" for v in verdicts):
        overall = "VERIFIED"
    elif any(v == "VERIFIED" for v in verdicts):
        overall = "PARTIAL"
    else:
        overall = "FAILED"

    logger.info(
        "[project-verify] Overall verdict: %s (%d phases: %s)",
        overall, len(verdicts), verdicts,
    )
    return {"overall_verdict": overall, "run_at": run_at}


async def _save_verify_result_node(
    state: ProjectVerifyState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Persist project_verification.json to the project store."""
    project_id = state.get("project_id", "unknown")
    project_path = state.get("project_path", ".spine/project")
    phase_results = state.get("phase_results", [])
    overall_verdict = state.get("overall_verdict", "FAILED")
    run_at = state.get("run_at", datetime.now(tz=timezone.utc).isoformat())

    doc: dict[str, Any] = {
        "project_id": project_id,
        "overall_verdict": overall_verdict,
        "run_at": run_at,
        "phase_results": phase_results,
        "summary": {
            "verified": sum(1 for r in phase_results if r.get("verdict") == "VERIFIED"),
            "partial": sum(1 for r in phase_results if r.get("verdict") == "PARTIAL"),
            "failed": sum(1 for r in phase_results if r.get("verdict") == "FAILED"),
            "total": len(phase_results),
        },
    }

    try:
        store = ProjectStore(base_path=project_path)
        store.save_result(project_id, "project_verification.json", doc)
        logger.info(
            "[project-verify] Saved project_verification.json for '%s'", project_id
        )
    except Exception as exc:
        logger.error("[project-verify] Failed to persist result: %s", exc, exc_info=True)

    return {}


# ── Graph ────────────────────────────────────────────────────────────────


def _build_project_verify_graph() -> Any:
    builder = StateGraph(ProjectVerifyState)

    builder.add_node("phase_verifier", _phase_verifier_node)
    builder.add_node("synthesize_verify", _synthesize_verify_node)
    builder.add_node("save_verify_result", _save_verify_result_node)

    builder.add_conditional_edges(
        START,
        _phase_verify_router,
        ["phase_verifier", "synthesize_verify"],
    )
    builder.add_edge("phase_verifier", "synthesize_verify")
    builder.add_edge("synthesize_verify", "save_verify_result")
    builder.add_edge("save_verify_result", END)

    return builder.compile(checkpointer=MemorySaver())


# ── Entry point ──────────────────────────────────────────────────────────


async def run_project_verify(
    project_id: str,
    config: SpineConfig,
    workspace_root: str = ".",
) -> dict[str, Any]:
    """Run the project integration verification pipeline.

    Loads the project spec and all member checkpoint states, then runs
    the parallel per-phase verifier graph.  The result is persisted to
    ``.spine/project/{project_id}/project_verification.json``.

    Args:
        project_id: The project slug.
        config: Loaded SpineConfig instance.
        workspace_root: Absolute path to the workspace root (default ``"."``).

    Returns:
        The project_verification.json document dict, or an error dict.
    """
    store = ProjectStore(base_path=config.project_path)
    spec = store.load_project(project_id)
    if spec is None:
        return {"error": f"Project '{project_id}' not found."}

    # Load aggregator coverage
    coverage = await aggregate_project_coverage(spec, config)

    # Load trimmed member checkpoint state (only the fields the verify tools read)
    from spine.persistence.checkpoint import CheckpointStore

    _KEEP = {"specification_json", "files_written", "verification_passed"}
    checkpoint_store = CheckpointStore(db_path=config.checkpoint_path)
    member_states: dict[str, Any] = {}
    try:
        for wid in spec.member_work_ids:
            raw = await checkpoint_store.get_state(wid)
            if isinstance(raw, dict):
                member_states[wid] = {k: v for k, v in raw.items() if k in _KEEP}
    finally:
        await checkpoint_store.close()

    initial_state: ProjectVerifyState = {
        "project_id": project_id,
        "workspace_root": workspace_root,
        "project_path": config.project_path,
        "spec_dict": spec.model_dump(),
        "coverage": coverage,
        "member_states": member_states,
        "phase_results": [],
    }

    graph = _build_project_verify_graph()
    thread_config = {"configurable": {"thread_id": f"project-verify-{project_id}"}}

    try:
        await graph.ainvoke(initial_state, config=thread_config)
    except Exception as exc:
        logger.error("[project-verify] Graph invocation failed: %s", exc, exc_info=True)
        return {"error": str(exc)}

    # Return the persisted result
    result = store.load_result(project_id, "project_verification.json")
    return result or {"error": "Verification completed but result was not persisted."}
