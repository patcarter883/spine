"""SPINE project-level adversarial review pipeline.

A single holistic red-team review of the completed project: integration
gaps, requirements not truly implemented end-to-end, hidden assumptions,
and cross-member contradictions.

Graph topology mirrors adversarial_subgraph::

    START → review_directive → adversarial_agent → save_review_result → END

Persisted result: ``.spine/project/{id}/project_review.json``
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from spine.agents.context import SpineContext
from spine.agents.plan_do import run_plan_node
from spine.agents.project_review_agent import build_project_review_agent
from spine.agents.retry import ainvoke_with_retry
from spine.config import SpineConfig
from spine.models.enums import PhaseName
from spine.persistence.project_store import ProjectStore
from spine.project.aggregator import aggregate_project_coverage

logger = logging.getLogger(__name__)


# ── State ────────────────────────────────────────────────────────────────


class ProjectReviewState(TypedDict, total=False):
    """State for the project adversarial review graph."""

    project_id: str
    workspace_root: str
    project_path: str        # ProjectStore base path (from SpineConfig.project_path)
    spec_dict: dict
    coverage: dict
    member_states: dict
    project_verification: dict | None

    review_directive: dict | None
    review_result: dict | None
    run_at: str


# ── Nodes ────────────────────────────────────────────────────────────────


async def _review_directive_node(
    state: ProjectReviewState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """No-tool planning step — identify attack angles before the do pass."""
    project_id = state.get("project_id", "unknown")
    workspace_root = state.get("workspace_root", ".")
    spec_dict = state.get("spec_dict", {})
    n_reqs = len(spec_dict.get("requirements", []))
    n_phases = len(spec_dict.get("roadmap", {}).get("phases", []))
    n_members = len(spec_dict.get("member_work_ids", []))

    task = (
        f"Plan a project-level adversarial review for '{project_id}' "
        f"({n_reqs} requirements, {n_phases} roadmap phases, {n_members} work items). "
        "The do node will call read_project_evidence and then write_project_review. "
        "Identify the most important attack angles: integration gaps, unverified "
        "requirements, shared-file conflicts, and cross-cutting concerns."
    )

    pseudo_state: dict = {
        "work_id": f"project-review-{project_id}",
        "workspace_root": workspace_root,
        "artifacts": {},
        "work_type": "critical_task",
    }

    directive = await run_plan_node(
        state=pseudo_state,
        config=config,
        phase_path=PhaseName.PROJECT_REVIEW.value,
        task_description=task,
        role_hint=f"adversarial reviewer for project '{project_id}'",
    )
    logger.info(
        "[project-review] Directive: approach=%r", directive.approach[:80]
    )
    return {"review_directive": directive.model_dump()}


async def _adversarial_agent_node(
    state: ProjectReviewState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Run the adversarial review agent."""
    project_id = state.get("project_id", "unknown")
    workspace_root = state.get("workspace_root", ".")

    logger.info("[project-review] Adversarial agent starting for '%s'", project_id)

    try:
        from spine.agents.plan_do import directive_from_state, format_directive_for_prompt

        agent = build_project_review_agent(
            project_id=project_id,
            spec_dict=state.get("spec_dict", {}),
            coverage=state.get("coverage", {}),
            member_states=state.get("member_states", {}),
            project_verification=state.get("project_verification"),
            workspace_root=workspace_root,
            config=config,
        )

        ctx = SpineContext(
            work_id=f"project-review-{project_id}",
            phase=PhaseName.PROJECT_REVIEW.value,
            workspace_root=workspace_root,
        )

        directive_block = format_directive_for_prompt(
            directive_from_state(dict(state), "review_directive")
        )
        prompt = (
            (directive_block + "\n\n" if directive_block else "")
            + "Call ``read_project_evidence`` first to load all project evidence, "
            "then red-team the project thoroughly.  Call ``write_project_review`` "
            "with your findings and verdict when done."
        )

        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.PROJECT_REVIEW.value,
            work_id=f"project-review-{project_id}",
            work_type="critical_task",
            context=ctx,
        )

        # Read the written result back from disk
        review_path = (
            Path(workspace_root)
            / ".spine"
            / "project"
            / project_id
            / "project_review.json"
        )
        review_result: dict | None = None
        if review_path.exists():
            try:
                review_result = json.loads(review_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.error("[project-review] Could not read result file: %s", exc)

        if review_result is None:
            logger.warning(
                "[project-review] Agent did not write a result — using fallback"
            )
            review_result = {
                "project_id": project_id,
                "verdict": "NEEDS_REVIEW",
                "summary": "Agent did not produce a structured review result.",
                "findings": [
                    {
                        "severity": "HIGH",
                        "category": "other",
                        "description": "Review agent failed to write structured output.",
                        "evidence": "project_review.json not found after agent invocation.",
                        "recommendation": "Re-run project review and check agent logs.",
                    }
                ],
            }

    except Exception as exc:
        logger.error(
            "[project-review] Adversarial agent failed: %s", exc, exc_info=True
        )
        review_result = {
            "project_id": project_id,
            "verdict": "NEEDS_REVIEW",
            "summary": f"Review agent raised an error: {exc}",
            "findings": [],
        }

    logger.info(
        "[project-review] Review complete: verdict=%s, %d finding(s)",
        review_result.get("verdict"),
        len(review_result.get("findings", [])),
    )
    return {"review_result": review_result}


async def _save_review_result_node(
    state: ProjectReviewState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Add run_at timestamp and persist to project store."""
    project_id = state.get("project_id", "unknown")
    project_path = state.get("project_path", ".spine/project")
    review_result = state.get("review_result") or {}
    run_at = datetime.now(tz=timezone.utc).isoformat()

    doc = {**review_result, "run_at": run_at}

    try:
        store = ProjectStore(base_path=project_path)
        store.save_result(project_id, "project_review.json", doc)
        logger.info(
            "[project-review] Saved project_review.json for '%s'", project_id
        )
    except Exception as exc:
        logger.error(
            "[project-review] Failed to persist result: %s", exc, exc_info=True
        )

    return {"run_at": run_at}


# ── Graph ────────────────────────────────────────────────────────────────


def _build_project_review_graph() -> Any:
    builder = StateGraph(ProjectReviewState)

    builder.add_node("review_directive", _review_directive_node)
    builder.add_node("adversarial_agent", _adversarial_agent_node)
    builder.add_node("save_review_result", _save_review_result_node)

    builder.add_edge(START, "review_directive")
    builder.add_edge("review_directive", "adversarial_agent")
    builder.add_edge("adversarial_agent", "save_review_result")
    builder.add_edge("save_review_result", END)

    return builder.compile(checkpointer=MemorySaver())


# ── Entry point ──────────────────────────────────────────────────────────


async def run_project_review(
    project_id: str,
    config: SpineConfig,
    workspace_root: str = ".",
) -> dict[str, Any]:
    """Run the project adversarial review pipeline.

    Args:
        project_id: The project slug.
        config: Loaded SpineConfig instance.
        workspace_root: Absolute path to the workspace root.

    Returns:
        The project_review.json document dict, or an error dict.
    """
    store = ProjectStore(base_path=config.project_path)
    spec = store.load_project(project_id)
    if spec is None:
        return {"error": f"Project '{project_id}' not found."}

    # Load aggregator coverage and member states
    coverage = await aggregate_project_coverage(spec, config)

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

    # Load prior verification result if available
    project_verification = store.load_result(project_id, "project_verification.json")

    initial_state: ProjectReviewState = {
        "project_id": project_id,
        "workspace_root": workspace_root,
        "project_path": config.project_path,
        "spec_dict": spec.model_dump(),
        "coverage": coverage,
        "member_states": member_states,
        "project_verification": project_verification,
    }

    graph = _build_project_review_graph()
    thread_config = {"configurable": {"thread_id": f"project-review-{project_id}"}}

    try:
        await graph.ainvoke(initial_state, config=thread_config)
    except Exception as exc:
        logger.error("[project-review] Graph invocation failed: %s", exc, exc_info=True)
        return {"error": str(exc)}

    result = store.load_result(project_id, "project_review.json")
    return result or {"error": "Review completed but result was not persisted."}
