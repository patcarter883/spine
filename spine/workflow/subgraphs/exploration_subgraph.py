"""SPECIFY/PLAN exploration subgraph — multi-node research loop.

Nodes:
- ``research_manager``: single LLM call to decide next topics or done
- ``explore``: researcher subagent (runs in parallel via Send API)
- ``aggregate``: deterministic merge — fan-in point for parallel results
- ``synthesize``: Deep Agent that writes the spec/plan artifact
- ``save_artifacts``: scans disk, materializes to state

Edges::

    START → research_manager
    research_manager → Send("explore", {topic}) × N  OR  → synthesize
    explore → aggregate
    aggregate → sufficiency check → research_manager (loop) OR synthesize
    synthesize → save_artifacts → END
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import json
import json as _json_mod
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from spine.models.enums import PhaseName
from spine.workflow.subgraph_state import ExplorationSubgraphState
from spine.agents.artifacts import (
    materialize_artifacts,
    materialize_phase_artifacts,
    scan_artifact_dir,
    artifact_path,
)
from spine.agents.helpers import extract_response
from spine.agents.retry import ainvoke_with_retry
from spine.agents.context import build_context

logger = logging.getLogger(__name__)
_MAX_ARTIFACT_STATE_CHARS = 500
_DEFAULT_MAX_ROUNDS = 3


# ── Node: research_manager ───────────────────────────────────────────────


async def _research_manager_node(
    state: ExplorationSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Call the research manager LLM to decide next topics or done."""
    from spine.agents.exploration_agents import run_research_manager

    result = await run_research_manager(dict(state), config)
    round_num = state.get("research_round", 0)
    return {
        **result,
        "research_round": round_num + 1,
    }


# ── Router: research_manager → explore (Send) or synthesize ─────────────


def _research_router(
    state: ExplorationSubgraphState,
) -> list[Send] | Literal["synthesize"]:
    """Fan-out to explore nodes via Send API, or proceed to synthesis.

    Returns a list of ``Send("explore", ...)`` objects when more
    research is needed, or the string ``"synthesize"`` when done.
    LangGraph executes all Send targets in parallel within the same
    super-step and waits for all to complete before proceeding.
    """
    decision = state.get("manager_decision", "done")
    topics: list[str] = state.get("topics", [])

    if decision == "done" or not topics:
        logger.info("Research complete — routing to synthesize")
        return "synthesize"  # type: ignore[return-value]

    sends = [Send("explore", {"topic": t}) for t in topics]
    logger.info("Dispatching %d explore node(s): %s", len(sends), topics)
    return sends


# ── Node: explore ───────────────────────────────────────────────────────


async def _explore_node(
    state: ExplorationSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Run a researcher subagent for one topic.

    The topic is injected into state by the Send API via
    ``Send("explore", {"topic": "area"})`` — it arrives as a state key,
    not a keyword argument.  The state merger injects ``topic`` alongside
    the normal subgraph state fields.
    """
    from spine.agents.exploration_agents import run_explore_node

    topic: str = state.get("topic", "")  # type: ignore[typeddict-unknown-key]
    return await run_explore_node(dict(state), config, topic=topic)


# ── Node: aggregate ────────────────────────────────────────────────────


async def _aggregate_node(
    state: ExplorationSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Fan-in point after all parallel explore nodes complete.

    Findings are already accumulated via ``operator.add`` on the
    ``findings`` field — no manual merging needed. This node exists
    as a routing checkpoint so the sufficiency gate can inspect the
    fully accumulated state.
    """
    findings = state.get("findings", [])
    logger.info("Aggregated %d findings total across all rounds", len(findings))
    return {}


# ── Router: aggregate → loop (research_manager) or done (synthesize) ──


def _sufficiency_router(
    state: ExplorationSubgraphState,
) -> Literal["loop", "done"]:
    """Check whether research is sufficient to proceed to synthesis.

    Returns ``"loop"`` to run another exploration round, or ``"done"``
    to exit the loop and begin synthesis.
    """
    decision = state.get("manager_decision", "done")
    max_rounds = state.get("max_rounds", _DEFAULT_MAX_ROUNDS)
    round_num = state.get("research_round", 0)

    if decision == "done":
        return "done"
    if round_num >= max_rounds:
        logger.info("Max rounds (%d) reached — proceeding to synthesis", max_rounds)
        return "done"
    return "loop"


# ── Node: synthesize (PLAN) ─────────────────────────────────────────────


async def _synthesize_plan(
    state: ExplorationSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Synthesize research findings into a plan.

    Uses the existing plan agent infrastructure — builds a Deep Agent
    with the ``write_structured_plan`` tool and research findings as context.
    Reads ``plan.json`` from disk after invocation and computes execution waves.
    """
    from spine.agents.plan_agent import build_plan_agent

    description = state.get("description", "")
    work_id = state.get("work_id", "unknown")
    work_type = state.get("work_type", "")
    workspace_root = state.get("workspace_root", ".")
    findings = state.get("findings", [])
    retry_count = state.get("retry_count", 0)
    feedback = state.get("feedback", [])

    logger.info(
        "[%s] Synthesize (plan): %d findings available, retry=%d",
        work_id,
        len(findings),
        retry_count,
    )

    try:
        agent = build_plan_agent(dict(state), config)
        materialize_artifacts(dict(state), workspace_root, work_id=work_id)

        findings_text = _format_findings(findings)
        rework_prefix = ""
        if retry_count > 0:
            rework_prefix = (
                "⚠ **REWORK PASS**: Your primary objective is to revise "
                "the prior plan. Address all points from the "
                "critic feedback.\n\n"
            )

        spec_path = artifact_path(work_id, PhaseName.SPECIFY.value) + "/specification.md"

        prompt = (
            f"{rework_prefix}Create a detailed technical plan with structured "
            f"feature slices, incorporating the codebase research findings below.\n\n"
            f"## Work Description\n{description}\n\n"
            f"## Codebase Research Findings\n{findings_text}\n\n"
            f"The full specification is available on disk at "
            f"`{spec_path}` — read it carefully with `read_file`.\n\n"
            f"After completing your research, call `write_structured_plan` to "
            f"produce both `plan.md` (narrative) and `plan.json` (structured "
            f"JSON with feature_slices).\n\n"
            f"Write the plan artifacts to `{artifact_path(work_id, PhaseName.PLAN.value)}/`."
        )

        if retry_count > 0 and feedback:
            feedback_text = "\n".join(
                f"- [{f.get('tier', 'unknown')}] {f.get('reason', '')}"
                for f in feedback
                if isinstance(f, dict)
            )
            prompt += f"\n\n## Previous Review Feedback\n{feedback_text}\n"

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
        logger.error("[%s] Synthesize (plan) failed: %s", work_id, e, exc_info=True)
        return {
            "messages": [],
            "agent_response": f"Synthesis error: {e}",
            "phase_status": "error",
        }


# ── Node: synthesize (SPECIFY) ──────────────────────────────────────────


async def _synthesize_specify(
    state: ExplorationSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Synthesize research findings into a specification.

    Uses the existing specify agent infrastructure — builds a Deep Agent
    with the ``write_specification`` tool and research findings as context.
    """
    from spine.agents.specify_agent import build_specify_agent

    description = state.get("description", "")
    work_id = state.get("work_id", "unknown")
    work_type = state.get("work_type", "")
    workspace_root = state.get("workspace_root", ".")
    findings = state.get("findings", [])
    retry_count = state.get("retry_count", 0)
    feedback = state.get("feedback", [])

    logger.info(
        "[%s] Synthesize (specify): %d findings available, retry=%d",
        work_id,
        len(findings),
        retry_count,
    )

    try:
        agent = build_specify_agent(dict(state), config)
        materialize_artifacts(dict(state), workspace_root, work_id=work_id)

        findings_text = _format_findings(findings)
        rework_prefix = ""
        if retry_count > 0:
            rework_prefix = (
                "⚠ **REWORK PASS**: Your primary objective is to revise "
                "the prior specification. Address all points from the "
                "critic feedback.\n\n"
            )

        prompt = (
            f"{rework_prefix}Create a detailed specification for the "
            f"following work, incorporating the codebase research "
            f"findings below.\n\n"
            f"## Work Description\n{description}\n\n"
            f"## Codebase Research Findings\n{findings_text}\n\n"
            f"Write the specification to "
            f"`{artifact_path(work_id, PhaseName.SPECIFY.value)}/specification.md` "
            f"using `write_file`."
        )

        if retry_count > 0 and feedback:
            feedback_text = "\n".join(
                f"- [{f.get('tier', 'unknown')}] {f.get('reason', '')}"
                for f in feedback
                if isinstance(f, dict)
            )
            prompt += f"\n\n## Previous Review Feedback\n{feedback_text}\n"

        ctx = build_context(dict(state), PhaseName.SPECIFY)
        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.SPECIFY.value,
            work_id=work_id,
            work_type=work_type,
            context=ctx,
        )

        return {
            "messages": result.get("messages", []),
            "agent_response": extract_response(result),
        }

    except Exception as e:
        logger.error("[%s] Synthesize (specify) failed: %s", work_id, e, exc_info=True)
        return {
            "messages": [],
            "agent_response": f"Synthesis error: {e}",
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


def _format_findings(findings: list[dict]) -> str:
    """Format accumulated findings for the synthesizer prompt.

    Keeps individual findings compact — the synthesizer can read files
    from disk if more detail is needed.
    """
    if not findings:
        return "(no codebase research was performed)"
    parts: list[str] = []
    for i, f in enumerate(findings):
        if isinstance(f, dict):
            topic = f.get("topic", "")
            summary = f.get("summary", "")
            patterns = f.get("patterns", [])
            file_map = f.get("file_map", {})
            deps = f.get("dependencies", [])
            header = f"### Finding {i + 1}"
            if topic:
                header += f" — Topic: {topic}"
            parts.append(f"{header}\n{summary}")
            if patterns:
                parts.append(f"Patterns: {', '.join(patterns)}")
            if file_map:
                parts.append(f"Key files: {_json_mod.dumps(file_map)}")
            if deps:
                parts.append(f"Dependencies: {', '.join(deps)}")
    return "\n\n".join(parts)


# ── Node: save_artifacts ────────────────────────────────────────────────


async def _save_exploration_artifacts(
    state: ExplorationSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Save artifacts from the exploration subgraph to disk and state."""
    workspace_root = state.get("workspace_root", ".")
    work_id = state.get("work_id", "unknown")
    phase = state.get("phase")
    if phase is None:
        raise ValueError(
            "Exploration subgraph state missing 'phase' key. "
            "This indicates a state mapper configuration error."
        )
    agent_response = state.get("agent_response", "")
    existing_phase_status = state.get("phase_status", "")

    if existing_phase_status in ("error", "needs_review"):
        return {
            "artifacts_output": {},
            "phase_status": existing_phase_status,
        }

    disk_artifacts = scan_artifact_dir(
        workspace_root,
        work_id,
        phase,
        max_preview_chars=_MAX_ARTIFACT_STATE_CHARS,
    )

    if not disk_artifacts and agent_response.strip():
        artifact_name = "specification.md" if PhaseName(phase) == PhaseName.SPECIFY else "plan.md"
        materialize_phase_artifacts(
            phase,
            {artifact_name: agent_response},
            workspace_root,
            work_id=work_id,
        )
        disk_artifacts = {artifact_name: agent_response[:_MAX_ARTIFACT_STATE_CHARS]}

    # PLAN-specific: merge plan.json into disk_artifacts if not already present
    plan_json_str = state.get("plan_json")
    execution_waves = state.get("execution_waves", [])

    if isinstance(disk_artifacts, dict) and "plan.json" not in disk_artifacts:
        plan_json_path = (
            Path(workspace_root) / ".spine" / "artifacts" / work_id / "plan" / "plan.json"
        )
        if plan_json_path.exists() and plan_json_str:
            disk_artifacts["plan.json"] = plan_json_str[:_MAX_ARTIFACT_STATE_CHARS]

    result: dict[str, Any] = {
        "artifacts_output": disk_artifacts,
        "phase_status": "success" if disk_artifacts else "needs_review",
    }

    if execution_waves:
        result["execution_waves"] = execution_waves

    return result


# ── Builder ──────────────────────────────────────────────────────────────


def build_exploration_subgraph(
    phase: str,
    max_rounds: int = _DEFAULT_MAX_ROUNDS,
) -> Any:
    """Build the multi-node exploration → synthesis subgraph.

    Args:
        phase: Which phase this subgraph is for (``"specify"`` or ``"plan"``).
        max_rounds: Maximum number of research_manager rounds (safety valve).

    Returns:
        Uncompiled StateGraph builder.  Call ``.compile()`` to get a
        runnable graph, or ``.compile(checkpointer=...)`` for per-phase
        checkpoint isolation.
    """
    synthesizer_map = {
        PhaseName.SPECIFY: _synthesize_specify,
        PhaseName.PLAN: _synthesize_plan,
    }
    synthesizer = synthesizer_map.get(PhaseName(phase))
    if synthesizer is None:
        raise ValueError(f"Unsupported phase for exploration subgraph: {phase!r}")

    builder = StateGraph(ExplorationSubgraphState)

    builder.add_node("research_manager", _research_manager_node)
    builder.add_node("explore", _explore_node)
    builder.add_node("aggregate", _aggregate_node)
    builder.add_node("synthesize", synthesizer)
    builder.add_node("save_artifacts", _save_exploration_artifacts)

    builder.add_edge(START, "research_manager")

    # research_manager → Send("explore", ...) or → synthesize
    builder.add_conditional_edges(
        "research_manager",
        _research_router,
        {"explore": "explore", "synthesize": "synthesize"},
    )

    # Explore → aggregate (fan-in — LangGraph waits for ALL Send targets)
    builder.add_edge("explore", "aggregate")

    # Aggregate → loop to research_manager or done → synthesize
    builder.add_conditional_edges(
        "aggregate",
        _sufficiency_router,
        {"loop": "research_manager", "done": "synthesize"},
    )

    builder.add_edge("synthesize", "save_artifacts")
    builder.add_edge("save_artifacts", END)

    return builder
