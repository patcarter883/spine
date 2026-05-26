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
from spine.agents.garbage_collector import calculate_safe_eviction
from spine.exceptions import CriticalContractFailure

from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)
_MAX_ARTIFACT_STATE_CHARS = 500
_DEFAULT_MAX_ROUNDS = 3


# ── Node: pre_research_gate ─────────────────────────────────────────────


async def _pre_research_gate(
    state: ExplorationSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Classify + recall once at entry. Sets fields used by ``_gate_router``.

    Replaces the round-0 classify+recall block previously embedded in
    ``_research_manager_node`` so the routing decision (skip exploration
    vs. enter the loop) happens BEFORE we spend a round on the research
    manager LLM call.

    Fail-open: if classification or recall throws, return empty fields so
    the router falls through to the existing exploration loop.
    """
    from spine.agents.classification import classify_task
    from spine.agents.tools.recall_tool import RecallTool
    from spine.config import SpineConfig

    description = state.get("description", "")
    work_id = state.get("work_id", "unknown")
    phase = state.get("phase", "")

    if not description:
        logger.info("[%s] pre_research_gate: no description — fall through", work_id)
        return {"classification_confidence": 0.0}

    try:
        classification = await classify_task(description, config)
        task_category = classification.category
        confidence = float(classification.confidence)
        logger.info(
            "[%s] pre_research_gate: %s confidence=%.2f",
            work_id, task_category, confidence,
        )
    except Exception as exc:
        logger.warning("[%s] pre_research_gate: classify failed — %s", work_id, exc)
        return {"classification_confidence": 0.0}

    retrieved: list[dict] = []
    try:
        cfg = SpineConfig.load()
        recall = RecallTool(db_path=cfg.checkpoint_path)
        # Only fetch full raw_code for SPECIFY (synth needs it inline). For
        # PLAN we always re-run the exploration loop, so summaries are
        # plenty — the loop builds its own context.
        summaries_only = phase != PhaseName.SPECIFY.value
        result_text = await recall._arun(
            query=description,
            k=cfg.recall_k,
            task_category=task_category,
            max_tokens=cfg.specify_context_token_budget,
            summaries_only=summaries_only,
        )
        retrieved = json.loads(result_text).get("results", [])
        logger.info(
            "[%s] pre_research_gate: recall returned %d chunks (summaries_only=%s)",
            work_id, len(retrieved), summaries_only,
        )
    except Exception as exc:
        logger.warning("[%s] pre_research_gate: recall failed — %s", work_id, exc)

    return {
        "task_category": task_category,
        "classification_confidence": confidence,
        "retrieved_context": retrieved,
    }


def _gate_router(
    state: ExplorationSubgraphState,
) -> Literal["skip_to_synth", "explore"]:
    """High-confidence + sufficient hits → synthesize directly.

    Only SPECIFY gets the short-circuit; PLAN always runs the loop
    (it needs the spec plus broader codebase research).
    """
    from spine.config import SpineConfig

    phase = state.get("phase", "")
    if phase != PhaseName.SPECIFY.value:
        return "explore"

    cfg = SpineConfig.load()
    confidence = float(state.get("classification_confidence", 0.0) or 0.0)
    hits = len(state.get("retrieved_context") or [])
    if confidence >= cfg.recall_gate_confidence and hits >= cfg.recall_gate_min_hits:
        logger.info(
            "Recall gate firing: confidence=%.2f hits=%d → skip exploration",
            confidence, hits,
        )
        return "skip_to_synth"
    return "explore"


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

    Raises ``CriticalContractFailure`` if structured data transfer from
    the research_manager is incomplete — silent defaults to ``"done"``
    mask upstream bugs that cause missing research.
    """
    decision = state.get("manager_decision")

    if decision is None:
        raise CriticalContractFailure(
            phase="exploration",
            reason="manager_decision is missing from state — "
                   "the research_manager_node did not write structured output. "
                   "This indicates a model invocation failure in the manager node.",
        )

    if decision == "done":
        logger.info("Research complete — routing to synthesize")
        return "synthesize"  # type: ignore[return-value]

    if decision != "explore":
        raise CriticalContractFailure(
            phase="exploration",
            reason=f"manager_decision has unexpected value {decision!r} — "
                   f"expected 'explore' or 'done'. The research_manager_node "
                   f"produced invalid structured output.",
        )

    topics: list[str] = state.get("topics", [])
    if not topics:
        raise CriticalContractFailure(
            phase="exploration",
            reason="manager_decision is 'explore' but topics list is empty — "
                   "the research_manager_node produced malformed structured output.",
        )

    findings: list[dict] = state.get("findings", [])
    explored_topics: set[str] = {
        f.get("topic", "") for f in findings if isinstance(f, dict) and f.get("topic")
    }

    new_topics = [t for t in topics if t not in explored_topics]
    if not new_topics:
        logger.info("All topics already explored — routing to synthesize")
        return "synthesize"  # type: ignore[return-value]

    sends = [Send("explore", {"topic": t, "phase": state.get("phase", "")}) for t in new_topics]
    logger.info("Dispatching %d explore node(s): %s", len(sends), new_topics)
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

        scratchpad = state.get("scratchpad", "")
        if scratchpad:
            prompt += f"\n\n## Working Memory Scratchpad\n{scratchpad}\n"

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

        # When the pre_research_gate fired (high confidence + sufficient
        # hits), it skips the exploration loop and we synthesize directly
        # from the recalled chunks.  Inject the raw per-symbol bodies
        # (tree-sitter-sliced — single function bodies, not whole files)
        # up to the configured token budget.
        recall_section = _format_retrieved_context(
            state.get("retrieved_context") or []
        )

        prompt = (
            f"{rework_prefix}Create a detailed specification for the "
            f"following work, incorporating the codebase research "
            f"findings below.\n\n"
            f"## Work Description\n{description}\n\n"
            f"## Codebase Research Findings\n{findings_text}\n\n"
            f"{recall_section}"
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

        scratchpad = state.get("scratchpad", "")
        if scratchpad:
            prompt += f"\n\n## Working Memory Scratchpad\n{scratchpad}\n"

        ctx = build_context(dict(state), PhaseName.SPECIFY)
        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name=PhaseName.SPECIFY.value,
            work_id=work_id,
            work_type=work_type,
            context=ctx,
        )

        # ── Read specification.json from disk (written by write_specification) ──
        spec_json_path = (
            Path(workspace_root) / ".spine" / "artifacts" / work_id / "specify" / "specification.json"
        )
        spec_json_str: str | None = None

        if spec_json_path.exists():
            try:
                spec_json_str = spec_json_path.read_text(encoding="utf-8")
                logger.info("[%s] Read specification.json (%d chars)", work_id, len(spec_json_str))
            except (OSError) as exc:
                logger.warning("[%s] Failed to read specification.json: %s", work_id, exc)

        return {
            "messages": result.get("messages", []),
            "agent_response": extract_response(result),
            "specification_json": spec_json_str,
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


def _count_tokens(text: str) -> int:
    """Best-effort token count via tiktoken with a 4-char-per-token fallback."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


def _format_retrieved_context(chunks: list[dict]) -> str:
    """Render recalled chunks as code blocks, capped by ``specify_context_token_budget``.

    Each chunk is a tree-sitter-sliced symbol body (Phase 1), so there is
    no per-chunk truncation — we simply stop appending when adding the
    next chunk would exceed the budget.
    """
    if not chunks:
        return ""

    from spine.config import SpineConfig

    budget = SpineConfig.load().specify_context_token_budget
    parts: list[str] = ["## Retrieved Codebase Context\n"]
    used = _count_tokens(parts[0])

    for i, chunk in enumerate(chunks, 1):
        symbol = chunk.get("symbol_name", "unknown")
        file_path = chunk.get("file_path", "unknown")
        lang = chunk.get("lang", "")
        raw = chunk.get("raw_code", "") or ""
        if not raw:
            # Summaries-only result — fall back to the enriched_summary.
            summary = chunk.get("enriched_summary", "")
            if not summary:
                continue
            block = f"### Chunk {i}: {symbol} ({file_path})\n{summary}\n\n"
        else:
            block = (
                f"### Chunk {i}: {symbol} ({file_path})\n"
                f"```{lang}\n{raw}\n```\n\n"
            )

        block_tokens = _count_tokens(block)
        if used + block_tokens > budget:
            logger.info(
                "Context budget reached at chunk %d/%d (%d tokens)",
                i - 1, len(chunks), used,
            )
            break
        parts.append(block)
        used += block_tokens

    return "".join(parts)


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

    # Fail-closed: SPECIFY requires specification.json (fail-closed like plan)
    if phase == PhaseName.SPECIFY.value:
        spec_json_path = Path(workspace_root) / ".spine" / "artifacts" / work_id / "specify" / "specification.json"
        if spec_json_path.exists():
            try:
                raw = spec_json_path.read_text(encoding="utf-8")
                spec_data = json.loads(raw)
                if not isinstance(spec_data, dict):
                    raise CriticalContractFailure(
                        phase="specify",
                        reason="specification.json is not a JSON object",
                    )
                for key in ("title", "summary", "requirements"):
                    if key not in spec_data:
                        raise CriticalContractFailure(
                            phase="specify",
                            reason=f"specification.json missing required key '{key}' — "
                                   f"keys found: {list(spec_data.keys())}",
                        )
            except (json.JSONDecodeError, OSError) as exc:
                raise CriticalContractFailure(
                    phase="specify",
                    reason=f"specification.json is malformed or unreadable: {exc}",
                )
            except CriticalContractFailure:
                raise
            except Exception as exc:
                raise CriticalContractFailure(
                    phase="specify",
                    reason=f"specification.json validation error: {exc}",
                )
        else:
            raise CriticalContractFailure(
                phase="specify",
                reason="specification.json does not exist — "
                       "the specify agent did not produce structured output via write_specification. "
                       "This indicates a model invocation failure.",
            )

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

    # SPECIFY-specific: merge specification.json into disk_artifacts
    spec_json_str = state.get("specification_json")
    if isinstance(disk_artifacts, dict) and "specification.json" not in disk_artifacts:
        spec_json_path = (
            Path(workspace_root) / ".spine" / "artifacts" / work_id / "specify" / "specification.json"
        )
        if spec_json_path.exists() and spec_json_str:
            disk_artifacts["specification.json"] = spec_json_str[:_MAX_ARTIFACT_STATE_CHARS]

    result: dict[str, Any] = {
        "artifacts_output": disk_artifacts,
        "phase_status": "success" if disk_artifacts else "needs_review",
    }

    if execution_waves:
        result["execution_waves"] = execution_waves

    return result


# ── Node: eviction_check ───────────────────────────────────────────────


async def _eviction_check_node(
    state: ExplorationSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Evict old search history when commit_findings was called.

    Calls :func:`calculate_safe_eviction` to identify messages that can
    be removed without breaking parallel-tool references, and extracts
    committed findings from the Boundary AIMessage's tool_calls for the
    scratchpad accumulator.
    """
    messages = state.get("messages", [])
    evictions = calculate_safe_eviction(messages)

    if not evictions:
        return {}

    # Extract committed findings from the Boundary AIMessage's tool_calls
    scratchpad_entry = ""
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "commit_findings_and_clear_search":
                    args = tc.get("args", {})
                    note = args.get("note", "")
                    code = args.get("relevant_code", "")
                    scratchpad_entry += (
                        "## Saved Findings\n"
                        f"**Summary:** {note}\n\n"
                        f"**Key Code/Paths:**\n{code}\n\n"
                        "---\n"
                    )

    logger.info("Evicting %d messages from exploration context", len(evictions))
    return {
        "messages": evictions,
        "scratchpad": scratchpad_entry,
    }


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

    builder.add_node("pre_research_gate", _pre_research_gate)
    builder.add_node("research_manager", _research_manager_node)
    builder.add_node("explore", _explore_node)
    builder.add_node("aggregate", _aggregate_node)
    builder.add_node("eviction_check", _eviction_check_node)
    builder.add_node("synthesize", synthesizer)
    builder.add_node("save_artifacts", _save_exploration_artifacts)

    builder.add_edge(START, "pre_research_gate")

    # pre_research_gate → skip exploration (high confidence + hits) OR
    # fall through to the existing research_manager loop.
    builder.add_conditional_edges(
        "pre_research_gate",
        _gate_router,
        {"skip_to_synth": "synthesize", "explore": "research_manager"},
    )

    # research_manager → Send("explore", ...) or → synthesize
    builder.add_conditional_edges(
        "research_manager",
        _research_router,
        {"explore": "explore", "synthesize": "synthesize"},
    )

    # Explore → aggregate (fan-in — LangGraph waits for ALL Send targets)
    builder.add_edge("explore", "aggregate")

    # Aggregate → eviction_check → sufficiency router (loop or synthesize)
    builder.add_edge("aggregate", "eviction_check")
    builder.add_conditional_edges(
        "eviction_check",
        _sufficiency_router,
        {"loop": "research_manager", "done": "synthesize"},
    )

    builder.add_edge("synthesize", "save_artifacts")
    builder.add_edge("save_artifacts", END)

    return builder
