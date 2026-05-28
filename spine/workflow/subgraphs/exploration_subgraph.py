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
from langgraph.types import Command, Send

from spine.agents._tokens import count_tokens as _count_tokens

from spine.models.enums import PhaseName
from spine.workflow.subgraph_state import ExplorationSubgraphState
from spine.agents.artifacts import (
    materialize_artifacts,
    materialize_phase_artifacts,
    scan_artifact_dir,
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
# Cap on concurrent Send("explore", …) dispatches per research round.
# Without this, _research_router fans out one branch per topic — observed
# runs spawned ~9 simultaneous researchers, each re-fetching the same
# hot symbols and crossing the per-branch token budget. Topics beyond
# the cap are deferred: the manager will re-propose them next round
# (and _new_topics will skip ones already covered).
_MAX_PARALLEL_EXPLORES = 4


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
    (it needs the spec plus broader codebase research). On a rework
    pass (retry_count > 0) the short-circuit is disabled — the critic
    rejected the prior output, so re-synthesizing from the same recall
    chunks without re-running the research manager (with prior findings
    + critic feedback) would just repeat the work.
    """
    from spine.config import SpineConfig

    phase = state.get("phase", "")
    if phase != PhaseName.SPECIFY.value:
        return "explore"

    if int(state.get("retry_count", 0) or 0) > 0:
        logger.info(
            "Recall gate skipped: retry_count>0 — routing through research_manager "
            "with prior findings + critic feedback",
        )
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


# ── Node: topic_lookup ──────────────────────────────────────────────────


def _new_topics(state: ExplorationSubgraphState) -> list[str]:
    """Return the topics not yet represented in ``findings`` (round-stable).

    ``state.topics`` holds bare topic strings, but ``finding['topic']`` is
    stamped with the enriched form ("<topic> — recall symbols: …") by
    ``run_explore_node``. Comparing them raw therefore always reports the
    topic as un-explored. ``_normalise_topic`` strips the recall suffix and
    case/whitespace so the membership check actually reflects what was
    dispatched on prior rounds.

    Layered dedup:

    1. Exact normalised-string match against ``finding["topic"]`` —
       fast path for the manager re-emitting an identical topic.
    2. Content-word overlap against every prior topic — catches the
       local-model paraphrase pattern (trace 019e6e53: round 1 "How
       does the CLI entrypoint currently parse and handle command-line
       arguments?" ↔ round 2 "How does the command-line interface
       parse and handle flags?") that the exact filter misses because
       the strings differ word-for-word.
    """
    from spine.agents.exploration_agents import (
        _RECALL_SUFFIX_MARKER,
        _normalise_topic,
        _topics_near_duplicate,
    )

    def _strip_suffix(s: str) -> str:
        if s and _RECALL_SUFFIX_MARKER in s:
            return s.split(_RECALL_SUFFIX_MARKER, 1)[0]
        return s

    topics: list[str] = state.get("topics", [])
    findings: list[dict] = state.get("findings", [])
    # Strip the enriched recall suffix off prior topics before any
    # comparison — its tokens (file paths, symbol names) would otherwise
    # pollute the content-word fingerprint used for paraphrase dedup and
    # let near-duplicates slip through.
    prior_topics: list[str] = [
        _strip_suffix(f.get("topic", ""))
        for f in findings
        if isinstance(f, dict) and f.get("topic")
    ]
    explored: set[str] = {_normalise_topic(p) for p in prior_topics}
    explored.discard("")

    kept: list[str] = []
    for t in topics:
        if _normalise_topic(t) in explored:
            continue
        if any(_topics_near_duplicate(t, p) for p in prior_topics):
            logger.info(
                "topic dedup: dropping near-duplicate topic=%r (paraphrase of prior round)",
                t,
            )
            continue
        kept.append(t)
    return kept


def _enrich_topic(topic: str, hits: list[dict]) -> str:
    """Append the recalled symbol references to a topic string.

    The explore subagent gets this enriched string as its research target,
    which gives it concrete symbols (and file paths) to anchor MCP lookups
    against — much better than a bare natural-language topic.
    """
    if not hits:
        return topic
    refs = ", ".join(
        f"{h.get('symbol_name', '?')} ({h.get('file_path', '?')})"
        for h in hits
    )
    return f"{topic} — recall symbols: {refs}"


def _is_test_artifact(hit: dict) -> bool:
    """Return True if the recall hit points at a test file/function.

    Test-file matches dominated the false-positive symbol recalls in
    trace 019e6974: every other topic ended up anchored on a
    ``test_*`` function (e.g. ``test_config_nonexistent_file`` attached
    to a "How is CLI argument parsing done?" topic). Test symbols have
    rich docstrings that overlap with question wording, so the
    embedding similarity passes but the researcher learns nothing
    about the production code. Filter them out unless the topic
    explicitly mentions tests — which the caller can re-enable by
    passing ``allow_tests=True``.
    """
    path = (hit.get("file_path") or "").lower()
    if path.startswith("tests/") or "/tests/" in path or path.endswith("/conftest.py"):
        return True
    name = hit.get("symbol_name") or ""
    return name.startswith("test_") or name.startswith("Test")


def _topic_mentions_tests(topic: str) -> bool:
    """Cheap heuristic: only keep test hits when the topic is about tests."""
    t = topic.lower()
    return any(kw in t for kw in (" test", "tests ", "tests,", "tests."))


async def _topic_lookup_node(
    state: ExplorationSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """For each NEW topic, recall the top-K symbols above the configured
    similarity threshold and stash them under ``topic_recall_hits``.

    Runs between ``research_manager`` and ``_research_router``. Short-
    circuits to an empty hits dict when:
      - the manager decided ``"done"``,
      - or every topic from the manager has already been explored.

    Both ``topic_lookup_top_k`` and ``topic_lookup_min_similarity`` come
    from :class:`spine.config.SpineConfig`. Recall failures for a single
    topic do not abort the round — the topic is simply sent on without
    annotations.
    """
    from spine.agents.tools.recall_tool import RecallTool
    from spine.config import SpineConfig

    work_id = state.get("work_id", "unknown")
    decision = state.get("manager_decision")
    topics_in_state: list[str] = state.get("topics", [])
    findings_count = len(state.get("findings", []) or [])

    # NOTE: deliberately using logger.warning for the topic_lookup lifecycle
    # lines. The CLI never calls logging.basicConfig() (see work-item
    # 131f2f1e.md — pending), so the root logger sits at WARNING and any
    # INFO call is dropped silently. WARNING is the only level guaranteed
    # to reach stderr today. Drop back to INFO once configure_logging()
    # lands.
    logger.warning(
        "[%s] topic_lookup: ENTER decision=%s state_topics=%d findings=%d",
        work_id, decision, len(topics_in_state), findings_count,
    )

    if decision != "explore":
        logger.warning(
            "[%s] topic_lookup: SKIP — manager_decision=%r (not 'explore')",
            work_id, decision,
        )
        return {"topic_recall_hits": {}}

    new_topics = _new_topics(state)
    if not new_topics:
        logger.warning(
            "[%s] topic_lookup: SKIP — every topic %s already in findings",
            work_id, topics_in_state,
        )
        return {"topic_recall_hits": {}}

    cfg = SpineConfig.load()
    top_k = max(1, int(cfg.topic_lookup_top_k))
    min_sim = float(cfg.topic_lookup_min_similarity)
    # Request more than top_k so we still have ≥top_k after threshold filtering.
    request_k = max(top_k * 3, cfg.recall_k)

    recall = RecallTool(
        db_path=cfg.checkpoint_path,
        embedding_provider=cfg.embedding_provider,
    )
    # NOTE: deliberately NOT passing task_category. CATEGORY_TO_SYMBOL_TYPES
    # in spine/agents/classification.py maps every non-Generic category to
    # symbol_types like "endpoint", "component", "middleware" — but the
    # indexer only writes "function" and "class". Filtering by category
    # eliminates 100% of rows and gives 0 results for every topic.
    # Semantic similarity is already the right discriminator here.
    hits_map: dict[str, list[dict]] = {}

    logger.warning(
        "[%s] topic_lookup: searching %d topic(s) — request_k=%d min_sim=%.2f top_k=%d "
        "db=%s",
        work_id, len(new_topics), request_k, min_sim, top_k, cfg.checkpoint_path,
    )

    for topic in new_topics:
        try:
            raw = await recall._arun(
                query=topic,
                k=request_k,
                task_category=None,
                max_tokens=cfg.specify_context_token_budget,
                summaries_only=True,
            )
            results = json.loads(raw).get("results", []) or []
        except Exception as exc:
            logger.warning(
                "[%s] topic_lookup: recall FAILED for topic=%r — %s",
                work_id, topic, exc,
            )
            hits_map[topic] = []
            continue

        sims = [float(r.get("similarity", 0.0)) for r in results if isinstance(r, dict)]
        if not results:
            logger.warning(
                "[%s] topic_lookup: topic=%r — recall returned 0 results "
                "(vector store empty or no matches at all)",
                work_id, topic,
            )
            hits_map[topic] = []
            continue

        filtered = [
            r for r in results
            if isinstance(r, dict) and float(r.get("similarity", 0.0)) >= min_sim
        ]
        # Drop test-file matches unless the topic itself is about tests.
        # See _is_test_artifact for the trace 019e6974 evidence.
        if not _topic_mentions_tests(topic):
            before_test_filter = len(filtered)
            filtered = [h for h in filtered if not _is_test_artifact(h)]
            test_dropped = before_test_filter - len(filtered)
        else:
            test_dropped = 0
        filtered.sort(
            key=lambda r: float(r.get("similarity", 0.0)),
            reverse=True,
        )
        kept = filtered[:top_k]
        hits_map[topic] = kept

        if not filtered:
            logger.warning(
                "[%s] topic_lookup: topic=%r — %d raw hits, similarities=%s, "
                "ALL below threshold %.2f (max=%.3f); test_dropped=%d",
                work_id, topic, len(results),
                [f"{s:.3f}" for s in sims], min_sim,
                max(sims) if sims else 0.0, test_dropped,
            )
        else:
            logger.warning(
                "[%s] topic_lookup: topic=%r — %d raw hits, %d ≥%.2f, "
                "test_dropped=%d, keeping top-%d: %s",
                work_id, topic, len(results), len(filtered) + test_dropped,
                min_sim, test_dropped, len(kept),
                [
                    f"{h.get('symbol_name', '?')}({h.get('file_path', '?')})"
                    f"@{float(h.get('similarity', 0.0)):.3f}"
                    for h in kept
                ],
            )

    total_kept = sum(len(v) for v in hits_map.values())
    logger.warning(
        "[%s] topic_lookup: EXIT — %d topic(s) processed, %d total hit(s) attached",
        work_id, len(hits_map), total_kept,
    )

    return {"topic_recall_hits": hits_map}


# ── Router: topic_lookup → explore (Send) or synthesize ─────────────────


def _research_router(
    state: ExplorationSubgraphState,
) -> list[Send] | Literal["synthesize"]:
    """Fan-out to explore nodes via Send API, or proceed to synthesis.

    Returns a list of ``Send("explore", ...)`` objects when more
    research is needed, or the string ``"synthesize"`` when done.
    LangGraph executes all Send targets in parallel within the same
    super-step and waits for all to complete before proceeding.

    Each Send's ``topic`` arg is enriched with the recall hits gathered
    by the upstream ``topic_lookup`` node so the explore subagent
    receives concrete symbol references alongside the topic.

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

    new_topics = _new_topics(state)
    if not new_topics:
        logger.info("All topics already explored — routing to synthesize")
        return "synthesize"  # type: ignore[return-value]

    hits_map: dict[str, list[dict]] = state.get("topic_recall_hits") or {}
    phase = state.get("phase", "")
    capped_topics = new_topics[:_MAX_PARALLEL_EXPLORES]
    deferred = new_topics[_MAX_PARALLEL_EXPLORES:]
    sends = [
        Send(
            # Two-node researcher: explore_do (tools) → summarise (no tools).
            # The plain edge explore_do→summarise threads each parallel
            # branch's evidence dossier to its own summariser before fan-in.
            "explore_do",
            {
                "topic": _enrich_topic(t, hits_map.get(t, [])),
                "phase": phase,
            },
        )
        for t in capped_topics
    ]
    if deferred:
        logger.info(
            "Dispatching %d explore node(s) (deferring %d to next round): %s | deferred=%s",
            len(sends), len(deferred), capped_topics, deferred,
        )
    else:
        logger.info("Dispatching %d explore node(s): %s", len(sends), capped_topics)
    return sends


# ── Nodes: explore_do (tools) → summarise (no tools) ───────────────────


async def _explore_do_node(
    state: ExplorationSubgraphState,
    config: RunnableConfig | None = None,
) -> Command:
    """Run the researcher's tool-using loop for one topic, then dispatch to summarise.

    The topic is injected into state by the Send API via
    ``Send("explore_do", {"topic": "area"})``. Returns a LangGraph
    ``Command`` whose ``goto`` dispatches a per-branch ``Send`` to the
    summarise node carrying that branch's ``exploration_evidence`` —
    this is required because parallel Send branches share the
    subgraph's channel space, so a plain ``return {"exploration_evidence": ...}``
    would collide N writes into the LastValue channel and crash with
    ``InvalidUpdateError``.

    The ``update`` dict is restricted to channels with reducers
    (``read_cache`` has ``_merge_read_cache``) so concurrent writes from
    sibling branches merge cleanly.
    """
    from spine.agents.exploration_agents import run_explore_do_node

    topic: str = state.get("topic", "")  # type: ignore[typeddict-unknown-key]
    raw = await run_explore_do_node(dict(state), config, topic=topic)

    evidence = raw.get("exploration_evidence") or {}
    cache = raw.get("read_cache")

    update: dict[str, Any] = {}
    if cache:
        update["read_cache"] = cache

    # Carry the BaseSubgraphState fields the downstream node will need
    # to resolve the researcher model and log against the right work_id.
    send_payload: dict[str, Any] = {
        "exploration_evidence": evidence,
        "topic": evidence.get("topic") or topic,
        "phase": state.get("phase"),
        "work_id": state.get("work_id"),
        "work_type": state.get("work_type", ""),
        "workspace_root": state.get("workspace_root", "."),
        "spec_path": state.get("spec_path", ""),
    }
    return Command(update=update, goto=Send("summarise", send_payload))


async def _summarise_node(
    state: ExplorationSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Convert the per-branch evidence dossier into a ResearchFindings.

    Runs with no tools attached — the model's only job is structural
    conversion of evidence to findings, aligned to the original topic.
    This split exists so smaller models stop oscillating between tool
    use and structured-output reasoning within a single node.
    """
    from spine.agents.exploration_agents import run_summarise_node

    return await run_summarise_node(dict(state), config)


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
    from spine.agents.plan_agent import build_plan_synthesizer

    description = state.get("description", "")
    work_id = state.get("work_id", "unknown")
    work_type = state.get("work_type", "")
    workspace_root = state.get("workspace_root", ".")
    findings = state.get("findings", [])
    retry_count = state.get("retry_count", 0)
    feedback = state.get("feedback", [])
    last_critic_review = state.get("last_critic_review") or {}

    logger.info(
        "[%s] Synthesize (plan): %d findings available, retry=%d",
        work_id,
        len(findings),
        retry_count,
    )

    try:
        agent = build_plan_synthesizer(dict(state), config)
        materialize_artifacts(dict(state), workspace_root, work_id=work_id)

        from spine.config import SpineConfig as _SpineConfig
        findings_text = _format_findings(
            findings,
            budget=_SpineConfig.load().synthesize_findings_token_budget,
        )
        rework_prefix = ""
        if retry_count > 0:
            rework_prefix = (
                "⚠ **REWORK PASS**: Your primary objective is to revise "
                "the prior plan. Address all points from the "
                "critic feedback.\n\n"
            )

        prompt = (
            f"{rework_prefix}Create a detailed technical plan with structured "
            f"feature slices, incorporating the codebase research findings below.\n\n"
            f"## Work Description\n{description}\n\n"
            f"## Codebase Research Findings\n{findings_text}\n\n"
            f"Call `read_prior_artifacts` to load the specification and prior context "
            f"in one call. Then synthesize the spec + findings into structured "
            f"feature_slices and call `write_structured_plan` exactly once — the tool "
            f"writes both plan.md and plan.json for you. Do not call write_file."
        )

        if retry_count > 0:
            feedback_text = _render_rework_feedback(last_critic_review, feedback)
            if feedback_text:
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
            "read_cache": result.get("read_cache") or {},
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
    from spine.agents.specify_agent import build_specify_synthesizer

    description = state.get("description", "")
    work_id = state.get("work_id", "unknown")
    work_type = state.get("work_type", "")
    workspace_root = state.get("workspace_root", ".")
    findings = state.get("findings", [])
    retry_count = state.get("retry_count", 0)
    feedback = state.get("feedback", [])
    last_critic_review = state.get("last_critic_review") or {}

    logger.info(
        "[%s] Synthesize (specify): %d findings available, retry=%d",
        work_id,
        len(findings),
        retry_count,
    )

    try:
        agent = build_specify_synthesizer(dict(state), config)
        materialize_artifacts(dict(state), workspace_root, work_id=work_id)

        from spine.config import SpineConfig as _SpineConfig
        findings_text = _format_findings(
            findings,
            budget=_SpineConfig.load().synthesize_findings_token_budget,
        )
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
            f"Call `read_work_context` once to load description, feedback, and any "
            f"prior spec. Then synthesize the work description + research findings "
            f"into the structured fields and call `write_specification` exactly once "
            f"(fields: title, summary, objectives, requirements, constraints, "
            f"scope_inclusions, scope_exclusions, known_risks). The tool renders "
            f"markdown and emits JSON for you — do not call write_file."
        )

        if retry_count > 0:
            feedback_text = _render_rework_feedback(last_critic_review, feedback)
            if feedback_text:
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
            "read_cache": result.get("read_cache") or {},
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


# _count_tokens has moved to spine.agents._tokens (imported at module top).


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


def _render_rework_feedback(
    last_critic_review: dict,
    feedback: list[dict],
) -> str:
    """Render the rework feedback block for the synthesizer prompt.

    Prefers ``last_critic_review`` (the single record written by the most
    recent critic) so the rework prompt always surfaces the exact verdict
    that caused the loopback. Falls back to the full ``feedback`` list when
    the field is absent (older state checkpoints).
    """
    if last_critic_review:
        status = last_critic_review.get("status", "needs_revision")
        tier = last_critic_review.get("tier", "unknown")
        reason = last_critic_review.get("reason", "")
        suggestions = last_critic_review.get("suggestions") or []
        attempt = last_critic_review.get("attempt", 1)
        lines = [f"- [{tier}] (attempt {attempt}, status={status}) {reason}"]
        for s in suggestions:
            lines.append(f"  - {s}")
        return "\n".join(lines)

    if not feedback:
        return ""
    return "\n".join(
        f"- [{f.get('tier', 'unknown')}] {f.get('reason', '')}"
        for f in feedback
        if isinstance(f, dict)
    )


def _format_findings(
    findings: list[dict], *, budget: int | None = None,
) -> str:
    """Format accumulated findings for the synthesizer prompt.

    Keeps individual findings compact — the synthesizer can read files
    from disk if more detail is needed.

    When ``budget`` is a positive int, accumulates token count per
    appended finding (via :func:`spine.agents._tokens.count_tokens`) and
    stops once the next finding would push the total over budget. A
    trailing marker tells the synthesizer how many findings were omitted
    so it can request specific symbols if needed.
    """
    if not findings:
        return "(no codebase research was performed)"

    use_budget = isinstance(budget, int) and budget > 0
    parts: list[str] = []
    included = 0
    omitted = 0
    used_tokens = 0

    for i, f in enumerate(findings):
        if not isinstance(f, dict):
            continue
        if f.get("error"):
            continue
        topic = f.get("topic", "")
        summary = f.get("summary", "")
        patterns = f.get("patterns", [])
        file_map = f.get("file_map", {})
        deps = f.get("dependencies", [])
        header = f"### Finding {i + 1}"
        if topic:
            header += f" — Topic: {topic}"
        block_parts: list[str] = [f"{header}\n{summary}"]
        if patterns:
            block_parts.append(f"Patterns: {', '.join(patterns)}")
        if file_map:
            block_parts.append(f"Key files: {_json_mod.dumps(file_map)}")
        if deps:
            block_parts.append(f"Dependencies: {', '.join(deps)}")
        block = "\n\n".join(block_parts)

        if use_budget:
            block_tokens = _count_tokens(block)
            if used_tokens + block_tokens > budget:
                # Count this finding + every later non-error finding as omitted.
                omitted = sum(
                    1 for g in findings[i:]
                    if isinstance(g, dict) and not g.get("error")
                )
                break
            used_tokens += block_tokens
        parts.append(block)
        included += 1

    if not parts:
        return "(no codebase research was performed)"

    rendered = "\n\n".join(parts)
    if omitted > 0:
        rendered += (
            f"\n\n[truncated: {omitted} more findings omitted "
            f"(over {budget}-token budget — request specific symbols if needed)]"
        )
    return rendered


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

    # ── Persist research log as artifact ───────────────────────────────
    topics: list[str] = state.get("topics", [])
    findings: list[dict] = state.get("findings", [])
    if topics or findings:
        try:
            from spine.persistence.artifacts import ArtifactStore
            from spine.config import SpineConfig

            cfg = SpineConfig.load()
            store = ArtifactStore(base_path=cfg.artifact_path)
            research_log = json.dumps({
                "topics": topics,
                "findings": [
                    {
                        "topic": f.get("topic", ""),
                        "summary": f.get("summary", ""),
                        "patterns": f.get("patterns", []),
                        "file_map": f.get("file_map", {}),
                        "dependencies": f.get("dependencies", []),
                    }
                    for f in findings
                    if isinstance(f, dict) and not f.get("error")
                ],
            }, indent=2, default=str)
            store.save_artifact(work_id, phase, "research_log.json", research_log)
        except Exception as exc:
            logger.warning("[%s] Could not persist research_log.json: %s", work_id, exc)

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
    builder.add_node("topic_lookup", _topic_lookup_node)
    builder.add_node("explore_do", _explore_do_node)
    builder.add_node("summarise", _summarise_node)
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

    # research_manager → topic_lookup (annotate topics with recall hits)
    builder.add_edge("research_manager", "topic_lookup")

    # topic_lookup → Send("explore_do", ...) (with enriched topics) or → synthesize
    builder.add_conditional_edges(
        "topic_lookup",
        _research_router,
        {"explore": "explore_do", "synthesize": "synthesize"},
    )

    # explore_do dispatches to summarise dynamically via Command(goto=Send)
    # so each parallel branch carries its own evidence dossier without
    # colliding on a shared LastValue channel. summarise → aggregate is a
    # plain fan-in edge that runs once on the merged ``findings`` channel
    # after all parallel summarises complete.
    builder.add_edge("summarise", "aggregate")

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
