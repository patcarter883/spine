"""SPINE exploration agents — research_manager and explore node agents.

These are lightweight agent functions for the exploration subgraph.
They do NOT use the full middleware stack — they are single-purpose
LLM calls, not orchestrator agents.

- ``run_research_manager``: single LLM ``ainvoke()`` — no tools, no agent loop.
- ``run_explore_node``: builds a researcher subagent via the existing
  ``build_subagent_spec`` + ``build_phase_agent`` machinery.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, Field

from spine.agents.helpers import resolve_model

logger = logging.getLogger(__name__)

# Truncate spec content for PLAN explore agents to reasonable size
_MAX_SPEC_CHARS = 8000

# Token budget for the cumulative Explore loop findings.  Once the
# manager's accumulated findings exceed this we force "done" rather than
# launching another researcher round — Explore is a survey loop, not an
# implementation loop, and beyond this threshold the manager starts
# re-exploring territory already covered by earlier rounds.
EXPLORE_TOKEN_BUDGET = 20_000


_RECALL_SUFFIX_MARKER = " — recall symbols:"


def _normalise_topic(topic: str) -> str:
    """Normalise a topic string for set-membership comparisons.

    ``_research_router`` decorates topics with a "— recall symbols: …"
    suffix before dispatch, and ``run_explore_node`` stamps that decorated
    string onto each finding as ``finding["topic"]``. The manager's
    ``state.topics`` list, by contrast, holds the bare topic. Exact-match
    comparisons therefore mis-classify every enriched topic as "new",
    which makes the topics-already-explored signal useless. Normalising on
    both sides keeps the comparison stable across that decoration.
    """
    if not topic:
        return ""
    if _RECALL_SUFFIX_MARKER in topic:
        topic = topic.split(_RECALL_SUFFIX_MARKER, 1)[0]
    return " ".join(topic.lower().split())


_CRITIC_RESEARCH_KEYWORDS: tuple[str, ...] = (
    "research",
    "explore",
    "investigate",
    "missing knowledge",
    "missing context",
    "unclear scope",
    "more information",
    "more info",
    "need to understand",
    "needs investigation",
    "find out",
    "discover",
)


def _critic_wants_more_research(last_critic_review: dict[str, Any]) -> bool:
    """Decide whether a critic verdict implies more exploration is needed.

    The synth-only rework path requires confidence that the critic is
    asking us to fix the artifact, not fill a research gap. We look at
    both the verdict's reason and its suggestions for explicit research
    cues — when neither mentions them, synthesis can take another swing
    with the existing findings.
    """
    if not last_critic_review:
        return False
    blob_parts = [last_critic_review.get("reason", "") or ""]
    sugs = last_critic_review.get("suggestions") or []
    blob_parts.extend(str(s) for s in sugs)
    blob = " ".join(blob_parts).lower()
    return any(kw in blob for kw in _CRITIC_RESEARCH_KEYWORDS)


def _approx_findings_tokens(findings: list[Any]) -> int:
    """Rough token estimate for accumulated Explore findings.

    Uses the 4-chars-per-token heuristic — Explore content is plain English
    + code snippets so this is within ~20 % of the real tokenisation for
    every provider SPINE supports. Good enough for an early-exit gate.
    """
    if not findings:
        return 0
    total_chars = 0
    for item in findings:
        if isinstance(item, str):
            total_chars += len(item)
        elif isinstance(item, dict):
            try:
                total_chars += len(json.dumps(item, default=str))
            except Exception:
                total_chars += len(repr(item))
        else:
            total_chars += len(str(item))
    return total_chars // 4

# ── Research manager structured output model ─────────────────────────────


class ResearchManagerDecision(BaseModel):
    """Structured output from the research manager — explore more or done.

    Used with ``model.with_structured_output()`` so the LLM returns a
    validated instance instead of raw JSON that needs post-hoc parsing
    and markdown-fence stripping.
    """

    decision: Literal["explore", "done"] = Field(
        description="'explore' to continue research, 'done' when sufficient"
    )
    topics: list[str] = Field(
        default_factory=list,
        description="2-4 specific research topics if decision is 'explore', empty if 'done'",
    )


# ── Research manager prompts ─────────────────────────────────────────────

_RESEARCH_MANAGER_SPECIFY = """\
You are an Architectural Research Manager. Your job is to map the codebase
structure so the SPECIFY orchestrator can write an accurate specification.

## Your mission

The SPECIFY orchestrator needs to understand what EXISTS:
1. Public interfaces and module boundaries — what are the contracts?
2. Dependency relationships — what imports what and why?
3. Conventions and patterns — naming, structure, idioms in use
4. Configuration and global state — what settings or singletons exist?

Your task is to assign research topics to subagents (Architectural Scouts)
that will discover this structural knowledge. Each topic should describe
a *question* about a specific area of the codebase — in plain natural
language, no symbol or file references.

## How to work

1. Read the **Retrieved Symbol Summaries** section below (if present).
   These are LLM-generated summaries of functions/classes most relevant
   to the work description, discovered by semantic vector search. Use
   them as background context — they tell you which areas of the codebase
   are likely relevant. Do NOT copy symbol names or file paths into your
   topics.

2. Phrase each topic as a natural-language research question about an
   area of the codebase. Examples:
   "How is CLI verbosity configured and threaded through to subcommands?"
   "How does the configuration loader resolve workspace roots?"
   "What logging conventions are used across the agent layer?"

3. Do NOT name specific functions, classes, methods, or file paths in
   topics. A downstream lookup step will resolve each topic to the
   relevant symbols via semantic search — naming them yourself risks
   hallucinating names that don't exist.

4. Topics should focus on STRUCTURE, not implementation:
   - What are the public interfaces? (function signatures, class contracts)
   - What are the dependency relationships? (imports, call chains)
   - What conventions and patterns exist? (naming, error handling, config)
   - What configuration or global state is involved? (singletons, env vars)
   Do NOT ask subagents to propose solutions or implementation details.

Given:
1. The work description
2. The retrieved symbol summaries (semantic search results, for context only)
3. A list of research topics already explored
4. The findings accumulated so far

Decide:
- Are we done? (decision: "done") — structural coverage is sufficient
- Or do we need more? (decision: "explore") — return 2-4 plain-language
  topics framed as research questions

Rules:
- Never return more than 4 topics in a single round.
- If you've already explored a topic, don't return it again.
- On any round after the first, every new topic MUST target a different
  module, layer, or concern than every topic in "Topics Already Explored".
  Rephrasing a prior topic with different words is NOT a new topic — if
  the "Files examined" coverage in Findings So Far already touches the
  area, do not propose another topic about that area.
- Topics MUST NOT contain symbol names, function names, class names, or
  file paths. Use plain English descriptions of the area to investigate.
- If the work description is self-contained (no codebase needed), decide "done".
- If findings already cover all key architectural areas, decide "done".
- Cover breadth before depth — ensure all implicated modules are touched
  before diving deeper into any one module.
- If "Findings So Far" is non-empty on Round 1 of N, you are resuming a
  rework — the prior pass already wrote a research log. Default to
  decision="done" so synthesis can re-run with the existing findings.
  Propose new topics ONLY when a critic suggestion explicitly identifies
  a coverage gap, and then cap at 1 topic addressing that gap.
"""

_RESEARCH_MANAGER_PLAN = """\
You are a Change Surface Research Manager. Your job is to identify the
codebase areas that will need modification so the PLAN orchestrator can
decompose the work into executable slices.

## Your mission

The PLAN orchestrator needs to understand what CHANGES:
1. Touch points — which areas of the codebase will need edits?
2. Dependency chains — what else will be affected by those changes?
3. Risks — are there complex data flows, global state, or tight coupling to flag?

Your task is to assign research topics to subagents (Blueprint Scouts)
that will map the specification requirements to the codebase surface area.
Each topic should describe — in plain natural language — an area of the
codebase where changes will happen, framed as a research question.

## How to work

1. Read the **Specification** section below (always present in PLAN phase).
   This describes what needs to be built or changed. Your research topics
   MUST directly target the codebase areas that the specification requires.

2. Read the **Retrieved Symbol Summaries** section below (if present).
   These are LLM-generated summaries of functions/classes most relevant
   to the work. Use them as background context to understand which areas
   are implicated. Do NOT copy symbol names or file paths into your topics.

3. Phrase each topic as a natural-language research question about an
   area of the codebase. Examples:
   "How does workspace-root resolution work for spec-path handling?"
   "How do the CLI subcommands wire arguments through to the agent factory?"

4. Do NOT name specific functions, classes, methods, or file paths in
   topics. A downstream lookup step will resolve each topic to the
   relevant symbols via semantic search — naming them yourself risks
   hallucinating names that don't exist.

5. Topics should focus on CHANGE SURFACE, not architecture:
   - Which areas will need modification? (touch points)
   - What imports or callers will be affected? (dependency chains)
   - Are there complex data flows, global state, or tight coupling? (risks)
   Do NOT ask subagents to propose solutions — only to identify what exists
   and what will be impacted.

Given:
1. The specification (what needs to change)
2. The retrieved symbol summaries (semantic search results, for context only)
3. A list of research topics already explored
4. The findings accumulated so far

Decide:
- Are we done? (decision: "done") — change surface is adequately mapped
- Or do we need more? (decision: "explore") — return 2-4 plain-language
  topics framed as research questions

Rules:
- Never return more than 4 topics in a single round.
- If you've already explored a topic, don't return it again.
- On any round after the first, every new topic MUST target a different
  module, layer, or concern than every topic in "Topics Already Explored".
  Rephrasing a prior topic with different words is NOT a new topic — if
  the "Files examined" coverage in Findings So Far already touches the
  area, do not propose another topic about that area.
- Topics MUST NOT contain symbol names, function names, class names, or
  file paths. Use plain English descriptions of the area to investigate.
- If findings already cover all key change areas, decide "done".
- Prioritize high-risk areas (glue code, shared state, widely-imported
  modules) before peripheral concerns.
- Cross-reference every topic against the specification — if the spec
  doesn't mention an area, don't explore it.
- If "Findings So Far" is non-empty on Round 1 of N, you are resuming a
  rework — the prior pass already wrote a research log. Default to
  decision="done" so synthesis can re-run with the existing findings.
  Propose new topics ONLY when a critic suggestion explicitly identifies
  a coverage gap, and then cap at 1 topic addressing that gap.
"""


async def run_research_manager(
    state: dict[str, Any],
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Run the research manager — single LLM call to decide next topics.

    On the first round, pre-runs a semantic recall (summaries only) to
    give the manager an instant architectural map. The summaries are
    background context only — the manager emits plain natural-language
    topics; the downstream ``topic_lookup`` node resolves each topic to
    concrete symbols via semantic search.

    Args:
        state: The ExplorationSubgraphState.
        config: LangGraph runtime config.

    Returns:
        Dict with ``manager_decision`` and ``topics`` keys.
    """
    description = state.get("description", "")
    existing_topics = state.get("topics", [])
    findings = state.get("findings", [])
    round_num = state.get("research_round", 0)
    max_rounds = state.get("max_rounds", 3)
    work_id = state.get("work_id", "unknown")
    retry_count = state.get("retry_count", 0)
    last_critic_review = state.get("last_critic_review") or {}
    phase = state.get("phase")
    if phase is None:
        raise ValueError(
            "Exploration subgraph state missing 'phase' key. "
            "This indicates a state mapper configuration error."
        )
    workspace_root = state.get("workspace_root", ".")

    # Safety valve — if we've hit max rounds, force done
    if round_num >= max_rounds:
        logger.info(
            "[%s] Research manager: max rounds (%d) reached — forcing done",
            work_id,
            max_rounds,
        )
        return {"manager_decision": "done", "topics": []}

    # Token-budget safety valve — survey loops should synthesise once their
    # accumulated findings approach the prompt-cache window. Forcing "done"
    # past EXPLORE_TOKEN_BUDGET keeps the manager from spinning on already-
    # explored ground when the findings are already context-heavy.
    if round_num > 0 and _approx_findings_tokens(findings) >= EXPLORE_TOKEN_BUDGET:
        logger.info(
            "[%s] Research manager: token budget (%d) reached — forcing done",
            work_id,
            EXPLORE_TOKEN_BUDGET,
        )
        return {"manager_decision": "done", "topics": []}

    # Critic-rework re-entry guard — when the state mapper seeded findings
    # from a prior attempt's research_log.json AND the critic's feedback
    # doesn't call out a research gap, skip exploration and let the
    # synthesiser take another swing with the existing findings.
    #
    # Trace 019e6974: PLAN was reworked 3× because critic_plan rejected
    # the slice metadata ("missing target_files", "malformed JSON"). On
    # each rework the manager fired again and re-emitted near-duplicate
    # topics — even though the research log already covered the relevant
    # files. ~3.3M PLAN prompt tokens were spent re-learning the same
    # facts. The synthesis fix-up was the actual unit of work.
    is_rework_entry = (
        round_num == 0
        and retry_count > 0
        and bool(findings)
        and bool(last_critic_review)
    )
    if is_rework_entry and not _critic_wants_more_research(last_critic_review):
        logger.warning(
            "[%s] Research manager: critic-rework entry — %d prior findings "
            "exist and critic feedback %r does not call for more research; "
            "skipping exploration so synthesis can rework with what we have.",
            work_id,
            len(findings),
            (last_critic_review.get("reason", "") or "")[:120],
        )
        return {"manager_decision": "done", "topics": []}

    model = resolve_model(config, session_id=work_id, phase="exploration/manager")

    # resolve_model may return a string spec or a pre-built BaseChatModel.
    if isinstance(model, str):
        from langchain.chat_models import init_chat_model

        model = init_chat_model(model)

    # ── Recall context from state ────────────────────────────────────
    # The pre_research_gate (exploration_subgraph._pre_research_gate)
    # already classified the task and pulled recall hits before the loop
    # started.  Use the chunks it put on state — no second classify+recall
    # call.  task_category may be None on the first round if the gate
    # failed (we fall through to the loop anyway).
    task_category = state.get("task_category")
    retrieved = state.get("retrieved_context") or []
    recall_section = ""
    if round_num == 0 and retrieved:
        recall_section = (
            "## Retrieved Symbol Summaries (semantic search, filtered by task category)\n"
            "These are the most relevant functions/classes discovered by "
            "vector search. Use them ONLY as background context to identify "
            "which areas of the codebase are relevant — do NOT copy symbol "
            "names or file paths into your topics.\n\n"
        )
        for i, chunk in enumerate(retrieved, 1):
            recall_section += (
                f"### {chunk.get('symbol_name', '?')} "
                f"({chunk.get('symbol_type', '?')} in {chunk.get('file_path', '?')})\n"
                f"{chunk.get('enriched_summary', '')[:400]}\n\n"
            )
        logger.info(
            "[%s] Research manager: using %d pre-recalled summaries from gate",
            work_id, len(retrieved),
        )

    # ── Build the context for the manager ────────────────────────────
    spec_section = ""
    if phase == "plan":
        spec_path = state.get("spec_path", "")
        if spec_path:
            full_path = Path(workspace_root) / spec_path / "specification.md"
            if full_path.exists():
                try:
                    raw_spec = full_path.read_text(encoding="utf-8")
                    spec_section = (
                        f"## Specification (use this to guide research decisions)\n"
                        f"{raw_spec[:_MAX_SPEC_CHARS]}\n\n"
                    )
                except OSError:
                    pass

    findings_summary = _summarize_findings(findings)

    # Prior-round framing: whenever any topic has already been dispatched
    # (round_num > 0 OR seeded from a prior research_log on rework), the
    # manager must treat the explored set as territory to *extend past*,
    # not territory to revisit. Without this, the LLM tends to rephrase
    # prior topics — same modules, different words — because the
    # natural-language "already explored" list alone is too soft a signal.
    #
    # When the critic also rejected the prior pass, the rework specifics
    # (verdict, suggestions) layer on top of the always-on framing.
    has_prior_round = round_num > 0 or bool(existing_topics) or bool(findings)
    prior_round_section = ""
    if has_prior_round:
        prior_round_section = (
            "## Prior-Round Coverage — Extend, Do Not Repeat\n"
            "The 'Topics Already Explored' and 'Findings So Far' sections "
            "below describe ground that has already been mapped. Every new "
            "topic you propose MUST target a different module / layer / "
            "concern than what is already covered there. Use the 'Files "
            "examined' line on each finding as the authoritative coverage "
            "signal — if a file or area appears there, that area is "
            "covered, regardless of how the prior topic was worded. "
            "Rephrasing a prior topic ('How does X configure Y?' → 'How "
            "are Y preferences managed in X?') is forbidden — those count "
            "as the same topic. If findings already cover all relevant "
            "areas, decide 'done'.\n\n"
            "Before returning each topic, silently check: which module or "
            "layer does this target, and is that module already in any "
            "prior finding's 'Files examined'? If yes, drop the topic.\n\n"
        )
    rework_section = ""
    if retry_count > 0 and last_critic_review:
        suggestions = last_critic_review.get("suggestions") or []
        sug_text = (
            "\n".join(f"  - {s}" for s in suggestions) if suggestions else "  (none)"
        )
        rework_section = (
            "## ⚠ Rework Pass — Critic Rejected Prior Output\n"
            f"Attempt: {last_critic_review.get('attempt', retry_count + 1)}\n"
            f"Verdict: {last_critic_review.get('status', 'needs_revision')} "
            f"(tier: {last_critic_review.get('tier', 'unknown')})\n"
            f"Reason: {last_critic_review.get('reason', '')}\n"
            f"Suggestions:\n{sug_text}\n\n"
            "Propose 2-4 NEW topics that close the specific gaps the critic "
            "flagged (in addition to the prior-round rules above). If the "
            "prior research is already sufficient to address the verdict, "
            "decide 'done' so synthesis can rework with the existing "
            "findings.\n\n"
        )

    context = (
        f"## Work Description\n{description}\n\n"
        f"{spec_section}"
        f"{recall_section}"
        f"{prior_round_section}"
        f"{rework_section}"
        f"## Round\n{round_num + 1} of max {max_rounds}\n\n"
        f"## Topics Already Explored\n{json.dumps(existing_topics)}\n\n"
        f"## Findings So Far\n{findings_summary}\n\n"
        "Decide: are we done, or do we need more research? "
        "If exploring, return 2-4 plain-language topics framed as research "
        "questions. Topics MUST NOT contain symbol names, function names, "
        "class names, or file paths — a downstream lookup step resolves "
        "each topic to the relevant symbols automatically."
    )

    # ── Select the phase-appropriate manager prompt ─────────────────
    # The SPECIFY manager maps architecture (what exists — interfaces,
    # boundaries, conventions). The PLAN manager maps change surface
    # (what will change — touch points, dependency chains, risks).
    # This mirrors the researcher subagent split: Architectural Scout
    # vs Blueprint Scout.
    if phase == "plan":
        manager_prompt = _RESEARCH_MANAGER_PLAN
    else:
        manager_prompt = _RESEARCH_MANAGER_SPECIFY

    try:
        # Use model.with_structured_output() for proper Pydantic validation.
        # The local vLLM can produce syntactically-valid but semantically-
        # garbage JSON (e.g. topics=[" ["]) that json.loads() accepts —
        # only Pydantic validation catches this.  We extract plain values
        # into a dict so the Pydantic instance never leaks into LangGraph
        # state (avoids the checkpoint serializer warning on AIMessage.parsed).
        structured_model = model.with_structured_output(ResearchManagerDecision)
        # LangChain's tracer Pydantic-serializes the raw AIMessage with
        # `.parsed` populated to our custom model, which trips Pydantic's
        # "unexpected value" warning every call. The warning is cosmetic
        # — suppress only this specific warning around the invocation.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*PydanticSerializationUnexpectedValue.*parsed.*",
            )
            warnings.filterwarnings(
                "ignore",
                message=r".*Expected `none`.*parsed.*",
            )
            response = await structured_model.ainvoke(
                [SystemMessage(content=manager_prompt), HumanMessage(content=context)],
            )

        # .with_structured_output() may return the Pydantic instance directly
        # (newer LangChain) or in AIMessage.parsed (legacy providers).
        if isinstance(response, ResearchManagerDecision):
            parsed = response
        elif hasattr(response, "parsed") and isinstance(response.parsed, ResearchManagerDecision):
            parsed = response.parsed
            response.parsed = None  # prevent Pydantic serialization warning
        else:
            raise ValueError(
                f"Unexpected structured output response type: {type(response).__name__}"
            )

        logger.info(
            "[%s] Research manager: decision=%s topics=%s",
            work_id, parsed.decision, parsed.topics,
        )
        result: dict[str, Any] = {"manager_decision": parsed.decision, "topics": parsed.topics}
        if task_category:
            result["task_category"] = task_category
        return result

    except Exception as e:
        logger.warning(
            "[%s] Research manager failed: %s — defaulting to explore with one topic",
            work_id, e,
        )
        # Fail-open: if we can't understand the response, do at least one
        # exploration round rather than silently skipping all research.
        return {
            "manager_decision": "explore",
            "topics": ["codebase structure and key files"],
        }


_FINDINGS_SUMMARY_BUDGET = 8000


def _summarize_findings(findings: list[dict]) -> str:
    """Create a coverage-oriented summary of accumulated research findings.

    The manager uses this to decide gaps for the next round. It must convey
    *what was covered* (topic + files touched) — not just *flavor* — or it
    will rephrase prior topics instead of extending coverage. Accumulates
    entries until the running string crosses ``_FINDINGS_SUMMARY_BUDGET``
    chars (well inside ``EXPLORE_TOKEN_BUDGET``).
    """
    if not findings:
        return "(no findings yet)"
    parts: list[str] = []
    total = 0
    for i, f in enumerate(findings):
        if not isinstance(f, dict):
            continue
        if f.get("error"):
            continue
        topic = _normalise_topic(f.get("topic", ""))
        summary = f.get("summary", "")
        patterns = f.get("patterns", [])
        deps = f.get("dependencies", [])
        file_map = f.get("file_map", {}) or {}
        files = list(file_map.keys()) if isinstance(file_map, dict) else []
        entry = f"Finding {i + 1} [topic: {topic[:160]}]: {summary[:600]}"
        if files:
            entry += f"\n  Files examined: {', '.join(files[:10])}"
        if patterns:
            entry += f"\n  Patterns: {', '.join(patterns[:8])}"
        if deps:
            entry += f"\n  Dependencies: {', '.join(deps[:8])}"
        total += len(entry)
        parts.append(entry)
        if total >= _FINDINGS_SUMMARY_BUDGET:
            remaining = len(findings) - (i + 1)
            if remaining > 0:
                parts.append(f"(... {remaining} additional finding(s) elided — coverage budget reached)")
            break
    return "\n\n".join(parts)


# ── Explore node ─────────────────────────────────────────────────────────


async def run_explore_node(
    state: dict[str, Any],
    config: RunnableConfig | None = None,
    *,
    topic: str = "",
) -> dict[str, Any]:
    """Run an explore node — invokes a researcher subagent for one topic.

    Uses ``build_subagent_spec("researcher", ...)`` to get the full
    subagent configuration (model, tools, MCP, response_format), then
    wraps it in a minimal agent for invocation.

    When the phase is ``"plan"``, the specification content is read from
    disk and injected into the subagent prompt so the researcher maps spec
    requirements to codebase files rather than doing generic exploration.

    Args:
        state: The ExplorationSubgraphState.
        config: LangGraph runtime config.
        topic: The specific research topic (set by Send API).

    Returns:
        Dict with ``findings`` key containing a list with one
        ResearchFindings dict (merged by operator.add).
    """
    from spine.agents.factory import build_phase_agent
    from spine.agents.subagents import build_subagent_spec
    from spine.models.enums import PhaseName

    work_id = state.get("work_id", "unknown")
    phase = state.get("phase")
    if phase is None:
        raise ValueError(
            "Explore node state missing 'phase' key. "
            "This indicates a state mapper configuration error."
        )
    workspace_root = state.get("workspace_root", ".")
    topic_str = topic or "general codebase investigation"

    logger.info("[%s] Explore node (phase=%s): researching topic=%r", work_id, phase, topic_str)

    result: dict[str, Any] = {}
    try:
        # Determine which PhaseName to use for model/skill resolution
        phase_enum = PhaseName(phase)

        # Build the researcher subagent spec with the correct phase
        subagent_spec = build_subagent_spec(
            name="researcher",
            phase=phase_enum,
            state=state,  # type: ignore[arg-type]
            config=config,
        )

        # Build a minimal agent for this subagent — no filesystem middleware,
        # the tools are injected directly as extra_tools from the subagent spec.
        # NOTE: commit_findings_and_clear_search is intentionally NOT injected
        # here. The researcher's message loop is local to this one ainvoke and
        # is discarded when the explore node returns; the eviction node at the
        # subgraph level operates on the parent state, not the researcher's
        # internal messages. Giving the tool to the researcher only causes its
        # EVICTION_ANCHOR message to fool the model into believing its tool
        # results are gone — after which the structured response collapses to
        # an empty patterns/file_map/dependencies on most models.
        extra_tools = list(subagent_spec.get("tools", []))
        agent = build_phase_agent(
            state=state,  # type: ignore[arg-type]
            config=config,
            phase=phase_enum,
            system_prompt=subagent_spec["system_prompt"],
            is_subagent=True,
            extra_tools=extra_tools,
            response_format=subagent_spec.get("response_format"),
            skip_filesystem_middleware=True,
        )

        # ── Build the research prompt ─────────────────────────────────
        # For PLAN phase: inject the specification content so the researcher
        # maps spec requirements to codebase files, not just the generic topic.
        spec_content = ""
        if phase_enum == PhaseName.PLAN:
            spec_path = state.get("spec_path", "")
            if spec_path:
                full_path = Path(workspace_root) / spec_path / "specification.md"
                if full_path.exists():
                    try:
                        spec_content = full_path.read_text(encoding="utf-8")
                    except OSError:
                        pass

        if spec_content:
            prompt = (
                f"## Plan Research Topic: {topic_str}\n\n"
                f"## Specification (read this FIRST — your research must map this to "
                f"the codebase)\n\n{spec_content[:_MAX_SPEC_CHARS]}\n\n"
                f"## Research Task\n"
                f"For the topic '{topic_str}', find the existing codebase files, "
                f"patterns, conventions, and dependencies that map to the specification "
                f"sections above. Use MCP tools for structural navigation. "
                f"Return your findings in the ResearchFindings format with specific "
                f"file paths and how they relate to the spec."
            )
        else:
            prompt = (
                f"## Research Topic\n{topic_str}\n\n"
                f"Investigate this specific area of the codebase. "
                f"Use MCP tools for structural navigation. "
                f"Return your findings in the ResearchFindings format."
            )

        # Inject scratchpad into researcher prompt if available
        scratchpad = state.get("scratchpad", "")
        if scratchpad:
            prompt = prompt + "\n\n## Working Memory Scratchpad\n" + scratchpad + "\n"

        # Build a SpineContext so ReadCacheMiddleware can dedupe MCP/read_file
        # calls inside this researcher loop. Without context= the middleware
        # bails on ctx=None and every lookup re-hits the live tool — which is
        # exactly the failure mode the f51d448 deduper was supposed to fix.
        from spine.agents.context import build_context as _build_context

        ctx = _build_context(state, phase_enum)
        # Cap the researcher's LangGraph recursion at ~50 steps (≈25
        # model+tool round-trips). LangGraph's default is 25; the prior
        # cap of 24 sat below that and routinely terminated survey-style
        # branches mid-investigation (telemetry showed branches still
        # productive at >50K prompt tokens when the wall hit). 50 gives
        # enough headroom for breadth-first topics while still bounding
        # runaway loops.
        result = await _ainvoke_explore_collecting(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            work_id=work_id,
            context=ctx,
            config={"recursion_limit": 50},
        )

        # Finalize: convert the free-form research conversation into a
        # ResearchFindings JSON. The researcher agent is intentionally run
        # WITHOUT response_format so the model actually explores with
        # tools (binding a schema causes the model to satisfy it on turn 1
        # and skip tool calls). After the loop completes, take the final
        # assistant message and convert it via the model's native
        # structured-output binding.
        await _finalize_research_findings(result, subagent_spec.get("model"))

        # Extract findings from the result
        findings = _extract_findings(result)
        # Inject the topic into each finding so the synthesizer knows
        # which area each finding addresses.
        for f in findings:
            if isinstance(f, dict) and "topic" not in f:
                f["topic"] = topic_str
        logger.info(
            "[%s] Explore node: topic=%r — %d findings entries",
            work_id,
            topic_str,
            len(findings),
        )

    except Exception as e:
        # Recursion-cap salvage: _ainvoke_explore_collecting attaches the
        # last-seen state dict to the exception. We build an evidence
        # dossier from ToolMessage results (the actual research data),
        # not from the last AIMessage text — when the cap fires the last
        # text message is typically the topic restatement, which the
        # finalize call would just paraphrase back as a fake summary.
        # Salvage is accepted only when the structured response contains
        # at least one concrete field (file_map / patterns / dependencies);
        # otherwise we fall through to the error sentinel so the user-
        # visible outcome is a real error rather than a hallucinated
        # restatement of the topic.
        partial_state = getattr(e, "partial_state", None) if isinstance(e, GraphRecursionError) else None
        salvaged: list[dict] | None = None
        if partial_state and partial_state.get("messages"):
            salvaged = await _attempt_research_salvage(
                partial_state, subagent_spec.get("model"), topic_str, work_id
            )
            if salvaged is not None:
                result = partial_state

        if salvaged is not None:
            findings = salvaged
            logger.info(
                "[%s] Explore node (salvaged): topic=%r — %d findings entries",
                work_id,
                topic_str,
                len(findings),
            )
        else:
            logger.error(
                "[%s] Explore node failed for topic=%r: %s",
                work_id,
                topic_str,
                e,
                exc_info=True,
            )
            # The summary field is intentionally a neutral marker — NEVER
            # embed the exception text here. Earlier versions wrote the raw
            # GraphRecursionError / Pydantic stack into `summary`, and even
            # though every render path filters error sentinels, the entry
            # is still in `state["findings"]` for coverage bookkeeping, so
            # any agent or sub-agent that introspects state directly was
            # being fed the noise (observed in production trace where the
            # synthesizer's input findings list carried "Research failed
            # for topic '…': GraphRecursionError: …"). Full traceback stays
            # in logs via exc_info=True. The structured `error_class` /
            # `error_topic` fields are available for diagnostic introspection
            # by anything that explicitly opts in.
            findings = [
                {
                    "topic": topic_str,
                    "summary": "(research did not converge on this topic)",
                    "patterns": [],
                    "file_map": attempted_files,
                    "dependencies": [],
                    # Marker consumed by _summarize_findings,
                    # _format_findings, _save_exploration_artifacts, and
                    # export.format_export_markdown to drop this sentinel
                    # from every LLM-facing or human-facing render path.
                    "error": True,
                    # Structured diagnostic fields — explicit, no free-text
                    # exception messages. The class name is the most you
                    # should ever surface to a downstream consumer.
                    "error_class": type(e).__name__,
                    "error_topic": topic_str,
                }
            ]

    # Bubble the post-invocation deduper cache back into subgraph state so
    # sibling Send() researchers (and the next rework cycle) skip queries we
    # already ran. The reducer (_merge_read_cache) handles merge semantics.
    cache_snapshot = result.get("read_cache") if isinstance(result, dict) else None
    update: dict[str, Any] = {"findings": findings}
    if cache_snapshot:
        update["read_cache"] = cache_snapshot
    return update


async def _ainvoke_explore_collecting(
    agent: Any,
    input_: dict[str, Any],
    *,
    work_id: str,
    context: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Drive the researcher agent via ``astream`` so partial state survives errors.

    Behaves like :func:`spine.agents.retry.ainvoke_with_retry` for transient
    errors (same backoff schedule via the imported helpers), but uses
    ``stream_mode="values"`` to accumulate the latest state dict locally.
    On ``GraphRecursionError`` it attaches the accumulated state to the
    exception as a ``partial_state`` attribute so the caller can salvage
    whatever messages the researcher produced before the cap fired.
    """
    from spine.agents.retry import (
        DEFAULT_BASE_DELAY,
        DEFAULT_MAX_DELAY,
        DEFAULT_MAX_RETRIES,
        _is_transient_error,
    )

    invoke_kwargs: dict[str, Any] = {"config": config}
    if context is not None:
        invoke_kwargs["context"] = context

    prefix = f"[{work_id}]" if work_id else ""

    last_exc: Exception | None = None
    for attempt in range(DEFAULT_MAX_RETRIES + 1):
        partial_state: dict[str, Any] = {}
        try:
            async for chunk in agent.astream(
                input_, stream_mode="values", **invoke_kwargs
            ):
                if isinstance(chunk, dict):
                    partial_state = chunk
            if context is not None and hasattr(context, "read_cache"):
                cache_snapshot = getattr(context, "read_cache", None) or {}
                if cache_snapshot:
                    partial_state["read_cache"] = dict(cache_snapshot)
            return partial_state
        except GraphRecursionError as exc:
            if context is not None and hasattr(context, "read_cache"):
                cache_snapshot = getattr(context, "read_cache", None) or {}
                if cache_snapshot:
                    partial_state["read_cache"] = dict(cache_snapshot)
            exc.partial_state = partial_state  # type: ignore[attr-defined]
            raise
        except Exception as exc:
            last_exc = exc
            if not _is_transient_error(exc) or attempt >= DEFAULT_MAX_RETRIES:
                raise

            delay = min(DEFAULT_BASE_DELAY * (2**attempt), DEFAULT_MAX_DELAY)
            import random as _random

            jitter = delay * 0.1
            sleep_time = max(delay + _random.uniform(-jitter, jitter), 0.5)
            logger.warning(
                f"{prefix} explore transient error "
                f"(attempt {attempt + 1}/{DEFAULT_MAX_RETRIES + 1}), "
                f"retrying in {sleep_time:.1f}s: {type(exc).__name__}: {exc}"
            )
            import asyncio as _asyncio

            await _asyncio.sleep(sleep_time)

    if last_exc:
        raise last_exc
    raise RuntimeError("_ainvoke_explore_collecting: unexpected state")


_TOOL_ERROR_MARKERS = (
    "tool validation error",
    "not a recognized function",
    "not a recognized tool",
    "not recognized as",
    "unknown tool",
    "no such tool",
    "tool call",
    "validation error",
    "failed validation",
    "execution failed",
    "not available",
    "available tools",
    "available toolset",
    "valid alternatives",
    "list of valid",
)


def _last_ai_narrative(messages: list[Any]) -> str:
    """Return content of the last AIMessage with non-empty string content.

    Walks messages in reverse and ignores ToolMessage / HumanMessage /
    SystemMessage entries. ToolMessages are tool outputs (including
    ``status="error"`` rebound messages from ToolSchemaValidator) — they
    are not the agent's own narration and must not be treated as the
    research report. Returns an empty string if no such message exists.
    """
    for msg in reversed(messages):
        if isinstance(msg, (ToolMessage, HumanMessage, SystemMessage)):
            continue
        if not isinstance(msg, AIMessage):
            # Defensive: fall through for unknown message types but still
            # skip anything whose role/type self-identifies as tool/user.
            role = getattr(msg, "type", "") or getattr(msg, "role", "")
            if role in {"tool", "user", "system"}:
                continue
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            # Anthropic-style content blocks — join text parts.
            text_parts = [
                blk.get("text", "")
                for blk in content
                if isinstance(blk, dict) and blk.get("type") == "text"
            ]
            content = "".join(text_parts)
        if isinstance(content, str) and content.strip():
            return content
    return ""


def _looks_like_pure_tool_error(text: str, messages: list[Any]) -> bool:
    """Heuristic: is ``text`` just a narration of tool errors, no findings?

    A real research report references files, modules, functions, or
    behaviour. An error narration just describes what tool failed and
    what was offered instead. We flag a report as a pure tool-error
    narration when:

    * It's short (<= 800 chars), AND
    * It contains tool-error marker phrases, AND
    * No successful (non-error) ToolMessage exists in the conversation.

    Any one of those alone is too aggressive — a long report that briefly
    mentions an error is fine, and a short report after successful tool
    calls is probably genuine if terse.
    """
    if len(text) > 800:
        return False
    lowered = text.lower()
    if not any(marker in lowered for marker in _TOOL_ERROR_MARKERS):
        return False
    for msg in messages:
        if isinstance(msg, ToolMessage):
            status = getattr(msg, "status", None)
            if status != "error":
                return False
    return True


async def _finalize_research_findings(result: dict, model: Any) -> None:
    """Convert the researcher's final markdown report into ResearchFindings.

    The researcher subagent runs without a bound response_format so that
    it actually explores with tools instead of satisfying the schema on
    turn 1. After the agent loop finishes we make ONE additional LLM call
    here using the model's native structured-output binding to coerce the
    accumulated conversation into the ResearchFindings schema.

    Mutates ``result`` in place: sets ``structured_response`` to a
    ``ResearchFindings`` instance on success. On failure, leaves
    ``result`` untouched so the markdown fallback in ``_extract_findings``
    still produces something.
    """
    from spine.agents.subagents import ResearchFindings

    if model is None:
        return
    messages = result.get("messages", [])
    final_content = _last_ai_narrative(messages)
    if not final_content:
        return
    if _looks_like_pure_tool_error(final_content, messages):
        # The researcher gave up after a tool error without producing
        # any actual codebase findings. Feeding this prose into the
        # structured-output model just produces a "summary" that
        # narrates the error (e.g. "The report indicates a tool
        # validation error where 'foo' is not a recognized function…"),
        # which then surfaces to the synthesizer as if it were a real
        # finding. Bail out so _extract_findings falls back to the
        # explicit "(no findings)" sentinel instead.
        logger.info(
            "Skipping structured finalize: researcher output is pure tool-error narration"
        )
        return

    try:
        structured_model = model.with_structured_output(ResearchFindings)
    except Exception:
        logger.debug("Model does not support with_structured_output; skipping finalize", exc_info=True)
        return

    finalize_prompt = (
        "Convert the research report below into the ResearchFindings JSON schema. "
        "Populate ALL fields: summary (2-3 paragraph synthesis), patterns "
        "(distinct list items), file_map (path -> description), dependencies "
        "(distinct list items). Do NOT invent — use only information present "
        "in the report.\n\n## Report\n" + final_content
    )
    try:
        parsed = await structured_model.ainvoke(
            [HumanMessage(content=finalize_prompt)]
        )
    except Exception as e:
        logger.warning("Structured finalization failed: %s", e)
        return

    result["structured_response"] = parsed


# Per-section cap for tool result bodies in the salvage evidence dossier.
# MCP results can be tens of KB (function source, search hits) — clamp each
# section so the salvage finalize prompt stays bounded regardless of how
# many turns ran before the cap fired.
_SALVAGE_SECTION_CHAR_CAP = 2000

# If the assembled evidence is shorter than this, treat it as "nothing
# usable accumulated" and skip salvage entirely. Calibrated above the
# length of a typical empty-tool-result error message so a single failed
# lookup doesn't get treated as a real finding.
_SALVAGE_MIN_EVIDENCE_CHARS = 200


def _collect_salvage_evidence(messages: list) -> str:
    """Build an evidence dossier from a researcher's partial message history.

    Walks ``messages`` in order and extracts ``ToolMessage`` contents (the
    real research data — symbol bodies, dependency lists, search hits) plus
    any non-empty ``AIMessage`` content (intermediate synthesis attempts).
    Skips ``HumanMessage`` / ``SystemMessage`` so the dossier doesn't echo
    the topic prompt back into the finalize call.

    Returns the concatenated dossier, or ``""`` if nothing substantive
    accumulated.
    """
    sections: list[str] = []
    for msg in messages:
        content = getattr(msg, "content", "")
        if not isinstance(content, str) or not content.strip():
            continue
        if isinstance(msg, ToolMessage):
            # Skip failed tool calls — their content is an error string from
            # ToolSchemaValidator, not real evidence. Including them would
            # let the salvage prompt frame "Tool 'X' execution failed: ..."
            # as research data and produce a hallucinated paraphrase.
            if getattr(msg, "status", None) == "error":
                continue
            tool_name = getattr(msg, "name", None) or "tool"
            body = content.strip()
            if len(body) > _SALVAGE_SECTION_CHAR_CAP:
                body = body[: _SALVAGE_SECTION_CHAR_CAP - 1] + "…"
            sections.append(f"### Tool result: {tool_name}\n{body}")
        elif isinstance(msg, AIMessage):
            sections.append(f"### Intermediate synthesis\n{content.strip()}")
    return "\n\n".join(sections)


async def _finalize_research_findings_from_evidence(
    result: dict, model: Any, evidence: str
) -> None:
    """Salvage-mode finalize: coerce a tool-result dossier into ResearchFindings.

    Unlike :func:`_finalize_research_findings`, which assumes the
    researcher produced a clean final synthesis message, this variant runs
    when the LangGraph recursion cap fired mid-investigation. The
    ``evidence`` argument is the raw tool-result dossier from
    :func:`_collect_salvage_evidence`; the prompt explicitly frames it as
    partial evidence (not a finished report) and forbids using the
    research topic to fabricate fields when the dossier has no support
    for them.

    Mutates ``result`` in place on success.
    """
    from spine.agents.subagents import ResearchFindings

    if model is None or not evidence:
        return

    try:
        structured_model = model.with_structured_output(ResearchFindings)
    except Exception:
        logger.debug(
            "Model does not support with_structured_output; skipping salvage finalize",
            exc_info=True,
        )
        return

    salvage_prompt = (
        "The researcher agent ran out of steps before producing a final "
        "synthesis. Below is the raw evidence it gathered — tool results "
        "from symbol lookups, dependency queries, and codebase searches, "
        "plus any intermediate synthesis attempts.\n\n"
        "Convert this evidence into the ResearchFindings JSON schema. "
        "STRICT RULES:\n"
        "- Use ONLY information present in the evidence below.\n"
        "- Do NOT use the research topic name to fill in fields. If the "
        "evidence does not mention a file, pattern, or dependency, do not "
        "invent one.\n"
        "- file_map paths MUST appear in the tool results (as file paths, "
        "symbol locations, or import targets).\n"
        "- If a field has no support in the evidence, leave it empty "
        "(``[]`` for lists, ``{}`` for file_map).\n"
        "- summary should describe what was actually discovered in the "
        "tool results, not what the agent was asked to investigate.\n\n"
        "## Evidence\n" + evidence
    )
    try:
        parsed = await structured_model.ainvoke(
            [HumanMessage(content=salvage_prompt)]
        )
    except Exception as e:
        logger.warning("Salvage finalization failed: %s", e)
        return

    result["structured_response"] = parsed


async def _attempt_research_salvage(
    partial_state: dict,
    model: Any,
    topic_str: str,
    work_id: str,
) -> list[dict] | None:
    """Decide whether a researcher's partial state yields usable findings.

    Returns a list of finding dicts (each stamped ``partial=True`` and
    ``topic``) on success, or ``None`` when the caller should fall through
    to the error sentinel — either because nothing substantive
    accumulated, because the salvage finalize call failed, or because the
    structured response contained no concrete fields and would just be a
    hallucinated paraphrase of the topic.
    """
    messages = partial_state.get("messages", [])
    msg_count = len(messages)
    evidence = _collect_salvage_evidence(messages)
    if len(evidence) < _SALVAGE_MIN_EVIDENCE_CHARS:
        logger.warning(
            "[%s] Recursion cap hit for topic=%r after %d messages; "
            "evidence too thin (%d chars) — skipping salvage",
            work_id,
            topic_str,
            msg_count,
            len(evidence),
        )
        return None

    logger.warning(
        "[%s] Recursion cap hit for topic=%r after %d messages; "
        "salvaging from %d chars of evidence",
        work_id,
        topic_str,
        msg_count,
        len(evidence),
    )
    try:
        await _finalize_research_findings_from_evidence(partial_state, model, evidence)
    except Exception:
        logger.warning(
            "[%s] Salvage finalize failed for topic=%r",
            work_id,
            topic_str,
            exc_info=True,
        )
        return None

    candidate = _extract_findings(partial_state)
    first = candidate[0] if candidate else {}
    has_real_content = (
        bool(first.get("file_map"))
        or bool(first.get("patterns"))
        or bool(first.get("dependencies"))
    )
    if not has_real_content:
        logger.warning(
            "[%s] Salvage rejected for topic=%r: "
            "no concrete file_map / patterns / dependencies "
            "in structured output — falling through to error sentinel",
            work_id,
            topic_str,
        )
        return None

    for f in candidate:
        if isinstance(f, dict):
            if "topic" not in f:
                f["topic"] = topic_str
            f["partial"] = True
    return candidate


def _extract_findings(result: dict) -> list[dict]:
    """Extract ResearchFindings from an agent result.

    If the agent returned structured output (via response_format),
    it'll be in the structured_response key. Otherwise fall back to
    parsing the final message content.
    """
    # Try structured output first (DA's response_format processing)
    structured = result.get("structured_response")
    if structured:
        if isinstance(structured, dict):
            return [structured]
        if hasattr(structured, "model_dump"):
            return [structured.model_dump()]

    # Fall back to messages — use the last assistant message content.
    # ToolMessages (especially status="error" rebound messages from
    # ToolSchemaValidator) must be skipped: they're tool outputs, not
    # the researcher's own narration, and treating them as findings
    # produced summaries like "Tool 'foo' is not a recognized function".
    messages = result.get("messages", [])
    content = _last_ai_narrative(messages)
    if content:
        # Try to parse as JSON first (models may output JSON directly)
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return [parsed]
        except (json.JSONDecodeError, TypeError):
            pass
        if _looks_like_pure_tool_error(content, messages):
            return [
                {
                    "summary": "(no findings)",
                    "patterns": [],
                    "file_map": {},
                    "dependencies": [],
                }
            ]
        return [
            {
                "summary": content,
                "patterns": [],
                "file_map": {},
                "dependencies": [],
            }
        ]

    return [{"summary": "(no findings)", "patterns": [], "file_map": {}, "dependencies": []}]
