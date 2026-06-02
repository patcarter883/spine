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
import re
import warnings
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, Field

from spine.agents._tokens import count_tokens as _count_tokens
from spine.agents.helpers import cap_completion_tokens, resolve_chat_model

try:  # openai is always present when the model is ChatOpenAI; guard for safety.
    from openai import LengthFinishReasonError as _LengthFinishReasonError
except Exception:  # pragma: no cover - openai missing → length salvage is a no-op
    class _LengthFinishReasonError(Exception):  # type: ignore[no-redef]
        """Fallback sentinel — never raised, so the except branch stays inert."""
from spine.agents.prompt_format import (
    Tag,
    hostage_layout,
    xml_block,
    xml_blocks,
)

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


# Stop words stripped before content-word overlap comparison. Short
# function words don't carry topic identity — keeping them in the
# similarity score inflates duplicate detection for unrelated topics.
_TOPIC_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "and", "any", "are", "as", "at", "be", "by", "do", "does",
    "for", "from", "has", "have", "how", "if", "in", "into", "is", "it",
    "its", "of", "on", "or", "that", "the", "their", "then", "there",
    "these", "this", "those", "to", "use", "used", "uses", "using", "via",
    "was", "we", "what", "when", "where", "which", "who", "why", "will",
    "with", "would",
    # SPINE-domain noise words — common in every research question without
    # actually distinguishing topics.
    "code", "codebase", "currently", "currently", "exist", "exists",
    "way", "ways", "approach", "approaches", "system",
})


def _topic_content_tokens(topic: str) -> frozenset[str]:
    """Extract the content-word fingerprint of a topic.

    Lowercases, splits on any non-alphanumeric run (so "CLI", "command-line",
    "command_line" all reduce to the same constituent words), strips
    short / stop tokens, and crudely de-pluralises trailing 's' on tokens
    ≥ 5 chars so "arguments"/"argument" and "patterns"/"pattern" collapse.
    Returns a frozenset for set-arithmetic in :func:`_topics_near_duplicate`.
    """
    if not topic:
        return frozenset()
    cleaned: list[str] = []
    buf: list[str] = []
    for ch in topic.lower():
        if ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                cleaned.append("".join(buf))
                buf = []
    if buf:
        cleaned.append("".join(buf))
    out: set[str] = set()
    for w in cleaned:
        if len(w) < 4:
            continue
        if w in _TOPIC_STOPWORDS:
            continue
        # Cheap singular collapse — only strip a trailing 's' so we don't
        # mangle words that legitimately end in 's' as the final letter
        # (e.g. "status" stays "status"; "arguments" → "argument").
        if len(w) >= 5 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.add(w)
    return frozenset(out)


# Topic-pair content-overlap threshold (Szymkiewicz-Simpson coefficient).
# Two topics whose smaller content-word set is ≥60 % covered by the
# other count as paraphrases. Calibrated against trace 019e6e53 where
# "How does the CLI entrypoint currently parse and handle command-line
# arguments?" and "How does the command-line interface parse and handle
# flags?" produced an overlap of 4/6 = 0.67 after stop-word stripping.
_TOPIC_DUPE_OVERLAP_THRESHOLD: float = 0.6


def _topics_near_duplicate(a: str, b: str) -> bool:
    """Return True when *a* and *b* are likely paraphrases of the same topic.

    Compares the content-word fingerprints using the overlap coefficient
    ``|A ∩ B| / min(|A|, |B|)``. This catches semantic paraphrases the
    exact-string :func:`_normalise_topic` check misses (observed in
    trace 019e6e53). The threshold is intentionally conservative — a
    false-positive merely drops one duplicate research question, which
    is recoverable, but a false-negative wastes a full explore round.
    """
    fa = _topic_content_tokens(a)
    fb = _topic_content_tokens(b)
    if not fa or not fb:
        return False
    overlap = len(fa & fb)
    if overlap == 0:
        return False
    return (overlap / min(len(fa), len(fb))) >= _TOPIC_DUPE_OVERLAP_THRESHOLD


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

_RESEARCH_MANAGER_SPECIFY = (
    xml_block(
        Tag.ROLE,
        "You are an Architectural Research Manager. Your job is to map the "
        "codebase structure so the SPECIFY orchestrator can write an accurate "
        "specification. The orchestrator needs to understand what EXISTS:\n"
        "1. Public interfaces and module boundaries — what are the contracts?\n"
        "2. Dependency relationships — what imports what and why?\n"
        "3. Conventions and patterns — naming, structure, idioms in use.\n"
        "4. Configuration and global state — what settings or singletons exist?\n\n"
        "Your task is to assign research topics to subagents (Architectural "
        "Scouts) that will discover this structural knowledge. Each topic "
        "should describe a *question* about a specific area of the codebase "
        "— in plain natural language, no symbol or file references.",
    )
    + "\n\n"
    + xml_block(
        Tag.WORKFLOW,
        "1. Read the <retrieved_code> block in the user message (if present). "
        "These are LLM-generated summaries of functions/classes most relevant "
        "to the work description, discovered by semantic vector search. Use "
        "them as background context — they tell you which areas of the "
        "codebase are likely relevant. Do NOT copy symbol names or file "
        "paths into your topics.\n\n"
        "2. Phrase each topic as a natural-language research question about "
        "an area of the codebase. Examples:\n"
        '   "How is CLI verbosity configured and threaded through to subcommands?"\n'
        '   "How does the configuration loader resolve workspace roots?"\n'
        '   "What logging conventions are used across the agent layer?"\n\n'
        "3. Do NOT name specific functions, classes, methods, or file paths "
        "in topics. A downstream lookup step will resolve each topic to the "
        "relevant symbols via semantic search — naming them yourself risks "
        "hallucinating names that don't exist.\n\n"
        "4. Topics should focus on STRUCTURE, not implementation:\n"
        "   - What are the public interfaces? (function signatures, class contracts)\n"
        "   - What are the dependency relationships? (imports, call chains)\n"
        "   - What conventions and patterns exist? (naming, error handling, config)\n"
        "   - What configuration or global state is involved? (singletons, env vars)\n"
        "   Do NOT ask subagents to propose solutions or implementation details.",
    )
    + "\n\n"
    + xml_block(
        Tag.CONSTRAINTS,
        "- Never return more than 4 topics in a single round.\n"
        "- If you've already explored a topic, don't return it again.\n"
        "- On any round after the first, every new topic MUST target a "
        "different module, layer, or concern than every topic in the "
        "topics_already_explored block. Rephrasing a prior topic with "
        "different words is NOT a new topic — if the 'Files examined' "
        "coverage already touches the area, do not propose another topic "
        "about that area.\n"
        "- Topics MUST NOT contain symbol names, function names, class "
        "names, or file paths. Use plain English descriptions of the area "
        "to investigate.\n"
        "- If the work description is self-contained (no codebase needed), "
        "decide \"done\".\n"
        "- If findings already cover all key architectural areas, decide \"done\".\n"
        "- Cover breadth before depth — ensure all implicated modules are "
        "touched before diving deeper into any one module.\n"
        "- If the <findings> block is non-empty on Round 1 of N, you are "
        "resuming a rework — the prior pass already wrote a research log. "
        "Default to decision=\"done\" so synthesis can re-run with the "
        "existing findings. Propose new topics ONLY when a critic suggestion "
        "explicitly identifies a coverage gap, and then cap at 1 topic "
        "addressing that gap.",
    )
)

_RESEARCH_MANAGER_PLAN = (
    xml_block(
        Tag.ROLE,
        "You are a Change Surface Research Manager. Your job is to identify "
        "the codebase areas that will need modification so the PLAN "
        "orchestrator can decompose the work into executable slices. The "
        "orchestrator needs to understand what CHANGES:\n"
        "1. Touch points — which areas of the codebase will need edits?\n"
        "2. Dependency chains — what else will be affected by those changes?\n"
        "3. Risks — complex data flows, global state, or tight coupling to flag?\n\n"
        "Your task is to assign research topics to subagents (Blueprint "
        "Scouts) that will map the specification requirements to the "
        "codebase surface area. Each topic should describe — in plain "
        "natural language — an area of the codebase where changes will "
        "happen, framed as a research question.",
    )
    + "\n\n"
    + xml_block(
        Tag.WORKFLOW,
        "1. Read the <specification> block in the user message (always "
        "present in PLAN phase). This describes what needs to be built or "
        "changed. Your research topics MUST directly target the codebase "
        "areas that the specification requires.\n\n"
        "2. Read the <retrieved_code> block (if present). These are LLM-"
        "generated summaries of functions/classes most relevant to the "
        "work. Use them as background context. Do NOT copy symbol names or "
        "file paths into your topics.\n\n"
        "3. Phrase each topic as a natural-language research question about "
        "an area of the codebase. Examples:\n"
        '   "How does workspace-root resolution work for spec-path handling?"\n'
        '   "How do the CLI subcommands wire arguments through to the agent factory?"\n\n'
        "4. Do NOT name specific functions, classes, methods, or file paths "
        "in topics. A downstream lookup step will resolve each topic to "
        "the relevant symbols via semantic search.\n\n"
        "5. Topics should focus on CHANGE SURFACE, not architecture:\n"
        "   - Which areas will need modification? (touch points)\n"
        "   - What imports or callers will be affected? (dependency chains)\n"
        "   - Are there complex data flows, global state, or tight coupling? (risks)\n"
        "   Do NOT ask subagents to propose solutions — only to identify "
        "what exists and what will be impacted.",
    )
    + "\n\n"
    + xml_block(
        Tag.CONSTRAINTS,
        "- Never return more than 4 topics in a single round.\n"
        "- If you've already explored a topic, don't return it again.\n"
        "- On any round after the first, every new topic MUST target a "
        "different module, layer, or concern than every topic in the "
        "topics_already_explored block. Rephrasing a prior topic with "
        "different words is NOT a new topic — if the 'Files examined' "
        "coverage already touches the area, do not propose another topic "
        "about that area.\n"
        "- Topics MUST NOT contain symbol names, function names, class "
        "names, or file paths. Use plain English descriptions of the area "
        "to investigate.\n"
        "- If findings already cover all key change areas, decide \"done\".\n"
        "- Prioritize high-risk areas (glue code, shared state, widely-"
        "imported modules) before peripheral concerns.\n"
        "- Cross-reference every topic against the specification — if the "
        "spec doesn't mention an area, don't explore it.\n"
        "- If a <prior_research> block is present, it describes the "
        "architectural map already gathered during SPECIFY. Treat its "
        "'Key files' entries as covered territory — propose PLAN topics "
        "that build on this map (touch points, change risk, dependency "
        "chains) rather than topics that would just re-discover the same "
        "files.\n"
        "- If the <findings> block is non-empty on Round 1 of N, you are "
        "resuming a rework — the prior pass already wrote a research log. "
        "Default to decision=\"done\" so synthesis can re-run with the "
        "existing findings. Propose new topics ONLY when a critic "
        "suggestion explicitly identifies a coverage gap, and then cap at "
        "1 topic addressing that gap.",
    )
)


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

    model = resolve_chat_model(config, session_id=work_id, phase="exploration/manager")

    # ── Recall context from state ────────────────────────────────────
    # The pre_research_gate (exploration_subgraph._pre_research_gate)
    # already classified the task and pulled recall hits before the loop
    # started.  Use the chunks it put on state — no second classify+recall
    # call.  task_category may be None on the first round if the gate
    # failed (we fall through to the loop anyway).
    task_category = state.get("task_category")
    retrieved = state.get("retrieved_context") or []
    retrieved_code_body = ""
    if round_num == 0 and retrieved:
        chunks = [
            (
                f"### {chunk.get('symbol_name', '?')} "
                f"({chunk.get('symbol_type', '?')} in {chunk.get('file_path', '?')})\n"
                f"{chunk.get('enriched_summary', '')[:400]}"
            )
            for chunk in retrieved
        ]
        retrieved_code_body = "\n\n".join(chunks)
        logger.info(
            "[%s] Research manager: using %d pre-recalled summaries from gate",
            work_id, len(retrieved),
        )

    # ── Build the context for the manager ────────────────────────────
    spec_body = ""
    if phase == "plan":
        spec_path = state.get("spec_path", "")
        if spec_path:
            full_path = Path(workspace_root) / spec_path / "specification.md"
            if full_path.exists():
                try:
                    spec_body = full_path.read_text(encoding="utf-8")[:_MAX_SPEC_CHARS]
                except OSError:
                    pass

    # SPECIFY's research log carried across by the PLAN state mapper. The
    # manager uses this to avoid proposing topics that would just re-map
    # files SPECIFY already examined — see _RESEARCH_MANAGER_PLAN rules.
    prior_research_body = ""
    if phase == "plan":
        prior_phase_findings = state.get("prior_phase_findings") or []
        if prior_phase_findings:
            from spine.config import SpineConfig as _SpineConfig

            prior_research_body = format_findings(
                prior_phase_findings,
                budget=_SpineConfig.load().prior_phase_findings_token_budget,
            )

    findings_summary = _summarize_findings(findings)

    # Conditional rules / framing aggregated into one <constraints> block so
    # the model sees "this turn's rules" as a single bounded region distinct
    # from "this turn's data".
    constraint_lines: list[str] = [
        f"Round {round_num + 1} of max {max_rounds}.",
    ]
    has_prior_round = round_num > 0 or bool(existing_topics) or bool(findings)
    if has_prior_round:
        constraint_lines.append(
            "Prior-round coverage: the topics_already_explored and findings "
            "blocks below describe ground already mapped. Every new topic "
            "MUST target a different module / layer / concern. Use the "
            "'Files examined' line on each finding as the authoritative "
            "coverage signal — if a file appears there, that area is "
            "covered regardless of how the prior topic was worded. "
            "Rephrasing a prior topic counts as the same topic. If findings "
            "already cover all relevant areas, decide 'done'. Before each "
            "candidate topic, silently check: which module does it target, "
            "and is that module already in a prior finding's 'Files "
            "examined'? If yes, drop it."
        )
    rework_body = ""
    if retry_count > 0 and last_critic_review:
        suggestions = last_critic_review.get("suggestions") or []
        sug_text = "\n".join(f"  - {s}" for s in suggestions) if suggestions else "  (none)"
        rework_body = (
            f"Attempt: {last_critic_review.get('attempt', retry_count + 1)}\n"
            f"Verdict: {last_critic_review.get('status', 'needs_revision')} "
            f"(tier: {last_critic_review.get('tier', 'unknown')})\n"
            f"Reason: {last_critic_review.get('reason', '')}\n"
            f"Suggestions:\n{sug_text}"
        )
        constraint_lines.append(
            "Rework pass — the critic rejected the prior output (see "
            "<critic_feedback> block). Propose 2-4 NEW topics that close the "
            "specific gaps the critic flagged (in addition to the prior-round "
            "rules above). If the prior research is already sufficient to "
            "address the verdict, decide 'done' so synthesis can rework with "
            "the existing findings."
        )
    constraints_body = "\n\n".join(constraint_lines)

    explored_roll_up = _render_explored_topic_roll_up(existing_topics, findings)

    context = hostage_layout(
        xml_blocks(
            (Tag.OBJECTIVE, description),
            (Tag.SPECIFICATION, spec_body),
            (Tag.PRIOR_RESEARCH, prior_research_body),
            (Tag.RETRIEVED_CODE, retrieved_code_body),
            (Tag.CRITIC_FEEDBACK, rework_body),
            (Tag.CONSTRAINTS, constraints_body),
            (Tag.TOPICS_ALREADY_EXPLORED, explored_roll_up),
            (Tag.FINDINGS, findings_summary),
        ),
        (
            "Decide: are we done, or do we need more research? If exploring, "
            "return 2-4 plain-language topics framed as research questions. "
            "Returning decision='explore' with an empty topics list is "
            "invalid — if you cannot articulate at least one concrete topic, "
            "decide 'done' instead. "
            "Topics MUST NOT contain symbol names, function names, class "
            "names, or file paths — a downstream lookup step resolves each "
            "topic to the relevant symbols automatically. Cross-reference "
            "each candidate topic against the topics_already_explored block "
            "before returning — if the same module / concern was already "
            "investigated or attempted, drop the candidate or decide 'done'."
        ),
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

    # ── Prepend onboarding orientation ──────────────────────────────
    # The research manager picks exploration topics that shape the entire
    # phase.  Giving it a bounded excerpt of the phase-relevant onboarding
    # doc grounds its topic selection in the project's actual architecture
    # without going through the build_phase_agent middleware stack (this
    # function uses a raw ainvoke path).
    try:
        from spine.agents.factory import _onboarding_injection_enabled
        from spine.agents.skills_resolver import (
            _PHASE_PRIMARY_DOC,
            load_onboarding_excerpt,
        )

        if _onboarding_injection_enabled():
            _primary_doc = _PHASE_PRIMARY_DOC.get(phase)
            if _primary_doc:
                _excerpt = load_onboarding_excerpt(
                    workspace_root, _primary_doc, max_bytes=6_000
                )
                if _excerpt:
                    manager_prompt = (
                        xml_block(Tag.ONBOARDING_DOCS, _excerpt) + "\n\n" + manager_prompt
                    )
    except Exception:
        pass  # fail-open — orientation is best-effort

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

        # Small local models (trace 019e72bc) sometimes pick "explore" while
        # leaving topics=[] — a structurally-valid but semantically-empty
        # response that the downstream router treats as a contract failure.
        # Treat the empty-topics shape as the model declining to research
        # further and downgrade to "done" so synthesis can proceed. The
        # alternative (generic-topic fallback) burns a whole research round
        # on trivial tasks where the model already signalled it has nothing
        # specific to investigate.
        decision = parsed.decision
        topics = parsed.topics
        if decision == "explore" and not topics:
            logger.warning(
                "[%s] Research manager: model returned explore with empty "
                "topics on round %d (phase=%s) — coercing to 'done' so the "
                "workflow can proceed. Treat this as the model declining to "
                "research further.",
                work_id, round_num + 1, phase,
            )
            decision = "done"
        result: dict[str, Any] = {"manager_decision": decision, "topics": topics}
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

    Error sentinels (``error=True``) get a one-line "attempted, no usable
    findings" entry rather than being hidden — the manager needs to see
    *that the topic was tried* so it doesn't re-propose the same question
    under different wording. The neutral marker is intentional: no error
    text is rendered (per the
    [[feedback_no_error_text_in_research_results]] rule).
    """
    if not findings:
        return "(no findings yet)"
    parts: list[str] = []
    total = 0
    for i, f in enumerate(findings):
        if not isinstance(f, dict):
            continue
        topic = _normalise_topic(f.get("topic", ""))
        if f.get("error"):
            entry = (
                f"Finding {i + 1} [topic: {topic[:160]}]: "
                "(attempted; no usable findings — do NOT re-propose this topic)"
            )
        else:
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


def _render_explored_topic_roll_up(
    existing_topics: list[str],
    findings: list[dict],
) -> str:
    """Render prior topics with their outcome inline.

    The legacy ``"## Topics Already Explored\n{json.dumps([...])}"``
    rendering presented topics as a passive list disconnected from the
    findings block — the model had to cross-reference manually and
    routinely rephrased prior questions instead of treating them as
    covered ground.

    This roll-up pairs each prior topic with one of three outcomes:

    * ``investigated; <N> files examined`` — substantive finding
    * ``attempted; no usable findings`` — error sentinel
    * ``proposed; no result recorded`` — topic in ``state.topics`` that
      never produced a finding (e.g. router dropped it as a near-dupe)

    Returns ``""`` when there is nothing to surface so the manager prompt
    can omit the section entirely on round 1.
    """
    if not existing_topics and not findings:
        return ""

    # Build a lookup keyed by normalised topic so enriched (suffix-stamped)
    # finding topics still match the bare topic the manager originally
    # emitted. Last finding for a given topic wins — typical case is one.
    by_topic: dict[str, dict] = {}
    for f in findings:
        if not isinstance(f, dict):
            continue
        t = _normalise_topic(f.get("topic", ""))
        if t:
            by_topic[t] = f

    lines: list[str] = []
    seen: set[str] = set()
    for topic in existing_topics:
        if not isinstance(topic, str) or not topic.strip():
            continue
        key = _normalise_topic(topic)
        if key in seen:
            continue
        seen.add(key)
        f = by_topic.get(key)
        if f is None:
            outcome = "proposed; no result recorded"
        elif f.get("error"):
            outcome = "attempted; no usable findings"
        else:
            file_map = f.get("file_map", {}) or {}
            n_files = len(file_map) if isinstance(file_map, dict) else 0
            outcome = f"investigated; {n_files} file(s) examined"
        lines.append(f"- {topic.strip()} — {outcome}")

    # Surface any finding whose topic never appeared in existing_topics
    # (e.g. seeded from a prior research_log on rework). The manager
    # otherwise has no way to know these were covered.
    for key, f in by_topic.items():
        if key in seen:
            continue
        seen.add(key)
        original = f.get("topic", "") or "(unknown topic)"
        if f.get("error"):
            outcome = "attempted; no usable findings"
        else:
            file_map = f.get("file_map", {}) or {}
            n_files = len(file_map) if isinstance(file_map, dict) else 0
            outcome = f"investigated; {n_files} file(s) examined"
        lines.append(f"- {original.strip()} — {outcome}")

    return "\n".join(lines)


# ── Explore node ─────────────────────────────────────────────────────────


async def run_explore_node(
    state: dict[str, Any],
    config: RunnableConfig | None = None,
    *,
    topic: str = "",
) -> dict[str, Any]:
    """Back-compat shim: run explore_do then summarise inline.

    The exploration subgraph now wires these as two separate graph nodes
    (see :func:`run_explore_do_node` and :func:`run_summarise_node`) so
    smaller models can switch cognitive modes between "use tools" and
    "convert evidence to findings". This wrapper preserves the old
    single-call contract for any direct caller.
    """
    do_update = await run_explore_do_node(state, config, topic=topic)
    merged = {**state, **do_update}
    summarise_update = await run_summarise_node(merged, config)
    # Merge: explore_do contributes read_cache, summarise contributes findings.
    out: dict[str, Any] = {}
    if "read_cache" in do_update:
        out["read_cache"] = do_update["read_cache"]
    if "findings" in summarise_update:
        out["findings"] = summarise_update["findings"]
    return out


async def run_explore_do_node(
    state: dict[str, Any],
    config: RunnableConfig | None = None,
    *,
    topic: str = "",
) -> dict[str, Any]:
    """Drive the researcher via a supervisor↔worker micro-loop.

    Replaces the legacy "one big agent.astream() with a recursion cap"
    pattern with a deterministic loop: the supervisor (no-tool LLM) emits
    a :class:`SupervisorDirective` each cycle; the worker (tool-using
    agent restricted to the supervisor's chosen tool class) executes one
    move and reports a :class:`StructuredFinding`. The loop terminates
    on ``directive.is_complete`` or the per-phase cycle cap fires.

    See :mod:`spine.agents.researcher_supervisor` for the schemas and
    helpers driving each cycle.

    Returns a dict with two keys:

    * ``exploration_evidence`` — a per-branch dict containing the
      assembled tool_results_text dossier, the supervisor's last
      reasoning, the topic, and a ``recursion_capped`` flag. Last-write-
      wins per Send branch. Shape is unchanged from the legacy node so
      :func:`run_summarise_node` consumes it without modification.
    * ``read_cache`` — propagated to the parent state for cross-branch
      dedupe (same as the legacy node).
    """
    # Note: ``build_phase_agent`` is no longer used here. Workers go
    # through the model.bind_tools() direct-invocation path (no agent
    # loop) — see _get_or_bind_model_for_class below. The supervisor
    # uses ``resolve_model`` directly via ``run_supervisor_node``.
    from spine.agents.researcher_supervisor import (
        FindingStatus,
        StructuredFinding,
        SupervisorDirective,
        ToolClass,
        filter_extra_tools_for_class,
        render_history_as_evidence,
        run_supervisor_node,
        run_worker_node,
    )
    from spine.agents.subagents import build_subagent_spec
    from spine.config import SpineConfig
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

    logger.info(
        "[%s] explore_do (phase=%s): researching topic=%r",
        work_id, phase, topic_str,
    )

    # ── Resolve per-phase cycle cap ──────────────────────────────────
    phase_enum = PhaseName(phase)
    conv_cfg = SpineConfig.load().convergence
    if phase_enum == PhaseName.PLAN:
        max_cycles = conv_cfg.researcher_supervisor_max_cycles_plan
    else:
        max_cycles = conv_cfg.researcher_supervisor_max_cycles_specify

    # ── Resolve spec content + prior SPECIFY findings (PLAN only) ────
    # These get folded into both the supervisor's global_goal framing
    # (so it understands what success looks like) and the topic string
    # passed to workers. Compact rendering — the loop is cycle-bounded,
    # so per-turn context bloat matters more than for one big stream.
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

    prior_findings_body = ""
    if phase_enum == PhaseName.PLAN:
        prior_findings = state.get("prior_phase_findings") or []
        if prior_findings:
            prior_findings_body = format_findings(
                prior_findings,
                budget=SpineConfig.load().prior_phase_findings_token_budget,
            )

    # The "global_goal" passed to the supervisor and the per-worker prompt
    # carry the topic plus (for PLAN) the spec + prior-findings context, all
    # bounded as XML blocks so the supervisor sees the same hostage-laid
    # payload the worker does. ``topic_str`` alone is the bare research
    # question; the enriched form prepends it as the OBJECTIVE block and
    # then layers SPECIFICATION / PRIOR_RESEARCH / SCRATCHPAD when present.
    enriched_topic = topic_str
    scratchpad = state.get("scratchpad", "")
    enriched_blocks = xml_blocks(
        (Tag.OBJECTIVE, topic_str),
        (Tag.SPECIFICATION, spec_content[:_MAX_SPEC_CHARS] if spec_content else ""),
        (Tag.PRIOR_RESEARCH, prior_findings_body),
        (Tag.SCRATCHPAD, scratchpad),
    )
    # When only OBJECTIVE is present (no spec, no prior research, no
    # scratchpad), keep the bare topic string so the supervisor's existing
    # short-circuit / re-rendering logic stays compact. When ANY auxiliary
    # block is present, ship the full XML payload.
    if spec_content or prior_findings_body or scratchpad:
        enriched_topic = enriched_blocks

    # ── Build worker agents lazily, one per ToolClass ────────────────
    # Each worker shares the researcher's system prompt + model; only
    # extra_tools differ. Built on first use of each class so unused
    # classes don't pay for agent construction.
    subagent_spec = build_subagent_spec(
        name="researcher",
        phase=phase_enum,
        state=state,  # type: ignore[arg-type]
        config=config,
    )
    full_extra_tools = list(subagent_spec.get("tools", []))
    worker_system_prompt = subagent_spec["system_prompt"]

    from spine.agents.context import build_context as _build_context

    ctx = _build_context(state, phase_enum)

    # ── Resolve the base chat model ONCE for the worker direct-bind path ──
    # Workers no longer go through build_phase_agent — they call the model
    # directly via ``model.bind_tools(...).ainvoke(...)`` per turn. This
    # collapses each worker invocation from N model calls (the agent loop
    # auto-cycled until a non-tool message) down to exactly ONE, which is
    # what the supervisor's per-cycle directive expects. See audit #2 on
    # trace 019e71b4 — the agent loop was the dominant prompt-bloat driver.
    _raw_worker_model = subagent_spec.get("model")
    if _raw_worker_model and not isinstance(_raw_worker_model, str):
        # Pre-built model injected by the caller (e.g. test harness).
        worker_base_model = _raw_worker_model
    else:
        worker_base_model = resolve_chat_model(
            config, session_id=work_id, phase=f"{phase_enum.value}/subagents/researcher"
        )

    # bound_models: ToolClass → (model_with_tools_bound, scoped_tools_list).
    # Built lazily on first use per tool class so unused classes pay nothing.
    bound_models: dict[ToolClass, tuple[Any, list[Any]]] = {}

    def _get_or_bind_model_for_class(tool_class: ToolClass) -> None:
        if tool_class in bound_models:
            return
        scoped_tools = filter_extra_tools_for_class(full_extra_tools, tool_class)
        if not scoped_tools:
            logger.warning(
                "[%s] explore_do: no tools available for class=%s — "
                "worker invocation will return EMPTY",
                work_id, tool_class.value,
            )
        bound = worker_base_model.bind_tools(scoped_tools) if scoped_tools else worker_base_model
        bound_models[tool_class] = (bound, scoped_tools)

    # ── Supervisor↔worker loop ───────────────────────────────────────
    history: list[StructuredFinding] = []
    latest: StructuredFinding | None = None
    last_directive: SupervisorDirective | None = None
    cycle = 0
    supervisor_phase_path = f"{phase_enum.value}/subagents/researcher/supervisor"

    while cycle < max_cycles:
        directive = await run_supervisor_node(
            state=state,
            config=config,
            phase_path=supervisor_phase_path,
            global_goal=enriched_topic,
            latest_finding=latest,
            evaluation_history=history,
            cycle_idx=cycle,
            max_cycles=max_cycles,
        )
        last_directive = directive
        if directive.is_complete:
            logger.info(
                "[%s] explore_do: supervisor complete after %d cycle(s) "
                "for topic=%r",
                work_id, cycle, topic_str,
            )
            break

        # Lazy-bind the model for this turn's tool class. Pre-populate
        # the bound_models map so run_worker_node finds it.
        if directive.allowed_tool_class is not None:
            _get_or_bind_model_for_class(directive.allowed_tool_class)

        # Direct-bind path: no agent loop, no middleware stack. The
        # worker does ONE model call + ONE tool execution per cycle.
        finding = await run_worker_node(
            state=state,
            config=config,
            topic=enriched_topic,
            directive=directive,
            bound_models=bound_models,
            system_prompt=worker_system_prompt,
            context=ctx,
        )
        history.append(finding)
        latest = finding
        cycle += 1

        logger.info(
            "[%s] explore_do: cycle %d/%d done — class=%s status=%s tool=%s",
            work_id, cycle, max_cycles,
            (directive.allowed_tool_class.value
             if directive.allowed_tool_class else "—"),
            finding.status.value,
            finding.tool_name,
        )

    cap_hit = cycle >= max_cycles and (
        last_directive is None or not last_directive.is_complete
    )

    # Off-by-one salvage: the supervisor runs BEFORE the worker each cycle, so
    # the final worker turn's finding never receives a completion verdict — the
    # loop exits on the counter first. Give the supervisor one last no-worker
    # evaluation against the full history; if it now judges the evidence
    # sufficient, this was a false cap and we proceed as a normal completion.
    if cap_hit and latest is not None:
        final_directive = await run_supervisor_node(
            state=state,
            config=config,
            phase_path=supervisor_phase_path,
            global_goal=enriched_topic,
            latest_finding=latest,
            evaluation_history=history,
            cycle_idx=max(cycle - 1, 0),
            max_cycles=max_cycles,
        )
        last_directive = final_directive
        if final_directive.is_complete:
            cap_hit = False
            logger.info(
                "[%s] explore_do: supervisor completed on final evaluation "
                "after %d cycle(s) for topic=%r",
                work_id, cycle, topic_str,
            )

    evidence_text = render_history_as_evidence(history)

    if cap_hit:
        # A capped run that still gathered substantial evidence is an expected
        # soft landing — summarise will emit a usable partial finding — not an
        # error. Reserve WARNING for the thin-evidence case that degrades to the
        # "(research did not converge)" sentinel.
        log = (
            logger.warning
            if len(evidence_text) < _SUMMARISE_MIN_EVIDENCE_CHARS
            else logger.info
        )
        log(
            "[%s] explore_do: cycle cap (%d) reached without supervisor "
            "completion for topic=%r — marking recursion_capped",
            work_id, max_cycles, topic_str,
        )
    # Supervisor's last reasoning becomes the narrative — gives summarise
    # a non-tool gloss on what the loop concluded.
    narrative = ""
    if last_directive is not None:
        narrative = (last_directive.analysis_and_reasoning or "").strip()

    evidence: dict[str, Any] = {
        "topic": topic_str,
        "phase": phase,
        "tool_results_text": evidence_text,
        "narrative": narrative,
        "recursion_capped": cap_hit,
        "error_class": None,
        "message_count": len(history),
        "supervisor_cycles": cycle,
    }
    logger.info(
        "[%s] explore_do: topic=%r — %d cycles, %d evidence chars, "
        "%d narrative chars",
        work_id, topic_str, cycle, len(evidence_text), len(narrative),
    )

    # Bubble the post-invocation deduper cache back into subgraph state so
    # sibling Send() researchers (and the next rework cycle) skip queries we
    # already ran. The reducer (_merge_read_cache) handles merge semantics.
    cache_snapshot: dict | None = None
    if ctx is not None and hasattr(ctx, "read_cache"):
        snap = getattr(ctx, "read_cache", None) or {}
        if snap:
            cache_snapshot = dict(snap)

    update: dict[str, Any] = {"exploration_evidence": evidence}
    if cache_snapshot:
        update["read_cache"] = cache_snapshot
    return update


# Minimum evidence dossier length before the summarise node will attempt
# a structured summarisation. Below this we emit the error sentinel so a
# recursion-capped or empty-tool-result run never produces a hallucinated
# paraphrase of the topic. Kept in sync with _SALVAGE_MIN_EVIDENCE_CHARS
# (defined further down) — same calibration rationale.
_SUMMARISE_MIN_EVIDENCE_CHARS = 200


def _empty_research_finding(topic_str: str, error_class: str | None = None) -> dict:
    """Build the canonical error-sentinel finding.

    The shape MUST match what ``_summarize_findings`` /
    ``_format_findings`` / ``_save_exploration_artifacts`` already drop
    via the ``error=True`` marker. The memory-pinned rule
    ``feedback_no_error_text_in_research_results`` requires that the
    ``summary`` field NEVER contain raw exception text — keep it as a
    neutral marker.
    """
    out: dict[str, Any] = {
        "topic": topic_str,
        "summary": "(research did not converge on this topic)",
        "patterns": [],
        "file_map": {},
        "dependencies": [],
        "error": True,
        "error_topic": topic_str,
    }
    if error_class:
        out["error_class"] = error_class
    return out


async def run_summarise_node(
    state: dict[str, Any],
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """No-tool summarisation: convert an evidence dossier into ResearchFindings.

    Reads ``state["exploration_evidence"]`` (populated by
    :func:`run_explore_do_node` in the same Send branch) and produces
    exactly one entry to merge into the shared ``findings`` channel
    via the existing ``operator.add`` reducer.

    Tools are deliberately not bound on this call — the model's only job
    here is structural conversion. This is the node that solves the
    "smaller model gets fixated on the original request" problem.
    """
    from spine.agents.subagents import build_subagent_spec
    from spine.models.enums import PhaseName

    work_id = state.get("work_id", "unknown")
    phase = state.get("phase")
    evidence = state.get("exploration_evidence") or {}
    topic_str = evidence.get("topic") or "general codebase investigation"
    evidence_text = evidence.get("tool_results_text") or ""
    narrative = evidence.get("narrative") or ""
    recursion_capped = bool(evidence.get("recursion_capped"))
    error_class = evidence.get("error_class")

    # Empty evidence and no narrative — emit the error sentinel. The
    # downstream filter chain (_summarize_findings, _format_findings,
    # _save_exploration_artifacts) already drops error=True entries
    # before they reach any LLM-facing or human-facing render path.
    if not evidence_text and (not narrative or _looks_like_pure_tool_error(narrative, [])):
        logger.warning(
            "[%s] summarise: topic=%r had no usable evidence — emitting sentinel",
            work_id, topic_str,
        )
        return {"findings": [_empty_research_finding(topic_str, error_class)]}

    # When evidence is below the salvage threshold and the explore_do
    # call was recursion-capped, prefer the sentinel over a hallucinated
    # paraphrase of the topic.
    if recursion_capped and len(evidence_text) < _SUMMARISE_MIN_EVIDENCE_CHARS and not narrative:
        logger.warning(
            "[%s] summarise: topic=%r recursion-capped with thin evidence — emitting sentinel",
            work_id, topic_str,
        )
        return {"findings": [_empty_research_finding(topic_str, error_class)]}

    if phase is None:
        logger.warning(
            "[%s] summarise: missing phase — emitting sentinel for topic=%r",
            work_id, topic_str,
        )
        return {"findings": [_empty_research_finding(topic_str, "MissingPhase")]}

    try:
        phase_enum = PhaseName(phase)
        subagent_spec = build_subagent_spec(
            name="researcher",
            phase=phase_enum,
            state=state,  # type: ignore[arg-type]
            config=config,
        )
        model = subagent_spec.get("model")
    except Exception:
        logger.warning(
            "[%s] summarise: could not resolve researcher model — emitting sentinel",
            work_id, exc_info=True,
        )
        return {"findings": [_empty_research_finding(topic_str, "ResolveModelFailed")]}

    if model is None:
        return {"findings": [_empty_research_finding(topic_str, "NoModel")]}

    finding = await summarise_evidence(
        model=model,
        topic=topic_str,
        evidence_text=evidence_text,
        narrative=narrative,
        recursion_capped=recursion_capped,
    )
    if finding is None:
        return {"findings": [_empty_research_finding(topic_str, "SummariseFailed")]}

    finding.setdefault("topic", topic_str)
    if recursion_capped:
        finding["partial"] = True
    logger.info(
        "[%s] summarise: topic=%r — %d file_map / %d patterns / %d dependencies",
        work_id,
        topic_str,
        len(finding.get("file_map") or {}),
        len(finding.get("patterns") or []),
        len(finding.get("dependencies") or []),
    )
    return {"findings": [finding]}


def _findings_structured_model(model: Any) -> Any | None:
    """Bind ``ResearchFindings`` structured output with a tight token cap.

    A findings JSON is small, but a local model can ramble to the global
    window cap (16K) — 207s of generation that then raises
    ``LengthFinishReasonError`` and gets discarded (trace 019e77fe). We copy
    the model with ``summarise_max_completion_tokens`` so a degenerate run is
    cut off in seconds and falls through to the sentinel, instead of burning
    the whole window. Returns ``None`` if the model lacks structured output.
    """
    from spine.agents.subagents import ResearchFindings
    from spine.config import SpineConfig

    if model is None:
        return None

    capped = model
    try:
        cap = SpineConfig.load().summarise_max_completion_tokens
        # ChatOpenAI exposes both fields; override whichever the builder set,
        # defaulting to max_completion_tokens. model_copy keeps with_structured_output.
        capped = cap_completion_tokens(model, cap)
    except Exception:
        logger.debug("findings cap: model_copy failed — using uncapped model", exc_info=True)
        capped = model

    try:
        return capped.with_structured_output(ResearchFindings)
    except Exception:
        logger.debug("findings cap: model lacks with_structured_output", exc_info=True)
        return None


def _partial_content_from_length_error(exc: Any) -> str:
    """Pull the partial assistant text out of a ``LengthFinishReasonError``.

    The exception carries a ``completion`` snapshot whose first choice holds
    the JSON the model had emitted before it hit the token ceiling. When
    streaming, ``usage`` is absent but ``message.content`` is still populated.
    Fail-open to ``""`` so the caller falls back to the sentinel.
    """
    completion = getattr(exc, "completion", None)
    if completion is None:
        return ""
    try:
        choices = getattr(completion, "choices", None) or []
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        return content if isinstance(content, str) else ""
    except Exception:  # pragma: no cover - defensive against snapshot shape drift
        return ""


def _extract_json_string_field(raw: str, key: str) -> str:
    """Best-effort extract of one string field from possibly-truncated JSON.

    Scans for ``"<key>": "`` and reads to the next unescaped quote (or EOF if
    the value itself was truncated mid-string), then JSON-unescapes the token.
    Used to recover the ``summary`` from a findings JSON the model never closed.
    """
    m = re.search(r'"' + re.escape(key) + r'"\s*:\s*"', raw)
    if not m:
        return ""
    buf: list[str] = []
    i, n = m.end(), len(raw)
    while i < n:
        c = raw[i]
        if c == "\\" and i + 1 < n:
            buf.append(raw[i : i + 2])
            i += 2
            continue
        if c == '"':
            break
        buf.append(c)
        i += 1
    token = "".join(buf)
    for candidate in (token, token.rstrip("\\")):
        try:
            return json.loads(f'"{candidate}"').strip()
        except Exception:
            continue
    return token.strip()


def _salvage_truncated_findings(raw: str) -> dict | None:
    """Recover a partial ResearchFindings dict from a truncated JSON payload.

    The model writes fields in schema order (``summary`` first under
    json_schema guided decoding), so a length-capped run almost always has a
    complete summary before it ran out filling ``patterns`` / ``file_map``.
    Returns a findings dict with the recovered summary (and any list fields
    that still parse), or ``None`` when nothing usable can be extracted.
    """
    if not raw:
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    summary = _extract_json_string_field(raw, "summary")
    if not summary:
        return None
    return {"summary": summary, "patterns": [], "file_map": {}, "dependencies": []}


async def summarise_evidence(
    *,
    model: Any,
    topic: str,
    evidence_text: str,
    narrative: str = "",
    recursion_capped: bool = False,
) -> dict | None:
    """Coerce an evidence dossier into a ResearchFindings dict.

    No tools are attached to this call. The model receives only the
    evidence (tool results + the researcher's own narration, if any) and
    the topic, and must produce a strict ResearchFindings JSON populated
    from what the evidence actually shows.

    Returns the parsed finding as a dict, or ``None`` if the model
    rejected the call or the structured output binding is unsupported.
    """
    if model is None:
        return None

    structured_model = _findings_structured_model(model)
    if structured_model is None:
        logger.debug("summarise_evidence: model lacks with_structured_output")
        return None

    constraints_body = (
        "- Use ONLY information present in the <findings> block below.\n"
        "- Do NOT use the topic name to fill in fields. If the evidence does "
        "not mention a file, pattern, or dependency, leave the field empty.\n"
        "- file_map paths MUST appear in the evidence (as file paths, symbol "
        "locations, or import targets).\n"
        "- If a field has no support in the evidence, leave it empty "
        "(``[]`` for lists, ``{}`` for file_map).\n"
        "- summary should describe what was actually discovered in the "
        "evidence, not what the agent was asked to investigate.\n"
        "- Keep summary to AT MOST 3 short paragraphs. Do NOT reproduce code, "
        "file contents, or long verbatim quotations — describe them concisely. "
        "This is a structured directive, not a report: brevity is required so "
        "the JSON stays within the token budget.\n"
        "- List at most 8 items each for patterns and dependencies; keep the "
        "most important and drop the rest."
    )
    if recursion_capped:
        constraints_body += (
            "\n- The researcher hit its step cap before producing a final "
            "report. Treat the evidence as partial — leave fields empty "
            "when unsupported."
        )

    summarise_prompt = hostage_layout(
        xml_blocks(
            (Tag.OBJECTIVE, topic),
            (Tag.CONSTRAINTS, constraints_body),
            (Tag.FINDINGS, evidence_text),
            (Tag.SCRATCHPAD, narrative.strip() if narrative else ""),
        ),
        (
            "Convert the evidence above into the ResearchFindings JSON "
            "schema following the constraints. Be concise — a short, accurate "
            "summary beats an exhaustive one."
        ),
    )

    try:
        parsed = await structured_model.ainvoke(
            [HumanMessage(content=summarise_prompt)]
        )
    except _LengthFinishReasonError as e:
        # The model ran past summarise_max_completion_tokens before closing the
        # JSON. Rather than discard the topic to the sentinel, salvage the
        # (usually complete) summary from the truncated payload — schema order
        # puts summary first, so it has almost always landed. Trace 019e8694
        # lost 2 SPECIFY topics this way before the salvage existed.
        salvaged = _salvage_truncated_findings(_partial_content_from_length_error(e))
        if salvaged is not None:
            logger.warning(
                "summarise_evidence: length cap hit for topic=%r — salvaged "
                "partial findings (%d summary chars)",
                topic,
                len(salvaged.get("summary", "")),
            )
            return salvaged
        logger.warning(
            "summarise_evidence: length cap hit for topic=%r — nothing "
            "salvageable, falling back to sentinel",
            topic,
        )
        return None
    except Exception as e:
        logger.warning("summarise_evidence: invocation failed: %s", e)
        return None

    if hasattr(parsed, "model_dump"):
        return parsed.model_dump()
    if isinstance(parsed, dict):
        return parsed
    return None


def collect_exploration_evidence(messages: list) -> str:
    """Public alias for :func:`_collect_salvage_evidence`.

    The salvage path historically owned this extractor; the explore/summarise
    split uses it on the happy path too. Keep the underscore-prefixed name
    available for the existing salvage tests.
    """
    return _collect_salvage_evidence(messages)


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
                # Attach any state accumulated by astream before the
                # exception fired so the caller can salvage messages
                # the same way it does for GraphRecursionError. Notably
                # rescues context-overflow BadRequestError after a
                # researcher has built up 80K of investigation.
                if partial_state:
                    if context is not None and hasattr(context, "read_cache"):
                        cache_snapshot = getattr(context, "read_cache", None) or {}
                        if cache_snapshot:
                            partial_state["read_cache"] = dict(cache_snapshot)
                    exc.partial_state = partial_state  # type: ignore[attr-defined]
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

    structured_model = _findings_structured_model(model)
    if structured_model is None:
        logger.debug("Model does not support with_structured_output; skipping finalize")
        return

    finalize_prompt = hostage_layout(
        xml_blocks(
            (Tag.FINDINGS, final_content),
            (
                Tag.CONSTRAINTS,
                "Populate ALL fields: summary (2-3 paragraph synthesis), "
                "patterns (distinct list items), file_map (path -> "
                "description), dependencies (distinct list items). Do NOT "
                "invent — use only information present in the report.",
            ),
        ),
        "Convert the report above into the ResearchFindings JSON schema.",
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
    if model is None or not evidence:
        return

    structured_model = _findings_structured_model(model)
    if structured_model is None:
        logger.debug(
            "Model does not support with_structured_output; skipping salvage finalize"
        )
        return

    salvage_prompt = hostage_layout(
        xml_blocks(
            (Tag.FINDINGS, evidence),
            (
                Tag.CONSTRAINTS,
                "- Use ONLY information present in the <findings> block.\n"
                "- Do NOT use the research topic name to fill in fields. If "
                "the evidence does not mention a file, pattern, or "
                "dependency, do not invent one.\n"
                "- file_map paths MUST appear in the tool results (as file "
                "paths, symbol locations, or import targets).\n"
                "- If a field has no support in the evidence, leave it "
                "empty (``[]`` for lists, ``{}`` for file_map).\n"
                "- summary should describe what was actually discovered in "
                "the tool results, not what the agent was asked to "
                "investigate.",
            ),
        ),
        (
            "The researcher agent ran out of steps before producing a final "
            "synthesis — the <findings> block above is the raw evidence it "
            "gathered. Convert it into the ResearchFindings JSON schema."
        ),
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


# ── Shared findings renderer ────────────────────────────────────────────


def format_findings(
    findings: list[dict], *, budget: int | None = None,
) -> str:
    """Render accumulated findings for inclusion in an LLM prompt.

    Used by:
    - synthesizer prompts (plan/specify) — see exploration_subgraph
    - PLAN per-topic researcher prompts (prior-phase findings inject)
    - PLAN research-manager prompt (prior-phase findings inject)

    Keeps individual findings compact — the consumer can read files
    from disk if more detail is needed. Skips error-sentinel entries
    (``error=True``) so they never leak into LLM-facing renders.

    When ``budget`` is a positive int, accumulates tokens per appended
    finding and stops once the next finding would push the total over
    budget. A trailing marker tells the consumer how many findings were
    omitted so it can request specific symbols if needed.
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
            block_parts.append(f"Key files: {json.dumps(file_map)}")
        if deps:
            block_parts.append(f"Dependencies: {', '.join(deps)}")
        block = "\n\n".join(block_parts)

        if use_budget:
            block_tokens = _count_tokens(block)
            if used_tokens + block_tokens > budget:
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


# Backwards-compatible alias for the legacy underscore name used by the
# synthesizers in exploration_subgraph.py.
_format_findings = format_findings
