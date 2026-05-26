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
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from spine.agents.helpers import resolve_model

logger = logging.getLogger(__name__)

# Truncate spec content for PLAN explore agents to reasonable size
_MAX_SPEC_CHARS = 8000

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


# ── Research manager prompt ──────────────────────────────────────────────

_RESEARCH_MANAGER_SYSTEM = """\
You are a research planning assistant. Your job is to map the codebase
territory and assign precise, scoped research topics to subagents.

## How to work

1. Read the **Retrieved Symbol Summaries** section below. These are
   LLM-generated summaries of functions/classes most relevant to the work
   description, discovered by semantic vector search. They give you an
   instant architectural map — file paths, symbol names, and what each
   symbol does — without raw code.

2. From these summaries, identify which specific symbols (functions,
   classes, files) are implicated in the work. Use these to craft
   precise topics.

3. Each topic you assign must reference at least 1-2 specific symbol
   names discovered in the summaries, so the subagent knows exactly
   what to look up with MCP tools. Example:
   "Investigate CLI verbosity setup — symbols: cli/__init__.py::index,
   spine/config.py::SpineConfig"

Given:
1. The work description
2. The retrieved symbol summaries (semantic search results)
3. A list of research topics already explored
4. The findings accumulated so far

Decide:
- Are we done? (decision: "done")
- Or do we need more? (decision: "explore") — return 2-4 topics, each
  referencing specific symbol names from the summaries

Rules:
- Never return more than 4 topics in a single round.
- If you've already explored a topic, don't return it again.
- Each topic MUST name specific symbols for the subagent to look up.
- If the work description is self-contained (no codebase needed), decide "done".
- If findings already cover all key areas, decide "done".
"""


async def run_research_manager(
    state: dict[str, Any],
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Run the research manager — single LLM call to decide next topics.

    On the first round, pre-runs a semantic recall (summaries only) to
    give the manager an instant architectural map. This avoids raw code
    bloat while still letting the manager identify precise symbol names
    to include in research topics.

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
            "vector search. Use these to name specific symbols in your "
            "research topics.\n\n"
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
    context = (
        f"## Work Description\n{description}\n\n"
        f"{spec_section}"
        f"{recall_section}"
        f"## Round\n{round_num + 1} of max {max_rounds}\n\n"
        f"## Topics Already Explored\n{json.dumps(existing_topics)}\n\n"
        f"## Findings So Far\n{findings_summary}\n\n"
        "Decide: are we done, or do we need more research? "
        "If exploring, each topic MUST name specific symbols from the summaries above."
    )

    try:
        # Use model.with_structured_output() for proper Pydantic validation.
        # The local vLLM can produce syntactically-valid but semantically-
        # garbage JSON (e.g. topics=[" ["]) that json.loads() accepts —
        # only Pydantic validation catches this.  We extract plain values
        # into a dict so the Pydantic instance never leaks into LangGraph
        # state (avoids the checkpoint serializer warning on AIMessage.parsed).
        structured_model = model.with_structured_output(ResearchManagerDecision)
        response = await structured_model.ainvoke(
            [SystemMessage(content=_RESEARCH_MANAGER_SYSTEM), HumanMessage(content=context)],
        )

        # .with_structured_output() may return the Pydantic instance directly
        # (newer LangChain) or in AIMessage.parsed (legacy providers).
        if isinstance(response, ResearchManagerDecision):
            parsed = response
        elif hasattr(response, "parsed") and isinstance(response.parsed, ResearchManagerDecision):
            parsed = response.parsed
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


def _summarize_findings(findings: list[dict]) -> str:
    """Create a compact summary of accumulated research findings."""
    if not findings:
        return "(no findings yet)"
    parts = []
    for i, f in enumerate(findings):
        if isinstance(f, dict):
            summary = f.get("summary", "")
            patterns = f.get("patterns", [])
            deps = f.get("dependencies", [])
            entry = f"Finding {i + 1}: {summary[:200]}"
            if patterns:
                entry += f"\n  Patterns: {', '.join(patterns[:5])}"
            if deps:
                entry += f"\n  Dependencies: {', '.join(deps[:5])}"
            parts.append(entry)
    return "\n\n".join(parts[:10])  # Keep compact


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
    from spine.agents.retry import ainvoke_with_retry
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
                f"sections above. Use MCP tools first for structural navigation, "
                f"then fall back to read_file/glob/grep if needed. "
                f"Return your findings in the ResearchFindings format with specific "
                f"file paths and how they relate to the spec."
            )
        else:
            prompt = (
                f"## Research Topic\n{topic_str}\n\n"
                f"Investigate this specific area of the codebase. "
                f"Use MCP tools first for structural navigation, "
                f"then fall back to read_file/glob/grep if needed. "
                f"Return your findings in the ResearchFindings format."
            )

        # Inject scratchpad into researcher prompt if available
        scratchpad = state.get("scratchpad", "")
        if scratchpad:
            prompt = prompt + "\n\n## Working Memory Scratchpad\n" + scratchpad + "\n"

        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name="explore",
            work_id=work_id,
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
        logger.error(
            "[%s] Explore node failed for topic=%r: %s",
            work_id,
            topic_str,
            e,
            exc_info=True,
        )
        findings = [
            {
                "summary": f"Research failed for topic '{topic_str}': {e}",
                "patterns": [],
                "file_map": {},
                "dependencies": [],
            }
        ]

    return {"findings": findings}


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
    final_content = ""
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip():
            final_content = content
            break
    if not final_content:
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

    # Fall back to messages — use the last assistant message content
    messages = result.get("messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip():
            # Try to parse as JSON first (models may output JSON directly)
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    return [parsed]
            except (json.JSONDecodeError, TypeError):
                pass
            return [
                {
                    "summary": content,
                    "patterns": [],
                    "file_map": {},
                    "dependencies": [],
                }
            ]

    return [{"summary": "(no findings)", "patterns": [], "file_map": {}, "dependencies": []}]
