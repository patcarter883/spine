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
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from spine.agents.helpers import resolve_model

logger = logging.getLogger(__name__)

# ── Research manager prompt ──────────────────────────────────────────────

_RESEARCH_MANAGER_SYSTEM = """\
You are a research planning assistant. Your job is to decide what areas
of a codebase still need investigation before writing a specification or plan.

Given:
1. The work description
2. A list of research topics already explored
3. The findings accumulated so far

Decide:
- Are we done? (decision: "done") — all key areas have been investigated
- Or do we need more? (decision: "explore") — return the next 2-4 topics

Respond with ONLY a JSON object:
{"decision": "explore" | "done", "topics": ["area1", "area2"]}

Rules:
- Never return more than 4 topics in a single round.
- If you've already explored a topic, don't return it again.
- Prefer targeted, specific topics over broad ones.
- If the work description is self-contained (no codebase needed), decide "done".
- If findings already cover all key areas, decide "done".
"""


async def run_research_manager(
    state: dict[str, Any],
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Run the research manager — a single LLM call to decide next topics.

    The manager has NO tools — it receives the description and accumulated
    findings as plain text and makes one decision: explore more areas or done.

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
    # Ensure we have a BaseChatModel instance for .ainvoke().
    if isinstance(model, str):
        from langchain.chat_models import init_chat_model
        model = init_chat_model(model)

    # Build the context for the manager
    findings_summary = _summarize_findings(findings)
    context = (
        f"## Work Description\n{description}\n\n"
        f"## Round\n{round_num + 1} of max {max_rounds}\n\n"
        f"## Topics Already Explored\n{json.dumps(existing_topics)}\n\n"
        f"## Findings So Far\n{findings_summary}\n\n"
        "Decide: are we done, or do we need more research?"
    )

    try:
        response = await model.ainvoke(
            [SystemMessage(content=_RESEARCH_MANAGER_SYSTEM), HumanMessage(content=context)]
        )
        raw = response.content if hasattr(response, "content") else str(response)
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        result = json.loads(raw)
        decision = result.get("decision", "done")
        topics = result.get("topics", [])
        if not isinstance(topics, list):
            topics = []
        logger.info(
            "[%s] Research manager: decision=%s topics=%s", work_id, decision, topics
        )
        return {"manager_decision": decision, "topics": topics}
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(
            "[%s] Research manager failed to parse response: %s — defaulting to done",
            work_id,
            e,
        )
        return {"manager_decision": "done", "topics": []}
    except Exception as e:
        logger.warning(
            "[%s] Research manager LLM call failed: %s — defaulting to done",
            work_id,
            e,
        )
        return {"manager_decision": "done", "topics": []}


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
    from spine.models.state import WorkflowState as _WS  # noqa: N814

    work_id = state.get("work_id", "unknown")
    topic_str = topic or "general codebase investigation"

    logger.info("[%s] Explore node: researching topic=%r", work_id, topic_str)

    try:
        # Build the researcher subagent spec
        subagent_spec = build_subagent_spec(
            name="researcher",
            phase=PhaseName.SPECIFY,
            state=state,  # type: ignore[arg-type]
            config=config,
        )

        # Build a minimal agent for this subagent — no filesystem middleware,
        # the tools are injected directly as extra_tools from the subagent spec.
        agent = build_phase_agent(
            state=state,  # type: ignore[arg-type]
            config=config,
            phase=PhaseName.SPECIFY,
            system_prompt=subagent_spec["system_prompt"],
            is_subagent=True,
            extra_tools=subagent_spec.get("tools", []),
            response_format=subagent_spec.get("response_format"),
            skip_filesystem_middleware=True,
        )

        prompt = (
            f"## Research Topic\n{topic_str}\n\n"
            "Investigate this specific area of the codebase. "
            "Use MCP tools first for structural navigation, "
            "then fall back to read_file/glob/grep if needed. "
            "Return your findings in the ResearchFindings format."
        )

        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name="explore",
            work_id=work_id,
        )

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
