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
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from spine.agents.helpers import resolve_model
from spine.agents.garbage_collector import commit_findings_and_clear_search

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
You are a research planning assistant. Your job is to decide what areas
of a codebase still need investigation before writing a specification or plan.

Given:
1. The work description
2. A list of research topics already explored
3. The findings accumulated so far

Decide:
- Are we done? (decision: "done") — all key areas have been investigated
- Or do we need more? (decision: "explore") — return the next 2-4 topics

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

    When the phase is ``"plan"``, the specification content is read from
    disk and included in the context so the manager bases its decisions on
    spec requirements rather than just the work description.

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
    # Ensure we have a BaseChatModel instance for .ainvoke().
    if isinstance(model, str):
        from langchain.chat_models import init_chat_model

        model = init_chat_model(model)

    # ── Build the context for the manager ────────────────────────────
    # For PLAN phase: include the specification so the manager can decide
    # research areas based on spec requirements, not just the description.
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
        f"## Round\n{round_num + 1} of max {max_rounds}\n\n"
        f"## Topics Already Explored\n{json.dumps(existing_topics)}\n\n"
        f"## Findings So Far\n{findings_summary}\n\n"
        "Decide: are we done, or do we need more research?"
    )

    try:
        # Use response_format kwarg to enforce JSON schema at the model
        # level without leaking a Pydantic instance into LangGraph state
        # (avoids the serializer warning that .with_structured_output() causes
        # when the AIMessage.parsed field is checkpointed).
        #
        # Falls back to raw JSON extraction for providers that don't
        # support response_format with json_schema.
        schema = ResearchManagerDecision.model_json_schema()
        response_format_kwarg = {
            "type": "json_schema",
            "json_schema": {
                "name": "research_manager_decision",
                "schema": schema,
            },
        }

        try:
            response = await model.ainvoke(
                [SystemMessage(content=_RESEARCH_MANAGER_SYSTEM), HumanMessage(content=context)],
                response_format=response_format_kwarg,
            )
        except (TypeError, ValueError) as exc:
            logger.debug(
                "[%s] Model does not support response_format kwarg: %s — "
                "falling back to raw JSON extraction",
                work_id, exc,
            )
            response = await model.ainvoke(
                [SystemMessage(content=_RESEARCH_MANAGER_SYSTEM), HumanMessage(content=context)]
            )

        raw = response.content if hasattr(response, "content") else str(response)
        # Handle content blocks (list of dicts from some model wrappers)
        if isinstance(raw, list):
            raw = "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in raw
            )
        # Extract JSON from anywhere in the response — handles leading
        # whitespace, code fences, and thinking/reasoning preamble.
        json_match = re.search(
            r'\{\s*"decision"\s*:\s*"(?:explore|done)"[^}]*\}', raw, re.DOTALL
        )
        if json_match:
            result_dict = json.loads(json_match.group(0))
        else:
            # Try stripping code fences with leading whitespace
            stripped = raw.strip()
            if stripped.startswith("```"):
                inner = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
                if inner.endswith("```"):
                    inner = inner[:-3]
                result_dict = json.loads(inner.strip())
            else:
                raise json.JSONDecodeError("No JSON found in response", raw, 0)

        decision = result_dict.get("decision", "done")
        topics = result_dict.get("topics", [])

        if not isinstance(topics, list):
            topics = []
        logger.info("[%s] Research manager: decision=%s topics=%s", work_id, decision, topics)
        return {"manager_decision": decision, "topics": topics}

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
        # Add the GC tool to the researcher's toolbar so it can save findings
        # and clear context when the window gets full.
        extra_tools = list(subagent_spec.get("tools", []))
        extra_tools.append(commit_findings_and_clear_search)
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
