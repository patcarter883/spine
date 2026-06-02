"""Supervisor↔worker micro-loop for the per-topic researcher.

Replaces the free-form "one big agent.astream() with a recursion cap"
researcher pattern with a deterministic loop:

1. ``run_supervisor_node`` — no-tool LLM call that emits a
   :class:`SupervisorDirective` (analysis + ``is_complete`` flag + the
   ``next_directive`` to execute + which :class:`ToolClass` the worker
   may use this turn).

2. ``run_worker_node`` — agent.invoke against a pre-built worker whose
   tool surface is restricted to the supervisor's chosen tool class.
   Returns a :class:`StructuredFinding` describing what the single turn
   produced.

The loop alternates these two until ``is_complete=True`` or the per-phase
cycle cap (``ConvergenceConfig.researcher_supervisor_max_cycles_*``)
fires. This module is the SHARED infrastructure for that pattern; the
loop itself is driven from
:func:`spine.agents.exploration_agents.run_explore_do_node`.

This mirrors :mod:`spine.agents.plan_do` (single plan→do hop). The
researcher case is a *looped* generalisation with one extra concept:
tool-class allowlisting per worker turn.
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import TYPE_CHECKING, Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field

from spine.agents.helpers import (
    bind_structured_output,
    cap_completion_tokens,
    resolve_chat_model,
)
from spine.agents.prompt_format import (
    Tag,
    hostage_layout,
    xml_block,
    xml_blocks,
)

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


# ── Schemas ────────────────────────────────────────────────────────────


class ToolClass(str, Enum):
    """Tool-class taxonomy the supervisor picks from each turn.

    The four classes partition the researcher's seven callable actions
    (codebase_query{find_symbol, get_source, get_dependencies,
    get_dependents, search}, ast_extract_symbol, search_codebase) into
    distinct investigation phases — name discovery, body inspection,
    call-graph tracing, content search.
    """

    SEARCH = "search"
    FIND_SYMBOL = "find_symbol"
    READ_SOURCE = "read_source"
    TRACE_DEPS = "trace_deps"


# Tool-level allowlist per class. ``codebase_query`` spans multiple
# classes (its ``action`` arg selects the behaviour), so action-level
# restriction is enforced via the worker prompt + the directive's
# ``next_directive`` text. Tool-level restriction here ensures the worker
# can't reach for, say, ``ast_extract_symbol`` during a TRACE_DEPS turn.
TOOL_CLASS_TO_TOOLNAMES: dict[ToolClass, frozenset[str]] = {
    ToolClass.SEARCH: frozenset({"codebase_query", "search_codebase"}),
    ToolClass.FIND_SYMBOL: frozenset({"codebase_query"}),
    ToolClass.READ_SOURCE: frozenset({"codebase_query", "ast_extract_symbol"}),
    ToolClass.TRACE_DEPS: frozenset({"codebase_query"}),
}


# Human-readable hint of the codebase_query action(s) appropriate for each
# class. Rendered into the worker prompt so the model picks the correct
# action and into the supervisor prompt so it understands its choices.
TOOL_CLASS_ACTION_HINT: dict[ToolClass, str] = {
    ToolClass.SEARCH: (
        "codebase_query(action='search', pattern=...) or "
        "search_codebase(queries=[...])"
    ),
    ToolClass.FIND_SYMBOL: "codebase_query(action='find_symbol', name=...)",
    ToolClass.READ_SOURCE: (
        "codebase_query(action='get_source', name=...) or "
        "ast_extract_symbol(symbol=...)"
    ),
    ToolClass.TRACE_DEPS: (
        "codebase_query(action='get_dependencies'|'get_dependents', name=...)"
    ),
}


class FindingStatus(str, Enum):
    """Status of a single worker turn's tool call."""

    SUCCESS = "success"
    EMPTY = "empty"
    ERROR = "error"


class StructuredFinding(BaseModel):
    """One worker turn's compact result, fed back into the supervisor.

    Distinct from :class:`spine.agents.subagents.ResearchFindings`, which
    is the FINAL per-topic synthesis written into the subgraph's
    ``findings`` channel. A ``StructuredFinding`` is per-cycle evidence
    used to advance the supervisor's decision-making within the loop.
    """

    tool_name: str = Field(description="Name of the tool the worker invoked.")
    tool_class: ToolClass = Field(description="Supervisor-assigned class for this turn.")
    status: FindingStatus
    target_path: str = Field(default="", description="File path the call landed on, if any.")
    matched_symbols: list[str] = Field(
        default_factory=list,
        description="Symbols, file matches, or other anchors surfaced by the tool.",
    )
    structured_code_block: str = Field(
        default="",
        description="Capped body snippet returned by the tool (truncated to ~2 KB).",
    )
    execution_error_details: str = Field(
        default="",
        description="Error text when status=error. Empty otherwise.",
    )


class SupervisorDirective(BaseModel):
    """Supervisor's verdict + next move (or terminate).

    When ``is_complete=True``, the loop exits and the accumulated
    findings get summarised. When False, ``next_directive`` and
    ``allowed_tool_class`` MUST be populated — the worker uses both.
    """

    analysis_and_reasoning: str = Field(
        description=(
            "Two to four sentences naming what the latest finding adds, "
            "what is still missing, and why the next move is the right "
            "one. Concrete, not abstract."
        )
    )
    is_complete: bool = Field(
        description=(
            "True only when the accumulated findings cover the topic well "
            "enough to synthesise a ResearchFindings (summary + patterns + "
            "file_map + dependencies). False otherwise."
        )
    )
    next_directive: str = Field(
        default="",
        description=(
            "When is_complete=False: ONE short sentence describing the "
            "single tool call the worker should make this turn. Empty "
            "when is_complete=True."
        ),
    )
    allowed_tool_class: ToolClass | None = Field(
        default=None,
        description=(
            "When is_complete=False: the tool class the worker may use this "
            "turn (SEARCH / FIND_SYMBOL / READ_SOURCE / TRACE_DEPS). "
            "Required when is_complete=False."
        ),
    )


# ── Constants ──────────────────────────────────────────────────────────


# Cap per-section code snippet at ~2 KB so a single worker turn returning
# a multi-thousand-line file body doesn't blow the supervisor's prompt
# on the next cycle. Matches the salvage-evidence cap used elsewhere.
_FINDING_SNIPPET_CHAR_CAP = 2000


# ── Supervisor ─────────────────────────────────────────────────────────


_SUPERVISOR_SYSTEM_PROMPT = (
    xml_block(
        Tag.ROLE,
        "You are the supervisor of a per-topic codebase researcher. The "
        "researcher investigates ONE topic at a time via a worker that "
        "executes a single tool call per turn from a restricted tool class. "
        "Your job each cycle: read the latest StructuredFinding plus the "
        "evaluation history, judge whether the accumulated evidence is "
        "sufficient to write a ResearchFindings (summary + patterns + "
        "file_map + dependencies), and emit a SupervisorDirective JSON.",
    )
    + "\n\n"
    + xml_block(
        Tag.TOOLS,
        "Tool classes the worker may use (you choose one per turn):\n"
        "- SEARCH       — codebase_query(action='search') or search_codebase: "
        "regex / keyword discovery across files.\n"
        "- FIND_SYMBOL  — codebase_query(action='find_symbol'): locate a "
        "single named symbol's definition.\n"
        "- READ_SOURCE  — codebase_query(action='get_source') or "
        "ast_extract_symbol: fetch a symbol's body.\n"
        "- TRACE_DEPS   — codebase_query(action='get_dependencies'|"
        "'get_dependents'): follow the call graph.",
    )
    + "\n\n"
    + xml_block(
        Tag.CONSTRAINTS,
        "- You do NOT have tools. Do not attempt to call one.\n"
        "- Be decisive. If the evidence is thin, pick a SINGLE next move "
        "that produces concrete new information.\n"
        "- One move per turn. next_directive is ONE short sentence naming "
        "ONE tool call (e.g. 'get_source for SpineConfig.load').\n"
        "- Set is_complete=True only when file_map, patterns, and "
        "dependencies can all be populated from the current history. When "
        "you set is_complete=True, leave next_directive empty and "
        "allowed_tool_class null.\n"
        "- When is_complete=False, allowed_tool_class is REQUIRED.",
    )
)


def _build_supervisor_user_message(
    *,
    global_goal: str,
    cycle_idx: int,
    max_cycles: int,
    latest_finding_block: str,
    history_count: int,
    history_summary: str,
) -> str:
    """Build the supervisor's hostage-layout user message.

    Sections (in order): OBJECTIVE → cycle counter (inside CONSTRAINTS) →
    LATEST_FINDING → HISTORY. Final plain-text directive sits AFTER every
    closing tag per the project's hostage-prompt convention.
    """
    cycle_line = f"Cycle {cycle_idx} of {max_cycles} (hard cap)."
    # Soft landing: in the last couple of cycles, push the supervisor to
    # converge on what it already has rather than spending the remaining
    # budget chasing marginal detail. Without this nudge the loop tends to
    # run to the cap and exit as recursion_capped even when the accumulated
    # evidence is already enough to write a ResearchFindings.
    if cycle_idx >= max_cycles - 1:
        cycle_line += (
            " You are near the hard cap. Unless a single critical fact is still "
            "missing, set is_complete=True now, leave next_directive empty and "
            "allowed_tool_class null, and keep analysis_and_reasoning to your "
            "usual 2-4 sentences. Do NOT write the findings yourself and do NOT "
            "request further tool calls just to be thorough — a later step "
            "synthesises the ResearchFindings from the accumulated evidence."
        )
    return hostage_layout(
        xml_blocks(
            (Tag.OBJECTIVE, global_goal),
            (
                Tag.CONSTRAINTS,
                cycle_line,
            ),
            (Tag.LATEST_FINDING, latest_finding_block),
            (
                Tag.HISTORY,
                (
                    f"{history_count} prior finding(s).\n{history_summary}"
                    if history_count
                    else history_summary
                ),
            ),
        ),
        "Analyse the structured payload above and emit your next directive.",
    )


def _render_finding_block(finding: StructuredFinding | None) -> str:
    """Render a StructuredFinding as a labelled block for the supervisor."""
    if finding is None:
        return "(none — initialization cycle)"
    matched = ", ".join(finding.matched_symbols) if finding.matched_symbols else "None"
    snippet = finding.structured_code_block or "[No code returned]"
    err = finding.execution_error_details or "None"
    return (
        f"- Tool Used: {finding.tool_name}\n"
        f"- Tool Class: {finding.tool_class.value}\n"
        f"- Execution Status: {finding.status.value}\n"
        f"- Target File/Path: {finding.target_path or 'None'}\n"
        f"- Extracted Symbols: {matched}\n"
        f"- Error Logs (if any): {err}\n"
        f"- Code Snippet:\n\"\"\"\n{snippet}\n\"\"\""
    )


def _render_history_summary(history: list[StructuredFinding]) -> str:
    """Compact one-line-per-cycle render of prior findings for the supervisor."""
    if not history:
        return "(no prior cycles)"
    lines: list[str] = []
    for i, f in enumerate(history, 1):
        matched = ", ".join(f.matched_symbols[:5]) if f.matched_symbols else "—"
        lines.append(
            f"{i}. [{f.tool_class.value}/{f.status.value}] "
            f"{f.tool_name} → path={f.target_path or '—'} symbols={matched}"
        )
    return "\n".join(lines)


def _initialization_directive(global_goal: str) -> SupervisorDirective:
    """Seed directive emitted on cycle 0 without an LLM call.

    Starts the loop with a SEARCH for the topic. This deterministically
    primes the first worker turn instead of paying for a supervisor LLM
    call that has no evidence to evaluate yet.
    """
    return SupervisorDirective(
        analysis_and_reasoning=(
            "Initialization pass — no findings yet. Begin with a discovery "
            "call to identify symbols and files relevant to the topic."
        ),
        is_complete=False,
        next_directive=(
            f"Use search_codebase or codebase_query(action='search') to "
            f"surface files and symbols related to: {global_goal}"
        ),
        allowed_tool_class=ToolClass.SEARCH,
    )


def _terminating_directive(reason: str) -> SupervisorDirective:
    """Emit a complete=True directive when the supervisor can't run.

    Used on resolve_model failure, structured-output unsupported, or any
    invocation error — better to exit the loop and let summarise emit an
    error sentinel than to spin without a supervisor.
    """
    return SupervisorDirective(
        analysis_and_reasoning=f"(supervisor unavailable: {reason}) — terminating loop",
        is_complete=True,
        next_directive="",
        allowed_tool_class=None,
    )


def _coerce_to_chat_model(model: Any) -> Any | None:
    """Promote a model spec string to a chat model instance.

    Duck-typed (not isinstance) — accepts any object with
    ``with_structured_output``. Mirrors ``plan_do._coerce_to_chat_model``.
    """
    if isinstance(model, str):
        try:
            from langchain.chat_models import init_chat_model
        except ImportError:
            return None
        try:
            return init_chat_model(model)
        except Exception:
            logger.warning(
                "researcher_supervisor: init_chat_model failed for %r",
                model,
                exc_info=True,
            )
            return None
    if model is None:
        return None
    if hasattr(model, "with_structured_output"):
        return model
    return None


async def run_supervisor_node(
    *,
    state: dict[str, Any],
    config: "RunnableConfig | None",
    phase_path: str,
    global_goal: str,
    latest_finding: StructuredFinding | None,
    evaluation_history: list[StructuredFinding],
    cycle_idx: int,
    max_cycles: int,
) -> SupervisorDirective:
    """Run one supervisor turn — judge progress and emit the next directive.

    Cycle 0 short-circuits to a deterministic seed directive (no LLM call
    while history is empty). Any failure during model resolution or
    structured-output invocation returns a terminating directive so the
    loop exits cleanly rather than spinning.
    """
    work_id = state.get("work_id", "unknown")

    # Cycle 0 → deterministic seed. No LLM call while we have nothing to
    # analyse.
    if cycle_idx == 0 and latest_finding is None and not evaluation_history:
        return _initialization_directive(global_goal)

    session_id = work_id if work_id and work_id != "unknown" else None

    try:
        model = resolve_chat_model(config, session_id=session_id, phase=phase_path)
    except Exception:
        logger.warning(
            "[%s] researcher_supervisor: resolve_chat_model failed for phase_path=%r",
            work_id,
            phase_path,
            exc_info=True,
        )
        return _terminating_directive("resolve_chat_model failed")

    # Cap the directive call's completion budget before binding structured
    # output. A SupervisorDirective is tiny, but the resolved model inherits the
    # global max_completion_tokens (e.g. 40K); without a tight cap a local model
    # that reads the near-cap soft-landing nudge as "write the findings now" can
    # ramble into the free-text analysis field for minutes before raising
    # LengthFinishReasonError (trace 019e8679, run 019e867a). The cap fails fast;
    # the except below turns the truncation into a terminating directive so the
    # loop exits cleanly. Mirrors exploration_agents._findings_structured_model.
    try:
        from spine.config import SpineConfig

        cap = SpineConfig.load().researcher_supervisor_max_completion_tokens
        model = cap_completion_tokens(model, cap)
    except Exception:
        logger.debug(
            "[%s] researcher_supervisor: token-cap copy failed — using uncapped model",
            work_id,
            exc_info=True,
        )

    try:
        structured = bind_structured_output(model, SupervisorDirective)
    except Exception:
        logger.debug(
            "[%s] researcher_supervisor: model %r lacks with_structured_output",
            work_id,
            type(model).__name__,
            exc_info=True,
        )
        return _terminating_directive("structured output unsupported")

    user_msg = _build_supervisor_user_message(
        global_goal=global_goal,
        cycle_idx=cycle_idx + 1,
        max_cycles=max_cycles,
        latest_finding_block=_render_finding_block(latest_finding),
        history_count=len(evaluation_history),
        history_summary=_render_history_summary(evaluation_history),
    )

    try:
        response: Any = await structured.ainvoke(
            [
                SystemMessage(content=_SUPERVISOR_SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ]
        )
    except Exception:
        logger.warning(
            "[%s] researcher_supervisor: invocation failed for phase_path=%r",
            work_id,
            phase_path,
            exc_info=True,
        )
        return _terminating_directive("supervisor invocation failed")

    return _validate_directive_response(response, work_id)


def _validate_directive_response(response: Any, work_id: str) -> SupervisorDirective:
    """Coerce a model response to a SupervisorDirective.

    Accepts the four shapes Pydantic structured output can return
    (instance, ``.parsed`` wrapper, dict, JSON string). Falls back to a
    terminating directive on unrecognisable output so the loop exits
    rather than stalling.
    """
    if isinstance(response, SupervisorDirective):
        return _enforce_directive_contract(response)
    if hasattr(response, "parsed") and isinstance(response.parsed, SupervisorDirective):
        parsed = response.parsed
        response.parsed = None  # prevent Pydantic serialization warning
        return _enforce_directive_contract(parsed)
    if isinstance(response, dict):
        try:
            return _enforce_directive_contract(
                SupervisorDirective.model_validate(response)
            )
        except Exception:
            logger.warning(
                "[%s] researcher_supervisor: dict response failed validation",
                work_id,
            )
            return _terminating_directive("invalid directive shape")
    if isinstance(response, str):
        try:
            return _enforce_directive_contract(
                SupervisorDirective.model_validate(json.loads(response))
            )
        except Exception:
            return _terminating_directive("non-JSON string response")
    return _terminating_directive("unrecognized response shape")


def _enforce_directive_contract(directive: SupervisorDirective) -> SupervisorDirective:
    """Enforce the is_complete/allowed_tool_class invariant.

    If the model returns ``is_complete=False`` but forgets to set a tool
    class, we can't dispatch the worker. Treat that as a terminating
    directive rather than guessing — keeps the loop deterministic.
    """
    if not directive.is_complete and directive.allowed_tool_class is None:
        logger.warning(
            "researcher_supervisor: directive missing allowed_tool_class "
            "while is_complete=False — terminating loop"
        )
        return _terminating_directive("missing allowed_tool_class")
    return directive


# ── Worker ─────────────────────────────────────────────────────────────


def _build_worker_user_message(
    *,
    topic: str,
    directive_block: str,
    tool_class: str,
    action_hint: str,
) -> str:
    """Build the worker's hostage-layout user message.

    Sections: OBJECTIVE (topic) → DIRECTIVE (supervisor's reasoning + next
    move) → TOOLS (action hint scoped to the chosen tool class). The plain-
    text directive at the tail tells the worker to make ONE tool call and
    then narrate briefly.
    """
    return hostage_layout(
        xml_blocks(
            (Tag.OBJECTIVE, topic),
            (Tag.DIRECTIVE, directive_block),
            (
                Tag.TOOLS,
                f"Allowed tool class this turn: {tool_class}.\n{action_hint}",
            ),
        ),
        (
            "Make ONE tool call now to satisfy the directive. After the "
            "tool result returns, give a brief one-paragraph narration of "
            "what you found (no further tool calls)."
        ),
    )


def _render_directive_for_worker(directive: SupervisorDirective) -> str:
    """Render the supervisor's directive as a worker-facing block.

    Returned text is the BODY content for a ``<directive>`` tag; the caller
    wraps it via :func:`xml_blocks` / :func:`_build_worker_user_message`.
    """
    return (
        f"**Reasoning:** {directive.analysis_and_reasoning.strip()}\n"
        f"**Do this:** {directive.next_directive.strip()}"
    )


def filter_extra_tools_for_class(
    extra_tools: list[Any],
    tool_class: ToolClass,
) -> list[Any]:
    """Return the subset of ``extra_tools`` allowed for ``tool_class``.

    Uses the tool's ``.name`` attribute to match against
    :data:`TOOL_CLASS_TO_TOOLNAMES`. Unknown tools (no ``.name``) are
    dropped — the worker should only see explicitly-allowlisted tools.
    """
    allowed_names = TOOL_CLASS_TO_TOOLNAMES.get(tool_class) or frozenset()
    out: list[Any] = []
    for t in extra_tools:
        name = getattr(t, "name", None)
        if name and name in allowed_names:
            out.append(t)
    return out


def _extract_finding_from_worker_messages(
    *,
    messages: list[Any],
    tool_class: ToolClass,
) -> StructuredFinding:
    """Distil a worker invocation's messages into a StructuredFinding.

    Walks the message list looking for the LAST ``ToolMessage`` (the
    single call the worker made this turn) and builds a finding from it.
    Falls back to a synthesised finding when no tool message is present
    (the model declined to call a tool — counts as EMPTY).
    """
    last_tool_msg: ToolMessage | None = None
    last_ai_msg: AIMessage | None = None
    for msg in messages:
        if isinstance(msg, ToolMessage):
            last_tool_msg = msg
        elif isinstance(msg, AIMessage):
            last_ai_msg = msg

    if last_tool_msg is None:
        narrative = ""
        if last_ai_msg is not None:
            content = getattr(last_ai_msg, "content", "")
            if isinstance(content, str):
                narrative = content[:_FINDING_SNIPPET_CHAR_CAP]
            elif isinstance(content, list):
                parts = [
                    blk.get("text", "")
                    for blk in content
                    if isinstance(blk, dict) and blk.get("type") == "text"
                ]
                narrative = "".join(parts)[:_FINDING_SNIPPET_CHAR_CAP]
        return StructuredFinding(
            tool_name="(none)",
            tool_class=tool_class,
            status=FindingStatus.EMPTY,
            target_path="",
            matched_symbols=[],
            structured_code_block=narrative,
            execution_error_details="worker produced no tool call",
        )

    tool_name = getattr(last_tool_msg, "name", None) or "tool"
    status_attr = getattr(last_tool_msg, "status", None)
    raw_content = getattr(last_tool_msg, "content", "") or ""
    if not isinstance(raw_content, str):
        raw_content = str(raw_content)

    if status_attr == "error":
        return StructuredFinding(
            tool_name=tool_name,
            tool_class=tool_class,
            status=FindingStatus.ERROR,
            target_path="",
            matched_symbols=[],
            structured_code_block="",
            execution_error_details=raw_content[:_FINDING_SNIPPET_CHAR_CAP],
        )

    body = raw_content.strip()
    if not body:
        return StructuredFinding(
            tool_name=tool_name,
            tool_class=tool_class,
            status=FindingStatus.EMPTY,
            target_path="",
            matched_symbols=[],
            structured_code_block="",
            execution_error_details="tool returned empty result",
        )

    target_path, matched = _extract_anchors_from_tool_payload(body)
    if len(body) > _FINDING_SNIPPET_CHAR_CAP:
        body = body[: _FINDING_SNIPPET_CHAR_CAP - 1] + "…"

    return StructuredFinding(
        tool_name=tool_name,
        tool_class=tool_class,
        status=FindingStatus.SUCCESS,
        target_path=target_path,
        matched_symbols=matched,
        structured_code_block=body,
        execution_error_details="",
    )


def _extract_anchors_from_tool_payload(body: str) -> tuple[str, list[str]]:
    """Best-effort: pull a primary file path + symbol names from tool output.

    The researcher's tools return JSON-ish payloads. Parse-once with
    ``json.loads`` and harvest the common keys ``file_path`` /
    ``symbol_name`` / ``symbol`` / ``path`` / ``results``. Fail-open:
    if the payload isn't JSON, return ``("", [])`` — the supervisor will
    rely on the snippet body.
    """
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return "", []

    target_path = ""
    symbols: list[str] = []

    if isinstance(data, dict):
        target_path = str(
            data.get("file_path") or data.get("path") or ""
        )
        single_symbol = data.get("symbol_name") or data.get("symbol")
        if single_symbol:
            symbols.append(str(single_symbol))
        results = data.get("results")
        if isinstance(results, list):
            for r in results[:10]:
                if isinstance(r, dict):
                    name = r.get("symbol_name") or r.get("name") or r.get("symbol")
                    if name:
                        symbols.append(str(name))
                    if not target_path:
                        target_path = str(
                            r.get("file_path") or r.get("path") or ""
                        )
    elif isinstance(data, list):
        for r in data[:10]:
            if isinstance(r, dict):
                name = r.get("symbol_name") or r.get("name")
                if name:
                    symbols.append(str(name))
                if not target_path:
                    target_path = str(r.get("file_path") or r.get("path") or "")

    # De-dupe preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            deduped.append(s)

    return target_path, deduped


async def _execute_worker_first_tool(
    ai_msg: Any,
    tool_by_name: dict[str, Any],
    tool_class: ToolClass,
) -> tuple[str, Any]:
    """Run the FIRST tool call on ``ai_msg`` and classify the outcome.

    Returns ``(kind, payload)``:
      - ``("finding", StructuredFinding)`` — the model emitted no tool call;
        its narration was turned into a finding (terminal, no retry).
      - ``("ok", (tool_name, ToolMessage))`` — the tool executed; the caller
        extracts a finding from ``[ai_msg, ToolMessage]``.
      - ``("error", (tool_id, tool_name, error_text))`` — the tool was missing
        from the scoped surface or raised. ``error_text`` is the teaching
        message the caller may feed back to the model for ONE retry.
    """
    tool_calls = getattr(ai_msg, "tool_calls", None) or []
    if not tool_calls:
        return "finding", _extract_finding_from_worker_messages(
            messages=[ai_msg], tool_class=tool_class,
        )

    tc = tool_calls[0]
    tool_name = tc.get("name") or "(unknown)"
    tool_args = tc.get("args") or {}
    tool_id = tc.get("id") or "manual_tool_call"

    tool = tool_by_name.get(tool_name)
    if tool is None:
        return "error", (
            tool_id,
            tool_name,
            f"tool {tool_name!r} is not in your scoped tool surface for class "
            f"{tool_class.value}; available tools: {sorted(tool_by_name)}. "
            f"Call exactly one of the available tools.",
        )

    try:
        tool_output = await tool.ainvoke(tool_args)
    except Exception as exc:
        return "error", (tool_id, tool_name, f"{type(exc).__name__}: {exc}")

    # Wrap the tool output as a ToolMessage so the existing extractor
    # (which expects a message list) keeps working unchanged.
    tool_msg = ToolMessage(
        content=str(tool_output) if tool_output is not None else "",
        name=tool_name,
        tool_call_id=tool_id,
    )
    return "ok", (tool_name, tool_msg)


async def run_worker_node(
    *,
    state: dict[str, Any],
    config: "RunnableConfig | None",
    topic: str,
    directive: SupervisorDirective,
    bound_models: dict[ToolClass, tuple[Any, list[Any]]],
    system_prompt: str,
    context: Any = None,  # noqa: ARG001  — kept for API compat; unused in direct-bind path
) -> StructuredFinding:
    """ONE-SHOT supervisor-directed worker turn.

    No agent loop. The previous implementation invoked a full
    :func:`langchain.agents.create_agent` instance, which auto-cycles
    model → tools → model → tools until the model produces a non-tool
    message. The "make ONE tool call" instruction in the worker prompt
    is soft and local models routinely ignored it — trace 019e71b4
    showed worker invocations averaging 3-5 model calls each, with each
    round paying the full prompt prefix PLUS the accumulated tool-result
    history (~4 K → 24 K input tokens by round 3). That silently undid
    the supervisor's per-cycle cap and was the dominant prompt-bloat
    driver.

    New shape:
      1. Look up the model + scoped_tools for the directive's tool class.
      2. ``model.bind_tools(scoped_tools).ainvoke(...)`` ONCE.
      3. Execute the FIRST tool call from the response manually.
      4. Return a :class:`StructuredFinding` distilled from the AI
         message + tool result. No further model rounds.

    The caller is responsible for handling the supervisor's
    ``is_complete=True`` branch — by the time this function runs, the
    directive has already been validated to carry a tool class.

    Args:
        bound_models: Map of ToolClass → (model_with_tools_bound,
            scoped_tools_list). Built once per ToolClass by
            ``run_explore_do_node`` and reused across cycles.
        system_prompt: The researcher subagent's system prompt
            (``SUBAGENT_PROMPTS["researcher"]`` / ``-plan``). Sent as a
            SystemMessage to the bound model.
        context: Unused. Kept in the signature for caller compatibility
            after the agent-loop removal — ReadCacheMiddleware no longer
            wraps the worker's tool execution because there is no
            middleware stack to wrap. MCP / search tools cache
            internally; the worker only does ONE tool call per turn
            anyway, so cross-turn dedupe wasn't doing useful work here.
    """
    work_id = state.get("work_id", "unknown")
    tool_class = directive.allowed_tool_class
    if tool_class is None:
        return StructuredFinding(
            tool_name="(none)",
            tool_class=ToolClass.SEARCH,
            status=FindingStatus.ERROR,
            execution_error_details="worker received directive without tool class",
        )

    entry = bound_models.get(tool_class)
    if entry is None:
        logger.warning(
            "[%s] researcher_supervisor: no bound model for tool_class=%s",
            work_id,
            tool_class.value,
        )
        return StructuredFinding(
            tool_name="(none)",
            tool_class=tool_class,
            status=FindingStatus.ERROR,
            execution_error_details=f"no bound model for tool_class={tool_class.value}",
        )
    bound_model, scoped_tools = entry

    action_hint = TOOL_CLASS_ACTION_HINT.get(tool_class, "")
    user_msg = _build_worker_user_message(
        topic=topic,
        directive_block=_render_directive_for_worker(directive),
        tool_class=tool_class.value,
        action_hint=action_hint,
    )

    try:
        ai_msg = await bound_model.ainvoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_msg),
            ]
        )
    except Exception as exc:
        logger.warning(
            "[%s] researcher_supervisor: bound-model invocation failed (%s): %s",
            work_id,
            tool_class.value,
            exc,
            exc_info=False,
        )
        return StructuredFinding(
            tool_name="(none)",
            tool_class=tool_class,
            status=FindingStatus.ERROR,
            execution_error_details=f"{type(exc).__name__}: {exc}",
        )

    tool_by_name = {
        getattr(t, "name", None): t for t in scoped_tools if getattr(t, "name", None)
    }
    base_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg),
    ]

    # Execute the FIRST tool call ONLY. The model may have emitted more, but
    # the supervisor governs convergence and explicitly asked for a single
    # move this turn — extra tool calls would just inflate the next cycle's
    # history block.
    kind, payload = await _execute_worker_first_tool(ai_msg, tool_by_name, tool_class)
    if kind == "finding":
        return payload  # no tool call — narration-only finding
    if kind == "ok":
        _name, tool_msg = payload
        return _extract_finding_from_worker_messages(
            messages=[ai_msg, tool_msg], tool_class=tool_class,
        )

    # kind == "error": the single move failed. The worker is one-and-done by
    # design (7296969), but a malformed call that raised a *teaching* error
    # (codebase_query / search_codebase emit one with a worked example) is
    # recoverable in ONE more shot: feed the error back as a ToolMessage and
    # let the model re-pick its arguments. Bounded to a single retry so the
    # token economy stays flat. Without it the cycle is wasted AND the error
    # text can leak into the evidence dossier (trace 019e784c).
    tool_id, tool_name, error_text = payload
    error_tool_msg = ToolMessage(
        content=error_text,
        name=tool_name,
        tool_call_id=tool_id,
        status="error",
    )
    try:
        retry_ai_msg = await bound_model.ainvoke([*base_messages, ai_msg, error_tool_msg])
    except Exception as exc:
        logger.warning(
            "[%s] researcher_supervisor: retry invocation failed (%s): %s",
            work_id, tool_class.value, exc,
        )
        return StructuredFinding(
            tool_name=tool_name,
            tool_class=tool_class,
            status=FindingStatus.ERROR,
            execution_error_details=f"retry invocation failed: {type(exc).__name__}: {exc}",
        )

    logger.info(
        "[%s] researcher_supervisor: worker retried %s after teachable tool error",
        work_id, tool_name,
    )
    kind2, payload2 = await _execute_worker_first_tool(
        retry_ai_msg, tool_by_name, tool_class
    )
    if kind2 == "finding":
        return payload2
    if kind2 == "ok":
        _name2, tool_msg2 = payload2
        return _extract_finding_from_worker_messages(
            messages=[retry_ai_msg, tool_msg2], tool_class=tool_class,
        )
    # Retry also failed — give up (no second retry). render_history_as_evidence
    # skips ERROR findings, so the error text stays out of the dossier.
    _tid2, tool_name2, error_text2 = payload2
    return StructuredFinding(
        tool_name=tool_name2,
        tool_class=tool_class,
        status=FindingStatus.ERROR,
        execution_error_details=error_text2,
    )


# ── Loop helpers (used by run_explore_do_node) ─────────────────────────


def render_history_as_evidence(history: list[StructuredFinding]) -> str:
    """Render the loop's history into the tool_results_text dossier shape
    that :func:`spine.agents.exploration_agents.run_summarise_node` already
    consumes.

    Mirrors :func:`spine.agents.exploration_agents.collect_exploration_evidence`'s
    output format (``### Tool result: <name>\\n<body>``) so the summarise
    node needs no changes.
    """
    if not history:
        return ""
    sections: list[str] = []
    for i, f in enumerate(history, 1):
        if f.status == FindingStatus.ERROR:
            # Skip error sections — summarise/sentinel logic already handles
            # the no-evidence case, and we don't want error narration to
            # become a "finding".
            continue
        if f.status == FindingStatus.EMPTY:
            continue
        body = f.structured_code_block or "(no body returned)"
        sections.append(
            f"### Tool result: {f.tool_name} "
            f"(class={f.tool_class.value}, target={f.target_path or '—'})\n"
            f"{body}"
        )
    return "\n\n".join(sections)
