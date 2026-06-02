"""Plan-before-do helper for SPINE subagent nodes.

Smaller local models (Qwen3-class) get fixated on the original request
and the accumulating chat history when one graph node must both reason
about approach AND drive tools. The fix is to bisect each subagent
node into two graph nodes:

1. A ``plan`` node that runs the model with NO tools attached and asks
   for a structured :class:`SubagentDirective`. Because there are no
   tools, the model has nothing to do except think about approach.
2. A ``do`` node — the existing agent — that prepends the directive to
   its prompt and then executes with its normal tool surface.

This module is the shared infrastructure for that pattern. The
exploration subgraph uses a different (do→summarise) split implemented
in :mod:`spine.agents.exploration_agents`.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from spine.agents.helpers import resolve_chat_model
from spine.agents.prompt_format import (
    Tag,
    hostage_layout,
    xml_block,
    xml_blocks,
)

logger = logging.getLogger(__name__)


class SubagentDirective(BaseModel):
    """Structured plan output produced by the plan node.

    Consumed by the do node, which prepends a rendered version of this
    directive to its task prompt. The do node is free to deviate when
    new information surfaces — the directive is advisory, not a contract.
    """

    approach: str = Field(
        description=(
            "One to three sentences describing how the do node should "
            "approach this task. Concrete, not abstract."
        )
    )
    target_files: list[str] = Field(
        default_factory=list,
        description=(
            "Files the do node should read, edit, or create. Use repo-"
            "relative paths."
        ),
    )
    tool_calls_to_make: list[str] = Field(
        default_factory=list,
        description=(
            "Short labels for tool calls the do node should issue, e.g. "
            "'codebase_query find_symbol X' or 'write_structured_plan'. "
            "Advisory only."
        ),
    )
    acceptance: list[str] = Field(
        default_factory=list,
        description=(
            "Bullets the do node's output must satisfy for this task to "
            "be considered done."
        ),
    )
    notes: str = Field(
        default="",
        description="Optional caveats, risks, or context the do node should keep in mind.",
    )


_PLAN_SYSTEM_PROMPT = (
    xml_block(
        Tag.ROLE,
        "You are the planner for a SPINE subagent step. You will be shown a "
        "task description. Your job is to think about the APPROACH and return "
        "a SubagentDirective JSON.",
    )
    + "\n\n"
    + xml_block(
        Tag.CONSTRAINTS,
        "- You do NOT have any tools. Do not attempt to call one.\n"
        "- Do not implement the task. Plan it.\n"
        "- Keep the directive short and concrete. The do node will read it.\n"
        "- If the task names specific files, list them in target_files.\n"
        "- If the task implies specific tool calls, list short labels in "
        "tool_calls_to_make. If unsure, leave the list empty.\n"
        "- acceptance bullets should be testable conditions, not vague goals.",
    )
)


def empty_directive(reason: str = "") -> SubagentDirective:
    """Return a stub directive used when the plan node fails or is skipped.

    The do node still runs against an empty directive — better to lose the
    planning step than block the whole subgraph.
    """
    return SubagentDirective(
        approach=(
            f"(no directive produced: {reason})" if reason else "(no directive produced)"
        ),
        target_files=[],
        tool_calls_to_make=[],
        acceptance=[],
        notes="",
    )


def format_directive_for_prompt(directive: SubagentDirective | dict) -> str:
    """Render a SubagentDirective wrapped in a ``<directive>`` block.

    Output is XML-bounded so it can be safely concatenated into a parent
    user prompt without conflating with the surrounding instructions —
    see :mod:`spine.agents.prompt_format` for the convention. Accepts
    either a SubagentDirective instance or the dict shape LangGraph
    round-trips through state.
    """
    if isinstance(directive, SubagentDirective):
        data = directive.model_dump()
    elif isinstance(directive, dict):
        data = directive
    else:
        return ""

    lines: list[str] = []
    approach = (data.get("approach") or "").strip()
    if approach:
        lines.append(f"**Approach:** {approach}")

    targets = data.get("target_files") or []
    if targets:
        lines.append("**Target files:**")
        for t in targets:
            lines.append(f"- {t}")

    calls = data.get("tool_calls_to_make") or []
    if calls:
        lines.append("**Suggested tool calls:**")
        for c in calls:
            lines.append(f"- {c}")

    acceptance = data.get("acceptance") or []
    if acceptance:
        lines.append("**Acceptance:**")
        for a in acceptance:
            lines.append(f"- {a}")

    notes = (data.get("notes") or "").strip()
    if notes:
        lines.append(f"**Notes:** {notes}")

    body = "\n".join(lines).strip()
    if not body:
        return ""
    return xml_block(Tag.DIRECTIVE, body)


def _coerce_to_chat_model(model: Any) -> Any | None:
    """Promote a model spec string to a chat model instance.

    Returns ``None`` if coercion fails — the caller falls back to
    :func:`empty_directive`. Duck-typed (not ``isinstance``) because the
    plan node only needs ``model.with_structured_output(...)``: any object
    that exposes that method is acceptable, including test fakes.
    """
    if isinstance(model, str):
        try:
            from langchain.chat_models import init_chat_model
        except ImportError:
            return None
        try:
            return init_chat_model(model)
        except Exception:
            logger.warning("plan_do: init_chat_model failed for %r", model, exc_info=True)
            return None
    if model is None:
        return None
    # Duck-type — accept anything with the required method.
    if hasattr(model, "with_structured_output"):
        return model
    return None


async def run_plan_node(
    *,
    state: dict[str, Any],
    config: RunnableConfig | None,
    phase_path: str,
    task_description: str,
    role_hint: str = "",
    workspace_root: str | None = None,
) -> SubagentDirective:
    """Run the no-tool planner for a subagent step.

    Args:
        state: The current subgraph state (for ``work_id`` logging only).
        config: LangGraph runtime config (used by :func:`resolve_model`).
        phase_path: The phase or subagent path used for model override
            resolution — e.g. ``"plan"``, ``"implement/subagents/slice-implementer"``.
        task_description: The task the do node will be asked to perform.
            The planner reads this and produces a directive.
        role_hint: Optional one-line description of which subagent the
            directive is for (rendered into the planner's user message).
        workspace_root: Project workspace root.  When provided and onboarding
            injection is enabled, a bounded excerpt of the phase-relevant
            onboarding document is prepended to the system prompt so the
            planner has project context even though it bypasses
            ``build_phase_agent``.

    Returns:
        A :class:`SubagentDirective`. On any failure returns
        :func:`empty_directive` so the do node can still proceed.
    """
    work_id = state.get("work_id", "unknown")
    session_id = work_id if work_id and work_id != "unknown" else None

    try:
        model = resolve_chat_model(config, session_id=session_id, phase=phase_path)
    except Exception:
        logger.warning(
            "[%s] plan_do: resolve_chat_model failed for phase_path=%r",
            work_id,
            phase_path,
            exc_info=True,
        )
        return empty_directive("resolve_chat_model failed")

    try:
        structured = model.with_structured_output(SubagentDirective)
    except Exception:
        logger.debug(
            "[%s] plan_do: model %r lacks with_structured_output — skipping plan",
            work_id,
            type(model).__name__,
            exc_info=True,
        )
        return empty_directive("structured output unsupported")

    # Build the effective system prompt, optionally prepending an onboarding
    # excerpt.  This function uses a raw ainvoke path that bypasses
    # build_phase_agent, so we handle the injection explicitly here.
    effective_system_prompt = _PLAN_SYSTEM_PROMPT
    if workspace_root:
        try:
            from spine.agents.factory import _onboarding_injection_enabled
            from spine.agents.skills_resolver import (
                _PHASE_PRIMARY_DOC,
                load_onboarding_excerpt,
            )

            if _onboarding_injection_enabled():
                _primary_doc = _PHASE_PRIMARY_DOC.get(phase_path)
                if _primary_doc:
                    _excerpt = load_onboarding_excerpt(
                        workspace_root, _primary_doc, max_bytes=4_000
                    )
                    if _excerpt:
                        effective_system_prompt = (
                            xml_block(Tag.ONBOARDING_DOCS, _excerpt)  # Tag from module imports
                            + "\n\n"
                            + _PLAN_SYSTEM_PROMPT
                        )
        except Exception:
            pass  # fail-open — orientation is best-effort

    human_content = hostage_layout(
        xml_blocks(
            (Tag.ROLE, role_hint.strip()),
            (Tag.OBJECTIVE, task_description.strip()),
        ),
        "Return a SubagentDirective JSON describing how the do node should proceed.",
    )

    try:
        response: Any = await structured.ainvoke(
            [
                SystemMessage(content=effective_system_prompt),
                HumanMessage(content=human_content),
            ]
        )
    except Exception:
        logger.warning(
            "[%s] plan_do: plan invocation failed for phase_path=%r",
            work_id,
            phase_path,
            exc_info=True,
        )
        return empty_directive("plan invocation failed")

    if isinstance(response, SubagentDirective):
        return response
    if hasattr(response, "parsed") and isinstance(response.parsed, SubagentDirective):
        parsed = response.parsed
        response.parsed = None  # prevent Pydantic serialization warning
        return parsed
    if isinstance(response, dict):
        try:
            return SubagentDirective.model_validate(response)
        except Exception:
            logger.warning(
                "[%s] plan_do: could not validate dict response as SubagentDirective",
                work_id,
            )
            return empty_directive("invalid directive shape")
    # Some models return a raw string with JSON inside.
    if isinstance(response, str):
        try:
            return SubagentDirective.model_validate(json.loads(response))
        except Exception:
            return empty_directive("non-JSON string response")

    return empty_directive("unrecognized response shape")


def directive_from_state(state: dict[str, Any], key: str) -> SubagentDirective | dict:
    """Pull a directive out of subgraph state.

    LangGraph round-trips Pydantic models as dicts on the way through
    channels. Return whichever shape we get — :func:`format_directive_for_prompt`
    accepts both.
    """
    raw = state.get(key)
    if raw is None:
        return empty_directive("no directive in state")
    if isinstance(raw, SubagentDirective):
        return raw
    if isinstance(raw, dict):
        return raw
    return empty_directive("directive had unexpected type")
