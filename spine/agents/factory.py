"""SPINE agent factory — shared Deep Agent construction for all phases.

Every phase agent builder (specify, plan, implement, etc.) was duplicating
the same pattern: resolve model, build backend, optionally add interpreter
middleware, assemble system_prompt.  This module consolidates the shared
logic and adds the context engineering features:

- **Artifacts on disk** — prior artifacts are materialized to the filesystem
  and referenced by path, not inlined into the prompt.
- **Memory** — workspace AGENTS.md files are loaded via DA's ``memory``
  parameter for always-injected project conventions.
- **Skills** — phase-specific and RLM skills are loaded via DA's ``skills``
  parameter for progressive disclosure.
- **Context schema** — ``SpineContext`` provides typed per-run context that
  propagates to subagents automatically.
- **Summarization middleware** — for long-running phases, the agent can
  proactively compress its context between tasks.

Usage::

    from spine.agents.factory import build_phase_agent

    agent = build_phase_agent(
        state=state,
        config=config,
        phase=PhaseName.IMPLEMENT,
        system_prompt="You are an implementation engineer...",
    )
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.context import SpineContext
from spine.agents.helpers import resolve_model, debug_enabled
from spine.agents.interpreter import build_interpreter_middleware, interpreter_enabled

logger = logging.getLogger(__name__)


def build_phase_agent(
    state: WorkflowState,
    config: RunnableConfig | None,
    phase: PhaseName,
    system_prompt: str,
    *,
    extra_middleware: list[Any] | None = None,
    add_summarization: bool = False,
    subagents: list[Any] | None = None,
    response_format: Any | None = None,
    is_subagent: bool = False,
) -> Any:
    """Build a Deep Agent for a SPINE phase with full context engineering.

    This is the single entry point for all phase agent construction.
    It handles model resolution, backend creation, interpreter middleware,
    memory, skills, context schema, and summarization — so individual
    agent builders don't have to.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.
        phase: The phase being executed.
        system_prompt: Phase-specific system prompt (role + task framing).
        extra_middleware: Additional middleware to include.
        add_summarization: Whether to add the summarization tool middleware
            (recommended for IMPLEMENT and VERIFY).
        subagents: Optional subagent specs for this phase.
        response_format: Optional Pydantic model or ResponseFormat for
            structured output (DA ≥0.5.3).
        is_subagent: When True, skips artifact materialization and
            interpreter/summarization middleware (subagents are leaf
            agents, not orchestrators).

    Returns:
        A compiled Deep Agent ready for invocation.
    """
    from deepagents import create_deep_agent

    from spine.agents.backend import build_backend
    from spine.agents.artifacts import materialize_artifacts
    from spine.agents.skills_resolver import resolve_skills, resolve_memory

    # ── Resolve model and backend ────────────────────────────────────
    # Use phase-aware model resolution so per-phase overrides in
    # providers.phases take effect.
    model = resolve_model(config, session_id=state.get("work_id"), phase=phase.value)
    workspace_root = state.get("workspace_root", ".")
    backend = build_backend(workspace_root)

    work_id = state.get("work_id", "")

    # ── Materialize prior artifacts to disk ──────────────────────────
    # Subagents skip this — the parent already materialized.
    if not is_subagent:
        materialize_artifacts(state, workspace_root, work_id=work_id)

    # ── Middleware ───────────────────────────────────────────────────
    middleware: list[Any] = list(extra_middleware or [])

    # Interpreter middleware (only for top-level phase agents)
    has_interpreter = False
    if not is_subagent:
        has_interpreter = interpreter_enabled()
        if has_interpreter:
            middleware.append(build_interpreter_middleware(phase.value))

    # Summarization tool middleware (for long-running phases)
    if add_summarization and not is_subagent:
        _add_summarization_middleware(middleware, model, backend)

    # Context editing: trim old tool results for long-running phases
    if phase in (PhaseName.TASKS, PhaseName.IMPLEMENT, PhaseName.VERIFY) and not is_subagent:
        from spine.agents.context_editing import ToolOutputTrimmer

        middleware.append(ToolOutputTrimmer(max_full_tool_results=20))

    # ── Memory ───────────────────────────────────────────────────────
    memory = resolve_memory(workspace_root, phase=phase.value)
    if memory:
        logger.debug("Phase %s: loading memory files: %s", phase.value, memory)

    # ── Skills ───────────────────────────────────────────────────────
    # Subagents don't use the interpreter, so never include RLM skills.
    include_rlm = has_interpreter if not is_subagent else False
    skills = resolve_skills(
        phase=phase.value,
        workspace_root=workspace_root,
        include_rlm=include_rlm,
    )
    if skills:
        logger.debug("Phase %s: loading skills: %s", phase.value, skills)

    # ── Context schema ───────────────────────────────────────────────
    # SpineContext is passed at invoke time via context= kwarg.
    # It propagates to subagents automatically.
    context_schema = SpineContext

    # ── Construct the agent ──────────────────────────────────────────
    agent_kwargs: dict[str, Any] = {
        "name": f"spine-{phase.value}",
        "model": model,
        "backend": backend,
        "system_prompt": system_prompt,
        "context_schema": context_schema,
        "debug": debug_enabled(),
    }
    if middleware:
        agent_kwargs["middleware"] = middleware
    if memory:
        agent_kwargs["memory"] = memory
    if skills:
        agent_kwargs["skills"] = skills
    if subagents:
        agent_kwargs["subagents"] = subagents
    if response_format:
        agent_kwargs["response_format"] = response_format

    agent = create_deep_agent(**agent_kwargs)

    return agent


_SPINE_SUMMARY_PROMPT = """\
You are summarizing the conversation history of an autonomous code agent \
(inside a SPINE workflow phase). The agent is NOT a chatbot — it is a \
phase executor that reads files, writes code, runs tests, and dispatches \
subagents. Your summary MUST preserve the agent's working state so it \
can continue seamlessly after compaction.

PRESERVE these in your summary (in this order):

1. **Active objective**: What is the agent currently working on? Include \
the exact work description and which phase (tasks/implement/verify).

2. **Files currently being modified**: List every absolute file path the \
agent has read or written. Mark which ones have UNCOMMITTED changes.

3. **Unresolved errors**: Any compiler errors, linter failures, or test \
failures the agent has NOT yet fixed. Include the exact error messages.

4. **Feature slice status**: For each slice being implemented/verified, \
note: slice name, status (not started / in progress / done), and any \
blockers.

5. **Subagent results**: Brief summary of any subagent (researcher, \
slice-implementer, slice-verifier) results received.

6. **Offloaded history path**: If the conversation was previously \
compacted, the offloaded history file path is referenced in the summary \
message. Preserve that path so the agent can page back if needed.

STRIP: Narration ("I will now..."), planning chatter, and repeated \
file contents that are available on disk. The agent can re-read files \
from disk — do not include full file contents in the summary.
"""


def _add_summarization_middleware(
    middleware: list[Any],
    model: Any,
    backend: Any,
) -> None:
    """Add DA summarization middleware with SPINE-specific configuration.

    Three key design decisions:
    1. Token-based trigger (80K) — model-independent, leaves 48K buffer.
    2. Custom state-extraction summary prompt — preserves file paths,
       errors, slice objectives, and offloaded history path.
    3. Keep window of 20 messages — covers full edit-test-fix cycle.
    """
    try:
        from deepagents.middleware.summarization import (
            create_summarization_middleware,
            create_summarization_tool_middleware,
        )

        try:
            # Auto-summarization with aggressive token trigger
            auto_mw = create_summarization_middleware(
                model,
                backend,
                trigger=("tokens", 80000),
                keep=("messages", 20),
                summary_prompt=_SPINE_SUMMARY_PROMPT,
            )
            middleware.append(auto_mw)

            # Manual compact_conversation tool for on-demand use
            tool_mw = create_summarization_tool_middleware(model, backend)
            middleware.append(tool_mw)

            logger.debug(
                "Added summarization middleware "
                "(trigger=80K tokens, keep=20 msgs, custom prompt)"
            )
        except Exception as exc:
            # Fallback: try just the tool middleware
            logger.debug(
                "Auto-summarization middleware failed, trying tool-only: %s", exc
            )
            try:
                tool_mw = create_summarization_tool_middleware(model, backend)
                middleware.append(tool_mw)
                logger.debug("Added summarization tool middleware (fallback)")
            except Exception as exc2:
                logger.debug(
                    "Summarization middleware could not be initialized "
                    "(skipping): %s", exc2
                )
    except ImportError:
        logger.debug(
            "Summarization middleware not available "
            "(requires deepagents >= 0.5.0)"
        )
