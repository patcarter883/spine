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


# ── RLM preamble — injected into system prompts when interpreter is active ──
# The Recursive Language Model pattern (arXiv:2512.24601) uses the interpreter
# as a workspace to keep intermediate data OUT of model context. This preamble
# instructs the agent to use eval for orchestration, not iterate tool-by-tool.

_RLM_PREAMBLE = (
    "\n\n## Interpreter Workspace (eval tool)\n\n"
    "You have a persistent QuickJS runtime via the `eval` tool. "
    "The interpreter is your **orchestration workspace** — use it "
    "aggressively to keep the model context lean.\n\n"
    "**Core rules:**\n"
    "1. **Decompose in code, not conversation.** When processing ≥3 files "
    "or spawning ≥3 subagents, write a JS program in `eval` that reads "
    "files into variables, extracts structure, and dispatches work. "
    "Intermediate data stays in the interpreter — only the final synthesis "
    "returns to the model.\n"
    "2. **Parallel dispatch.** Use `Promise.all(tools.task(...))` for "
    "independent subagents. Use `Promise.allSettled()` for error-tolerant "
    "batches with retry logic.\n"
    "3. **Interpreter as working memory.** File contents, grep results, "
    "and subagent outputs live in JS variables (`window.results = ...`). "
    "Variables persist across turns (snapshots). Do NOT type raw data "
    "into conversation — process it in code.\n"
    "4. **Before each manual `read_file`/`grep` call, ask:** can I write "
    "one eval program that does this work and returns only what's needed?\n"
    "5. **Keep results compact.** The interpreter caps output at ~4000 "
    "chars. Synthesize findings — don't dump raw data.\n\n"
    "**Tools available in eval via PTC:** `tools.task(...)` for subagent "
    "delegation. Filesystem and shell tools are called directly — not "
    "from eval — but their results can be loaded into eval for processing.\n"
)


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

    # ── Memory ───────────────────────────────────────────────────────
    memory = resolve_memory(workspace_root)
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
    # RLM preamble: append interpreter instructions when available
    if has_interpreter:
        system_prompt += _RLM_PREAMBLE

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


def _add_summarization_middleware(
    middleware: list[Any],
    model: Any,
    backend: Any,
) -> None:
    """Add the DA summarization tool middleware to the middleware list.

    The summarization tool lets the agent proactively compress its context
    at opportune times (e.g. between implementation waves) instead of
    waiting for the automatic 85% context threshold.

    Falls back silently if the middleware is not available or if model
    initialization fails (e.g. missing API keys in test environments).
    """
    try:
        from deepagents.middleware.summarization import (
            create_summarization_tool_middleware,
        )

        # create_summarization_tool_middleware may internally resolve the
        # model string, which can fail in test environments without API
        # keys.  Catch and skip gracefully.
        try:
            mw = create_summarization_tool_middleware(model, backend)
            middleware.append(mw)
            logger.debug("Added summarization tool middleware")
        except Exception as exc:
            logger.debug(
                "Summarization tool middleware could not be initialized "
                "(skipping): %s", exc
            )
    except ImportError:
        logger.debug(
            "Summarization tool middleware not available "
            "(requires deepagents >= 0.6.0)"
        )
