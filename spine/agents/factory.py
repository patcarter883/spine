"""SPINE agent factory — custom agent construction with full middleware control.

Every phase agent builder (specify, plan, implement, etc.) was duplicating
the same pattern: resolve model, build backend, optionally add interpreter
middleware, assemble system_prompt.  This module consolidates the shared
logic and adds the context engineering features:

- **Artifacts on disk** — prior artifacts are materialized to the filesystem
  and referenced by path, not inlined into the prompt.
- **Memory** — workspace AGENTS.md files are loaded via ``MemoryMiddleware``
  for always-injected project conventions.
- **Skills** — phase-specific and RLM skills are loaded via ``SkillsMiddleware``
  for progressive disclosure.
- **Context schema** — ``SpineContext`` provides typed per-run context that
  propagates to subagents automatically.
- **Read cache** — ``ReadCacheMiddleware`` prevents re-read amnesia by
  returning cached metadata summaries for already-read files. The cache
  is shared across subagents via ``SpineContext``.

## Why ``create_agent`` instead of ``create_deep_agent``?

SPINE uses ``langchain.agents.create_agent`` (LangChain 1.0) directly instead
of ``deepagents.create_deep_agent``.  The Deep Agents harness auto-adds
``FilesystemMiddleware``, ``TodoListMiddleware``, ``SubAgentMiddleware``, etc.
with no way to customise the ``FilesystemMiddleware`` system prompt.  The
default prompt says **"All file paths must start with a /"**, which conflicts
with ``virtual_mode=True`` on our ``LocalShellBackend`` — absolute paths get
double-nested under the workspace root.

By assembling the middleware stack ourselves, we gain:

1. **Custom filesystem prompt** — relative-path guidance that works with
   ``virtual_mode=True``.
2. **Explicit stack** — every middleware is visible, no hidden auto-wiring.
3. **Trimmed prompts** — we only inject what SPINE needs, avoiding DA's
   conversational framing that fights SPINE's phase-executor model.
4. **Interpreter in the stack** — the ``CodeInterpreterMiddleware`` (eval tool)
   is always present for phases that enable it, with instructions in the
   filesystem prompt about how to use it for orchestration.

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

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.runnables import RunnableConfig

from deepagents.graph import BASE_AGENT_PROMPT
from deepagents.middleware.filesystem import FilesystemMiddleware, supports_execution
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware

from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.context import SpineContext
from spine.agents.helpers import resolve_model, debug_enabled
from spine.agents.interpreter import build_interpreter_middleware, interpreter_enabled

# ── Recursion limit ─────────────────────────────────────────────────
# Removed per-phase limits (2026-05). Each SPINE phase runs an entire
# Deep Agent loop inside one LangGraph node — the agent's own internal
# recursion (model→tools→model cycles) all counts toward the LangGraph
# recursion counter.  Per-phase caps like 200 were too low for agents
# that make 5-10 tool calls per turn over many turns, causing
# GraphRecursionError before the agent could finish its work.
#
# The default 9_999 is safe — real runaway loops hit the LLM token
# budget or the stall timeout long before 9_999 LangGraph steps.

logger = logging.getLogger(__name__)


# ── Model detection helpers ─────────────────────────────────────────────


def _is_anthropic_model(model: Any) -> bool:
    """Return True if the model is an Anthropic Claude instance or spec.

    Used to conditionally include ``AnthropicPromptCachingMiddleware`` —
    its ``cache_control`` breakpoints are only meaningful on Anthropic's
    API.  On other providers (OpenRouter, OpenAI, local) the middleware
    adds per-turn overhead with no benefit.

    Args:
        model: A model string (``"anthropic:claude-sonnet-4-20250514"``) or
            a pre-built ``BaseChatModel`` instance.

    Returns:
        ``True`` when the model targets Anthropic.
    """
    # Pre-built ChatAnthropic instance
    if hasattr(model, "__class__") and model.__class__.__name__ == "ChatAnthropic":
        return True
    # String model spec with anthropic provider prefix
    if isinstance(model, str) and model.startswith("anthropic:"):
        return True
    return False


# ── SPINE filesystem system prompt ───────────────────────────────────────
# Replaces DA's default "All file paths must start with a /" prompt.
# Under virtual_mode=True, paths are resolved relative to the workspace root.
# Relative paths like ".spine/artifacts/..." map correctly; absolute paths
# like "/home/user/project/.spine/..." double-nest.  This prompt teaches the
# agent the correct convention.

SPINE_FILESYSTEM_PROMPT = """\
## Filesystem Tools — ls, read_file, write_file, edit_file, glob, grep

You have access to a virtual filesystem rooted at the project workspace.

**Path conventions — VIOLATIONS BREAK EVERYTHING:**
- Use **relative paths** from the workspace root: `.spine/artifacts/file.md`, `spine/ui/pages.py`.
- A leading `/` is treated as workspace-relative (e.g. `/spine/ui/pages.py` resolves correctly).
- **NEVER use absolute Linux paths** like `/home/user/project/spine/ui/pages.py` — the
  virtual filesystem treats `/home/user/...` as a virtual path, so it gets double-nested
  under the workspace root and your files land at a path that does not exist and will never
  be found by subsequent phases. Write `spine/ui/pages.py` NOT `/home/pat/Projects/spine/spine/ui/pages.py`.
- **Path traversal** (`..`, `~`) is BLOCKED. Use relative paths only.
- **Verify paths exist** before modifying: use `ls` on parent directories, or `search_codebase`
  if available. Do not invent paths like `src/main.py` or `api/routes.py` — confirm they exist
  or that the parent directory exists for new files.
- Use **offset/limit** when reading large files. Read only what you need.

Tools:
- ls: list files in a directory
- read_file: read a file (supports offset/limit for large files, plus images/PDFs)
- write_file: create or overwrite a file at a relative path
- edit_file: find-and-replace within a file (supports replace_all)
- glob: find files matching a pattern (e.g. `**/*.py`)
- grep: search file contents with multiple output modes

## Batch reads
Never read one file per turn. Always batch: read ≥3 files or use search_codebase
instead. Sequential single-file reads waste turns and bloat context.

## Large Tool Results
When a tool result is too large, it may be offloaded to `/large_tool_results/<tool_call_id>` \
instead of being returned inline. Use `read_file` to inspect in chunks, or `grep` within \
`/large_tool_results/` to search across offloaded results."""

SPINE_FILESYSTEM_EXEC_PROMPT = SPINE_FILESYSTEM_PROMPT + """

## Execute Tool — execute

You have access to an `execute` tool for running shell commands.
Use it for commands, scripts, tests, builds, and other shell operations.
Commands run in the workspace root directory.
All paths in commands MUST be relative (e.g. `pytest tests/unit/` NOT `pytest /home/user/project/tests/unit/`).

- execute: run a shell command (returns output and exit code)"""


# ── SpineProjectMemoryMiddleware ───────────────────────────────────────────


class SpineProjectMemoryMiddleware(MemoryMiddleware):
    """Custom project memory middleware that injects AGENTS.md conventions.

    Bypasses deepagents' default, conversational `<memory_guidelines>` block
    as SPINE agents run programmatically without an interactive user, saving
    substantial context tokens and preventing early-yield confusion.
    """

    def modify_request(self, request: Any) -> Any:
        from langchain_core.messages import SystemMessage
        from deepagents.middleware._utils import append_to_system_message

        contents = request.state.get("memory_contents", {})
        if not contents:
            return request

        sections = []
        for path in self.sources:
            content = contents.get(path)
            if content:
                # Use a clean, concise XML enclosure for documentation injection
                sections.append(
                    f"<project_documentation path=\"{path}\">\n"
                    f"{content}\n"
                    f"</project_documentation>"
                )

        if not sections:
            return request

        memory_body = "\n\n".join(sections)
        new_system_message = append_to_system_message(request.system_message, memory_body)

        # Apply prompt caching breakpoint if Anthropic and enabled
        if self._add_cache_control and type(request.model).__name__ == "ChatAnthropic" and hasattr(new_system_message, "content_blocks") and new_system_message.content_blocks:
            blocks = list(new_system_message.content_blocks)
            last_block: Any = blocks[-1]
            base = last_block if isinstance(last_block, dict) else {}
            blocks[-1] = {**base, "cache_control": {"type": "ephemeral"}}  # type: ignore[invalid-assignment]
            new_system_message = SystemMessage(content_blocks=blocks)

        return request.override(system_message=new_system_message)


# ── Main factory function ────────────────────────────────────────────────

def build_phase_agent(
    state: WorkflowState,
    config: RunnableConfig | None,
    phase: PhaseName,
    system_prompt: str,
    *,
    extra_middleware: list[Any] | None = None,
    subagents: list[Any] | None = None,
    response_format: Any | None = None,
    is_subagent: bool = False,
    allowed_tools: list[str] | None = None,
    extra_tools: list[Any] | None = None,
    skip_filesystem_middleware: bool = False,
) -> Any:
    """Build a LangChain agent for a SPINE phase with full context engineering.

    This is the single entry point for all phase agent construction.
    It handles model resolution, backend creation, full middleware assembly,
    memory, skills, and context schema — so individual
    phase builders don't have to.

    Uses ``create_agent`` directly with a custom middleware stack, giving
    full control over the ``FilesystemMiddleware`` system prompt and stack
    ordering.  This replaces the previous ``create_deep_agent`` approach
    which could not customise the filesystem prompt.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.
        phase: The phase being executed.
        system_prompt: Phase-specific system prompt (role + task framing).
        extra_middleware: Additional middleware to include.
        subagents: Optional subagent specs for this phase.
        response_format: Optional Pydantic model or ResponseFormat for
            structured output (DA ≥0.5.3).
        is_subagent: When True, skips artifact materialization and
            interpreter/read-cache middleware (subagents are leaf
            agents, not orchestrators).
        extra_tools: Additional BaseTool instances injected directly into
            create_agent(tools=[...]). These sit alongside tools from
            middleware. Use for custom orchestrator tools (e.g. ReadSliceFilesTool)
            that replace generic filesystem access.
        skip_filesystem_middleware: When True, FilesystemMiddleware is omitted
            entirely. Pair with extra_tools to replace all filesystem access
            with purpose-built tools, removing any generic read/write fallback
            from the model's tool surface.

    Returns:
        A compiled agent (CompiledStateGraph) ready for invocation.
    """

    from spine.agents.backend import build_backend
    from spine.agents.artifacts import materialize_artifacts
    from spine.agents.skills_resolver import resolve_skills, resolve_memory

    # ── Resolve model and backend ────────────────────────────────────
    model = resolve_model(config, session_id=state.get("work_id"), phase=phase.value)
    workspace_root = state.get("workspace_root", ".")
    backend = build_backend(workspace_root)
    work_id = state.get("work_id", "")

    # ── MCP tools ──────────────────────────────────────────────────────
    from spine.config import SpineConfig
    from spine.mcp.client import get_mcp_tools

    mcp_tools: list = []
    try:
        config_obj = SpineConfig.load()
        mcp_tools = get_mcp_tools(
            config_obj.mcp_servers,
            cache_key=work_id or "default",
            workspace_root=workspace_root,
        )
    except Exception:
        logger.debug("MCP tool loading failed (non-fatal)", exc_info=True)

    # ── Resolve profile for prompt assembly ──────────────────────────
    # The HarnessProfile is registered per-provider by ensure_spine_profiles().
    # It holds base_system_prompt (SPINE_BASE_PROMPT) and tool_description_overrides.
    # _harness_profile_for_model needs a BaseChatModel, so if our resolve_model
    # returned a string, we pass it through DA's resolve_model first.
    _model_spec = model if isinstance(model, str) else None
    resolved_model = _resolve_model_for_profile(model)
    profile = _resolve_profile(resolved_model, _model_spec)
    base_prompt = _apply_profile_prompt(profile, BASE_AGENT_PROMPT)
    tool_desc_overrides = _get_tool_description_overrides(profile)

    # ── Materialize prior artifacts to disk ──────────────────────────
    if not is_subagent:
        materialize_artifacts(state, workspace_root, work_id=work_id)

    # ── Assemble final system prompt ─────────────────────────────────
    # Assembly order: user system_prompt → base prompt (CUSTOM slot)
    if system_prompt is None:
        final_system_prompt: str = base_prompt
    else:
        final_system_prompt = system_prompt + "\n\n" + base_prompt

    # ── MCP tool guidance ────────────────────────────────────────────
    if mcp_tools:
        mcp_names = [getattr(t, "name", "?") for t in mcp_tools]
        mcp_guidance = (
            "\n\n## Codebase Navigation Tools (MCP)\n"
            "You have access to MCP tools for efficient codebase navigation. "
            "Use these for symbol lookup, dependency analysis, and change impact "
            "assessment. They are MUCH more token-efficient than reading entire "
            "files with glob/grep/read — use them FIRST when exploring the codebase.\n"
            f"Available MCP tools: {', '.join(mcp_names[:10])}"
        )
        if len(mcp_names) > 10:
            mcp_guidance += f" and {len(mcp_names) - 10} more"
        final_system_prompt += mcp_guidance

    # ── Memory ───────────────────────────────────────────────────────
    memory = resolve_memory(workspace_root, phase=phase.value)
    if memory:
        logger.debug("Phase %s: loading memory files: %s", phase.value, memory)

    # ── Skills ───────────────────────────────────────────────────────
    # Check if interpreter will be in the stack (needed before building
    # skills, because include_rlm affects skill resolution)
    has_interpreter = not is_subagent and interpreter_enabled()
    include_rlm = has_interpreter if not is_subagent else False
    skills = resolve_skills(
        phase=phase.value,
        workspace_root=workspace_root,
        include_rlm=include_rlm,
    )
    if skills:
        logger.debug("Phase %s: loading skills: %s", phase.value, skills)

    # ── Build the middleware stack ───────────────────────────────────
    middleware = _build_middleware_stack(
        backend=backend,
        model=model,
        phase=phase,
        is_subagent=is_subagent,
        has_interpreter=has_interpreter,
        extra_middleware=extra_middleware,
        tool_desc_overrides=tool_desc_overrides,
        profile=profile,
        subagents=subagents,
        memory=memory,
        skills=skills,
        allowed_tools=allowed_tools,
        skip_filesystem_middleware=skip_filesystem_middleware,
    )

    # ── Context schema ───────────────────────────────────────────────
    context_schema = SpineContext

    # ── Construct the agent ──────────────────────────────────────────
    all_tools = list(extra_tools) if extra_tools else []
    all_tools.extend(mcp_tools)

    agent = create_agent(
        model,
        system_prompt=final_system_prompt,
        tools=all_tools,  # Custom tools + MCP tools + middleware tools
        middleware=middleware,
        response_format=response_format,
        context_schema=context_schema,
        debug=debug_enabled(),
        name=f"spine-{phase.value}",
    ).with_config(
        {
            "recursion_limit": 9999,
            "metadata": {
                "ls_integration": "spine",
                "lc_agent_name": f"spine-{phase.value}",
            },
        }
    )

    return agent


# ── Middleware stack builder ─────────────────────────────────────────────

def _build_middleware_stack(
    *,
    backend: Any,
    model: Any,
    phase: PhaseName,
    is_subagent: bool,
    has_interpreter: bool,
    extra_middleware: list[Any] | None,
    tool_desc_overrides: dict[str, str],
    profile: Any,
    subagents: list[Any] | None,
    memory: list[str] | None,
    skills: list[str] | None,
    allowed_tools: list[str] | None = None,
    skip_filesystem_middleware: bool = False,
) -> list[Any]:
    """Assemble the full middleware stack for a SPINE phase agent.

    Replaces create_deep_agent's auto-wiring with explicit, auditable
    middleware construction. Each entry is intentional and documented.

    Stack order (from outermost to innermost in the middleware chain):

    1.  TodoListMiddleware         — task planning state
    2.  SkillsMiddleware           — progressive skill disclosure
    3.  FilesystemMiddleware       — filesystem tools (skipped when skip_filesystem_middleware=True)
    4.  SubAgentMiddleware         — task delegation (BEFORE interpreter so tools.task is PTC-visible)
    5.  CodeInterpreterMiddleware  — eval tool (when enabled); sees task tool from step 4
    6.  ReadCacheMiddleware        — prevents re-read amnesia via SpineContext cache
    7.  PatchToolCallsMiddleware   — tool call normalization
    8.  SPINE-specific middleware  — ToolSchemaValidator (ToolOutputTrimmer removed 2026-05)
    9.  User extra_middleware
    10. Profile extra_middleware
    11. AnthropicPromptCachingMiddleware — prompt caching (no-op for non-Anthropic)
    12. MemoryMiddleware           — AGENTS.md injection (when memory present)

    CRITICAL ordering constraint: SubAgentMiddleware (4) MUST precede
    CodeInterpreterMiddleware (5). The interpreter's PTC system calls
    filter_tools_for_ptc(request.tools, ptc_allowlist) to bind tools onto
    globalThis.tools in the QuickJS sandbox. If SubAgentMiddleware hasn't
    run yet, `task` is not in request.tools and tools.task is undefined,
    causing "TypeError: not a function" on every tools.task() call in eval.
    """
    middleware: list[Any] = []

    # 1. Todo list — task planning state
    middleware.append(TodoListMiddleware())

    # 2. Skills — progressive disclosure of skill directories
    if skills:
        middleware.append(SkillsMiddleware(backend=backend, sources=skills))

    # 3. Filesystem — core tool surface with custom SPINE prompt
    #    Skipped when skip_filesystem_middleware=True, which is used by
    #    the implement orchestrator to replace generic filesystem access
    #    with purpose-built tools (ReadSliceFilesTool, WriteImplementationReportTool)
    #    that enforce dispatch-only behaviour at the tool level.
    if not skip_filesystem_middleware:
        fs_prompt = _get_filesystem_prompt(backend)
        fs_mw = FilesystemMiddleware(
            backend=backend,
            system_prompt=fs_prompt,
            custom_tool_descriptions=tool_desc_overrides,
        )
        # Optional tool filter — when the phase declares an allowed_tools
        # whitelist, drop everything else from the middleware's tool list.
        # Used by orchestrator phases (implement, verify) to physically
        # prevent the orchestrator from writing source code itself.
        if allowed_tools is not None:
            _filter_filesystem_tools(fs_mw, allowed_tools, phase)
        middleware.append(fs_mw)
    else:
        logger.debug(
            "Phase %s: FilesystemMiddleware skipped (skip_filesystem_middleware=True)",
            phase.value,
        )

    # 4. Subagent delegation — MUST come before interpreter so that the
    #    `task` tool is in request.tools when CodeInterpreterMiddleware's
    #    _prepare_for_call runs filter_tools_for_ptc. If SubAgentMiddleware
    #    is after the interpreter, `task` is not yet in the tool registry
    #    and tools.task resolves to undefined in the QuickJS sandbox,
    #    causing "TypeError: not a function" on every Promise.allSettled call.
    if subagents:
        middleware.append(SubAgentMiddleware(
            backend=backend,
            subagents=subagents,
            task_description=tool_desc_overrides.get("task"),
        ))

    # 5. Interpreter (eval tool) — only for top-level phase agents.
    #    SubAgentMiddleware must be above this in the stack (position 4)
    #    so that the `task` tool is visible to PTC when the interpreter
    #    installs its globalThis.tools bindings.
    if has_interpreter:
        middleware.append(build_interpreter_middleware(phase.value))

    # 6. Read cache — prevents re-reading already-seen files.  Checks
    #    SpineContext.read_cache before allowing read_file to execute.
    #    Only for top-level orchestrators, not subagent leaves.
    if not is_subagent:
        middleware.append(_build_read_cache_middleware())

    # 7. Patch tool calls — normalisation
    middleware.append(PatchToolCallsMiddleware())

    # 8. SPINE-specific middleware (tool validation, context editing)
    if not is_subagent:
        _add_spine_middleware(middleware, phase)

    # 9. User-provided extra middleware
    if extra_middleware:
        middleware.extend(extra_middleware)

    # 10. Profile extra middleware
    if profile is not None:
        try:
            extra = profile.materialize_extra_middleware()
            if extra:
                middleware.extend(extra)
        except Exception:
            pass

    # 11. Prompt caching — only for Anthropic models.
    #     Non-Anthropic models (OpenRouter, OpenAI, local) don't support
    #     Anthropic cache_control breakpoints.  Skip entirely to avoid
    #     per-turn isinstance check overhead and misleading "ran 30 times"
    #     metrics in traces.
    if _is_anthropic_model(model):
        middleware.append(AnthropicPromptCachingMiddleware())

    # 12. Memory — AGENTS.md injection (when memory sources provided)
    if memory:
        middleware.append(SpineProjectMemoryMiddleware(
            backend=backend,
            sources=memory,
            add_cache_control=True,
        ))

    return middleware


def _filter_filesystem_tools(
    fs_mw: Any,
    allowed_tools: list[str],
    phase: PhaseName,
) -> None:
    """Restrict a FilesystemMiddleware to only expose the named tools.

    Mutates the middleware's ``tools`` list in place. Used by orchestrator
    phases (implement, verify) that must dispatch work to subagents rather
    than touch source code themselves.

    Args:
        fs_mw: The FilesystemMiddleware instance to filter.
        allowed_tools: Tool names to keep (e.g. ``["ls", "read_file",
            "glob", "grep", "write_file"]``). Tools not in this list are
            dropped from the middleware.
        phase: Phase name, used only for logging.
    """
    if not hasattr(fs_mw, "tools"):
        logger.warning(
            "Phase %s: FilesystemMiddleware has no .tools attribute; "
            "cannot apply allowed_tools filter (%s)",
            phase.value, allowed_tools,
        )
        return

    allowed = set(allowed_tools)
    original_names = [t.name for t in fs_mw.tools]
    fs_mw.tools = [t for t in fs_mw.tools if t.name in allowed]
    kept = [t.name for t in fs_mw.tools]
    dropped = [n for n in original_names if n not in allowed]
    logger.debug(
        "Phase %s: filtered filesystem tools — kept=%s dropped=%s",
        phase.value, kept, dropped,
    )


def _add_spine_middleware(middleware: list[Any], phase: PhaseName) -> None:
    """Add SPINE-specific middleware for tool validation and context editing."""
    import os as _os

    # Tool schema validation — rebound loop for self-correction
    _validation_enabled = _os.getenv(
        "SPINE_TOOL_SCHEMA_VALIDATION", "true"
    ).lower() not in ("0", "false", "no")
    if _validation_enabled:
        from spine.agents.tool_schema_validator import ToolSchemaValidator
        middleware.append(ToolSchemaValidator())

    # Note: ToolOutputTrimmer was removed (2026-05) — excessive trimming of
    # filesystem results was causing agents to lose critical context during
    # implementation/verification phases. ReadCacheMiddleware now handles
    # the re-read problem at source, while ToolOutputTrimmer is retired.


def _build_read_cache_middleware() -> Any:
    """Build the ReadCacheMiddleware that prevents re-reading files.

    Each ``read_file`` call checks ``SpineContext.read_cache`` before executing.
    If the file was already read this phase, returns a compact metadata summary
    instead of re-reading. The cache is shared across subagents via context
    propagation.
    """
    from spine.agents.context_editing import ReadCacheMiddleware
    return ReadCacheMiddleware()


# ── Profile helpers ──────────────────────────────────────────────────────

def _resolve_model_for_profile(model: Any) -> Any:
    """Resolve a model identifier to a BaseChatModel for profile lookup.

    SPINE's ``resolve_model`` may return a string identifier (e.g.
    ``"openrouter:openai/gpt-4o-mini"``) or a pre-built ``ChatOpenRouter``
    instance.  DA's ``_harness_profile_for_model`` expects a
    ``BaseChatModel`` so we need to materialise strings into model instances
    first.  This uses DA's ``resolve_model`` which wraps
    ``init_chat_model`` — it will fail if the provider API key is not
    configured, which is fine because profile resolution is best-effort.
    """
    if isinstance(model, str):
        try:
            from deepagents._models import resolve_model as da_resolve_model
            return da_resolve_model(model)
        except Exception:
            return model  # Return the string — _resolve_profile handles it
    return model


def _resolve_profile(model: Any, model_spec: str | None = None) -> Any:
    """Resolve the HarnessProfile for the current model.

    Args:
        model: A BaseChatModel instance (preferred) or a string identifier.
        model_spec: The original model string, used for key-based lookup.

    Returns the profile registered for this model's provider, or None.
    """
    try:
        from deepagents.profiles.harness.harness_profiles import _harness_profile_for_model
        return _harness_profile_for_model(model, model_spec)
    except (ImportError, Exception):
        return None


def _apply_profile_prompt(profile: Any, default_prompt: str) -> str:
    """Apply the profile's prompt overlay to produce the base prompt.

    If the profile has a ``base_system_prompt`` (SPINE's HarnessProfile
    sets this to SPINE_BASE_PROMPT), use it.  Otherwise fall back to
    DA's default BASE_AGENT_PROMPT.

    If the profile has a ``system_prompt_suffix``, append it.
    """
    if profile is not None and profile.base_system_prompt is not None:
        prompt = profile.base_system_prompt
    else:
        prompt = default_prompt

    if profile is not None and profile.system_prompt_suffix is not None:
        prompt = prompt + "\n\n" + profile.system_prompt_suffix

    return prompt


def _get_tool_description_overrides(profile: Any) -> dict[str, str]:
    """Extract tool description overrides from the profile."""
    if profile is None:
        return {}
    try:
        return dict(profile.tool_description_overrides)
    except (TypeError, AttributeError):
        return {}


def _get_filesystem_prompt(backend: Any) -> str:
    """Choose the appropriate filesystem prompt based on backend capabilities.

    If the backend supports execution (LocalShellBackend, sandbox backends),
    include the execute tool section.  Otherwise, use the base filesystem
    prompt only.
    """
    try:
        if supports_execution(backend):
            return SPINE_FILESYSTEM_EXEC_PROMPT
    except Exception:
        pass

    return SPINE_FILESYSTEM_PROMPT