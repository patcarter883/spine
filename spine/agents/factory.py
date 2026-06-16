"""SPINE agent factory — custom agent construction with full middleware control.

Every phase agent builder (specify, plan, implement, etc.) was duplicating
the same pattern: resolve model, build backend, assemble system_prompt.
This module consolidates the shared logic and adds the context engineering
features:

- **Artifacts on disk** — prior artifacts are materialized to the filesystem
  and referenced by path, not inlined into the prompt.
- **Memory** — workspace AGENTS.md files are loaded via ``MemoryMiddleware``
  for always-injected project conventions.
- **Context schema** — ``SpineContext`` provides typed per-run context.
- **Read cache** — ``ReadCacheMiddleware`` prevents re-read amnesia by
  returning cached metadata summaries for already-read files.

Parallel work (researcher subagents, slice-implementers, slice-verifiers)
is dispatched at the **graph layer** via the LangGraph ``Send`` API from
the per-phase subgraph routers (``exploration_subgraph``, ``implement_subgraph``,
``verify_subgraph``).  Phase agents themselves have no ``eval`` interpreter
and no ``task`` subagent-dispatch tool — those were removed because models
used them as escape hatches around the curated phase tool surfaces.

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
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.runnables import RunnableConfig

from deepagents.graph import BASE_AGENT_PROMPT
from deepagents.middleware.filesystem import FilesystemMiddleware, supports_execution
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware

from spine.agents.prompt_format import Tag, xml_block

from spine.models.enums import PhaseName
from spine.models.state import WorkflowState
from spine.agents.context import SpineContext
from spine.agents.helpers import resolve_model, debug_enabled

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


def _supports_cache_control(model: Any) -> bool:
    """Return True if the model is Anthropic-backed (direct or via OpenRouter).

    OpenRouter passes Anthropic ``cache_control`` markers through to the
    upstream Anthropic API for the ``anthropic/*`` model family. Marking
    a static-prefix breakpoint costs nothing for non-Anthropic providers
    (they ignore the field) but unlocks prefix caching when the trace
    actually does land on Anthropic.
    """
    if _is_anthropic_model(model):
        return True
    raw = ""
    if isinstance(model, str):
        raw = model
    elif hasattr(model, "model_name"):
        raw = str(getattr(model, "model_name", ""))
    elif hasattr(model, "model"):
        raw = str(getattr(model, "model", ""))
    raw = raw.lower()
    # openrouter:anthropic/claude-*, openrouter:anthropic/...
    if raw.startswith("openrouter:") and "anthropic/" in raw:
        return True
    return False


# ── Static-prefix cache marker ──────────────────────────────────────────


class StaticPrefixCacheMiddleware:
    """Mark the static system prefix with an Anthropic ephemeral cache breakpoint.

    The factory assembles the system prompt as ``user_prompt +
    BASE_AGENT_PROMPT``. The *base prompt* portion is identical across every
    turn of a given phase agent, so it is the right thing to
    cache. We split the system message into two content blocks so the
    cacheable static prefix lives in its own block, then stamp the
    ``cache_control`` breakpoint on that block.

    Implemented as a plain class rather than subclassing AgentMiddleware
    because LangChain's middleware base lives at multiple import paths
    depending on version — we duck-type with ``modify_request`` only.
    """

    def __init__(self, static_prefix: str) -> None:
        self._static_prefix = static_prefix.strip()

    def modify_request(self, request: Any) -> Any:
        from langchain_core.messages import SystemMessage

        if not self._static_prefix:
            return request
        sys_msg = getattr(request, "system_message", None)
        if sys_msg is None:
            return request

        content = getattr(sys_msg, "content", None)
        if not isinstance(content, str) or self._static_prefix not in content:
            return request

        idx = content.index(self._static_prefix)
        head = content[:idx].rstrip()
        tail = content[idx + len(self._static_prefix):].lstrip()

        blocks: list[dict[str, Any]] = []
        if head:
            blocks.append({"type": "text", "text": head})
        blocks.append({
            "type": "text",
            "text": self._static_prefix,
            "cache_control": {"type": "ephemeral"},
        })
        if tail:
            blocks.append({"type": "text", "text": tail})

        new_sys = SystemMessage(content=blocks)
        return request.override(system_message=new_sys)

    # LangChain calls ``awrap_model_call`` or ``modify_request`` depending on
    # version. Provide a passthrough wrap_model_call that delegates to
    # modify_request semantics, since AgentMiddleware's default flow is to
    # call modify_request before the model call.
    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        return await handler(self.modify_request(request))

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        return handler(self.modify_request(request))


# ── SPINE filesystem system prompt ───────────────────────────────────────
# Replaces DA's default "All file paths must start with a /" prompt.
# Under virtual_mode=True, paths are resolved relative to the workspace root.
# Relative paths like ".spine/artifacts/..." map correctly; absolute paths
# like "/home/user/project/.spine/..." double-nest.  This prompt teaches the
# agent the correct convention.

SPINE_FILESYSTEM_PROMPT = (
    xml_block(
        Tag.TOOLS,
        "Filesystem tools available: ls, read_file, write_file, edit_file, "
        "glob, grep.\n\n"
        "Descriptions:\n"
        "- ls: list files in a directory\n"
        "- read_file: read a file (supports offset/limit for large files, "
        "plus images/PDFs)\n"
        "- write_file: create or overwrite a file at a relative path\n"
        "- edit_file: find-and-replace within a file (supports replace_all)\n"
        "- glob: find files matching a pattern (e.g. `**/*.py`)\n"
        "- grep: search file contents with multiple output modes\n\n"
        "You have access to a virtual filesystem rooted at the project "
        "workspace.",
    )
    + "\n\n"
    + xml_block(
        Tag.CONSTRAINTS,
        "Path conventions — VIOLATIONS BREAK EVERYTHING:\n"
        "- Use relative paths from the workspace root: "
        "`.spine/artifacts/file.md`, `spine/ui/pages.py`.\n"
        "- A leading `/` is treated as workspace-relative (e.g. "
        "`/spine/ui/pages.py` resolves correctly).\n"
        "- NEVER use absolute Linux paths like "
        "`/home/user/project/spine/ui/pages.py` — the virtual filesystem "
        "treats `/home/user/...` as a virtual path, so it gets double-"
        "nested under the workspace root and your files land at a path "
        "that does not exist and will never be found by subsequent phases. "
        "Write `spine/ui/pages.py` NOT "
        "`/home/pat/Projects/spine/spine/ui/pages.py`.\n"
        "- Path traversal (`..`, `~`) is BLOCKED. Use relative paths only.\n"
        "- Verify paths exist before modifying: use `ls` on parent "
        "directories, or `search_codebase` if available. Do not invent "
        "paths — confirm they exist or that the parent directory exists "
        "for new files.\n"
        "- Use offset/limit when reading large files. Read only what you "
        "need.\n"
        "- Batch reads: never read one file per turn. Always read ≥3 files "
        "or use search_codebase instead. Sequential single-file reads "
        "waste turns and bloat context.\n"
        "- Large tool results may be offloaded to "
        "`/large_tool_results/<tool_call_id>` — use `read_file` to inspect "
        "in chunks, or `grep` within `/large_tool_results/` to search "
        "across offloaded results.",
    )
)

SPINE_FILESYSTEM_EXEC_PROMPT = (
    SPINE_FILESYSTEM_PROMPT
    + "\n\n"
    + xml_block(
        Tag.TOOLS,
        "Additional tool available: execute.\n\n"
        "- execute: run a shell command (returns output and exit code).\n\n"
        "Use it for commands, scripts, tests, builds, and other shell "
        "operations. Commands run in the workspace root directory. All "
        "paths in commands MUST be relative (e.g. `pytest tests/unit/` "
        "NOT `pytest /home/user/project/tests/unit/`).",
    )
)


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
                    f'<project_documentation path="{path}">\n{content}\n</project_documentation>'
                )

        if not sections:
            return request

        memory_body = "\n\n".join(sections)
        new_system_message = append_to_system_message(request.system_message, memory_body)

        # Apply prompt caching breakpoint if Anthropic and enabled
        if (
            self._add_cache_control
            and type(request.model).__name__ == "ChatAnthropic"
            and hasattr(new_system_message, "content_blocks")
            and new_system_message.content_blocks
        ):
            blocks = list(new_system_message.content_blocks)
            last_block: Any = blocks[-1]
            base = last_block if isinstance(last_block, dict) else {}
            blocks[-1] = {**base, "cache_control": {"type": "ephemeral"}}  # type: ignore[invalid-assignment]
            new_system_message = SystemMessage(content_blocks=blocks)

        return request.override(system_message=new_system_message)


# ── Main factory function ────────────────────────────────────────────────


def _onboarding_injection_enabled() -> bool:
    """Whether onboarding-document injection is enabled (config flag, default on).

    Reads ``SpineConfig.onboarding_context_injection`` the same way the rest of
    the factory reads config (``SpineConfig.load()``). Any load failure defaults
    to enabled so the feature is on out of the box.
    """
    try:
        from spine.config import SpineConfig

        return bool(SpineConfig.load().onboarding_context_injection)
    except Exception:
        return True


def build_phase_agent(
    state: WorkflowState,
    config: RunnableConfig | None,
    phase: PhaseName,
    system_prompt: str,
    *,
    extra_middleware: list[Any] | None = None,
    response_format: Any | None = None,
    is_subagent: bool = False,
    allowed_tools: list[str] | None = None,
    extra_tools: list[Any] | None = None,
    skip_filesystem_middleware: bool = False,
    completion_token_cap: int | None = None,
) -> Any:
    """Build a LangChain agent for a SPINE phase with full context engineering.

    This is the single entry point for all phase agent construction.
    It handles model resolution, backend creation, full middleware assembly,
    memory, skills, and context schema — so individual
    phase builders don't have to.

    Phase agents do not get the ``eval`` interpreter or the ``task``
    subagent-dispatch tool. Parallel work is orchestrated by the per-phase
    subgraph routers via the LangGraph ``Send`` API.

    Args:
        state: The current workflow state.
        config: LangGraph runtime config.
        phase: The phase being executed.
        system_prompt: Phase-specific system prompt (role + task framing).
        extra_middleware: Additional middleware to include.
        response_format: Optional Pydantic model or ResponseFormat for
            structured output (DA ≥0.5.3).
        is_subagent: When True, skips artifact materialization and
            read-cache middleware (subagents are leaf agents).
        extra_tools: Additional BaseTool instances injected directly into
            create_agent(tools=[...]). These sit alongside tools from
            middleware. Use for custom orchestrator tools (e.g. ReadSliceFilesTool)
            that replace generic filesystem access.
        skip_filesystem_middleware: When True, FilesystemMiddleware is omitted
            entirely. Pair with extra_tools to replace all filesystem access
            with purpose-built tools, removing any generic read/write fallback
            from the model's tool surface.
        completion_token_cap: When > 0, clamp the model's completion-token
            request to this value (via ``cap_completion_tokens``). Used by
            the SPECIFY/PLAN synthesizers so a structured 2-4K JSON output
            doesn't request the global 30K budget and push the prompt over
            a finite context window (trace 019eb3dd).

    Returns:
        A compiled agent (CompiledStateGraph) ready for invocation.
    """

    from spine.agents.backend import build_backend
    from spine.agents.artifacts import materialize_artifacts
    from spine.agents.skills_resolver import (
        build_onboarding_reference,
        resolve_memory,
        resolve_onboarding_docs,
    )

    # ── Resolve model and backend ────────────────────────────────────
    model = resolve_model(config, session_id=state.get("work_id"), phase=phase.value)
    if completion_token_cap and completion_token_cap > 0:
        if isinstance(model, str):
            # String specs are resolved by create_agent without provider
            # kwargs; they only occur for cloud providers that declare no
            # finite context_window, so the cap is a no-op there.
            logger.debug(
                "Phase %s: completion_token_cap=%d skipped (string model spec)",
                phase.value, completion_token_cap,
            )
        else:
            from spine.agents.helpers import cap_completion_tokens

            model = cap_completion_tokens(model, completion_token_cap)
            logger.info(
                "Phase %s: completion tokens clamped to %d for synthesis",
                phase.value, completion_token_cap,
            )
    workspace_root = state.get("workspace_root", ".")
    backend = build_backend(workspace_root)
    work_id = state.get("work_id", "")

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

    # ── Onboarding documentation (hybrid injection) ──────────────────
    # The relevant onboarding document for this phase is injected in full (via
    # the memory middleware, below); the remaining documents are referenced by
    # path for on-demand reading. Disabled when onboarding_context_injection is
    # off. Subagents are leaf workers and skip this — their parent already
    # carries the context.
    onboarding_inject: str | None = None
    onboarding_ref_block = ""
    if not is_subagent and _onboarding_injection_enabled():
        onboarding_inject, onboarding_ref, inject_excerpt = resolve_onboarding_docs(
            workspace_root, phase.value
        )
        # When the primary doc exceeds the full-inject cap, resolve_onboarding_docs
        # returns an excerpt string instead of a path.  Write it to a tempfile so
        # the memory middleware can load it identically to a full doc.  The file
        # must persist for the lifetime of this process (the middleware reads it
        # per-request, not at construction time); atexit handles cleanup so we
        # don't pollute the workspace .spine/onboarding/ directory.
        if onboarding_inject is None and inject_excerpt:
            import atexit
            import os
            import tempfile

            fd, _excerpt_path = tempfile.mkstemp(
                suffix=".md", prefix="spine_onboarding_excerpt_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as _fh:
                    _fh.write(inject_excerpt)
                onboarding_inject = _excerpt_path
                atexit.register(
                    lambda p=_excerpt_path: os.unlink(p) if os.path.exists(p) else None
                )
            except OSError:
                try:
                    os.close(fd)
                    os.unlink(_excerpt_path)
                except OSError:
                    pass
        onboarding_ref_block = build_onboarding_reference(onboarding_ref)

    # ── Assemble final system prompt ─────────────────────────────────
    # Assembly order: user system_prompt → onboarding references → base prompt.
    # The reference block sits before the base prompt so the base prompt stays
    # the trailing static chunk the StaticPrefixCacheMiddleware stamps.
    prompt_parts: list[str] = []
    if system_prompt is not None:
        prompt_parts.append(system_prompt)
    if onboarding_ref_block:
        prompt_parts.append(onboarding_ref_block)
    prompt_parts.append(base_prompt)
    final_system_prompt: str = "\n\n".join(prompt_parts)

    # The static prefix — the base prompt — is what we want to mark with
    # an ephemeral cache breakpoint.  It is identical across every turn of
    # this phase agent, and lives entirely inside the system message.  Pass
    # it through to the middleware stack so the StaticPrefixCacheMiddleware
    # can find and stamp it.
    static_cacheable_prefix = base_prompt.strip()

    # ── Memory ───────────────────────────────────────────────────────
    memory = resolve_memory(workspace_root, phase=phase.value)
    # Inject the phase's primary onboarding doc in full, reusing the memory
    # middleware's <project_documentation> wrapper + cache breakpoint.
    if onboarding_inject:
        memory = [*memory, onboarding_inject]
    if memory:
        logger.debug("Phase %s: loading memory files: %s", phase.value, memory)

    # ── Build the middleware stack ───────────────────────────────────
    middleware = _build_middleware_stack(
        backend=backend,
        model=model,
        phase=phase,
        is_subagent=is_subagent,
        extra_middleware=extra_middleware,
        tool_desc_overrides=tool_desc_overrides,
        profile=profile,
        memory=memory,
        allowed_tools=allowed_tools,
        skip_filesystem_middleware=skip_filesystem_middleware,
        static_cacheable_prefix=static_cacheable_prefix,
    )

    # ── Context schema ───────────────────────────────────────────────
    context_schema = SpineContext

    # ── Construct the agent ──────────────────────────────────────────
    all_tools = list(extra_tools) if extra_tools else []

    agent = create_agent(
        model,
        system_prompt=final_system_prompt,
        tools=all_tools,  # Custom tools + middleware tools
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
    extra_middleware: list[Any] | None,
    tool_desc_overrides: dict[str, str],
    profile: Any,
    memory: list[str] | None,
    allowed_tools: list[str] | None = None,
    skip_filesystem_middleware: bool = False,
    static_cacheable_prefix: str | None = None,
) -> list[Any]:
    """Assemble the full middleware stack for a SPINE phase agent.

    Stack order (from outermost to innermost in the middleware chain):

    1.  FilesystemMiddleware       — filesystem tools (skipped when skip_filesystem_middleware=True)
    2.  ReadCacheMiddleware        — prevents re-read amnesia via SpineContext cache
    3.  PatchToolCallsMiddleware   — tool call normalization
    4.  SPINE-specific middleware  — ToolSchemaValidator
    5.  User extra_middleware
    6.  Profile extra_middleware
    7.  AnthropicPromptCachingMiddleware — prompt caching (no-op for non-Anthropic)
    8.  MemoryMiddleware           — AGENTS.md injection (when memory present)
    """
    middleware: list[Any] = []

    # 1. Filesystem — core tool surface with custom SPINE prompt
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

    # 2. Read cache — prevents re-reading already-seen files and re-issuing
    #    identical search_codebase / MCP lookups.  Checks SpineContext.read_cache
    #    before allowing the cached tools to execute.  Applied to subagents too:
    #    Explore researcher loops were re-running the same MCP lookups within a
    #    single invocation and leaking duplicate output into findings.
    middleware.append(_build_read_cache_middleware())

    # 3. Patch tool calls — normalisation
    middleware.append(PatchToolCallsMiddleware())

    # 4. SPINE-specific middleware (tool validation, context editing).  Applied
    #    to subagents too so that tool runtime errors become a compact
    #    ToolMessage(status="error") instead of a raw traceback that ends up
    #    serialized into ResearchFindings.summary.
    _add_spine_middleware(middleware, phase)

    # 5. User-provided extra middleware
    if extra_middleware:
        middleware.extend(extra_middleware)

    # 6. Profile extra middleware
    if profile is not None:
        try:
            extra = profile.materialize_extra_middleware()
            if extra:
                middleware.extend(extra)
        except Exception:
            pass

    # 7. Prompt caching — mark the static system prefix with an Anthropic
    #    ephemeral cache breakpoint. Honored by direct Anthropic API and
    #    OpenRouter Anthropic routes; silently ignored by other providers
    #    (the field is added once per turn, negligible overhead).
    if static_cacheable_prefix and _supports_cache_control(model):
        middleware.append(StaticPrefixCacheMiddleware(static_cacheable_prefix))
    if _is_anthropic_model(model):
        middleware.append(AnthropicPromptCachingMiddleware())

    # 8. Memory — AGENTS.md injection (when memory sources provided)
    if memory:
        middleware.append(
            SpineProjectMemoryMiddleware(
                backend=backend,
                sources=memory,
                add_cache_control=True,
            )
        )

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
            phase.value,
            allowed_tools,
        )
        return

    allowed = set(allowed_tools)
    original_names = [t.name for t in fs_mw.tools]
    fs_mw.tools = [t for t in fs_mw.tools if t.name in allowed]
    kept = [t.name for t in fs_mw.tools]
    dropped = [n for n in original_names if n not in allowed]
    logger.debug(
        "Phase %s: filtered filesystem tools — kept=%s dropped=%s",
        phase.value,
        kept,
        dropped,
    )


def _add_spine_middleware(middleware: list[Any], phase: PhaseName) -> None:
    """Add SPINE-specific middleware for tool validation and context editing."""
    import os as _os

    # Tool schema validation — rebound loop for self-correction
    _validation_enabled = _os.getenv("SPINE_TOOL_SCHEMA_VALIDATION", "true").lower() not in (
        "0",
        "false",
        "no",
    )
    if _validation_enabled:
        from spine.agents.tool_schema_validator import ToolSchemaValidator

        middleware.append(ToolSchemaValidator())

    # Note: ToolOutputTrimmer was removed (2026-05) — excessive trimming of
    # filesystem results was causing agents to lose critical context during
    # implementation/verification phases. ReadCacheMiddleware now handles
    # the re-read problem at source, while ToolOutputTrimmer is retired.

    # Resolve the model's declared context window once — both the eviction
    # threshold and the per-turn completion clamp are window-relative.
    from spine.config import SpineConfig

    _spine_cfg = SpineConfig.load()
    try:
        _provider_cfg = _spine_cfg.resolve_provider_config(phase=phase.value)
        _window = int(_provider_cfg.get("context_window") or 0)
    except Exception:  # noqa: BLE001 — config resolution is best-effort here
        _window = 0
    # Room to keep free below the eviction trigger (and above the prompt) for
    # the completion request + tokenizer/template overhead.
    _gen_reserve = _spine_cfg.implement_max_completion_tokens + _spine_cfg.synthesize_overhead_tokens

    # TokenBudgetCompactor — token-threshold eviction with hardened
    # preservation rules (always keeps the last N tool results and any
    # write/edit/artifact-writer outputs). Enabled via
    # .spine/config.yaml `token_compaction.enabled` (the SPINE_TOKEN_COMPACTION
    # env var still overrides it — see _parse_token_compaction_config).
    cfg = _spine_cfg.token_compaction
    if cfg.enabled:
        from spine.agents.context_editing import TokenBudgetCompactor
        from spine.agents.synthesis_budget import window_aware_compaction_threshold

        configured = cfg.thresholds.get(phase.value, cfg.default_threshold)
        # A fixed threshold is meaningless when it exceeds the window — clamp
        # it below the window ceiling so eviction fires before the provider
        # 400s on a finite-window model (trace 019ece87).
        threshold = window_aware_compaction_threshold(
            window=_window,
            configured_threshold=configured,
            reserve=_gen_reserve,
        )
        if threshold > 0:
            middleware.append(
                TokenBudgetCompactor(
                    threshold_tokens=threshold,
                    keep_recent=cfg.keep_recent,
                    preserved_tools=frozenset(cfg.preserved_tools),
                )
            )
            logger.info(
                "TokenBudgetCompactor: phase=%s threshold=%d (configured=%d "
                "window=%d) keep_recent=%d",
                phase.value, threshold, configured, _window, cfg.keep_recent,
            )
        else:
            logger.info(
                "TokenBudgetCompactor: phase=%s SKIPPED (threshold=%d)",
                phase.value, threshold,
            )
    else:
        logger.info(
            "TokenBudgetCompactor: phase=%s DISABLED (token_compaction.enabled=false)",
            phase.value,
        )

    # DynamicCompletionCapMiddleware — defence in depth against the same
    # overflow: even with eviction, a single large tool result or a burst of
    # growth between trims can push prompt + reserved_cap past the window. This
    # recomputes max_tokens against the measured prompt every turn so the
    # request always fits. No-op for providers without a declared window.
    if _window > 0:
        from spine.agents.context_editing import DynamicCompletionCapMiddleware

        middleware.append(
            DynamicCompletionCapMiddleware(
                window=_window,
                overhead=_spine_cfg.synthesize_overhead_tokens,
            )
        )
        logger.info(
            "DynamicCompletionCap: phase=%s window=%d overhead=%d",
            phase.value, _window, _spine_cfg.synthesize_overhead_tokens,
        )


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
