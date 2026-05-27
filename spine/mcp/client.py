"""MCP tool loader for SPINE using langchain-mcp-adapters.

Wraps LangChain's ``MultiServerMCPClient`` to load MCP tools from the
SPINE config.  Tools are namespaced with the server name to prevent
collisions (e.g. ``mcp_codebase-index_find_symbol``).

The ``MultiServerMCPClient`` is stateless by default — each tool call
creates a fresh MCP session, executes, and cleans up.  This is correct
for SPINE's multi-agent architecture where different subagents may
call MCP tools concurrently from different event loops.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from spine.agents import symbol_cache

logger = logging.getLogger(__name__)


# ── Result-size caps for chatty index tools ───────────────────────────────
# A bare `mcp_codebase-index_search_codebase` call has been observed dumping
# ~190 KB into the agent transcript in one shot. We post-process the tool
# result string to enforce a hard ceiling and force the agent to refine its
# regex / file globs instead. Tunable per deployment via these constants.

SEARCH_CODEBASE_MAX_BYTES = 8192
SEARCH_CODEBASE_MAX_HITS = 50

# Tools whose results are line-shaped (one match per line) and benefit from
# both byte + hit caps.
_HIT_CAPPED_TOOLS: frozenset[str] = frozenset({
    "mcp_codebase-index_search_codebase",
})

# Tools that can return very large blobs (full source bodies). Apply only
# the byte cap.
_BYTE_CAPPED_TOOLS: frozenset[str] = frozenset({
    "mcp_codebase-index_get_function_source",
    "mcp_codebase-index_get_class_source",
})


# ── Path-prefix exclusions for indexed-but-uninteresting directories ─────
# The external indexer is configured against PROJECT_ROOT and has no
# awareness of spine's own scratch directories. Without this filter the
# researcher reads back its own prior artifacts as if they were source.
EXCLUDED_INDEX_PATHS: tuple[str, ...] = (
    ".spine/artifacts/",
    ".spine/checkpoints/",
    ".spine/spine.db",
)

# Matches a path-like prefix at the start of a line (after optional bullet/
# whitespace decoration). Anchored to line start so prose lines that merely
# mention a path are not stripped.
_PATH_LINE_RE = re.compile(
    r"^(?P<lead>\s*[-*•]?\s*)(?P<path>(?:\./)?[\w./-]+)"
)


def _line_starts_with_excluded_path(line: str) -> bool:
    """Return True when *line* leads with a path inside EXCLUDED_INDEX_PATHS."""
    m = _PATH_LINE_RE.match(line)
    if not m:
        return False
    path = m.group("path")
    if path.startswith("./"):
        path = path[2:]
    return any(path.startswith(prefix) for prefix in EXCLUDED_INDEX_PATHS)


def _strip_excluded_paths(text: str) -> tuple[str, int]:
    """Drop lines whose leading path falls under an excluded directory.

    Returns (filtered_text, n_dropped). Forgiving — non-path lines pass
    through untouched so prose/grouping headers are preserved.
    """
    if not text or not any(prefix in text for prefix in EXCLUDED_INDEX_PATHS):
        return text, 0
    kept: list[str] = []
    dropped = 0
    for line in text.splitlines():
        if _line_starts_with_excluded_path(line):
            dropped += 1
            continue
        kept.append(line)
    return ("\n".join(kept), dropped)


def _cap_result(tool_name: str, result: str) -> str:
    """Apply byte / hit caps to a tool result string."""
    if not isinstance(result, str):
        return result

    apply_hits = tool_name in _HIT_CAPPED_TOOLS
    apply_bytes = tool_name in _HIT_CAPPED_TOOLS or tool_name in _BYTE_CAPPED_TOOLS

    if not (apply_hits or apply_bytes):
        return result

    capped = result
    truncation_notes: list[str] = []

    if apply_hits:
        lines = capped.splitlines()
        total = len(lines)
        if total > SEARCH_CODEBASE_MAX_HITS:
            capped = "\n".join(lines[:SEARCH_CODEBASE_MAX_HITS])
            truncation_notes.append(
                f"{SEARCH_CODEBASE_MAX_HITS}/{total} hits"
            )

    if apply_bytes:
        raw = capped.encode("utf-8", errors="replace")
        if len(raw) > SEARCH_CODEBASE_MAX_BYTES:
            capped = raw[:SEARCH_CODEBASE_MAX_BYTES].decode("utf-8", errors="ignore")
            truncation_notes.append(
                f"{SEARCH_CODEBASE_MAX_BYTES}/{len(raw)} bytes"
            )

    if truncation_notes:
        capped = (
            f"{capped}\n[truncated: showed {', '.join(truncation_notes)} — "
            "refine regex with anchors / file globs / case sensitivity, "
            "or call a structural tool like find_symbol / get_dependencies]"
        )
    return capped


def _post_process_result(tool_name: str, result: Any) -> Any:
    """Apply path exclusion + size caps to a tool's textual result.

    Operates on strings (which MCP adapters return for most server tools).
    Non-string payloads pass through unchanged so future structured results
    are not mangled.
    """
    if not isinstance(result, str):
        return result
    filtered, dropped = _strip_excluded_paths(result)
    if dropped:
        logger.debug(
            "MCP %s: dropped %d result lines pointing at excluded paths",
            tool_name,
            dropped,
        )
    return _cap_result(tool_name, filtered)


def _wrap_for_postprocessing(tool: BaseTool) -> BaseTool:
    """Patch *tool* so its `_run` / `_arun` outputs are post-processed.

    Patches in place rather than subclassing to avoid Pydantic model
    constraints on BaseTool. The original bound methods are captured in a
    closure so the wrapper is idempotent (re-wrapping a wrapped tool is a
    no-op since we check the marker attribute).
    """
    name = getattr(tool, "name", None)
    if not isinstance(name, str) or not name.startswith("mcp_codebase-index_"):
        return tool
    if getattr(tool, "_spine_postprocess_wrapped", False):
        return tool

    try:
        original_run = getattr(tool, "_run", None)
        original_arun = getattr(tool, "_arun", None)

        if callable(original_arun):
            @functools.wraps(original_arun)
            async def _wrapped_arun(*args: Any, **kwargs: Any) -> Any:
                result = await original_arun(*args, **kwargs)
                return _post_process_result(name, result)
            object.__setattr__(tool, "_arun", _wrapped_arun)

        if callable(original_run):
            @functools.wraps(original_run)
            def _wrapped_run(*args: Any, **kwargs: Any) -> Any:
                result = original_run(*args, **kwargs)
                return _post_process_result(name, result)
            object.__setattr__(tool, "_run", _wrapped_run)

        object.__setattr__(tool, "_spine_postprocess_wrapped", True)
    except Exception:
        logger.debug("Failed to wrap MCP tool %s for post-processing", name, exc_info=True)
    return tool


def _convert_server_config(spine_server: dict[str, Any]) -> dict[str, Any]:
    """Convert SPINE server config to langchain-mcp-adapters format.

    SPINE config: {'command': '...', 'args': [...], 'env': {...}, ...}
    Adapter config: {'transport': 'stdio', 'command': '...', 'args': [...], ...}
    """
    adapter: dict[str, Any] = {
        "transport": spine_server.get("transport", "stdio"),
        "command": spine_server["command"],
        "args": spine_server.get("args", []),
    }
    if spine_server.get("env"):
        adapter["env"] = spine_server["env"]
    return adapter


def _namespace_tool(tool: BaseTool, server_name: str) -> BaseTool:
    """Add server-name prefix to a tool to prevent collisions.

    The adapter returns tools with their original names (e.g. ``find_symbol``).
    We prefix with ``mcp_{server_name}_`` so tools from different servers
    don't collide and prompts can reference them unambiguously.
    """
    # Create a copy with a namespaced name
    import copy

    namespaced = copy.copy(tool)
    namespaced.name = f"mcp_{server_name}_{tool.name}"
    # Preserve original metadata
    if hasattr(tool, "metadata") and tool.metadata:
        namespaced.metadata = dict(tool.metadata)  # type: ignore[attr-defined]
        namespaced.metadata["mcp_server"] = server_name  # type: ignore[attr-defined]
        namespaced.metadata["mcp_tool"] = tool.name  # type: ignore[attr-defined]
    return namespaced


# ── Module-level client (initialised once, reused for tool loading) ──

_client: MultiServerMCPClient | None = None
_client_config_hash: int = 0


def _get_client(server_configs: dict[str, dict[str, Any]]) -> MultiServerMCPClient:
    """Get or create the MultiServerMCPClient for the given configs."""
    global _client, _client_config_hash

    config_hash = hash(
        frozenset(
            (k, frozenset((fk, str(fv)) for fk, fv in v.items())) for k, v in server_configs.items()
        )
    )

    if _client is None or config_hash != _client_config_hash:
        adapter_configs = {
            name: _convert_server_config(cfg) for name, cfg in server_configs.items()
        }
        _client = MultiServerMCPClient(adapter_configs)
        _client_config_hash = config_hash
        # Opt every configured MCP server into cross-branch result sharing
        # for its deterministic read-only tools. The default suffix set
        # (get_/find_/list_/search_) covers the common naming convention;
        # mutating tools (create_/update_/delete_/run_/...) are excluded.
        for server_name in server_configs:
            symbol_cache.register_cacheable_server(server_name)
        logger.info("MCP client created for %d server(s)", len(adapter_configs))

    return _client


def get_mcp_tools(
    server_configs: dict[str, dict[str, Any]] | None,
    cache_key: str = "default",
    workspace_root: str | None = None,
) -> list[BaseTool]:
    """Load MCP tools from server configs, returning LangChain tools.

    Uses ``MultiServerMCPClient`` from ``langchain-mcp-adapters`` to
    connect to configured MCP servers and convert all discovered tools
    to LangChain ``BaseTool`` instances.  Tools are namespaced with
    the server name prefix (e.g. ``mcp_codebase-index_find_symbol``).

    The adapter is stateless — each tool invocation creates a fresh
    MCP session.  This is safe for SPINE's multi-agent architecture
    where subagents may call MCP tools from different event loops.

    Args:
        server_configs: Dict of server_name → {command, args, env, ...}.
            If ``None`` or empty, returns an empty list.
        cache_key: Ignored (kept for backward compat with custom client).
        workspace_root: Project root to inject as ``PROJECT_ROOT`` in every
            MCP server's environment. Resolved to an absolute path and
            *overrides* any PROJECT_ROOT the user may have set in their
            ``mcp_servers`` config — keeping the MCP index and spine
            pointed at the same project is the whole point of the wiring.

    Returns:
        List of LangChain ``BaseTool`` instances ready for agent injection.
    """
    if not server_configs:
        return []

    # Force every MCP server's PROJECT_ROOT to match spine's workspace_root.
    # We override unconditionally — a typo'd PROJECT_ROOT in user config
    # (e.g. lowercase vs capital P) silently breaks every codebase-index
    # query, so the workspace spine is running against wins.
    if workspace_root:
        resolved_root = str(Path(workspace_root).resolve())
        for _name, cfg in server_configs.items():
            env = cfg.setdefault("env", {})
            existing = env.get("PROJECT_ROOT")
            if existing and existing != resolved_root:
                logger.info(
                    "MCP %s: overriding configured PROJECT_ROOT %r with spine workspace %r",
                    _name,
                    existing,
                    resolved_root,
                )
            env["PROJECT_ROOT"] = resolved_root

    try:
        client = _get_client(server_configs)
        tools = _run_async(client.get_tools())
    except Exception:
        logger.warning("Failed to load MCP tools", exc_info=True)
        return []

    # Namespace tools to prevent collisions between servers.
    # The adapter returns tools from all servers without server tags.
    # For a single server: all tools get that server's prefix.
    # For multiple servers: we'd need per-server get_tools() calls;
    # for now, prefix with the first server name which is correct
    # for the common single-server (codebase-index) case.
    primary_server = next(iter(server_configs)) if server_configs else None
    namespaced: list[BaseTool] = []
    if primary_server:
        for tool in tools:
            ns_tool = _namespace_tool(tool, primary_server)
            namespaced.append(_wrap_for_postprocessing(ns_tool))

    logger.info("Loaded %d MCP tools from %d server(s)", len(namespaced), len(server_configs))
    return namespaced


def _run_async(coro: Any) -> Any:
    """Run an async coroutine in a sync context.

    Creates a fresh event loop to avoid conflicts with running loops
    (e.g. when called from within a LangGraph subgraph that already
    has an active event loop).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # Running inside an event loop — run in a separate thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result(timeout=60)
    else:
        return asyncio.run(coro)
