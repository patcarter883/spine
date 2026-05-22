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
import logging
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)


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
        workspace_root: Optional project root to set as ``PROJECT_ROOT``
            in server environments that don't have it configured.

    Returns:
        List of LangChain ``BaseTool`` instances ready for agent injection.
    """
    if not server_configs:
        return []

    # Auto-inject PROJECT_ROOT for servers that don't have it configured
    if workspace_root:
        for _name, cfg in server_configs.items():
            env = cfg.setdefault("env", {})
            env.setdefault("PROJECT_ROOT", str(workspace_root))

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
            namespaced.append(_namespace_tool(tool, primary_server))

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
