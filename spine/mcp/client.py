"""MCP client: connect to MCP servers via stdio, discover tools, execute calls.

Manages MCP server lifecycle — spawn subprocess, establish session, discover tools,
execute tool calls — using the ``mcp`` Python SDK's stdio client transport.

All public methods are synchronous (blocking) but use an internal asyncio
event loop for the MCP protocol, which is inherently async.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)


class MCPClient:
    """Manages a single MCP server connection via stdio transport.

    Handles process lifecycle (spawn, connect, close) and tool discovery.
    All calls are blocking but use an internal event loop for the MCP protocol.

    Args:
        name: Human-readable server name (e.g. ``"codebase-index"``).
        command: Executable to spawn (e.g. ``"mcp-codebase-index"``).
        args: Optional list of arguments passed to the command.
        env: Extra environment variables for the subprocess (merged with
            a filtered copy of the current environment).
        timeout: Per-tool-call timeout in seconds (default 120).
        connect_timeout: Timeout for initial connection and tool discovery
            (default 60).
    """

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 120,
        connect_timeout: int = 60,
    ):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._tools: list[dict[str, Any]] = []
        self._connected = False
        self._loop: asyncio.AbstractEventLoop | None = None

    # ── Connection lifecycle ────────────────────────────────────────

    def connect(self) -> None:
        """Start the MCP server subprocess and establish a session.

        Spawns the configured command as a subprocess, connects via stdio,
        initializes the MCP session, and discovers available tools via
        ``list_tools()``.

        Raises:
            Exception: If the server process fails to start, the connection
                times out, or tool discovery fails.
        """
        if self._connected:
            return

        loop = asyncio.new_event_loop()
        self._loop = loop

        # Build the subprocess environment: filtered copy of current
        # environment + user-specified overrides.
        child_env = os.environ.copy()
        child_env.update(self.env)

        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=child_env,
        )

        async def _connect() -> None:
            self._exit_stack = AsyncExitStack()
            transport = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read_stream, write_stream = transport
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()
            result = await self._session.list_tools()
            self._tools = [tool.model_dump() for tool in result.tools]

        try:
            loop.run_until_complete(
                asyncio.wait_for(_connect(), timeout=self.connect_timeout)
            )
            self._connected = True
            logger.info(
                "MCP server '%s' connected: %d tools discovered",
                self.name,
                len(self._tools),
            )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Close the MCP session and terminate the server process.

        Idempotent — safe to call multiple times or on an unconnected client.
        """
        self._connected = False
        self._tools = []
        if self._exit_stack:
            try:
                self._loop.run_until_complete(self._exit_stack.aclose())
            except Exception:
                pass
            self._exit_stack = None
            self._session = None
        if self._loop and not self._loop.is_closed():
            self._loop.close()
            self._loop = None

    # ── Tool discovery ──────────────────────────────────────────────

    def list_tools(self) -> list[dict[str, Any]]:
        """Return discovered tools as raw dicts.

        Each dict has keys: ``name``, ``description``, ``inputSchema``.

        Returns:
            List of tool descriptors.  Empty if not yet connected.
        """
        if not self._connected:
            self.connect()
        return list(self._tools)

    # ── Tool execution ───────────────────────────────────────────────

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute an MCP tool and return the result as a string.

        Args:
            name: MCP tool name (unprefixed — as returned by ``list_tools``).
            arguments: Keyword arguments matching the tool's input schema.

        Returns:
            The tool's text response as a single string.

        Raises:
            RuntimeError: If the tool returns an error.
            Exception: On connection failure or timeout.
        """
        if not self._connected:
            self.connect()

        async def _call() -> str:
            assert self._session is not None
            result = await self._session.call_tool(name, arguments=arguments)
            if result.isError:
                raise RuntimeError(
                    f"MCP tool '{name}' returned error: {result.content}"
                )
            parts: list[str] = []
            for content in result.content:
                if hasattr(content, "text"):
                    parts.append(content.text)
                elif hasattr(content, "data"):
                    parts.append(f"[binary data: {len(content.data)} bytes]")
            return "\n".join(parts) if parts else str(result.content)

        try:
            return self._loop.run_until_complete(
                asyncio.wait_for(_call(), timeout=self.timeout)
            )
        except Exception:
            logger.exception("MCP tool '%s' (server '%s') failed", name, self.name)
            raise

    # ── Context manager ─────────────────────────────────────────────

    def __enter__(self) -> MCPClient:
        self.connect()
        return self

    def __exit__(self, *args: object) -> bool:
        self.close()
        return False


# ── Multi-server manager ─────────────────────────────────────────────────

class MCPClientManager:
    """Manages multiple MCP server connections.

    Loads server configs and creates/reuses ``MCPClient`` instances.
    Intended to be instantiated once per cache key (e.g. work item ID)
    and shared across phase agents within that work item.

    Args:
        server_configs: Dict mapping server name → config dict with keys
            ``command``, ``args``, ``env``, ``timeout``, ``connect_timeout``.
    """

    def __init__(self, server_configs: dict[str, dict[str, Any]]):
        self._clients: dict[str, MCPClient] = {}
        self._server_configs = server_configs

    def get_client(self, server_name: str) -> MCPClient:
        """Get or create a client for the named server.

        Args:
            server_name: Server name as it appears in the config.

        Returns:
            A connected ``MCPClient`` instance.

        Raises:
            KeyError: If no config exists for the given server name.
        """
        if server_name not in self._clients:
            cfg = self._server_configs.get(server_name)
            if not cfg:
                raise KeyError(f"No config for MCP server '{server_name}'")
            client = MCPClient(
                name=server_name,
                command=cfg["command"],
                args=cfg.get("args", []),
                env=cfg.get("env", {}),
                timeout=cfg.get("timeout", 120),
                connect_timeout=cfg.get("connect_timeout", 60),
            )
            client.connect()
            self._clients[server_name] = client
        return self._clients[server_name]

    def get_all_tools(self) -> list[Any]:
        """Return all tools from all connected servers as ``MCPTool`` instances.

        Returns:
            List of ``MCPTool`` dataclass instances, one per discovered tool
            across all managed servers.
        """
        from spine.mcp.tools import MCPTool

        all_tools: list[Any] = []
        for name in self._server_configs:
            client = self.get_client(name)
            for tool_dict in client.list_tools():
                tool = MCPTool(
                    server_name=name,
                    name=tool_dict["name"],
                    description=tool_dict.get("description", ""),
                    input_schema=tool_dict.get("inputSchema", {}),
                    client=client,
                )
                all_tools.append(tool)
        return all_tools

    def close_all(self) -> None:
        """Close all managed server connections."""
        for client in self._clients.values():
            client.close()
        self._clients.clear()


# ── Module-level cache ───────────────────────────────────────────────────

_MCP_CACHE: dict[str, MCPClientManager] = {}


def get_mcp_tools(
    server_configs: dict[str, dict[str, Any]] | None,
    cache_key: str = "default",
    workspace_root: str | None = None,
) -> list[Any]:
    """Load MCP tools from server configs, returning LangChain tools.

    Tools are cached per *cache_key* (typically the work item ID) so the
    same MCP server connections are reused across phase agents within a
    single work item.

    Args:
        server_configs: Dict of server_name → {command, args, env, ...}.
            If ``None`` or empty, returns an empty list.
        cache_key: Key for client manager reuse (use ``work_id``).
        workspace_root: Optional project root to set as ``PROJECT_ROOT``
            in server environments that don't have it configured.

    Returns:
        List of LangChain ``BaseTool`` instances ready for agent injection.
    """
    if not server_configs:
        return []

    # Auto-inject PROJECT_ROOT for codebase-indexer servers that
    # don't have it explicitly configured in their env.
    if workspace_root:
        for _name, cfg in server_configs.items():
            env = cfg.setdefault("env", {})
            env.setdefault("PROJECT_ROOT", str(workspace_root))

    global _MCP_CACHE
    if cache_key not in _MCP_CACHE:
        mgr = MCPClientManager(server_configs)
        _MCP_CACHE[cache_key] = mgr
    else:
        mgr = _MCP_CACHE[cache_key]

    mcp_tools = mgr.get_all_tools()
    from spine.mcp.tools import mcp_tool_to_langchain

    lc_tools = [mcp_tool_to_langchain(t) for t in mcp_tools]
    return lc_tools
