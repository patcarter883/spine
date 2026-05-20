"""Unit tests for MCP client: connection lifecycle, tool discovery, execution."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spine.mcp.client import MCPClient, MCPClientManager, get_mcp_tools


# ── Mock helpers ─────────────────────────────────────────────────────────────


def _make_stdio_context() -> MagicMock:
    """Create a mock for stdio_client that acts as an async context manager.

    Returns ``(read_stream, write_stream)`` tuple on ``__aenter__`` so the
    ``transport = await self._exit_stack.enter_async_context(stdio_client(...))``
    pattern unpacks correctly.
    """
    read_stream = AsyncMock()
    write_stream = AsyncMock()

    @asynccontextmanager
    async def _fake_stdio(*args, **kwargs):
        yield (read_stream, write_stream)

    mock = MagicMock(side_effect=_fake_stdio)
    return mock


def _make_session_mock(tools: list[dict] | None = None) -> MagicMock:
    """Create a mock ClientSession with list_tools and call_tool configured.

    The mock is structured so that ``await exit_stack.enter_async_context(session)``
    returns a usable mock (the ``_inner`` object) where ``call_tool`` and
    ``list_tools`` are properly configured.  Tests that need to set
    ``call_tool.return_value`` should do so on the ``_inner`` attribute.
    """
    if tools is None:
        tools = []
    mock_tool_objs = []
    for t in tools:
        obj = MagicMock()
        obj.model_dump.return_value = t
        mock_tool_objs.append(obj)

    # The session object that __aenter__ returns (what gets stored as self._session)
    inner = AsyncMock()
    inner.list_tools.return_value.tools = mock_tool_objs

    # The outer mock acts as the ClientSession class/constructor.
    outer = MagicMock()
    outer.__aenter__ = AsyncMock(return_value=inner)
    outer._inner = inner  # So tests can configure call_tool on the inner mock
    return outer


# ── Tests ────────────────────────────────────────────────────────────────────


class TestMCPClient:
    """Tests for MCPClient — single server connection management."""

    def test_init_stores_config(self) -> None:
        """Client should store all constructor args."""
        client = MCPClient(
            name="test-server",
            command="test-command",
            args=["--verbose"],
            env={"KEY": "val"},
            timeout=30,
            connect_timeout=15,
        )
        assert client.name == "test-server"
        assert client.command == "test-command"
        assert client.args == ["--verbose"]
        assert client.env == {"KEY": "val"}
        assert client.timeout == 30
        assert client.connect_timeout == 15
        assert client._connected is False
        assert len(client._tools) == 0

    def test_init_defaults(self) -> None:
        """Default timeout is 120, connect_timeout is 60."""
        client = MCPClient(name="d", command="c")
        assert client.timeout == 120
        assert client.connect_timeout == 60
        assert client.args == []
        assert client.env == {}

    @patch("spine.mcp.client.ClientSession")
    @patch("spine.mcp.client.stdio_client")
    def test_connect_discovers_tools(
        self, mock_stdio: MagicMock, mock_session_cls: MagicMock
    ) -> None:
        """connect() should spawn server, initialize session, and discover tools."""
        mock_stdio.side_effect = _make_stdio_context().side_effect
        mock_session_cls.return_value = _make_session_mock([
            {"name": "tool1", "description": "First tool"},
            {"name": "tool2", "description": "Second tool"},
        ])

        client = MCPClient(name="test", command="cmd")
        client.connect()

        assert client._connected is True
        assert len(client._tools) == 2
        assert client._tools[0]["name"] == "tool1"
        assert client._tools[1]["name"] == "tool2"
        client.close()

    @patch("spine.mcp.client.ClientSession")
    @patch("spine.mcp.client.stdio_client")
    def test_connect_is_idempotent(
        self, mock_stdio: MagicMock, mock_session_cls: MagicMock
    ) -> None:
        """Calling connect() twice should not re-connect."""
        mock_stdio.side_effect = _make_stdio_context().side_effect
        mock_session_cls.return_value = _make_session_mock([])

        client = MCPClient(name="test", command="cmd")
        client.connect()
        first_tools_id = id(client._tools)
        client.connect()  # Should be no-op
        assert id(client._tools) == first_tools_id
        client.close()

    @patch("spine.mcp.client.ClientSession")
    @patch("spine.mcp.client.stdio_client")
    def test_list_tools_before_connect(
        self, mock_stdio: MagicMock, mock_session_cls: MagicMock
    ) -> None:
        """list_tools() should auto-connect if not connected."""
        mock_stdio.side_effect = _make_stdio_context().side_effect
        mock_session_cls.return_value = _make_session_mock([
            {"name": "auto_tool", "description": "Auto discovered"}
        ])

        client = MCPClient(name="test", command="cmd")
        tools = client.list_tools()
        assert client._connected is True
        assert len(tools) == 1
        assert tools[0]["name"] == "auto_tool"
        client.close()

    @patch("spine.mcp.client.ClientSession")
    @patch("spine.mcp.client.stdio_client")
    def test_call_tool_relays_arguments(
        self, mock_stdio: MagicMock, mock_session_cls: MagicMock
    ) -> None:
        """call_tool should forward args to session.call_tool."""
        mock_stdio.side_effect = _make_stdio_context().side_effect

        mock_session = _make_session_mock([])
        # Configure call_tool on the inner mock (what connect stores as self._session)
        mock_result = MagicMock()
        mock_result.isError = False
        mock_result.content = [MagicMock(text="result text")]
        mock_session._inner.call_tool.return_value = mock_result
        mock_session_cls.return_value = mock_session

        client = MCPClient(name="test", command="cmd")
        client.connect()

        result = client.call_tool("my_tool", {"param1": "value1"})
        assert result == "result text"
        mock_session._inner.call_tool.assert_called_once_with(
            "my_tool", arguments={"param1": "value1"}
        )
        client.close()

    @patch("spine.mcp.client.ClientSession")
    @patch("spine.mcp.client.stdio_client")
    def test_call_tool_raises_on_error(
        self, mock_stdio: MagicMock, mock_session_cls: MagicMock
    ) -> None:
        """call_tool should raise RuntimeError when tool returns error."""
        mock_stdio.side_effect = _make_stdio_context().side_effect

        mock_session = _make_session_mock([])
        mock_result = MagicMock()
        mock_result.isError = True
        mock_result.content = "Something went wrong"
        mock_session._inner.call_tool.return_value = mock_result
        mock_session_cls.return_value = mock_session

        client = MCPClient(name="test", command="cmd")
        client.connect()

        with pytest.raises(RuntimeError, match="returned error"):
            client.call_tool("bad_tool", {})
        client.close()

    def test_close_on_unconnected(self) -> None:
        """close() should be safe on an unconnected client."""
        client = MCPClient(name="test", command="cmd")
        client.close()  # should not raise
        assert client._connected is False

    @patch("spine.mcp.client.ClientSession")
    @patch("spine.mcp.client.stdio_client")
    def test_context_manager(
        self, mock_stdio: MagicMock, mock_session_cls: MagicMock
    ) -> None:
        """Client should work as a context manager."""
        mock_stdio.side_effect = _make_stdio_context().side_effect
        mock_session_cls.return_value = _make_session_mock([])

        with MCPClient(name="test", command="cmd") as client:
            assert client._connected is True
        # After exit, should be closed
        assert client._connected is False

    @patch("spine.mcp.client.ClientSession")
    @patch("spine.mcp.client.stdio_client")
    def test_connect_failure_calls_close(
        self, mock_stdio: MagicMock, mock_session_cls: MagicMock
    ) -> None:
        """Failed connect should clean up via close()."""
        mock_stdio.side_effect = _make_stdio_context().side_effect

        # Make ClientSession constructor raise
        mock_session_cls.side_effect = RuntimeError("Boom")

        client = MCPClient(name="test", command="cmd", connect_timeout=5)
        with pytest.raises(RuntimeError):
            client.connect()
        assert client._connected is False


class TestMCPClientManager:
    """Tests for MCPClientManager — multi-server management."""

    @patch("spine.mcp.client.ClientSession")
    @patch("spine.mcp.client.stdio_client")
    def test_get_client_creates_and_caches(
        self, mock_stdio: MagicMock, mock_session_cls: MagicMock
    ) -> None:
        """get_client should create, connect, cache, and reuse clients."""
        mock_stdio.side_effect = _make_stdio_context().side_effect
        mock_session_cls.return_value = _make_session_mock([])

        mgr = MCPClientManager({
            "server-a": {
                "command": "cmd-a", "args": [], "env": {},
                "timeout": 60, "connect_timeout": 30,
            },
        })
        c1 = mgr.get_client("server-a")
        c2 = mgr.get_client("server-a")
        assert c1 is c2  # Cached
        mgr.close_all()

    def test_get_client_unknown_raises(self) -> None:
        """Requesting unknown server should raise KeyError."""
        mgr = MCPClientManager({})
        with pytest.raises(KeyError, match="No config"):
            mgr.get_client("nonexistent")

    @patch("spine.mcp.client.ClientSession")
    @patch("spine.mcp.client.stdio_client")
    def test_get_all_tools(
        self, mock_stdio: MagicMock, mock_session_cls: MagicMock
    ) -> None:
        """get_all_tools should return MCPTool instances from all servers."""
        mock_stdio.side_effect = _make_stdio_context().side_effect
        mock_session_cls.return_value = _make_session_mock([
            {
                "name": "find_symbol",
                "description": "Find a symbol",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ])

        mgr = MCPClientManager({
            "idx": {
                "command": "idx", "args": [], "env": {},
                "timeout": 60, "connect_timeout": 30,
            },
        })
        tools = mgr.get_all_tools()
        assert len(tools) == 1
        assert tools[0].name == "find_symbol"
        assert tools[0].server_name == "idx"
        mgr.close_all()

    @patch("spine.mcp.client.ClientSession")
    @patch("spine.mcp.client.stdio_client")
    def test_close_all(
        self, mock_stdio: MagicMock, mock_session_cls: MagicMock
    ) -> None:
        """close_all should close and clear all clients."""
        mock_stdio.side_effect = _make_stdio_context().side_effect
        mock_session_cls.return_value = _make_session_mock([])

        mgr = MCPClientManager({
            "a": {
                "command": "a", "args": [], "env": {},
                "timeout": 60, "connect_timeout": 30,
            },
            "b": {
                "command": "b", "args": [], "env": {},
                "timeout": 60, "connect_timeout": 30,
            },
        })
        mgr.get_client("a")
        mgr.get_client("b")
        assert len(mgr._clients) == 2
        mgr.close_all()
        assert len(mgr._clients) == 0


class TestGetMCPTools:
    """Tests for get_mcp_tools() — the module-level convenience function."""

    def test_returns_empty_for_none_config(self) -> None:
        """None or empty config should return empty list."""
        assert get_mcp_tools(None) == []
        assert get_mcp_tools({}) == []

    @patch("spine.mcp.client.ClientSession")
    @patch("spine.mcp.client.stdio_client")
    def test_auto_injects_project_root(
        self, mock_stdio: MagicMock, mock_session_cls: MagicMock
    ) -> None:
        """Workspace root should be injected into env if not configured."""
        mock_stdio.side_effect = _make_stdio_context().side_effect
        mock_session_cls.return_value = _make_session_mock([])

        configs = {
            "idx": {
                "command": "idx", "args": [], "timeout": 60, "connect_timeout": 30,
            }
        }
        result = get_mcp_tools(
            configs, cache_key="test", workspace_root="/test/project"
        )
        assert isinstance(result, list)
        # PROJECT_ROOT should have been injected
        assert configs["idx"]["env"]["PROJECT_ROOT"] == "/test/project"
