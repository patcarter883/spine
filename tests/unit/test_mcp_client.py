"""Unit tests for MCP client using langchain-mcp-adapters.

Tests the thin wrapper in ``spine/mcp/client.py`` that loads tools
via ``MultiServerMCPClient`` from ``langchain-mcp-adapters``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.tools import StructuredTool

from spine.mcp.client import _convert_server_config, _namespace_tool, get_mcp_tools


class TestConvertServerConfig:
    """Tests for _convert_server_config — SPINE → adapter config conversion."""

    def test_minimal_config(self) -> None:
        """Minimal config (command only) should default transport to stdio."""
        result = _convert_server_config({"command": "my-server"})
        assert result["transport"] == "stdio"
        assert result["command"] == "my-server"
        assert result["args"] == []
        assert "env" not in result

    def test_full_config(self) -> None:
        """Full config should pass through all fields."""
        result = _convert_server_config(
            {
                "transport": "http",
                "command": "my-server",
                "args": ["--verbose"],
                "env": {"KEY": "val"},
            }
        )
        assert result["transport"] == "http"
        assert result["command"] == "my-server"
        assert result["args"] == ["--verbose"]
        assert result["env"] == {"KEY": "val"}

    def test_empty_args_and_env_omitted(self) -> None:
        """Empty args should be present (required by adapter), empty env omitted."""
        result = _convert_server_config(
            {
                "command": "cmd",
                "args": [],
                "env": {},
            }
        )
        assert result["transport"] == "stdio"
        assert result["command"] == "cmd"
        assert result["args"] == []
        assert "env" not in result


class TestNamespaceTool:
    """Tests for _namespace_tool — adding server-name prefix."""

    def test_prefix_added(self) -> None:
        """Tool name should get mcp_{server_name}_ prefix."""
        tool = StructuredTool.from_function(
            func=lambda x: x,
            name="find_symbol",
            description="Find a symbol",
        )
        result = _namespace_tool(tool, "my-server")
        assert result.name == "mcp_my-server_find_symbol"

    def test_description_preserved(self) -> None:
        """Original description should be preserved."""
        tool = StructuredTool.from_function(
            func=lambda x: x,
            name="search",
            description="Search the codebase",
        )
        result = _namespace_tool(tool, "idx")
        assert result.description == "Search the codebase"

    def test_metadata_preserved(self) -> None:
        """Original metadata should be preserved and augmented."""
        tool = StructuredTool.from_function(
            func=lambda x: x,
            name="tool1",
            description="desc",
        )
        tool.metadata = {"existing": "value"}  # type: ignore[attr-defined]
        result = _namespace_tool(tool, "srv")
        assert result.metadata == {  # type: ignore[attr-defined]
            "existing": "value",
            "mcp_server": "srv",
            "mcp_tool": "tool1",
        }

    def test_callable_after_namespacing(self) -> None:
        """Namespaced tool should still be callable."""
        tool = StructuredTool.from_function(
            func=lambda tool_input: f"result: {tool_input}",
            name="test_tool",
            description="Test",
        )
        result = _namespace_tool(tool, "srv")
        assert result.name == "mcp_srv_test_tool"
        output = result.invoke({"tool_input": "hello"})
        assert output == "result: hello"


class TestGetMCPTools:
    """Tests for get_mcp_tools() — the module-level loader."""

    def test_returns_empty_for_none_config(self) -> None:
        """None or empty config should return empty list."""
        assert get_mcp_tools(None) == []
        assert get_mcp_tools({}) == []

    def test_auto_injects_project_root(self) -> None:
        """Workspace root should be injected into env if not configured."""
        with patch("spine.mcp.client.MultiServerMCPClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get_tools = AsyncMock(return_value=[])
            mock_client_cls.return_value = mock_client

            configs = {
                "idx": {"command": "idx", "args": [], "transport": "stdio"},
            }
            result = get_mcp_tools(configs, cache_key="test", workspace_root="/test/project")
            assert isinstance(result, list)
            assert configs["idx"]["env"]["PROJECT_ROOT"] == "/test/project"

    def test_workspace_root_overrides_existing_project_root(self) -> None:
        """A wrong PROJECT_ROOT in user config must not silently win.

        Regression: trace 019e6974 had ``PROJECT_ROOT: /home/pat/projects/spine``
        (lowercase) configured but spine was running at ``/home/pat/Projects/spine``.
        ``setdefault`` left the wrong value in place and every codebase-index
        call returned ``not found``. Spine's workspace_root is the source of truth.
        """
        with patch("spine.mcp.client.MultiServerMCPClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get_tools = AsyncMock(return_value=[])
            mock_client_cls.return_value = mock_client

            configs = {
                "idx": {
                    "command": "idx",
                    "args": [],
                    "transport": "stdio",
                    "env": {"PROJECT_ROOT": "/wrong/path", "OTHER_VAR": "keep"},
                },
            }
            get_mcp_tools(configs, cache_key="test", workspace_root="/test/project")
            assert configs["idx"]["env"]["PROJECT_ROOT"] == "/test/project"
            assert configs["idx"]["env"]["OTHER_VAR"] == "keep"

    def test_workspace_root_resolved_to_absolute(self, tmp_path) -> None:
        """Relative workspace_root should be resolved to an absolute path
        before being passed to the MCP server (which spawns in a separate
        process and won't share spine's cwd)."""
        with patch("spine.mcp.client.MultiServerMCPClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.get_tools = AsyncMock(return_value=[])
            mock_client_cls.return_value = mock_client

            configs = {"idx": {"command": "idx", "args": [], "transport": "stdio"}}
            get_mcp_tools(configs, cache_key="test", workspace_root=str(tmp_path))
            injected = configs["idx"]["env"]["PROJECT_ROOT"]
            assert injected == str(tmp_path.resolve())
            # And for relative paths
            get_mcp_tools(configs, cache_key="test", workspace_root=".")
            assert configs["idx"]["env"]["PROJECT_ROOT"].startswith("/")

    @patch("spine.mcp.client.MultiServerMCPClient")
    def test_loads_tools_with_namespacing(self, mock_client_cls: MagicMock) -> None:
        """Tools should be loaded and namespaced with server prefix."""
        mock_client = MagicMock()
        # Create real StructuredTool instances for the mock to return
        tool1 = StructuredTool.from_function(
            func=lambda x: "ok", name="find_symbol", description="Find"
        )
        tool2 = StructuredTool.from_function(
            func=lambda x: "ok", name="get_dependencies", description="Deps"
        )
        mock_client.get_tools = AsyncMock(return_value=[tool1, tool2])
        mock_client_cls.return_value = mock_client

        configs = {
            "codebase-index": {
                "transport": "stdio",
                "command": "mcp-codebase-index",
                "args": [],
            },
        }
        result = get_mcp_tools(configs, cache_key="test")

        assert len(result) == 2
        assert result[0].name == "mcp_codebase-index_find_symbol"
        assert result[1].name == "mcp_codebase-index_get_dependencies"

    @patch("spine.mcp.client.MultiServerMCPClient")
    def test_load_failure_returns_empty(self, mock_client_cls: MagicMock) -> None:
        """Failed MCP connection should return empty list gracefully."""
        mock_client_cls.side_effect = RuntimeError("Server not found")

        configs = {
            "broken": {
                "transport": "stdio",
                "command": "nonexistent",
            },
        }
        result = get_mcp_tools(configs)
        assert result == []


class TestMultiServerNamespacing:
    """Per-server attribution (live failure: with two servers configured,
    every tool got the FIRST server's prefix, so the codebase-memory
    backend's tools were unfindable for a whole Phase-1 run)."""

    @patch("spine.mcp.client.MultiServerMCPClient")
    def test_each_server_gets_own_prefix(self, mock_client_cls: MagicMock) -> None:
        idx_tool = StructuredTool.from_function(
            func=lambda x: "ok", name="find_symbol", description="Find"
        )
        cbm_tool = StructuredTool.from_function(
            func=lambda x: "ok", name="search_graph", description="Graph"
        )

        async def per_server(server_name=None):
            return {"codebase-index": [idx_tool], "codebase-memory": [cbm_tool]}[
                server_name
            ]

        mock_client = MagicMock()
        mock_client.get_tools = per_server
        mock_client_cls.return_value = mock_client

        configs = {
            "codebase-index": {"transport": "stdio", "command": "a", "args": []},
            "codebase-memory": {"transport": "stdio", "command": "b", "args": []},
        }
        result = get_mcp_tools(configs, cache_key="test")
        names = sorted(t.name for t in result)
        assert names == [
            "mcp_codebase-index_find_symbol",
            "mcp_codebase-memory_search_graph",
        ]

    @patch("spine.mcp.client.MultiServerMCPClient")
    def test_one_failing_server_degrades_alone(self, mock_client_cls: MagicMock) -> None:
        ok_tool = StructuredTool.from_function(
            func=lambda x: "ok", name="search_graph", description="Graph"
        )

        async def per_server(server_name=None):
            if server_name == "broken":
                raise RuntimeError("spawn failed")
            return [ok_tool]

        mock_client = MagicMock()
        mock_client.get_tools = per_server
        mock_client_cls.return_value = mock_client

        configs = {
            "broken": {"transport": "stdio", "command": "x", "args": []},
            "codebase-memory": {"transport": "stdio", "command": "b", "args": []},
        }
        result = get_mcp_tools(configs, cache_key="test")
        assert [t.name for t in result] == ["mcp_codebase-memory_search_graph"]
