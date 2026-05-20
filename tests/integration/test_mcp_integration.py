"""Integration tests for MCP client with the real mcp-codebase-index server.

These tests require mcp-codebase-index to be installed. They create a small
test project, index it, and verify the tools work end-to-end.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from spine.mcp.client import MCPClient, get_mcp_tools
from spine.mcp.tools import MCPTool, mcp_tool_to_langchain
from spine.config import SpineConfig


@pytest.fixture
def sample_python_project() -> str:
    """Create a temporary Python project with a few files for indexing."""
    tmp = tempfile.mkdtemp()
    root = Path(tmp)

    # Write a couple Python files
    (root / "utils.py").write_text('''
"""Utility functions."""

import os
from typing import Optional


def read_config(path: str) -> dict:
    """Read configuration from a file."""
    with open(path) as f:
        return {"data": f.read()}


class ConfigLoader:
    """Loads and validates configuration."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self._loaded = False

    def load(self) -> Optional[dict]:
        """Load config from base directory."""
        path = os.path.join(self.base_dir, "config.yaml")
        return read_config(path)
''')

    (root / "main.py").write_text('''
"""Main entry point."""

from utils import ConfigLoader


def main():
    loader = ConfigLoader(".")
    config = loader.load()
    print(config)


if __name__ == "__main__":
    main()
''')

    return tmp


@pytest.fixture
def codebase_index_client(sample_python_project: str):  # noqa: ANN201
    """Create a connected MCP client for the sample project."""
    client = MCPClient(
        name="test-codebase-index",
        command="mcp-codebase-index",
        args=[],
        env={"PROJECT_ROOT": sample_python_project},
        timeout=30,
        connect_timeout=30,
    )
    client.connect()
    yield client
    client.close()


class TestRealMCPCodebaseIndex:
    """End-to-end tests with a real mcp-codebase-index server."""

    def test_connect_discovers_18_tools(self, codebase_index_client: MCPClient) -> None:
        """Should discover all 18 tools."""
        tools = codebase_index_client.list_tools()
        assert len(tools) == 18, f"Expected 18 tools, got {len(tools)}"
        names = {t["name"] for t in tools}
        assert "find_symbol" in names
        assert "get_project_summary" in names
        assert "get_dependencies" in names
        assert "get_dependents" in names
        assert "get_change_impact" in names
        assert "get_call_chain" in names
        assert "search_codebase" in names
        assert "get_function_source" in names

    def test_get_project_summary(self, codebase_index_client: MCPClient) -> None:
        """get_project_summary should return high-level stats."""
        result = codebase_index_client.call_tool("get_project_summary", {})
        assert len(result) > 100
        assert "Files:" in result
        assert "Lines:" in result
        assert "Functions:" in result
        assert "Classes:" in result

    def test_find_symbol(self, codebase_index_client: MCPClient) -> None:
        """find_symbol should locate a function definition."""
        result = codebase_index_client.call_tool("find_symbol", {"name": "read_config"})
        assert "read_config" in result
        assert "utils.py" in result

    def test_find_class_symbol(self, codebase_index_client: MCPClient) -> None:
        """find_symbol should find class definitions."""
        result = codebase_index_client.call_tool("find_symbol", {"name": "ConfigLoader"})
        assert "ConfigLoader" in result
        assert "utils.py" in result

    def test_get_function_source(self, codebase_index_client: MCPClient) -> None:
        """get_function_source should return the function body."""
        result = codebase_index_client.call_tool(
            "get_function_source", {"name": "read_config"}
        )
        assert "def read_config" in result
        assert "with open" in result

    def test_get_class_source(self, codebase_index_client: MCPClient) -> None:
        """get_class_source should return the class definition."""
        result = codebase_index_client.call_tool(
            "get_class_source", {"name": "ConfigLoader"}
        )
        assert "class ConfigLoader" in result
        assert "__init__" in result

    def test_get_dependencies(self, codebase_index_client: MCPClient) -> None:
        """get_dependencies should return what a function calls."""
        result = codebase_index_client.call_tool(
            "get_dependencies", {"name": "ConfigLoader.load"}
        )
        # load() calls read_config
        assert "read_config" in result or "ConfigLoader" in result

    def test_get_dependents(self, codebase_index_client: MCPClient) -> None:
        """get_dependents should return what calls a function."""
        result = codebase_index_client.call_tool(
            "get_dependents", {"name": "read_config"}
        )
        # Called by ConfigLoader.load and indirectly by main
        assert len(result) > 10

    def test_get_call_chain(self, codebase_index_client: MCPClient) -> None:
        """get_call_chain should find path between two symbols."""
        result = codebase_index_client.call_tool(
            "get_call_chain", {"from_name": "main", "to_name": "read_config"}
        )
        # Should find a path
        assert "main" in result.lower() or len(result) > 5

    def test_get_functions(self, codebase_index_client: MCPClient) -> None:
        """get_functions should list functions in a file."""
        result = codebase_index_client.call_tool(
            "get_functions", {"file_path": "utils.py"}
        )
        assert "read_config" in result
        assert "__init__" in result
        assert "load" in result

    def test_search_codebase(self, codebase_index_client: MCPClient) -> None:
        """search_codebase should find regex matches across files."""
        result = codebase_index_client.call_tool(
            "search_codebase", {"pattern": r"def\s+\w+", "max_results": 10}
        )
        assert len(result) > 20
        assert "def main" in result or "def read_config" in result

    def test_get_usage_stats(self, codebase_index_client: MCPClient) -> None:
        """get_usage_stats should report session stats."""
        result = codebase_index_client.call_tool("get_usage_stats", {})
        assert "Session duration" in result or "Total queries" in result

    def test_list_files(self, codebase_index_client: MCPClient) -> None:
        """list_files should return indexed file list."""
        result = codebase_index_client.call_tool(
            "list_files", {"pattern": "*.py", "max_results": 10}
        )
        assert "utils.py" in result
        assert "main.py" in result

    def test_list_files_no_pattern(self, codebase_index_client: MCPClient) -> None:
        """list_files without pattern should return all files."""
        result = codebase_index_client.call_tool("list_files", {})
        assert len(result) > 5

    def test_reindex(self, codebase_index_client: MCPClient) -> None:
        """reindex should complete successfully."""
        result = codebase_index_client.call_tool("reindex", {})
        assert result  # Should have some output


class TestMCPToolBridgeIntegration:
    """Integration tests for the LangChain tool bridge with real MCP tools."""

    def test_tool_conversion_and_call(
        self, codebase_index_client: MCPClient
    ) -> None:
        """Convert a real MCP tool to LangChain and invoke it."""
        tools = codebase_index_client.list_tools()
        find_symbol_dict = next(t for t in tools if t["name"] == "find_symbol")

        mcp_tool = MCPTool(
            server_name="test-codebase-index",
            name=find_symbol_dict["name"],
            description=find_symbol_dict.get("description", ""),
            input_schema=find_symbol_dict.get("inputSchema", {}),
            client=codebase_index_client,
        )

        lc_tool = mcp_tool_to_langchain(mcp_tool)
        assert lc_tool.name == "mcp_test-codebase-index_find_symbol"

        result = lc_tool.invoke({"tool_input": '{"name": "main"}'})  # type: ignore[arg-type]
        assert "main" in result or len(result) > 10
        assert isinstance(result, str)

    def test_get_mcp_tools_end_to_end(self, sample_python_project: str) -> None:
        """get_mcp_tools() should return LangChain tools from config."""
        from spine.mcp.client import _MCP_CACHE

        # Clear cache for a fresh test
        _MCP_CACHE.clear()

        configs = {
            "test-idx": {
                "command": "mcp-codebase-index",
                "args": [],
                "env": {"PROJECT_ROOT": sample_python_project},
                "timeout": 30,
                "connect_timeout": 30,
            }
        }
        tools = get_mcp_tools(configs, cache_key="integration-test")
        assert len(tools) == 18
        # All should be BaseTool
        from langchain_core.tools import BaseTool
        for t in tools:
            assert isinstance(t, BaseTool)
            assert t.name.startswith("mcp_test-idx_")

        # Test calling one
        find_symbol = next(t for t in tools if t.name.endswith("find_symbol"))
        result = find_symbol.invoke({"tool_input": '{"name": "read_config"}'})  # type: ignore[arg-type]
        assert "utils.py" in result

        # Clean up
        _MCP_CACHE.clear()


class TestSpineConfigMCPIntegration:
    """Integration tests for SpineConfig with MCP config."""

    def test_config_loads_mcp_servers(self) -> None:
        """SpineConfig.load() should pick up mcp_servers from disk."""
        config = SpineConfig.load()
        assert isinstance(config.mcp_servers, dict)

    def test_config_exposes_mcp_servers_to_ui(self) -> None:
        """get_config() should include mcp_servers for UI rendering."""
        config = SpineConfig.load()
        mcp_servers = config.mcp_servers
        assert isinstance(mcp_servers, dict)
        if mcp_servers:
            server_name = next(iter(mcp_servers))
            server_cfg = mcp_servers[server_name]
            assert "command" in server_cfg
            assert isinstance(server_cfg["args"], list)
            assert isinstance(server_cfg["env"], dict)
