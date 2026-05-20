"""Integration tests for MCP tools via langchain-mcp-adapters."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from spine.mcp.client import get_mcp_tools
from spine.config import SpineConfig


@pytest.fixture
def sample_python_project() -> str:
    """Create a temporary Python project with a few files for indexing."""
    tmp = tempfile.mkdtemp()
    root = Path(tmp)

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


def _tool_result_text(result) -> str:
    """Extract text from an adapter tool result (may be list of content blocks)."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return "\n".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in result
        )
    return str(result)


@pytest.fixture
def mcp_tools_async(sample_python_project: str):  # noqa: ANN201
    """Load MCP tools from mcp-codebase-index for the sample project."""
    configs = {
        "test-idx": {
            "transport": "stdio",
            "command": "mcp-codebase-index",
            "args": [],
            "env": {"PROJECT_ROOT": sample_python_project},
        },
    }
    tools = get_mcp_tools(configs, cache_key="integration-test-async")
    assert len(tools) == 18, f"Expected 18 tools, got {len(tools)}"
    return tools


class TestRealMCPCodebaseIndex:
    """End-to-end tests with real mcp-codebase-index via the adapter."""

    def test_all_18_tools_namespaced(self, mcp_tools_async: list) -> None:
        """All 18 tools should have the server prefix."""
        for t in mcp_tools_async:
            assert t.name.startswith("mcp_test-idx_"), f"Bad name: {t.name}"

    def test_tools_are_callable(self, mcp_tools_async: list) -> None:
        """Tools should be LangChain BaseTool instances."""
        from langchain_core.tools import BaseTool
        for t in mcp_tools_async:
            assert isinstance(t, BaseTool)

    @pytest.mark.asyncio
    async def test_find_symbol(self, mcp_tools_async: list) -> None:
        """find_symbol should locate a function definition."""
        tool = next(t for t in mcp_tools_async if t.name.endswith("find_symbol"))
        result = await tool.ainvoke({"name": "read_config"})
        text = _tool_result_text(result)
        assert "utils.py" in text

    @pytest.mark.asyncio
    async def test_get_project_summary(self, mcp_tools_async: list) -> None:
        """get_project_summary should return high-level stats."""
        tool = next(t for t in mcp_tools_async if t.name.endswith("get_project_summary"))
        result = await tool.ainvoke({})
        text = _tool_result_text(result)
        assert len(text) > 100
        assert "Files:" in text

    @pytest.mark.asyncio
    async def test_get_function_source(self, mcp_tools_async: list) -> None:
        """get_function_source should return the function body."""
        tool = next(t for t in mcp_tools_async if t.name.endswith("get_function_source"))
        result = await tool.ainvoke({"name": "read_config"})
        text = _tool_result_text(result)
        assert "def read_config" in text

    @pytest.mark.asyncio
    async def test_get_dependencies(self, mcp_tools_async: list) -> None:
        """get_dependencies should return what a function calls."""
        tool = next(t for t in mcp_tools_async if t.name.endswith("get_dependencies"))
        result = await tool.ainvoke({"name": "ConfigLoader.load"})
        text = _tool_result_text(result)
        assert len(text) > 5

    @pytest.mark.asyncio
    async def test_search_codebase(self, mcp_tools_async: list) -> None:
        """search_codebase should find regex matches."""
        tool = next(t for t in mcp_tools_async if t.name.endswith("search_codebase"))
        result = await tool.ainvoke({"pattern": r"def\s+\w+", "max_results": 10})
        text = _tool_result_text(result)
        assert len(text) > 20

    @pytest.mark.asyncio
    async def test_get_call_chain(self, mcp_tools_async: list) -> None:
        """get_call_chain should find path between symbols."""
        tool = next(t for t in mcp_tools_async if t.name.endswith("get_call_chain"))
        result = await tool.ainvoke({"from_name": "main", "to_name": "read_config"})
        text = _tool_result_text(result)
        assert len(text) > 5


class TestSpineConfigMCPIntegration:
    """Integration tests for SpineConfig with MCP config."""

    def test_config_loads_mcp_servers(self) -> None:
        """SpineConfig.load() should pick up mcp_servers from disk."""
        config = SpineConfig.load()
        assert isinstance(config.mcp_servers, dict)

    def test_config_has_transport_field(self) -> None:
        """MCP server configs should include transport field."""
        config = SpineConfig.load()
        for name, cfg in config.mcp_servers.items():
            assert "transport" in cfg, f"Server {name!r} missing 'transport'"
            assert cfg["transport"] in ("stdio", "http")
            assert "command" in cfg

    @pytest.mark.asyncio
    async def test_get_mcp_tools_end_to_end(self, sample_python_project: str) -> None:
        """get_mcp_tools() should return namespaced LangChain tools."""
        import spine.mcp.client as mcp_client
        mcp_client._client = None

        configs = {
            "test-idx": {
                "transport": "stdio",
                "command": "mcp-codebase-index",
                "args": [],
                "env": {"PROJECT_ROOT": sample_python_project},
            },
        }
        tools = get_mcp_tools(configs, cache_key="integration-test-e2e")

        assert len(tools) == 18
        from langchain_core.tools import BaseTool
        for t in tools:
            assert isinstance(t, BaseTool)
            assert t.name.startswith("mcp_test-idx_")

        find_symbol = next(t for t in tools if t.name.endswith("find_symbol"))
        result = await find_symbol.ainvoke({"name": "read_config"})
        text = _tool_result_text(result)
        assert "utils.py" in text
