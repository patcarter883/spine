"""Unit tests for MCP → LangChain tool bridge."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from langchain_core.tools import BaseTool

from spine.mcp.tools import MCPTool, _build_param_docs, mcp_tool_to_langchain


class TestBuildParamDocs:
    """Tests for _build_param_docs() — JSON Schema → human-readable docs."""

    def test_empty_schema(self) -> None:
        """Empty schema should return empty string."""
        assert _build_param_docs({}) == ""

    def test_none_schema(self) -> None:
        """None schema should return empty string."""
        assert _build_param_docs(None) == ""  # type: ignore[arg-type]

    def test_non_object_schema(self) -> None:
        """Non-object type schema should return empty string."""
        assert _build_param_docs({"type": "string"}) == ""

    def test_single_param(self) -> None:
        """Single required parameter should be documented."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The symbol name"}
            },
            "required": ["name"],
        }
        result = _build_param_docs(schema)
        assert "name" in result
        assert "string" in result
        assert "required" in result
        assert "The symbol name" in result

    def test_multiple_params_mixed_required(self) -> None:
        """Both required and optional params should show."""
        schema = {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Optional file path to narrow search"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results"
                },
            },
            "required": ["max_results"],
        }
        result = _build_param_docs(schema)
        assert "file_path" in result
        assert "optional" in result
        assert "max_results" in result
        assert "required" in result

    def test_param_without_type_falls_back_to_any(self) -> None:
        """Params without a type field should show 'any'."""
        schema = {
            "type": "object",
            "properties": {
                "raw": {"description": "Some arbitrary data"}
            },
        }
        result = _build_param_docs(schema)
        assert "any" in result


class TestMCPToolToLangChain:
    """Tests for mcp_tool_to_langchain() — MCPTool → BaseTool conversion."""

    def make_mock_client(self) -> MagicMock:
        """Helper: create a mock client that records calls."""
        client = MagicMock()
        client.call_tool.return_value = "mock result"
        return client

    def test_converts_to_langchain_tool(self) -> None:
        """MCPTool should become a callable LangChain BaseTool."""
        client = self.make_mock_client()
        mcp_tool = MCPTool(
            server_name="my-server",
            name="find_symbol",
            description="Find a symbol by name",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Symbol name"}
                },
                "required": ["name"],
            },
            client=client,
        )

        lc_tool = mcp_tool_to_langchain(mcp_tool)
        assert isinstance(lc_tool, BaseTool)
        # Hyphens are preserved in tool names
        assert lc_tool.name == "mcp_my-server_find_symbol"
        assert "Find a symbol" in lc_tool.description

    def test_tool_is_callable_with_json(self) -> None:
        """The LangChain tool should accept a JSON string input."""
        client = self.make_mock_client()
        mcp_tool = MCPTool(
            server_name="idx",
            name="get_dependencies",
            description="Get deps",
            input_schema={"type": "object", "properties": {}},
            client=client,
        )
        lc_tool = mcp_tool_to_langchain(mcp_tool)
        result = lc_tool.invoke({"tool_input": json.dumps({"name": "my_func"})})
        assert result == "mock result"
        client.call_tool.assert_called_once_with(
            "get_dependencies", arguments={"name": "my_func"}
        )

    def test_tool_is_callable_with_empty_input(self) -> None:
        """Calling with empty input should work."""
        client = self.make_mock_client()
        mcp_tool = MCPTool(
            server_name="srv",
            name="no_params",
            description="No params tool",
            input_schema={},
            client=client,
        )
        lc_tool = mcp_tool_to_langchain(mcp_tool)
        result = lc_tool.invoke({"tool_input": ""})
        assert result == "mock result"
        client.call_tool.assert_called_once_with(
            "no_params", arguments={}
        )

    def test_tool_calls_are_forwarded(self) -> None:
        """JSON string args should be parsed and forwarded to client.call_tool."""
        client = self.make_mock_client()
        mcp_tool = MCPTool(
            server_name="srv",
            name="search",
            description="Search",
            input_schema={},
            client=client,
        )
        lc_tool = mcp_tool_to_langchain(mcp_tool)
        lc_tool.invoke({"tool_input": json.dumps({"query": "test", "limit": 10})})
        client.call_tool.assert_called_once_with(
            "search", arguments={"query": "test", "limit": 10}
        )

    def test_tool_name_with_hyphens_preserved(self) -> None:
        """Server names with hyphens should keep them in the tool name."""
        client = self.make_mock_client()
        mcp_tool = MCPTool(
            server_name="codebase-index",
            name="get_project_summary",
            description="Summary",
            input_schema={},
            client=client,
        )
        lc_tool = mcp_tool_to_langchain(mcp_tool)
        # Hyphens are preserved: "mcp_codebase-index_get_project_summary"
        assert lc_tool.name == "mcp_codebase-index_get_project_summary"

    def test_tool_metadata_set(self) -> None:
        """Metadata should track the original server and tool name."""
        client = self.make_mock_client()
        mcp_tool = MCPTool(
            server_name="abc",
            name="xyz",
            description="desc",
            input_schema={},
            client=client,
        )
        lc_tool = mcp_tool_to_langchain(mcp_tool)
        assert lc_tool.metadata == {"mcp_server": "abc", "mcp_tool": "xyz"}  # type: ignore[attr-defined]

    def test_empty_input_schema(self) -> None:
        """Tool should work with an empty input schema."""
        client = self.make_mock_client()
        mcp_tool = MCPTool(
            server_name="srv",
            name="no_params",
            description="No params tool",
            input_schema={},
            client=client,
        )
        lc_tool = mcp_tool_to_langchain(mcp_tool)
        assert "none" in lc_tool.description.lower()
        # Should still work
        result = lc_tool.invoke({"tool_input": ""})
        assert result == "mock result"
