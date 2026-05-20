"""Bridge: convert MCP tools to LangChain BaseTool instances.

Each discovered MCP tool is wrapped as a LangChain ``StructuredTool``.
The tool name follows the convention ``mcp_{server_name}_{tool_name}``
to prevent collisions between tools from different MCP servers.

Because MCP tools have dynamic JSON Schema inputs (varying per tool),
we cannot use a typed Python function signature.  Instead we construct
a ``StructuredTool`` that accepts a single ``tool_input`` JSON string and
forwards it to the MCP server.  The LLM learns the expected parameters
from the tool description, which includes the full JSON Schema docs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool


@dataclass
class MCPTool:
    """Represents a discovered MCP tool before LangChain conversion.

    Attributes:
        server_name: The MCP server this tool belongs to.
        name: Original MCP tool name (e.g. ``"find_symbol"``).
        description: Human-readable tool description.
        input_schema: JSON Schema for the tool's parameters.
        client: Reference to the ``MCPClient`` that owns this tool.
    """

    server_name: str
    name: str
    description: str
    input_schema: dict[str, Any]
    client: Any  # MCPClient (avoid circular import)


def mcp_tool_to_langchain(mcp_tool: MCPTool) -> BaseTool:
    """Convert an ``MCPTool`` to a LangChain ``BaseTool``.

    Creates a callable LangChain tool that accepts keyword arguments,
    forwards them to the MCP server's ``call_tool()``, and returns the
    text result.

    Tool name: ``mcp_{server_name}_{tool_name}``
    Hyphens in names are preserved (valid in LangChain tool names).

    Args:
        mcp_tool: The discovered MCP tool to wrap.

    Returns:
        A LangChain ``BaseTool`` instance ready for agent injection.
    """
    lc_name = f"mcp_{mcp_tool.server_name}_{mcp_tool.name}"

    # Build a description with parameter docs so the LLM knows what args to pass.
    param_docs = _build_param_docs(mcp_tool.input_schema)
    lc_description = (
        f"MCP tool from server '{mcp_tool.server_name}'. "
        f"{mcp_tool.description}\n\n"
        f"Parameters: {param_docs if param_docs else 'none'}"
    )

    def _call_mcp_tool(tool_input: str = "") -> str:
        """Execute the MCP tool with the given JSON arguments.

        Args:
            tool_input: JSON object string with the tool's parameters.
                E.g. ``'{"name": "my_func", "max_results": 10}'``.
                Empty string for parameterless tools.
        """
        client = mcp_tool.client
        if tool_input and tool_input.strip():
            arguments = json.loads(tool_input)
        else:
            arguments = {}
        return client.call_tool(mcp_tool.name, arguments=arguments)

    tool = StructuredTool.from_function(
        func=_call_mcp_tool,
        name=lc_name,
        description=lc_description,
    )
    tool.metadata = {  # type: ignore[attr-defined]
        "mcp_server": mcp_tool.server_name,
        "mcp_tool": mcp_tool.name,
    }
    return tool


def _build_param_docs(input_schema: dict[str, Any]) -> str:
    """Build human-readable parameter documentation from JSON Schema.

    Args:
        input_schema: The tool's ``inputSchema`` dict (JSON Schema).

    Returns:
        A multi-line string describing each parameter, or an empty
        string if the schema is empty or not an object type.
    """
    if not input_schema or input_schema.get("type") != "object":
        return ""

    props = input_schema.get("properties", {})
    required = input_schema.get("required", [])
    lines: list[str] = []

    for param_name, param_schema in props.items():
        param_type = param_schema.get("type", "any")
        param_desc = param_schema.get("description", "")
        is_required = "required" if param_name in required else "optional"
        lines.append(
            f"  - {param_name} ({param_type}, {is_required}): {param_desc}"
        )

    return "\n".join(lines)
