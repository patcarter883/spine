"""SPINE MCP client — connects to MCP servers and bridges tools into LangChain.

Provides MCP session management (spawn, connect, tool discovery, tool execution)
and automatic conversion of discovered MCP tools to LangChain ``BaseTool`` instances
for injection into phase agents.
"""

from spine.mcp.client import MCPClient, MCPClientManager, get_mcp_tools
from spine.mcp.tools import MCPTool, mcp_tool_to_langchain

__all__ = [
    "MCPClient",
    "MCPClientManager",
    "MCPTool",
    "get_mcp_tools",
    "mcp_tool_to_langchain",
]