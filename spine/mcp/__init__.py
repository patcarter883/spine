"""SPINE MCP — loads MCP tools via langchain-mcp-adapters.

Thin wrapper around LangChain's ``MultiServerMCPClient`` that loads
MCP tools from the SPINE config and namespaces them with the server
name to prevent collisions.
"""

from spine.mcp.client import get_mcp_tools

__all__ = ["get_mcp_tools"]
