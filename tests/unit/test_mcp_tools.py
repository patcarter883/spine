"""Unit tests for MCP tool integration via langchain-mcp-adapters.

The adapter (MultiServerMCPClient) returns LangChain StructuredTool
instances directly — no manual conversion needed. These tests verify
the wrapper's config conversion, namespacing, and error handling.
"""
