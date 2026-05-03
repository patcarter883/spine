"""Tools Provider implementations."""

import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from .base import Provider, ProviderType


@dataclass
class ToolCall:
    """Represents a tool invocation."""
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0


class ToolsProvider(Provider):
    """Base class for tools providers."""
    provider_type = ProviderType.TOOLS

    @abstractmethod
    def list_tools(self) -> list[dict[str, Any]]:
        """Return available tools with schemas."""
        pass

    @abstractmethod
    def invoke(self, call: ToolCall) -> ToolCall:
        """Execute tool and return result."""
        pass

    @abstractmethod
    def has_tool(self, name: str) -> bool:
        """Check if tool is available."""
        pass


class MCPProvider(ToolsProvider):
    """MCP (Model Context Protocol) tools provider."""

    def __init__(self, server_name: str):
        self._server_name = server_name
        self._client = None
        self._tools: list[dict[str, Any]] = []

    def configure(self, config: dict[str, Any]) -> None:
        self._config = config
        self._tools = config.get("tools", [
            {"name": "browser_use", "description": "Browser automation tool", "parameters": {"type": "object"}},
            {"name": "file_read", "description": "Read file contents", "parameters": {"type": "object"}},
            {"name": "web_search", "description": "Web search capability", "parameters": {"type": "object"}},
        ])

    def validate(self) -> bool:
        return self._config is not None

    @property
    def name(self) -> str:
        return f"mcp:{self._server_name}"

    @property
    def enabled(self) -> bool:
        return self._config is not None

    def list_tools(self) -> list[dict[str, Any]]:
        return self._tools

    def has_tool(self, name: str) -> bool:
        return any(t.get("name") == name for t in self._tools)

    def invoke(self, call: ToolCall) -> ToolCall:
        start = time.time()
        try:
            if not self.has_tool(call.tool_name):
                call.error = f"Tool '{call.tool_name}' not found"
                return call

            call.result = {"status": "simulated", "tool": call.tool_name, "arguments": call.arguments}
        except Exception as e:
            call.error = str(e)
        finally:
            call.execution_time = time.time() - start
        return call

    def execute(self, tool_name: str, **kwargs) -> Any:
        call = ToolCall(tool_name=tool_name, arguments=kwargs)
        result = self.invoke(call)
        if result.error:
            raise RuntimeError(result.error)
        return result.result