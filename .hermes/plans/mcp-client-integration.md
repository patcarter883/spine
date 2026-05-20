# MCP Client Integration — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Integrate a native MCP client into SPINE that connects to MCP servers (specifically mcp-codebase-index) and exposes their tools to SPINE phase agents.

**Architecture:** A new `spine/mcp/` package manages MCP server lifecycle (spawn, connect, tool discovery, call) using the `mcp` Python SDK. Tools discovered from MCP servers are wrapped as LangChain `StructuredTool` instances and injected into phase agents via `extra_tools` in the agent factory.

**Tech Stack:** `mcp` Python SDK (stdlib), `mcp-codebase-index` (PyPI), LangChain `BaseTool`/`StructuredTool`

---

## Architecture Overview

```
SPINE PHASE AGENT (create_agent)
  │
  ├── FilesystemMiddleware (read, write, edit, execute, ...)
  ├── SubAgentMiddleware (task)
  ├── CodeInterpreterMiddleware (eval)
  ├── ToolSchemaValidator (rebound)
  ├── ToolOutputTrimmer (context eviction)
  │
  └── MCP Tools (injected via extra_tools)  ← NEW
      ├── mcp_codebase_index_get_project_summary
      ├── mcp_codebase_index_find_symbol
      ├── mcp_codebase_index_get_dependencies
      ├── mcp_codebase_index_get_change_impact
      ├── mcp_codebase_index_search_codebase
      └── ... (13 more tools)
```

### Integration Points

```
.spine/config.yaml
  └── mcp_servers:              → SpineConfig parses MCP config
      └── codebase-index:       → MCP client manages server lifecycle
          command: "..."
          args: [...]
          env: {...}

spine/mcp/client.py             → MCP session management (connect, list_tools, call_tool, close)
spine/mcp/tools.py              → MCP → LangChain tool conversion
spine/agents/factory.py         → Load MCP tools and inject via extra_tools
spine/config.py                 → Add mcp_servers config field
```

### Lifecycle

1. **Config load**: `SpineConfig.load()` reads `mcp_servers` from `.spine/config.yaml`
2. **Server start**: `MCPClient` spawns server process, connects via stdio
3. **Tool discovery**: `client.list_tools()` → MCP tools discovered
4. **Tool wrapping**: Each MCP tool → LangChain `StructuredTool(function=...)`
5. **Agent injection**: `build_phase_agent()` receives MCP tools via `extra_tools`
6. **Tool call**: Agent calls `mcp_codebase_index_find_symbol(name="foo")` → MCP client relays
7. **Server lifecycle**: Persists across phase agents for the work item lifetime

---

### Task 1: Add `mcp` dependency to pyproject.toml

**Objective:** Add the `mcp` Python SDK as an optional dependency.

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add MCP dependency**

Add `"mcp>=1.0.0"` to the `dependencies` list. The `mcp` SDK provides the client-side transports needed to connect to MCP servers (stdio, SSE, streamable HTTP).

**Step 2: Install and verify**

```bash
uv sync
python -c "from mcp.client.stdio import stdio_client; print('MCP SDK available')"
```

---

### Task 2: Add `mcp_servers` config to SpineConfig

**Objective:** Extend `SpineConfig` to parse MCP server configurations.

**Files:**
- Modify: `spine/config.py`

**Step 1: Add `mcp_servers` field to SpineConfig dataclass**

```python
@dataclass
class SpineConfig:
    # ... existing fields ...
    mcp_servers: dict = field(default_factory=dict)
    # Each key is a server name (e.g., "codebase-index")
    # Each value is: {command, args, env, timeout, connect_timeout}
```

**Step 2: Parse mcp_servers from config.yaml**

In `SpineConfig.load()`, parse the `mcp_servers` key after loading the YAML:

```python
mcp_servers = {}
raw_servers = config_data.get("mcp_servers", {})
for name, server_cfg in raw_servers.items():
    if not isinstance(server_cfg, dict):
        continue
    mcp_servers[name] = {
        "command": server_cfg.get("command", ""),
        "args": server_cfg.get("args", []),
        "env": server_cfg.get("env", {}),
        "timeout": server_cfg.get("timeout", 120),
        "connect_timeout": server_cfg.get("connect_timeout", 60),
    }
```

**Step 3: Environment variable override**

Allow `SPINE_MCP_SERVERS` env var (JSON string) to override/merge:

```python
env_mcp = os.environ.get("SPINE_MCP_SERVERS")
if env_mcp:
    import json
    mcp_servers.update(json.loads(env_mcp))
```

**Step 4: Write unit test**

```python
def test_mcp_servers_config():
    config = SpineConfig.load(raw={
        "mcp_servers": {
            "codebase-index": {
                "command": "mcp-codebase-index",
                "env": {"PROJECT_ROOT": "/test"},
            }
        }
    })
    assert "codebase-index" in config.mcp_servers
    assert config.mcp_servers["codebase-index"]["command"] == "mcp-codebase-index"
```

Run: `pytest tests/unit/test_config.py -k test_mcp -v`

---

### Task 3: Create MCP Client module (`spine/mcp/client.py`)

**Objective:** Create a client that connects to MCP servers via stdio, discovers tools, and executes calls.

**Files:**
- Create: `spine/mcp/__init__.py`
- Create: `spine/mcp/client.py`

**Step 1: Create package init**

```python
# spine/mcp/__init__.py
"""SPINE MCP client — connects to MCP servers and bridges tools into LangChain."""

from spine.mcp.client import MCPClient, MCPClientManager, get_mcp_tools
from spine.mcp.tools import mcp_tool_to_langchain

__all__ = ["MCPClient", "MCPClientManager", "get_mcp_tools", "mcp_tool_to_langchain"]
```

**Step 2: Implement MCPClient class**

```python
# spine/mcp/client.py
"""MCP client: connect to MCP servers via stdio, discover tools, execute calls."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

logger = logging.getLogger(__name__)


class MCPClient:
    """Manages a single MCP server connection via stdio transport.

    Handles process lifecycle (spawn, connect, close) and tool discovery.
    All calls are blocking but use an internal event loop for the MCP protocol.
    """

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 120,
        connect_timeout: int = 60,
    ):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self._session: ClientSession | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._tools: list[dict] = []
        self._connected = False
        self._loop: asyncio.AbstractEventLoop | None = None

    def connect(self) -> None:
        """Start the MCP server subprocess and establish a session."""
        if self._connected:
            return

        loop = asyncio.new_event_loop()
        self._loop = loop

        # Build filtered environment
        child_env = os.environ.copy()
        child_env.update(self.env)

        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=child_env,
        )

        async def _connect():
            self._exit_stack = AsyncExitStack()
            transport = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read_stream, write_stream = transport
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()
            # Discover tools
            result = await self._session.list_tools()
            self._tools = [tool.model_dump() for tool in result.tools]

        try:
            loop.run_until_complete(asyncio.wait_for(_connect(), timeout=self.connect_timeout))
            self._connected = True
            logger.info(
                "MCP server '%s' connected: %d tools discovered",
                self.name, len(self._tools),
            )
        except Exception:
            self.close()
            raise

    def list_tools(self) -> list[dict]:
        """Return discovered tools as dicts {name, description, inputSchema}."""
        if not self._connected:
            self.connect()
        return list(self._tools)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute an MCP tool and return the result as a string."""
        if not self._connected:
            self.connect()

        async def _call():
            assert self._session is not None
            result = await self._session.call_tool(name, arguments=arguments)
            if result.isError:
                raise RuntimeError(f"MCP tool '{name}' returned error: {result.content}")
            parts = []
            for content in result.content:
                if hasattr(content, "text"):
                    parts.append(content.text)
                elif hasattr(content, "data"):
                    parts.append(f"[binary data: {len(content.data)} bytes]")
            return "\n".join(parts) if parts else str(result.content)

        try:
            return self._loop.run_until_complete(
                asyncio.wait_for(_call(), timeout=self.timeout)
            )
        except Exception:
            logger.exception("MCP tool '%s' failed", name)
            raise

    def close(self) -> None:
        """Close the MCP session and terminate the server process."""
        self._connected = False
        self._tools = []
        if self._exit_stack:
            try:
                self._loop.run_until_complete(self._exit_stack.aclose())
            except Exception:
                pass
            self._exit_stack = None
            self._session = None
        if self._loop and not self._loop.is_closed():
            self._loop.close()
            self._loop = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
        return False
```

**Step 3: Implement MCPClientManager (multi-server)**

```python
class MCPClientManager:
    """Manages multiple MCP server connections.

    Loads server configs from SpineConfig and creates/reuses clients.
    Intended to be instantiated once per work item and shared across
    phase agents within that work item.
    """

    def __init__(self, server_configs: dict[str, dict[str, Any]]):
        self._clients: dict[str, MCPClient] = {}
        self._server_configs = server_configs

    def get_client(self, server_name: str) -> MCPClient:
        """Get or create a client for the named server."""
        if server_name not in self._clients:
            cfg = self._server_configs.get(server_name)
            if not cfg:
                raise KeyError(f"No config for MCP server '{server_name}'")
            client = MCPClient(
                name=server_name,
                command=cfg["command"],
                args=cfg.get("args", []),
                env=cfg.get("env", {}),
                timeout=cfg.get("timeout", 120),
                connect_timeout=cfg.get("connect_timeout", 60),
            )
            client.connect()
            self._clients[server_name] = client
        return self._clients[server_name]

    def get_all_tools(self) -> list["MCPTool"]:
        """Return all tools from all connected servers."""
        from spine.mcp.tools import MCPTool
        all_tools = []
        for name in self._server_configs:
            client = self.get_client(name)
            for tool_dict in client.list_tools():
                tool = MCPTool(
                    server_name=name,
                    name=tool_dict["name"],
                    description=tool_dict.get("description", ""),
                    input_schema=tool_dict.get("inputSchema", {}),
                    client=client,
                )
                all_tools.append(tool)
        return all_tools

    def close_all(self) -> None:
        for client in self._clients.values():
            client.close()
        self._clients.clear()
```

**Verification:** Run unit test: `pytest tests/unit/test_mcp_client.py -v`

---

### Task 4: Create MCP → LangChain tool bridge (`spine/mcp/tools.py`)

**Objective:** Convert MCP tools (with JSON Schema input) into LangChain `BaseTool` instances.

**Files:**
- Create: `spine/mcp/tools.py`

**Step 1: Implement MCPTool dataclass and conversion**

```python
# spine/mcp/tools.py
"""Bridge: convert MCP tools to LangChain BaseTool instances."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool


@dataclass
class MCPTool:
    """Represents a discovered MCP tool (before LangChain conversion)."""
    server_name: str
    name: str
    description: str
    input_schema: dict[str, Any]
    client: Any  # MCPClient reference for calling


def mcp_tool_to_langchain(tool: MCPTool) -> BaseTool:
    """Convert an MCPTool to a LangChain StructuredTool.

    Uses a generic 'run' function since MCP tools have arbitrary JSON Schema
    inputs — we can't create a typed Pydantic model for every possible schema.
    Instead, we accept a single JSON string arg and parse it server-side.

    The tool name follows the convention: mcp_{server_name}_{tool_name}
    """
    lc_name = f"mcp_{tool.server_name}_{tool.name}"

    # Build a description with parameter info
    param_docs = _build_param_docs(tool.input_schema)
    lc_description = (
        f"MCP tool from server '{tool.server_name}'. "
        f"{tool.description}\n\n"
        f"Parameters: {param_docs if param_docs else 'none'}"
    )

    def _call_mcp_tool(**kwargs) -> str:
        """Execute the MCP tool with the given arguments."""
        client = tool.client
        return client.call_tool(tool.name, arguments=kwargs)

    # Create the LangChain tool
    from langchain_core.tools import tool as lc_tool

    @lc_tool(lc_name, description=lc_description)
    def _wrapper(**kwargs) -> str:
        return _call_mcp_tool(**kwargs)

    # Set metadata so the model knows it's an MCP tool
    _wrapper.metadata = {
        "mcp_server": tool.server_name,
        "mcp_tool": tool.name,
    }

    return _wrapper


def _build_param_docs(input_schema: dict) -> str:
    """Build human-readable parameter documentation from JSON Schema."""
    if not input_schema or input_schema.get("type") != "object":
        return ""
    props = input_schema.get("properties", {})
    required = input_schema.get("required", [])
    lines = []
    for param_name, param_schema in props.items():
        param_type = param_schema.get("type", "any")
        param_desc = param_schema.get("description", "")
        is_required = "required" if param_name in required else "optional"
        lines.append(f"  - {param_name} ({param_type}, {is_required}): {param_desc}")
    return "\n".join(lines)
```

**Step 2: Handle the dynamic function signature problem**

LangChain's `@tool` decorator introspects the function signature. Since MCP tools have dynamic args, we need to accept `**kwargs`. This means the model receives the parameter info from the description text, not from the function signature. This is the standard approach for dynamic-tool bridges.

**Verification:** Test that the tool wrapper correctly:
1. Accepts keyword args
2. Routes to the client's `call_tool()`
3. Returns the string result
4. Raises on MCP error

---

### Task 5: Add `get_mcp_tools()` loader function

**Objective:** Create a function that loads MCP tools from config and injects them into the agent factory.

**Files:**
- Modify: `spine/mcp/client.py` (add `get_mcp_tools`)

**Step 1: Implement get_mcp_tools()**

```python
def get_mcp_tools(
    server_configs: dict[str, dict[str, Any]] | None,
    cache_key: str = "default",
) -> list[BaseTool]:
    """Load MCP tools from server configs, returning LangChain tools.

    Tools are cached per cache_key (typically work_id) so the same MCP
    server connections are reused across phase agents within a work item.

    Args:
        server_configs: Dict of server_name → {command, args, env, ...}
        cache_key: Key for client manager reuse (use work_id).
    """
    if not server_configs:
        return []

    global _MCP_CACHE
    if cache_key not in _MCP_CACHE:
        mgr = MCPClientManager(server_configs)
        _MCP_CACHE[cache_key] = mgr
    else:
        mgr = _MCP_CACHE[cache_key]

    mcp_tools = mgr.get_all_tools()
    lc_tools = [mcp_tool_to_langchain(t) for t in mcp_tools]
    return lc_tools

_MCP_CACHE: dict[str, MCPClientManager] = {}
```

---

### Task 6: Integrate MCP tools into build_phase_agent

**Objective:** Wire MCP tools into the agent factory so they're available to phase agents.

**Files:**
- Modify: `spine/agents/factory.py`

**Step 1: Load MCP tools in build_phase_agent**

Add after the model/backend resolution (after line 234):

```python
# ── MCP tools ──────────────────────────────────────────────────────
from spine.mcp.client import get_mcp_tools
from spine.config import SpineConfig

config_obj = SpineConfig.load()
mcp_tools = get_mcp_tools(
    config_obj.mcp_servers,
    cache_key=state.get("work_id", "default"),
)
```

**Step 2: Merge MCP tools with extra_tools**

```python
# In the create_agent call, merge MCP tools with user-provided extra_tools:
all_extra_tools = list(extra_tools) if extra_tools else []
all_extra_tools.extend(mcp_tools)

agent = create_agent(
    model,
    system_prompt=final_system_prompt,
    tools=all_extra_tools,
    middleware=middleware,
    # ... rest unchanged
)
```

**Step 3: Add MCP guidance to system prompt**

When MCP tools are available, append a brief usage reminder to the system prompt:

```python
if mcp_tools:
    mcp_guidance = (
        "\n\n## Codebase Navigation Tools (MCP)\n"
        "You have access to MCP tools for efficient codebase navigation. "
        "Use these for symbol lookup, dependency analysis, and change impact "
        "assessment. They are much more token-efficient than reading entire "
        "files with glob/grep/read."
    )
    final_system_prompt += mcp_guidance
```

---

### Task 7: Create default config for mcp-codebase-index

**Objective:** Ensure SPINE ships with a sensible default for mcp-codebase-index.

**Files:**
- Create: `spine/mcp/defaults.py`

**Step 1: Create defaults module**

```python
# spine/mcp/defaults.py
"""Default MCP server configurations for SPINE."""

DEFAULT_MCP_SERVERS = {
    "codebase-index": {
        "command": "mcp-codebase-index",
        "args": [],
        "env": {},  # PROJECT_ROOT is set from workspace_root at runtime
        "timeout": 120,
        "connect_timeout": 60,
    }
}
```

**Step 2: Apply defaults when config doesn't specify**

In `get_mcp_tools()` or `SpineConfig.load()`, merge defaults with user config (user config wins):

```python
from spine.mcp.defaults import DEFAULT_MCP_SERVERS

merged = {**DEFAULT_MCP_SERVERS, **user_servers}
```

---

### Task 8: Write tests

**Objective:** Comprehensive test coverage for the MCP integration.

**Files:**
- Create: `tests/unit/test_mcp_client.py`
- Create: `tests/unit/test_mcp_tools.py`
- Create: `tests/unit/test_mcp_config.py`
- Create: `tests/integration/test_mcp_integration.py`

**Step 1: Unit test MCPClient (mocked MCP session)**

```python
class TestMCPClient:
    def test_connect_discovers_tools(self):
        """Connect should discover tools via list_tools."""
        ...

    def test_call_tool_relays_arguments(self):
        """call_tool should forward args to the session."""
        ...

    def test_close_cleans_up(self):
        """close should terminate session and process."""
        ...

    def test_reconnect_after_close(self):
        """Calling after close should re-connect."""
        ...
```

**Step 2: Unit test tool conversion**

```python
class TestMCPToolConversion:
    def test_converts_to_langchain_tool(self):
        """MCPTool should become a callable LangChain BaseTool."""
        ...

    def test_tool_name_prefix(self):
        """Tool name should be mcp_{server}_{tool}."""
        ...

    def test_passes_kwargs_to_client(self):
        """Keyword args should reach client.call_tool."""
        ...
```

**Step 3: Integration test with a mock MCP server**

```python
@pytest.mark.asyncio
async def test_end_to_end_with_test_server(tmp_path):
    """Full integration: config → client → tool → call → result."""
    # Create a minimal test MCP server that implements one tool
    # Connect the client
    # Call the tool
    # Verify the result
    ...
```

---

### Task 9: Add MCP tool usage guidelines to SPINE skills

**Objective:** Update the SPINE agent skills documentation so agents know how and when to use MCP tools.

**Files:**
- Create: `spine/skills/mcp-codebase-index/SKILL.md`

**Step 1: Create the skill file**

```markdown
---
name: mcp-codebase-index
description: Codebase navigation via MCP — find symbols, analyze dependencies, assess change impact
phases: [plan, tasks, implement, verify]
---

# MCP Codebase Index

You have access to MCP tools for efficient codebase navigation. These tools are MUCH more token-efficient than reading entire files — use them FIRST.

## Available Tools

| Tool | Use When |
|------|----------|
| `get_project_summary` | You need a high-level overview |
| `find_symbol` | You need to locate where something is defined |
| `get_function_source` | You need to see a function's code |
| `get_class_source` | You need to see a class definition |
| `get_dependencies` | You need to know what a function calls |
| `get_dependents` | You need to know who calls a function |
| `get_change_impact` | You're planning a refactor — what breaks? |
| `get_call_chain` | You need to trace execution flow between two symbols |
| `search_codebase` | You need to find all occurrences of a pattern |

## Usage Rules

1. ALWAYS try `find_symbol` before reading files with glob/grep
2. Use `get_dependencies`/`get_dependents` to understand relationships
3. Use `get_change_impact` before modifying signatures or APIs
4. Fall back to read_file ONLY when MCP doesn't have what you need
```

Run: `pytest tests/ -v` to confirm all tests pass.

---

### Task 10: Document the integration

**Objective:** Add MCP configuration documentation.

**Files:**
- Create: `docs/mcp-integration.md` (or add to existing docs)

Content to cover:
- What MCP tools are available
- How to configure mcp-codebase-index in `.spine/config.yaml`
- How to add additional MCP servers
- Performance characteristics (sub-millisecond queries)
- Token savings vs reading files

---

## Edge Cases and Error Handling

| Scenario | Behavior |
|----------|----------|
| MCP server not installed | Graceful degradation — no MCP tools loaded, log warning |
| Server crashes mid-session | Reconnect on next tool call, log error |
| Tool call timeout | Raise to agent, model handles the error |
| No mcp_servers in config | No tools loaded, no error |
| Multiple servers configured | All tools available simultaneously |
| env var `PROJECT_ROOT` missing | Server starts but indexes CWD (server's default behavior) |

## Verification Checklist

- [ ] `mcp` package installs cleanly via uv
- [ ] Config parsing works with `mcp_servers` key
- [ ] MCPClient connects to a real mcp-codebase-index server
- [ ] Tools are discovered (18 expected)
- [ ] Tool naming follows `mcp_{server}_{tool}` convention
- [ ] `call_tool()` returns correct string results
- [ ] LangChain tool wrapper is callable with kwargs
- [ ] build_phase_agent injects MCP tools via extra_tools
- [ ] Agent system prompt includes MCP usage guidance
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Ruff linting passes: `ruff check spine/`
- [ ] No regression in existing tests
