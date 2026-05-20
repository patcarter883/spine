"""Default MCP server configurations for SPINE.

These defaults are merged with user-provided config in ``.spine/config.yaml``
(user config values take priority).  The ``PROJECT_ROOT`` env var is set at
runtime from the SPINE ``workspace_root`` when the MCP client is initialised.
"""

DEFAULT_MCP_SERVERS: dict[str, dict] = {
    "codebase-index": {
        "command": "mcp-codebase-index",
        "args": [],
        "env": {},
        "timeout": 120,
        "connect_timeout": 60,
    }
}
