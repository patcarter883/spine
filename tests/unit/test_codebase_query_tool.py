"""Tests for the consolidated CodebaseQueryTool.

Verifies the guards added in response to trace
``019e6cc4-f57d-7652-a718-15d04278ad5c``:
  * action enum dispatch to the correct underlying MCP tool
  * whitespace-only ``name`` rejected with a clear message
  * tool-call markup in ``pattern`` rejected with a clear message
  * ``name`` and ``pattern`` mutually exclusive
  * unknown action surfaces a useful error
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import ToolException

from spine.agents.tools.codebase_query import (
    _ACTION_TO_MCP,
    CodebaseQueryTool,
)


# ── Test double for the MCP tool map ─────────────────────────────────────


def _make_fake_mcp_tool(name: str, sentinel: str = "ok") -> MagicMock:
    """Mimic a BaseTool dispatched via the public ``.invoke`` / ``.ainvoke``
    API (the surface CodebaseQueryTool actually calls — direct ``_run`` /
    ``_arun`` access misses the ``config`` kwarg StructuredTool requires).
    """
    tool = MagicMock()
    tool.name = name
    tool.invoke = MagicMock(return_value=f"{name}::{sentinel}")
    tool.ainvoke = AsyncMock(return_value=f"{name}::async-{sentinel}")
    return tool


@pytest.fixture
def tool_with_fake_backend(monkeypatch):
    """A CodebaseQueryTool wired to a fake MCP backend map.

    Bypasses the real MCP loader by pre-populating ``_tool_map`` so the
    tool's lazy loader is a no-op.
    """
    cqt = CodebaseQueryTool(workspace_root="/tmp/x", mcp_servers={})
    cqt._tool_map = {
        backend: _make_fake_mcp_tool(backend) for backend in _ACTION_TO_MCP.values()
    }
    return cqt


# ── Happy path: action dispatch ──────────────────────────────────────────


def test_find_symbol_dispatches_via_invoke(tool_with_fake_backend):
    """Dispatch goes via ``.invoke(args)`` — the public BaseTool surface —
    NOT direct ``_run(**args)``, because StructuredTool._run requires a
    ``config`` kwarg the langchain framework supplies for us via invoke."""
    out = tool_with_fake_backend._run(action="find_symbol", name="MyClass")
    assert out.startswith("mcp_codebase-index_find_symbol::")
    backing = tool_with_fake_backend._tool_map["mcp_codebase-index_find_symbol"]
    backing.invoke.assert_called_once_with({"name": "MyClass"})


def test_search_dispatches_via_invoke(tool_with_fake_backend):
    out = tool_with_fake_backend._run(action="search", pattern="foo|bar", max_results=10)
    assert out.startswith("mcp_codebase-index_search_codebase::")
    backing = tool_with_fake_backend._tool_map["mcp_codebase-index_search_codebase"]
    backing.invoke.assert_called_once_with({"pattern": "foo|bar", "max_results": 10})


@pytest.mark.asyncio
async def test_async_dispatch_via_ainvoke(tool_with_fake_backend):
    out = await tool_with_fake_backend._arun(action="find_symbol", name="MyClass")
    assert "::async-" in out
    backing = tool_with_fake_backend._tool_map["mcp_codebase-index_find_symbol"]
    backing.ainvoke.assert_awaited_once_with({"name": "MyClass"})


@pytest.mark.asyncio
async def test_structured_tool_config_kwarg_regression(tool_with_fake_backend):
    """Regression for trace 019e6d27: backing StructuredTool._arun requires
    a ``config`` kwarg. CodebaseQueryTool must NOT call ``_arun(**args)``
    directly. We assert ``.ainvoke(args)`` was used (which handles config
    plumbing internally)."""
    await tool_with_fake_backend._arun(
        action="search", pattern="foo", max_results=5,
    )
    backing = tool_with_fake_backend._tool_map["mcp_codebase-index_search_codebase"]
    backing.ainvoke.assert_awaited_once()
    backing._arun.assert_not_called()  # never called directly
    backing._run.assert_not_called()


# ── Guards: required-arg checks ──────────────────────────────────────────


def test_search_without_pattern_raises(tool_with_fake_backend):
    with pytest.raises(ToolException, match="'pattern' is required"):
        tool_with_fake_backend._run(action="search")


def test_find_symbol_without_name_raises(tool_with_fake_backend):
    with pytest.raises(ToolException, match="'name' is required"):
        tool_with_fake_backend._run(action="find_symbol")


# ── Guards: mutually exclusive name vs pattern ───────────────────────────


def test_search_rejects_name(tool_with_fake_backend):
    with pytest.raises(ToolException, match="'name' must not be supplied"):
        tool_with_fake_backend._run(action="search", name="X", pattern="P")


def test_find_symbol_rejects_pattern(tool_with_fake_backend):
    with pytest.raises(ToolException, match="'pattern' must not be supplied"):
        tool_with_fake_backend._run(action="find_symbol", name="X", pattern="P")


# ── Guards: whitespace / markup ──────────────────────────────────────────


def test_whitespace_name_rejected(tool_with_fake_backend):
    with pytest.raises(ToolException, match="whitespace-only"):
        tool_with_fake_backend._run(action="get_source", name="\n   ")


def test_markup_in_pattern_rejected(tool_with_fake_backend):
    with pytest.raises(ToolException, match="tool-call markup"):
        tool_with_fake_backend._run(
            action="search", pattern="spine.*</tool_call>\n",
        )


def test_markup_in_name_rejected(tool_with_fake_backend):
    with pytest.raises(ToolException, match="tool-call markup"):
        tool_with_fake_backend._run(action="find_symbol", name="X<arg_value>Y")


# ── Backing tool missing ──────────────────────────────────────────────────


def test_missing_backend_raises_clear_error():
    cqt = CodebaseQueryTool(workspace_root="/tmp/x", mcp_servers={})
    cqt._tool_map = {}  # nothing loaded
    with pytest.raises(ToolException, match="backing tool"):
        cqt._run(action="find_symbol", name="X")
