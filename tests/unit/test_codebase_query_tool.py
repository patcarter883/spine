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

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from langchain_core.tools import ToolException

from spine.agents.tools.ast_extract import Edge
from spine.agents.tools.codebase_query import (
    _ACTION_TO_MCP,
    CodebaseQueryTool,
)
from spine.persistence.vector_store import VectorStore


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


# ── Local-index fallback (PHP / empty MCP results) ────────────────────────
#
# mcp-codebase-index has no PHP analyzer; these verify the three fallback
# triggers: PHP-only short-circuit, unloaded backend, empty MCP result.

_PHP_SRC = """<?php
class Invoice {
    public function total() {
        TaxCalc::apply($this);
        return format_money($this->sum);
    }
}
"""


@pytest.fixture
def php_index(tmp_path: Path) -> tuple[str, str]:
    """Workspace with a PHP file, an indexed PHP symbol, and its edges."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "invoice.php").write_text(_PHP_SRC)

    db_path = str(tmp_path / "spine.db")
    store = VectorStore(db_path)
    store.ensure_schema()
    store.insert(
        file_path="src/invoice.php",
        symbol_name="Invoice.total",
        symbol_type="method",
        enriched_summary="Computes the invoice total via TaxCalc",
        raw_code="public function total() { TaxCalc::apply($this); }",
        embedding=np.ones(VectorStore.EMBEDDING_DIM, dtype=np.float32),
        lang="php",
    )
    store.replace_file_edges("src/invoice.php", [
        Edge("src/invoice.php", "Invoice.total", "static_call", "TaxCalc.apply", "php"),
        Edge("src/invoice.php", "Invoice.total", "call", "format_money", "php"),
    ])
    store.close()
    return str(tmp_path), db_path


def _tool(php_index, tool_map):
    workspace, db_path = php_index
    cqt = CodebaseQueryTool(workspace_root=workspace, mcp_servers={}, db_path=db_path)
    cqt._tool_map = tool_map
    return cqt


def test_php_symbol_short_circuits_mcp(php_index):
    """A symbol the index knows only as PHP never reaches the MCP backend."""
    backing = _make_fake_mcp_tool("mcp_codebase-index_find_symbol")
    cqt = _tool(php_index, {"mcp_codebase-index_find_symbol": backing})
    out = json.loads(cqt._run(action="find_symbol", name="total"))
    assert out["source"] == "local_index"
    assert out["matches"][0]["symbol_name"] == "Invoice.total"
    backing.invoke.assert_not_called()


def test_unloaded_backend_serves_php_locally(php_index):
    """MCP unavailable + indexed PHP symbol → local result, no ToolException."""
    cqt = _tool(php_index, {})
    out = json.loads(cqt._run(action="get_source", name="Invoice.total"))
    assert out["source"] == "local_index"
    assert "total" in out["matches"][0]["raw_code"]


def test_unloaded_backend_still_raises_when_no_local_match(php_index):
    cqt = _tool(php_index, {})
    with pytest.raises(ToolException, match="backing tool"):
        cqt._run(action="find_symbol", name="not_indexed_anywhere")


def test_empty_mcp_result_falls_back(php_index):
    backing = _make_fake_mcp_tool("mcp_codebase-index_search_codebase")
    backing.invoke = MagicMock(return_value="No matches found.")
    cqt = _tool(php_index, {"mcp_codebase-index_search_codebase": backing})
    out = json.loads(cqt._run(action="search", pattern="TaxCalc"))
    assert out["source"] == "local_index"
    backing.invoke.assert_called_once()


def test_nonempty_mcp_result_is_returned_untouched(php_index):
    backing = _make_fake_mcp_tool("mcp_codebase-index_search_codebase", "real hit " * 30)
    cqt = _tool(php_index, {"mcp_codebase-index_search_codebase": backing})
    out = cqt._run(action="search", pattern="TaxCalc")
    assert out.startswith("mcp_codebase-index_search_codebase::")


def test_regex_metachar_pattern_searches_locally(php_index):
    """Regex metacharacters must not break the FTS fallback."""
    cqt = _tool(php_index, {})
    out = json.loads(cqt._run(action="search", pattern="TaxCalc::apply\\(.*\\)"))
    assert out["source"] == "local_index"
    assert any("TaxCalc" in (m["snippet"] or "") for m in out["matches"])


def test_php_dependencies_served_from_edges(php_index):
    cqt = _tool(php_index, {})
    out = json.loads(cqt._run(action="get_dependencies", name="Invoice.total"))
    assert out["source"] == "local_index"
    uses = {e["uses"] for e in out["dependencies"]}
    assert uses == {"TaxCalc.apply", "format_money"}


def test_php_dependents_served_from_edges(php_index):
    cqt = _tool(php_index, {})
    # format_money is called by Invoice.total but is not itself indexed in
    # symbol_metadata — index it so the php-lang gate sees it.
    workspace, db_path = php_index
    store = VectorStore(db_path)
    store.ensure_schema()
    store.insert(
        file_path="src/money.php",
        symbol_name="format_money",
        symbol_type="function",
        enriched_summary="formats money",
        raw_code="function format_money($x) {}",
        embedding=np.ones(VectorStore.EMBEDDING_DIM, dtype=np.float32),
        lang="php",
    )
    store.close()
    out = json.loads(cqt._run(action="get_dependents", name="format_money"))
    assert out["source"] == "local_index"
    assert out["dependents"][0]["symbol"] == "Invoice.total"


def test_php_dependencies_without_edge_rows_suggests_reindex(php_index):
    workspace, db_path = php_index
    store = VectorStore(db_path)
    store.replace_file_edges("src/invoice.php", [])  # simulate pre-edges index
    store.close()
    cqt = _tool(php_index, {})
    out = json.loads(cqt._run(action="get_dependencies", name="Invoice.total"))
    assert out["status"] == "unavailable"
    assert "Re-run indexing" in out["detail"]
