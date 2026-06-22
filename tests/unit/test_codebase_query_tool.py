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
from spine.agents.tools import codebase_query as cq_module
from spine.agents.tools.codebase_query import (
    _ACTION_TO_MCP,
    CodebaseQueryTool,
    list_files,
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


def test_small_search_result_passes_through_unbounded(tool_with_fake_backend):
    # PR-C: a short search result must reach the model unchanged (no marker).
    out = tool_with_fake_backend._run(action="search", pattern="foo|bar", max_results=10)
    assert "truncated" not in out


def test_oversized_search_result_truncated_with_expand_hint(tool_with_fake_backend):
    # PR-C: a large search result is bounded and points the model at
    # expand-on-demand (get_source) instead of dumping every match inline.
    big = "\n".join(
        f"match {i}: spine/mod_{i}.py:42  def func_{i}(): ..." for i in range(400)
    )
    backend = tool_with_fake_backend._tool_map["mcp_codebase-index_search_codebase"]
    backend.invoke = MagicMock(return_value=big)
    out = tool_with_fake_backend._run(action="search", pattern="func", max_results=50)
    assert "search results truncated" in out
    assert "get_source" in out
    assert len(out) < len(big)


def test_oversized_get_source_is_not_truncated(tool_with_fake_backend):
    # PR-C: get_source returns a body the model explicitly asked for — never bound.
    big = "def huge():\n" + "\n".join(f"    x{i} = {i}" for i in range(2000))
    backend = tool_with_fake_backend._tool_map["mcp_codebase-index_get_function_source"]
    backend.invoke = MagicMock(return_value=big)
    out = tool_with_fake_backend._run(action="get_source", name="huge")
    assert "truncated" not in out
    assert out == big


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


# ── list_files facade (single entry point for file discovery) ────────────


def _fake_list_files_backend(paths: list[str]) -> MagicMock:
    tool = MagicMock()
    tool.name = "mcp_codebase-index_list_files"
    tool.ainvoke = AsyncMock(return_value=json.dumps(paths))
    return tool


@pytest.fixture
def mixed_repo(tmp_path):
    """A repo with Python, PHP, a dot-folder, and a vendored dir."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1\n")
    (tmp_path / "src" / "Invoice.php").write_text("<?php class Invoice {}\n")
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "hidden.php").write_text("<?php\n")
    (tmp_path / "vendor").mkdir()
    (tmp_path / "vendor" / "dep.php").write_text("<?php\n")
    return tmp_path


_EXTS = frozenset({".py", ".php"})
_SKIPS = frozenset({"vendor"})


@pytest.mark.asyncio
async def test_list_files_supplements_php_via_walk(mixed_repo, monkeypatch):
    """Regression: MCP returns a non-empty py-only list; PHP must still appear."""
    import spine.mcp.client as mcp_client

    monkeypatch.setattr(
        mcp_client, "get_mcp_tools",
        lambda *a, **k: [_fake_list_files_backend(["src/app.py"])],
    )
    files = await list_files(
        str(mixed_repo), {"codebase-index": {}}, extensions=_EXTS, skip_dirs=_SKIPS
    )
    assert "src/app.py" in files
    assert "src/Invoice.php" in files


@pytest.mark.asyncio
async def test_list_files_walk_only_when_mcp_unconfigured(mixed_repo):
    files = await list_files(str(mixed_repo), {}, extensions=_EXTS, skip_dirs=_SKIPS)
    assert files == ["src/Invoice.php", "src/app.py"]


@pytest.mark.asyncio
async def test_list_files_walk_only_when_mcp_load_fails(mixed_repo, monkeypatch):
    import spine.mcp.client as mcp_client

    def _boom(*a, **k):
        raise RuntimeError("server down")

    monkeypatch.setattr(mcp_client, "get_mcp_tools", _boom)
    files = await list_files(
        str(mixed_repo), {"codebase-index": {}}, extensions=_EXTS, skip_dirs=_SKIPS
    )
    assert "src/app.py" in files and "src/Invoice.php" in files


@pytest.mark.asyncio
async def test_list_files_drops_dot_folder_and_skip_dir_paths(mixed_repo, monkeypatch):
    """MCP results inside dot-folders or skip_dirs never reach the caller."""
    import spine.mcp.client as mcp_client

    monkeypatch.setattr(
        mcp_client, "get_mcp_tools",
        lambda *a, **k: [_fake_list_files_backend([
            "src/app.py",
            ".venv/lib/pkg.py",
            ".spine/artifacts/old.py",
            "vendor/dep.php",
        ])],
    )
    files = await list_files(
        str(mixed_repo), {"codebase-index": {}}, extensions=_EXTS, skip_dirs=_SKIPS
    )
    assert not any(".venv" in f or ".spine" in f or f.startswith("vendor/") for f in files)
    assert not any(".cache" in f for f in files)  # walk side also prunes dot-dirs


@pytest.mark.asyncio
async def test_list_files_dedupes_mcp_walk_overlap(mixed_repo, monkeypatch):
    import spine.mcp.client as mcp_client

    monkeypatch.setattr(
        mcp_client, "get_mcp_tools",
        lambda *a, **k: [_fake_list_files_backend(["src/app.py", "src/Invoice.php"])],
    )
    files = await list_files(
        str(mixed_repo), {"codebase-index": {}}, extensions=_EXTS, skip_dirs=_SKIPS
    )
    assert files == sorted(set(files))
    assert files.count("src/app.py") == 1


# ── Facade lockdown guard ────────────────────────────────────────────────

def test_codebase_query_local_only_imported_by_facade():
    """codebase_query_local must be reachable only via codebase_query."""
    import subprocess

    root = Path(cq_module.__file__).resolve().parents[3]
    out = subprocess.run(
        ["grep", "-rl", "codebase_query_local", str(root / "spine"), "--include=*.py"],
        capture_output=True, text=True,
    ).stdout.splitlines()
    allowed = {
        str(root / "spine" / "agents" / "tools" / "codebase_query.py"),
        str(root / "spine" / "agents" / "tools" / "codebase_query_local.py"),
    }
    assert set(out) <= allowed, f"direct codebase_query_local usage: {set(out) - allowed}"


def test_no_direct_mcp_codebase_index_invocations():
    """Only the facade / mcp client / cache infra may reference mcp tool names."""
    import subprocess

    root = Path(cq_module.__file__).resolve().parents[3]
    out = subprocess.run(
        ["grep", "-rl", "mcp_codebase-index_", str(root / "spine"), "--include=*.py"],
        capture_output=True, text=True,
    ).stdout.splitlines()
    allowed_suffixes = (
        "spine/agents/tools/codebase_query.py",
        "spine/mcp/client.py",
        "spine/agents/symbol_cache.py",
        "spine/agents/context_editing.py",  # name-pattern infra (comments)
        "spine/agents/subagents.py",        # prompt text only
        "spine/work/onboarding/templates.py",  # template comment only
    )
    bad = [f for f in out if not f.endswith(allowed_suffixes)]
    assert not bad, f"direct mcp_codebase-index_ references: {bad}"


# ── resolve_backing_call / canonical_backing_call ───────────────────────


def test_resolve_backing_call_symbol_actions():
    from spine.agents.tools.codebase_query import resolve_backing_call

    name, args = resolve_backing_call("get_source", " render ", None)
    assert name == "mcp_codebase-index_get_function_source"
    assert args == {"name": "render"}


def test_resolve_backing_call_search_clamps_max_results():
    from spine.agents.tools.codebase_query import resolve_backing_call

    name, args = resolve_backing_call("search", None, "def foo", max_results=0)
    assert name == "mcp_codebase-index_search_codebase"
    assert args == {"pattern": "def foo", "max_results": 1}


def test_resolve_backing_call_search_ignores_nullish_name():
    """Regression (trace 019ed870): a small model emitted name="None" next to a
    valid pattern; the literal placeholder must not trip mutual exclusivity."""
    from spine.agents.tools.codebase_query import resolve_backing_call

    for placeholder in ("None", "null", "NONE", " none "):
        backing, args = resolve_backing_call("search", placeholder, "def foo")
        assert backing == "mcp_codebase-index_search_codebase"
        assert args == {"pattern": "def foo", "max_results": 20}


def test_resolve_backing_call_symbol_action_ignores_nullish_pattern():
    """The inverse: a name action with a placeholder pattern still resolves."""
    from spine.agents.tools.codebase_query import resolve_backing_call

    backing, args = resolve_backing_call("get_source", "render", "None")
    assert backing == "mcp_codebase-index_get_function_source"
    assert args == {"name": "render"}


def test_resolve_backing_call_real_name_search_still_rejected():
    """A genuine (non-placeholder) name on a search action is still an error."""
    from langchain_core.tools import ToolException

    from spine.agents.tools.codebase_query import resolve_backing_call

    with pytest.raises(ToolException, match="'name' must not be supplied"):
        resolve_backing_call("search", "render", "def foo")


def test_canonical_backing_call_coalesces_nullish_name_for_search():
    """Cache keying must collapse name=None and name='None' to one backing call."""
    from spine.agents.tools.codebase_query import canonical_backing_call

    with_placeholder = canonical_backing_call(
        {"action": "search", "pattern": "def foo", "name": "None"}
    )
    without = canonical_backing_call({"action": "search", "pattern": "def foo"})
    assert with_placeholder == without
    assert with_placeholder == (
        "mcp_codebase-index_search_codebase",
        {"pattern": "def foo", "max_results": 20},
    )


def test_resolve_backing_call_unknown_action_raises():
    from langchain_core.tools import ToolException

    from spine.agents.tools.codebase_query import resolve_backing_call

    with pytest.raises(ToolException, match="unknown action"):
        resolve_backing_call("read_file", "x", None)


def test_canonical_backing_call_drops_unknown_keys():
    from spine.agents.tools.codebase_query import canonical_backing_call

    assert canonical_backing_call(
        {"action": "get_source", "name": "render", "file_hint": "a/b.py"}
    ) == ("mcp_codebase-index_get_function_source", {"name": "render"})


def test_canonical_backing_call_returns_none_on_invalid():
    from spine.agents.tools.codebase_query import canonical_backing_call

    assert canonical_backing_call({"action": "search"}) is None
    assert canonical_backing_call({"action": "get_source"}) is None
    assert canonical_backing_call({}) is None
    assert canonical_backing_call({"action": "get_source", "name": "x", "pattern": "y"}) is None
