"""codebase-memory backend (Phase 1): arg mapping + response adaptation.

Fixtures are REAL response shapes captured from codebase-memory-mcp v0.9.0
during the 2026-07-17 evaluation spike (docs/codebase-memory-mcp-migration-
plan.md, Phase 0 results) — the adapter is the isolation layer for upstream
schema drift, so its tests must pin actual wire shapes, not invented ones.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.tools import BaseTool

from spine.agents.tools import codebase_memory_backend as cbm
from spine.agents.tools.codebase_query import CodebaseQueryTool

P = "home-pat-projects-spine"

# ── real v0.9.0 payloads (trimmed) ───────────────────────────────────────────
SEARCH_GRAPH_HIT = {
    "total": 1,
    "results": [{
        "name": "compute_streaks",
        "qualified_name": f"{P}.spine.workflow.critic_convergence.compute_streaks",
        "label": "Function",
        "file_path": "spine/workflow/critic_convergence.py",
        "in_degree": 10, "out_degree": 7, "is_exported": True,
    }],
}
SNIPPET = {
    "name": "compute_streaks",
    "qualified_name": f"{P}.spine.workflow.critic_convergence.compute_streaks",
    "label": "Function",
    "file_path": "/home/pat/projects/spine/spine/workflow/critic_convergence.py",
    "start_line": 241, "end_line": 354,
    "source": "def compute_streaks(\n    prior_review, current_review):\n    ...",
}
QUERY_ROWS = {
    "columns": ["a.qualified_name", "a.file_path", "type(r)"],
    "rows": [
        [f"{P}.spine.workflow.compose.__file__", "spine/workflow/compose.py", "CALLS"],
        [f"{P}.tests.unit.test_critic_convergence.TestX.test_y",
         "tests/unit/test_critic_convergence.py", "CALLS"],
    ],
    "total": 2,
}
QUERY_EMPTY = {"columns": [], "rows": [], "total": 0,
               "hint": "Query returned no results. Use get_graph_schema()..."}
SEARCH_CODE_HIT = {
    "results": [{
        "node": "compute_streaks",
        "qualified_name": f"{P}.spine.workflow.critic_convergence.compute_streaks",
        "label": "Function", "file": "spine/workflow/critic_convergence.py",
        "start_line": 241, "end_line": 354, "match_lines": [241],
    }],
    "raw_matches": [], "total_grep_matches": 1, "total_results": 1,
}


# ── project naming + arg mapping ─────────────────────────────────────────────
def test_project_name_rule():
    assert cbm.project_name_for("/home/pat/projects/spine") == P
    # Dots survive (live sandbox: /home/pat/projects/.agripath-spine-sandbox-x
    # indexed as home-pat-projects-.agripath-spine-sandbox-x).
    assert (
        cbm.project_name_for("/home/pat/projects/.agripath-spine-sandbox-d7fb94ab")
        == "home-pat-projects-.agripath-spine-sandbox-d7fb94ab"
    )
    # Consecutive dashes collapse (observed: /tmp/claude-1000/-home-... ->
    # tmp-claude-1000-home-...).
    assert cbm.project_name_for("/tmp/claude-1000/-home-x") == "tmp-claude-1000-home-x"


def test_backing_args_find_symbol_bare_vs_qualified():
    bare = cbm.backing_args_for("find_symbol", P, "compute_streaks", None, 20)
    assert bare == {"project": P, "name_pattern": "compute_streaks", "limit": 20}
    q = cbm.backing_args_for("find_symbol", P, "UIApi.get_providers", None, 20)
    assert q["qn_pattern"] == "*UIApi.get_providers"


def test_backing_args_dependents_cypher_uses_leaf_and_escapes():
    args = cbm.backing_args_for("get_dependents", P, "UIApi.o'brien", None, 20)
    assert "b.name = 'o\\'brien'" in args["query"]
    assert "CALLS|USAGE|IMPORTS|TESTS" in args["query"]
    deps = cbm.backing_args_for("get_dependencies", P, "compute_streaks", None, 20)
    assert "a.name = 'compute_streaks'" in deps["query"]
    assert "TESTS" not in deps["query"]


def test_backing_args_search_is_regex_bounded():
    args = cbm.backing_args_for("search", P, None, "def compute_\\w+", 7)
    assert args == {"project": P, "pattern": "def compute_\\w+",
                    "regex": True, "limit": 7}


# ── response adaptation (real shapes) ────────────────────────────────────────
def test_adapt_find_symbol_lists_and_strips_project():
    out = cbm.adapt_response("find_symbol", P, json.dumps(SEARCH_GRAPH_HIT))
    assert "spine.workflow.critic_convergence.compute_streaks" in out
    assert P + "." not in out  # project prefix stripped
    assert "[Function]" in out and "critic_convergence.py" in out


def test_adapt_find_symbol_empty_says_not_found():
    out = cbm.adapt_response("find_symbol", P, {"total": 0, "results": []})
    assert out == "not found"  # _looks_empty phrasing → local fallback fires


def test_adapt_get_source_prefixes_location():
    out = cbm.adapt_response("get_source", P, SNIPPET)
    assert out.startswith("# /home/pat/projects/spine/spine/workflow/critic_convergence.py:241")
    assert "def compute_streaks(" in out


def test_adapt_dependents_lines_and_empty():
    out = cbm.adapt_response("get_dependents", P, QUERY_ROWS)
    assert "spine.workflow.compose.__file__  (CALLS)" in out
    assert cbm.adapt_response("get_dependents", P, QUERY_EMPTY) == "no results"


def test_adapt_search_compact_pointers():
    out = cbm.adapt_response("search", P, SEARCH_CODE_HIT)
    assert "spine/workflow/critic_convergence.py:241" in out
    assert "match lines: [241]" in out
    assert cbm.adapt_response("search", P, {"results": [], "raw_matches": []}) == "no matches"


def test_adapt_handles_mcp_text_envelope():
    envelope = [{"type": "text", "text": json.dumps(SEARCH_GRAPH_HIT)}]
    out = cbm.adapt_response("find_symbol", P, envelope)
    assert "compute_streaks" in out


def test_resolve_qualified_name_exact_and_ambiguous():
    assert cbm.resolve_qualified_name(P, "compute_streaks", SEARCH_GRAPH_HIT) \
        == f"{P}.spine.workflow.critic_convergence.compute_streaks"
    two = {"results": [
        {"name": "run", "qualified_name": f"{P}.a.run"},
        {"name": "run", "qualified_name": f"{P}.b.run"},
    ]}
    assert cbm.resolve_qualified_name(P, "run", two) is None
    assert cbm.resolve_qualified_name(P, "b.run", two) == f"{P}.b.run"


# ── facade dispatch through a fake tool map ──────────────────────────────────
class _FakeBackingTool(BaseTool):
    name: str = "fake"
    description: str = "fake"
    response: str = ""
    calls: list = []

    def _run(self, **kwargs):  # noqa: D102
        self.calls.append(kwargs)
        return self.response


def _tool_with_backend(monkeypatch, responses: dict[str, str]) -> CodebaseQueryTool:
    tool = CodebaseQueryTool(
        workspace_root="/home/pat/projects/spine",
        mcp_servers={},
        backend="codebase-memory",
    )
    fake_map = {}
    for tool_name, resp in responses.items():
        f = _FakeBackingTool(response=resp)
        f.calls = []
        fake_map[tool_name] = f
    monkeypatch.setattr(tool, "_ensure_loaded", lambda: fake_map)
    tool._cbm_indexed = True  # skip the index hook in unit tests
    return tool


def test_facade_find_symbol_via_cbm(monkeypatch):
    t = _tool_with_backend(
        monkeypatch, {cbm.ACTION_TO_CBM["find_symbol"]: json.dumps(SEARCH_GRAPH_HIT)}
    )
    out = t._run(action="find_symbol", name="compute_streaks")
    assert "critic_convergence.py" in out
    call = t._ensure_loaded()[cbm.ACTION_TO_CBM["find_symbol"]].calls[0]
    assert call["project"] == P and call["name_pattern"] == "compute_streaks"


def test_facade_uses_server_project_name_from_index_response(monkeypatch):
    # The index hook's response names the project authoritatively; queries
    # must use IT, not the path-derived fallback (naming drift = every
    # query misses with "project not found", live 2026-07-17).
    t = CodebaseQueryTool(
        workspace_root="/home/pat/projects/.dotted-sandbox",
        mcp_servers={},
        backend="codebase-memory",
    )
    index_tool = _FakeBackingTool(
        response=json.dumps({"project": "server-authoritative-name", "status": "indexed"})
    )
    index_tool.calls = []
    finder = _FakeBackingTool(response=json.dumps(SEARCH_GRAPH_HIT))
    finder.calls = []
    fake_map = {cbm.INDEX_TOOL: index_tool, cbm.ACTION_TO_CBM["find_symbol"]: finder}
    monkeypatch.setattr(t, "_ensure_loaded", lambda: fake_map)

    t._run(action="find_symbol", name="compute_streaks")

    assert index_tool.calls == [{"repo_path": "/home/pat/projects/.dotted-sandbox"}]
    assert finder.calls[0]["project"] == "server-authoritative-name"


def test_facade_get_source_two_step(monkeypatch):
    t = _tool_with_backend(monkeypatch, {
        cbm.RESOLVE_TOOL: json.dumps(SEARCH_GRAPH_HIT),
        cbm.ACTION_TO_CBM["get_source"]: json.dumps(SNIPPET),
    })
    out = t._run(action="get_source", name="compute_streaks")
    assert "def compute_streaks(" in out
    snip_call = t._ensure_loaded()[cbm.ACTION_TO_CBM["get_source"]].calls[0]
    assert snip_call["qualified_name"].endswith(".compute_streaks")


def test_facade_validation_guards_still_apply(monkeypatch):
    import pytest
    from langchain_core.tools import ToolException

    t = _tool_with_backend(monkeypatch, {})
    with pytest.raises(ToolException):
        t._run(action="search", name="notallowed")  # name forbidden for search
    with pytest.raises(ToolException):
        t._run(action="nonsense", name="x")


def test_facade_search_cap_applies_to_cbm(monkeypatch):
    big = {"results": [], "raw_matches": [
        {"file": f"f{i}.py", "line": i, "text": "x" * 80} for i in range(400)
    ]}
    t = _tool_with_backend(
        monkeypatch, {cbm.ACTION_TO_CBM["search"]: json.dumps(big)}
    )
    t.search_result_char_cap = 500
    out = t._run(action="search", pattern="x+")
    assert len(out) < 900
    assert "truncated" in out


# ── .cbmignore hang guard (the 12-minute-index root cause: a single 6.2MB
# pickle spun the v0.9.0 indexer indefinitely) ──


def test_ensure_cbmignore_creates_with_guards(tmp_path):
    assert cbm.ensure_cbmignore(str(tmp_path)) is True
    text = (tmp_path / ".cbmignore").read_text()
    for g in cbm.CBMIGNORE_GUARDS:
        assert g in text.splitlines()


def test_ensure_cbmignore_never_writes_in_git_worktree(tmp_path):
    # Live 2026-07-17: the guard write dirtied the clone -> spine's own
    # clean-tree preflight rejected the run; in a sandbox it would leak
    # into patch diffs. Git worktrees warn instead of writing.
    (tmp_path / ".git").mkdir()
    assert cbm.ensure_cbmignore(str(tmp_path)) is False
    assert not (tmp_path / ".cbmignore").exists()
    # With guards already committed, it reports True and still writes nothing.
    (tmp_path / ".cbmignore").write_text(
        "".join(f"{g}\n" for g in cbm.CBMIGNORE_GUARDS)
    )
    before = (tmp_path / ".cbmignore").read_text()
    assert cbm.ensure_cbmignore(str(tmp_path)) is True
    assert (tmp_path / ".cbmignore").read_text() == before


def test_indexing_hazards_lists_visible_guard_matches(tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    big = tmp_path / "legacy-cache.pkl"
    big.write_bytes(b"x" * (600 * 1024))
    small = tmp_path / "tiny.pkl"
    small.write_bytes(b"x")
    (tmp_path / "code.py").write_text("x = 1\n")
    hazards = cbm.indexing_hazards(str(tmp_path))
    assert hazards == ["legacy-cache.pkl"]  # big + visible; small filtered
    # Gitignored blobs are invisible to the indexer -> not hazards.
    (tmp_path / ".gitignore").write_text("legacy-cache.pkl\n")
    assert "legacy-cache.pkl" not in cbm.indexing_hazards(str(tmp_path))


def test_ensure_cbmignore_appends_only_missing_and_is_idempotent(tmp_path):
    (tmp_path / ".cbmignore").write_text("vendor/\n*.pkl\n")
    cbm.ensure_cbmignore(str(tmp_path))
    text = (tmp_path / ".cbmignore").read_text()
    assert text.startswith("vendor/\n*.pkl\n")  # existing content untouched
    assert text.splitlines().count("*.pkl") == 1  # not duplicated
    assert ".spine/" in text.splitlines()
    cbm.ensure_cbmignore(str(tmp_path))
    assert (tmp_path / ".cbmignore").read_text() == text  # idempotent


def test_resolve_prefers_code_over_doc_headings():
    """Runs 019f6e2d/019f81c1: markdown headings share the graph with code
    symbols, so 'Handler' resolved to {class Handler, doc heading} and
    bounced 'did not resolve to exactly one symbol'. Code rows win."""
    rows = {"results": [
        {"name": "Handler", "qualified_name": f"{P}.docs.API-GUIDE",
         "file_path": "docs/API-GUIDE.md"},
        {"name": "8.2 Exception Handler", "qualified_name": f"{P}.docs.ARCH",
         "file_path": "docs/ARCHITECTURE.md"},
        {"name": "Handler", "qualified_name": f"{P}.app.Exceptions.Handler",
         "file_path": "app/Exceptions/Handler.php"},
    ]}
    assert cbm.resolve_qualified_name(P, "Handler", rows) \
        == f"{P}.app.Exceptions.Handler"


def test_resolve_doc_only_pool_still_resolves():
    # An agent may genuinely ask for a doc section — docs compete only
    # against each other.
    rows = {"results": [
        {"name": "Deployment", "qualified_name": f"{P}.docs.DEPLOY",
         "file_path": "docs/DEPLOY.md"},
    ]}
    assert cbm.resolve_qualified_name(P, "Deployment", rows) == f"{P}.docs.DEPLOY"


def test_resolve_code_ambiguity_still_none():
    # FarmScope.apply vs RelationshipFarmScope.apply — genuine code-side
    # ambiguity must still report candidates, not guess.
    rows = {"results": [
        {"name": "apply", "qualified_name": f"{P}.a.FarmScope.apply",
         "file_path": "app/Infrastructure/Scopes/FarmScope.php"},
        {"name": "apply", "qualified_name": f"{P}.b.RelationshipFarmScope.apply",
         "file_path": "app/Infrastructure/Scopes/RelationshipFarmScope.php"},
        {"name": "Apply Global Scopes", "qualified_name": f"{P}.docs.SCOPES",
         "file_path": "docs/SCOPES.md"},
    ]}
    assert cbm.resolve_qualified_name(P, "apply", rows) is None
    # But the dotted form disambiguates as before.
    assert cbm.resolve_qualified_name(P, "FarmScope.apply", rows) \
        == f"{P}.a.FarmScope.apply"


def test_is_doc_row_shapes():
    assert cbm.is_doc_row({"name": "1. Farm Managers", "file_path": "x.php"}) is True
    assert cbm.is_doc_row({"name": "Intro", "file_path": "README.md"}) is True
    assert cbm.is_doc_row({"name": "generate_erd", "file_path": "docs/generate_erd.py"}) is False
    assert cbm.is_doc_row({"name": "RouteServiceProvider.boot",
                           "file_path": "app/Providers/RouteServiceProvider.php"}) is False


def test_adapt_find_symbol_lists_code_first():
    payload = {"results": [
        {"name": "8.2 Exception Handler", "qualified_name": f"{P}.docs.ARCH",
         "label": "doc", "file_path": "docs/ARCHITECTURE.md"},
        {"name": "Handler", "qualified_name": f"{P}.app.Exceptions.Handler",
         "label": "class", "file_path": "app/Exceptions/Handler.php", "start_line": 12},
    ]}
    out = cbm.adapt_response("find_symbol", P, payload)
    first, second = out.splitlines()[:2]
    assert "Handler.php" in first
    assert "ARCHITECTURE.md" in second
