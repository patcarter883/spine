"""Unit tests for :mod:`spine.agents.tools.ast_extract_symbol`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spine.agents.tools.ast_extract_symbol import AstExtractSymbolTool
from spine.persistence.vector_store import VectorStore


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def workspace_with_index(tmp_path: Path) -> tuple[Path, str]:
    """Create a workspace with a small Python file and a vector index row."""
    pkg = tmp_path / "src"
    pkg.mkdir()
    sample = pkg / "greeter.py"
    sample.write_text(
        "def greet(name):\n"
        "    return 'hello ' + name\n"
        "\n"
        "class Greeter:\n"
        "    def say(self, name):\n"
        "        return 'hi ' + name\n"
    )

    db_path = tmp_path / "vector.db"
    store = VectorStore(str(db_path))
    store.ensure_schema()
    store.insert(
        file_path="src/greeter.py",
        symbol_name="greet",
        symbol_type="function",
        enriched_summary="greets by name",
        raw_code="def greet(name):\n    return 'hello ' + name\n",
        embedding=np.ones(VectorStore.EMBEDDING_DIM, dtype=np.float32),
        lang="python",
    )
    store.close()
    return tmp_path, str(db_path)


def test_vector_index_hit_returns_symbol(workspace_with_index: tuple[Path, str]) -> None:
    workspace, db_path = workspace_with_index
    tool = AstExtractSymbolTool(workspace_root=str(workspace), db_path=db_path)
    result = json.loads(tool._run(symbol_name="greet"))

    assert result["status"] == "ok"
    assert result["source"] == "index"
    assert len(result["matches"]) == 1
    match = result["matches"][0]
    assert match["symbol_name"] == "greet"
    assert match["file_path"] == "src/greeter.py"
    assert "return 'hello ' + name" in match["raw_code"]


def test_filesystem_fallback_when_symbol_not_in_index(
    workspace_with_index: tuple[Path, str],
) -> None:
    workspace, db_path = workspace_with_index
    tool = AstExtractSymbolTool(workspace_root=str(workspace), db_path=db_path)
    # Greeter (class) and say (method) are present in the file but NOT indexed.
    result = json.loads(tool._run(symbol_name="Greeter"))

    assert result["status"] == "ok"
    assert result["source"] == "fallback_walk"
    assert any(m["symbol_name"] == "Greeter" for m in result["matches"])


def test_not_found_returns_clear_status(workspace_with_index: tuple[Path, str]) -> None:
    workspace, db_path = workspace_with_index
    tool = AstExtractSymbolTool(workspace_root=str(workspace), db_path=db_path)
    result = json.loads(tool._run(symbol_name="not_a_real_symbol_xyz"))
    assert result["status"] == "not_found"


def test_missing_index_falls_back_to_walk(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "x.py").write_text("def find_me():\n    pass\n")
    tool = AstExtractSymbolTool(
        workspace_root=str(tmp_path), db_path=str(tmp_path / "missing.db")
    )
    result = json.loads(tool._run(symbol_name="find_me"))
    assert result["status"] == "ok"
    assert result["source"] == "fallback_walk"
