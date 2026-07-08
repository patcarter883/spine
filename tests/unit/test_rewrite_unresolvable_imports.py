"""Deterministic module-path repair for imports of real symbols.

The dominant editor authoring failure across the artifact_exists validation
runs (3 of 7 attempts) was a hallucinated module path for a symbol that
exists: `from spine.artifacts import ArtifactStore` (0eabad7d), `from
artifact_store import ArtifactStore` (7cb8dd73). The codebase index knows
which file exports the symbol, so the write exit rewrites the import to the
real module path; names the index doesn't know stay put and fail loudly in
the pytest evidence.
"""

from __future__ import annotations

import json

import pytest

from spine.agents.tools.read_edit_lint import ReadEditLintTool


@pytest.fixture()
def fake_index(monkeypatch, tmp_path):
    """Point the repair at a fake index mapping ArtifactStore to its file."""
    monkeypatch.setattr(
        "spine.agents.tools.read_edit_lint._index_db_path", lambda: "db"
    )

    def _find(db, name):
        if name == "ArtifactStore":
            return json.dumps(
                {"matches": [{"file_path": "spine/persistence/artifacts.py",
                              "symbol_name": "ArtifactStore"}]}
            )
        return None

    monkeypatch.setattr("spine.agents.tools.codebase_query.find_symbol", _find)
    # Make the hallucinated module genuinely unresolvable in the workspace.
    return ReadEditLintTool(workspace_root=str(tmp_path), target_files=["t.py"])


def test_hallucinated_module_path_is_rewritten(fake_index, tmp_path):
    code = (
        "from artifact_store import ArtifactStore\n"
        "\n"
        "\n"
        "def test_x():\n"
        "    assert ArtifactStore\n"
    )
    out = fake_index._run("t.py", full_replace=code)
    text = (tmp_path / "t.py").read_text()
    assert "from spine.persistence.artifacts import ArtifactStore" in text
    assert "from artifact_store import" not in text
    assert "imports_rewritten" in out
    compile(text, "t.py", "exec")


def test_unknown_names_stay_put(fake_index, tmp_path):
    code = (
        "from nowhere_real import TotallyUnknownThing\n"
        "\n"
        "\n"
        "def test_x():\n"
        "    assert TotallyUnknownThing\n"
    )
    out = fake_index._run("t.py", full_replace=code)
    text = (tmp_path / "t.py").read_text()
    # Nothing mappable — the import is left for pytest evidence to flag.
    assert "from nowhere_real import TotallyUnknownThing" in text
    assert "imports_rewritten" not in out


def test_resolvable_workspace_module_untouched(fake_index, tmp_path):
    (tmp_path / "helpers.py").write_text("VALUE = 1\n", encoding="utf-8")
    code = "from helpers import VALUE\n\n\ndef test_x():\n    assert VALUE\n"
    out = fake_index._run("t.py", full_replace=code)
    assert "imports_rewritten" not in out
    assert "from helpers import VALUE" in (tmp_path / "t.py").read_text()


def test_mixed_import_splits_mapped_and_unmapped(fake_index, tmp_path):
    code = (
        "from artifact_store import ArtifactStore, GhostHelper\n"
        "\n"
        "\n"
        "def test_x():\n"
        "    assert ArtifactStore and GhostHelper\n"
    )
    fake_index._run("t.py", full_replace=code)
    text = (tmp_path / "t.py").read_text()
    assert "from spine.persistence.artifacts import ArtifactStore" in text
    assert "from artifact_store import GhostHelper" in text
    compile(text, "t.py", "exec")
