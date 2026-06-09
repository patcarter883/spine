"""Unit tests for the index manifest and git-committable snapshot.

Covers the two guarantees added so the vector index can be committed to git
and trusted across machines:

* an embedding-model fingerprint stamped into the db and enforced at query
  time (a mismatched model must hard-fail, not silently return garbage), and
* a compressed snapshot that round-trips the index (export → restore) so a
  fresh clone restores instead of re-indexing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spine.persistence.vector_store import (
    VectorStore,
    restore_snapshot,
    snapshot_path_for,
)

_DIM = 8
_MODEL = "nomic-embed-text-v2-moe-GGUF"
_GOOD_CFG = {
    "model": _MODEL,
    "query_prefix": "search_query: ",
    "document_prefix": "search_document: ",
}


def _build_store(db_path: Path) -> VectorStore:
    """A tiny dim-8 index with two rows and a stamped manifest."""
    store = VectorStore(str(db_path))
    store.set_embedding_dim(_DIM)
    store.ensure_schema()
    store.insert("a.py", "foo", "function", "summary of foo", "def foo(): ...",
                 np.ones(_DIM, dtype=np.float32))
    store.insert("b.py", "bar", "function", "summary of bar", "def bar(): ...",
                 np.zeros(_DIM, dtype=np.float32))
    store.write_manifest({
        "embedding_model": _MODEL,
        "embedding_dim": _DIM,
        "query_prefix": "search_query: ",
        "document_prefix": "search_document: ",
        "index_version": "1",
    })
    return store


def test_manifest_round_trips(tmp_path: Path) -> None:
    store = _build_store(tmp_path / "spine.db")
    manifest = store.read_manifest()
    assert manifest["embedding_model"] == _MODEL
    assert manifest["embedding_dim"] == str(_DIM)
    assert manifest["query_prefix"] == "search_query: "


def test_compatible_model_accepted(tmp_path: Path) -> None:
    store = _build_store(tmp_path / "spine.db")
    # Should not raise.
    store.assert_embedding_compatible(_GOOD_CFG, query_dim=_DIM)


def test_wrong_model_rejected(tmp_path: Path) -> None:
    store = _build_store(tmp_path / "spine.db")
    with pytest.raises(ValueError, match="does not match"):
        store.assert_embedding_compatible(
            {**_GOOD_CFG, "model": "Qwen3-Embedding-8B"}, query_dim=_DIM
        )


def test_wrong_dimension_rejected(tmp_path: Path) -> None:
    store = _build_store(tmp_path / "spine.db")
    with pytest.raises(ValueError, match="does not match"):
        store.assert_embedding_compatible(_GOOD_CFG, query_dim=4096)


def test_legacy_db_warns_but_still_dim_checks(tmp_path: Path) -> None:
    store = _build_store(tmp_path / "legacy.db")
    # Simulate a pre-fingerprinting database.
    store._get_connection().execute("DROP TABLE index_manifest")
    # No manifest: matching dim is tolerated (warn only)...
    store.assert_embedding_compatible(_GOOD_CFG, query_dim=_DIM)
    # ...but a dimension mismatch still hard-fails.
    with pytest.raises(ValueError, match="does not match this index"):
        store.assert_embedding_compatible(_GOOD_CFG, query_dim=4096)


def test_snapshot_export_restore_preserves_rows_and_manifest(tmp_path: Path) -> None:
    db_path = tmp_path / "spine.db"
    store = _build_store(db_path)
    snap = snapshot_path_for(str(db_path))
    store.export_snapshot(snap)
    store.close()
    assert Path(snap).exists()

    # Restore is a no-op while the live db is present.
    assert restore_snapshot(str(db_path), snap) is False

    # Delete the live db; restore from the snapshot; data + manifest survive.
    db_path.unlink()
    assert restore_snapshot(str(db_path), snap) is True
    restored = VectorStore(str(db_path))
    restored.ensure_schema()
    assert restored.get_stats()["total_chunks"] == 2
    assert restored.read_manifest()["embedding_model"] == _MODEL
    restored.assert_embedding_compatible(_GOOD_CFG, query_dim=_DIM)


def test_snapshot_path_helper() -> None:
    assert snapshot_path_for(".spine/spine.db") == ".spine/spine.db.xz"
