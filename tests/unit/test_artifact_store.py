"""Tests for ArtifactStore — sidecar and orphan discovery fixes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import importlib.util

# Load artifacts.py directly, bypassing __init__.py (which imports
# CheckpointStore → langgraph.checkpoint.sqlite, not always installed)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location(
    "spine.persistence.artifacts",
    str(_PROJECT_ROOT / "spine" / "persistence" / "artifacts.py"),
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ArtifactStore = _mod.ArtifactStore


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    """Create an ArtifactStore rooted in a temp directory."""
    return ArtifactStore(base_path=str(tmp_path / "arts"))


class TestSaveArtifactSidecar:
    """Sidecar (.meta.json) must always be written, even when content is skipped."""

    def test_sidecar_written_on_new_artifact(self, store: ArtifactStore, tmp_path: Path) -> None:
        """Normal case: save_artifact writes both file and sidecar."""
        store.save_artifact("w1", "tasks", "tasks.md", "hello world content")

        base = tmp_path / "arts" / "w1" / "tasks"
        assert (base / "tasks.md").exists()
        assert (base / "tasks.md.meta.json").exists()
        meta = json.loads((base / "tasks.md.meta.json").read_text())
        assert meta["phase"] == "tasks"
        assert meta["name"] == "tasks.md"
        assert meta["size"] == 19  # len("hello world content")

    def test_sidecar_written_when_content_skipped(
        self, store: ArtifactStore, tmp_path: Path
    ) -> None:
        """BUG FIX: sidecar must be written even when overwrite-shorter guard fires.

        Scenario:
        1. Agent writes full content via write_file (5000 chars)
        2. Dispatcher calls save_artifact with truncated 500-char preview
        3. Guard fires (500 < 5000) — content skipped
        4. Sidecar MUST still be written so list_artifacts can find it
        """
        base = tmp_path / "arts" / "w1" / "tasks"
        base.mkdir(parents=True, exist_ok=True)

        # Simulate agent writing full content directly
        full_content = "x" * 5000
        (base / "tasks.md").write_text(full_content, encoding="utf-8")

        # Dispatcher saves truncated preview — guard should skip content, but
        # sidecar must still be written
        truncated = "x" * 500
        store.save_artifact("w1", "tasks", "tasks.md", truncated)

        # Content must NOT be overwritten
        assert (base / "tasks.md").read_text(encoding="utf-8") == full_content

        # Sidecar MUST exist
        assert (base / "tasks.md.meta.json").exists()
        meta = json.loads((base / "tasks.md.meta.json").read_text())
        assert meta["phase"] == "tasks"
        assert meta["name"] == "tasks.md"
        # Size should reflect actual on-disk content, not truncated version
        assert meta["size"] == 5000

    def test_sidecar_updated_when_content_overwritten(
        self, store: ArtifactStore, tmp_path: Path
    ) -> None:
        """When content IS written, sidecar size reflects the new content."""
        store.save_artifact("w1", "tasks", "tasks.md", "short")
        store.save_artifact(
            "w1", "tasks", "tasks.md", "much longer content here", overwrite_shorter=True
        )

        meta = json.loads((tmp_path / "arts" / "w1" / "tasks" / "tasks.md.meta.json").read_text())
        assert meta["size"] == 24  # len("much longer content here")


class TestListArtifactsOrphanDiscovery:
    """list_artifacts must discover orphan files with no sidecar."""

    def test_discovers_via_sidecar(self, store: ArtifactStore, tmp_path: Path) -> None:
        """Normal case: sidecar-based discovery."""
        store.save_artifact("w1", "tasks", "tasks.md", "hello")
        artifacts = store.list_artifacts("w1")
        assert len(artifacts) == 1
        assert artifacts[0]["name"] == "tasks.md"

    def test_discovers_orphan_files(self, store: ArtifactStore, tmp_path: Path) -> None:
        """Orphan files (no sidecar) must be discovered and sidecars created.

        This is the scenario when the agent writes files via write_file but
        the dispatcher hasn't called save_artifact yet (or the sidecar was
        never written due to the overwrite-shorter bug).
        """
        base = tmp_path / "arts" / "w1" / "tasks"
        base.mkdir(parents=True, exist_ok=True)
        (base / "tasks.md").write_text("task content here", encoding="utf-8")
        (base / "slice-auth.md").write_text("auth slice details", encoding="utf-8")

        # No sidecars exist yet
        assert not list(base.glob("*.meta.json"))

        artifacts = store.list_artifacts("w1")
        assert len(artifacts) == 2
        names = {a["name"] for a in artifacts}
        assert "tasks.md" in names
        assert "slice-auth.md" in names

        # Sidecars should now exist
        assert (base / "tasks.md.meta.json").exists()
        assert (base / "slice-auth.md.meta.json").exists()

    def test_mixed_sidecar_and_orphan(self, store: ArtifactStore, tmp_path: Path) -> None:
        """Some files have sidecars, some don't — all must be discovered."""
        base = tmp_path / "arts" / "w1" / "tasks"
        base.mkdir(parents=True, exist_ok=True)

        # One file via save_artifact (has sidecar)
        store.save_artifact("w1", "tasks", "tasks.md", "task content")

        # One file written directly by agent (no sidecar)
        (base / "slice-auth.md").write_text("auth slice", encoding="utf-8")

        artifacts = store.list_artifacts("w1")
        assert len(artifacts) == 2
        names = {a["name"] for a in artifacts}
        assert names == {"tasks.md", "slice-auth.md"}

    def test_skips_meta_json_files(self, store: ArtifactStore, tmp_path: Path) -> None:
        """Meta.json sidecar files themselves should not be treated as artifacts."""
        store.save_artifact("w1", "tasks", "tasks.md", "hello")
        artifacts = store.list_artifacts("w1")
        names = [a["name"] for a in artifacts]
        # No .meta.json entries should appear as artifact names
        assert not any(n.endswith(".meta.json") for n in names)

    def test_empty_work_dir(self, store: ArtifactStore, tmp_path: Path) -> None:
        """Non-existent work dir returns empty list."""
        assert store.list_artifacts("nonexistent") == []

    def test_orphan_sidecar_uses_file_mtime(self, store: ArtifactStore, tmp_path: Path) -> None:
        """Orphan sidecar created_time should come from the file's mtime."""
        base = tmp_path / "arts" / "w1" / "tasks"
        base.mkdir(parents=True, exist_ok=True)
        (base / "tasks.md").write_text("content", encoding="utf-8")

        artifacts = store.list_artifacts("w1")
        assert len(artifacts) == 1
        # Modified field should be an ISO timestamp (from file mtime)
        assert "T" in artifacts[0]["modified"]


class TestDeleteArtifact:
    """delete_artifact must remove both file and sidecar."""

    def test_deletes_both(self, store: ArtifactStore, tmp_path: Path) -> None:
        store.save_artifact("w1", "tasks", "tasks.md", "hello")
        base = tmp_path / "arts" / "w1" / "tasks"
        assert (base / "tasks.md").exists()
        assert (base / "tasks.md.meta.json").exists()

        assert store.delete_artifact("w1", "tasks", "tasks.md") is True
        assert not (base / "tasks.md").exists()
        assert not (base / "tasks.md.meta.json").exists()

    def test_deletes_orphan_file(self, store: ArtifactStore, tmp_path: Path) -> None:
        """Deleting an orphan (no sidecar) still succeeds."""
        base = tmp_path / "arts" / "w1" / "tasks"
        base.mkdir(parents=True, exist_ok=True)
        (base / "tasks.md").write_text("orphan", encoding="utf-8")

        assert store.delete_artifact("w1", "tasks", "tasks.md") is True
        assert not (base / "tasks.md").exists()

    def test_nonexistent_returns_false(self, store: ArtifactStore) -> None:
        assert store.delete_artifact("w1", "tasks", "nope.md") is False
