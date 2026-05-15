"""SPINE artifact store — file-based artifact persistence."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ArtifactStore:
    """Manages workflow artifacts on the filesystem.

    Artifacts are stored under ``.spine/artifacts/{work_id}/{phase}/{name}``.
    Each artifact file is accompanied by a ``.meta.json`` sidecar with
    metadata (phase, timestamp, size).
    """

    def __init__(self, base_path: str = ".spine/artifacts") -> None:
        self._base = Path(base_path)

    def save_artifact(
        self,
        work_id: str,
        phase: str,
        name: str,
        content: str,
        *,
        overwrite_shorter: bool = False,
    ) -> Path:
        """Save an artifact to disk.

        Args:
            work_id: The work item ID.
            phase: The phase that produced the artifact.
            name: The artifact filename.
            content: The artifact content.
            overwrite_shorter: When False (default), skip writing if the
                file already exists on disk and is longer than the new
                content.  This prevents truncated state previews (capped
                at ``_MAX_ARTIFACT_STATE_CHARS``) from overwriting the
                full files the agent originally wrote via ``write_file``.

        Returns:
            The path where the artifact was saved.
        """
        artifact_dir = self._base / work_id / phase
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / name

        # Guard: never overwrite a full on-disk artifact with a shorter
        # (truncated) version from workflow state.  The agent writes full
        # files via write_file; state only stores 500-char previews.
        if (
            not overwrite_shorter
            and artifact_path.exists()
            and len(content) < len(artifact_path.read_text(encoding="utf-8"))
        ):
            return artifact_path

        artifact_path.write_text(content, encoding="utf-8")

        # Write metadata sidecar
        meta = {
            "work_id": work_id,
            "phase": phase,
            "name": name,
            "size": len(content),
            "modified": datetime.now().isoformat(),
        }
        meta_path = artifact_dir / f"{name}.meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return artifact_path

    def load_artifact(self, work_id: str, phase: str, name: str) -> str | None:
        """Load an artifact from disk.

        Args:
            work_id: The work item ID.
            phase: The phase that produced the artifact.
            name: The artifact filename.

        Returns:
            The artifact content, or None if not found.
        """
        artifact_path = self._base / work_id / phase / name
        if artifact_path.exists():
            return artifact_path.read_text(encoding="utf-8")
        return None

    def list_artifacts(self, work_id: str) -> list[dict[str, Any]]:
        """List all artifacts for a work item.

        Args:
            work_id: The work item ID.

        Returns:
            A list of dicts with keys ``path``, ``phase``, ``name``, ``size``, ``modified``.
        """
        work_dir = self._base / work_id
        if not work_dir.exists():
            return []

        artifacts: list[dict[str, Any]] = []
        for meta_file in sorted(work_dir.rglob("*.meta.json")):
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                artifacts.append(meta)
            except (json.JSONDecodeError, OSError):
                continue

        return artifacts

    def delete_artifact(self, work_id: str, phase: str, name: str) -> bool:
        """Delete an artifact and its metadata.

        Args:
            work_id: The work item ID.
            phase: The phase that produced the artifact.
            name: The artifact filename.

        Returns:
            True if the artifact was deleted, False if not found.
        """
        artifact_path = self._base / work_id / phase / name
        meta_path = self._base / work_id / phase / f"{name}.meta.json"
        deleted = False

        if artifact_path.exists():
            artifact_path.unlink()
            deleted = True
        if meta_path.exists():
            meta_path.unlink()

        return deleted
