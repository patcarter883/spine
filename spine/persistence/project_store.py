"""SPINE project store — file-based persistence for the project/milestone layer.

A project is a persistent envelope grouping many top-level work items. Each
project is stored under ``.spine/project/{project_id}/spec.json`` with a
``spec.json.meta.json`` sidecar, mirroring :class:`ArtifactStore`'s conventions
and onboarding's stable (work_id-independent) doc layout.

Writes are atomic (write to a temp file, then ``os.replace``) so a crash mid-write
cannot leave a half-written ``spec.json``. The store is the source of truth for
project membership; the denormalized ``project_id`` column on ``work_entries`` is
only a reverse-lookup convenience.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from spine.models.types import ProjectSpec


class ProjectStore:
    """Manages :class:`ProjectSpec` documents on the filesystem."""

    def __init__(self, base_path: str = ".spine/project") -> None:
        self._base = Path(base_path)

    def _project_dir(self, project_id: str) -> Path:
        return self._base / project_id

    def save_project(self, spec: ProjectSpec) -> Path:
        """Persist a project spec atomically and write its metadata sidecar.

        Returns the path to the written ``spec.json``.
        """
        project_dir = self._project_dir(spec.id)
        project_dir.mkdir(parents=True, exist_ok=True)
        spec_path = project_dir / "spec.json"

        content = spec.model_dump_json(indent=2)
        # Atomic write: a half-written structured doc is worse than none, so
        # never write spec.json in place.
        tmp_path = spec_path.with_suffix(".json.tmp")
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, spec_path)

        meta = {
            "project_id": spec.id,
            "member_count": len(spec.member_work_ids),
            "size": len(content),
            "modified": datetime.now().isoformat(),
        }
        (project_dir / "spec.json.meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        return spec_path

    def load_project(self, project_id: str) -> ProjectSpec | None:
        """Load a project spec, or None if it does not exist / is unreadable."""
        spec_path = self._project_dir(project_id) / "spec.json"
        if not spec_path.exists():
            return None
        try:
            return ProjectSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def list_projects(self) -> list[str]:
        """Return the IDs of all stored projects, sorted."""
        if not self._base.exists():
            return []
        ids = [
            spec_file.parent.name
            for spec_file in self._base.glob("*/spec.json")
        ]
        return sorted(ids)

    def add_members(self, project_id: str, work_ids: list[str]) -> ProjectSpec:
        """Idempotently add work_ids to a project's membership (set union).

        Preserves existing order and appends new members; bumps ``updated_at``.

        Raises:
            KeyError: if the project does not exist.
        """
        spec = self.load_project(project_id)
        if spec is None:
            raise KeyError(f"Project '{project_id}' not found")

        existing = set(spec.member_work_ids)
        added = [w for w in work_ids if w and w not in existing]
        if added:
            spec.member_work_ids = spec.member_work_ids + added
            spec.updated_at = datetime.now().isoformat()
            self.save_project(spec)
        return spec
