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
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from spine.models.types import ProjectSpec

logger = logging.getLogger(__name__)

try:
    import fcntl  # POSIX advisory file locking
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]


class ProjectStore:
    """Manages :class:`ProjectSpec` documents on the filesystem."""

    def __init__(self, base_path: str = ".spine/project") -> None:
        self._base = Path(base_path)

    def _project_dir(self, project_id: str) -> Path:
        return self._base / project_id

    @contextmanager
    def _project_lock(self, project_id: str) -> Iterator[None]:
        """Hold an exclusive inter-process lock for a single project.

        Membership mutations (:meth:`add_members` / :meth:`remove_members`) are
        read-modify-write cycles on ``spec.json``. Without a lock, two
        concurrent submissions — parallel ``spine run --project`` calls, or a
        CLI submission racing a UI edit, each in its own process — can both
        read the same spec and the later write clobbers the earlier one's new
        members (a lost-update anomaly). An exclusive ``flock`` on a per-project
        lock file serialises the whole cycle across processes so every member
        is preserved.

        The lock file lives at ``{base}/{project_id}.lock`` (a sibling of the
        project directory, so it never shows up in ``list_projects``'s
        ``*/spec.json`` glob). On a non-POSIX platform without ``fcntl`` the
        lock degrades to a no-op (logged once) — the same best-effort behaviour
        as before this guard existed.
        """
        self._base.mkdir(parents=True, exist_ok=True)
        lock_path = self._base / f"{project_id}.lock"
        if fcntl is None:
            logger.warning(
                "fcntl unavailable; project membership updates for '%s' are not "
                "lock-protected against concurrent writers.",
                project_id,
            )
            yield
            return
        with open(lock_path, "w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

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
        with self._project_lock(project_id):
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

    def remove_members(self, project_id: str, work_ids: list[str]) -> ProjectSpec:
        """Remove work_ids from a project's membership (set difference).

        Also strips the removed ids from each roadmap phase's
        ``member_work_ids`` so coverage stays consistent. Bumps ``updated_at``
        only when something actually changed.

        Raises:
            KeyError: if the project does not exist.
        """
        with self._project_lock(project_id):
            spec = self.load_project(project_id)
            if spec is None:
                raise KeyError(f"Project '{project_id}' not found")

            to_remove = {w for w in work_ids if w}
            new_members = [w for w in spec.member_work_ids if w not in to_remove]
            changed = len(new_members) != len(spec.member_work_ids)

            for phase in spec.roadmap.phases:
                new_phase_members = [w for w in phase.member_work_ids if w not in to_remove]
                if len(new_phase_members) != len(phase.member_work_ids):
                    phase.member_work_ids = new_phase_members
                    changed = True

            if changed:
                spec.member_work_ids = new_members
                spec.updated_at = datetime.now().isoformat()
                self.save_project(spec)
            return spec

    def delete_project(self, project_id: str) -> bool:
        """Delete a project's on-disk directory.

        Returns True if the project existed and was removed, False otherwise.
        """
        import shutil

        project_dir = self._project_dir(project_id)
        if not project_dir.exists():
            return False
        shutil.rmtree(project_dir)
        return True
