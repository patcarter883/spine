"""SPINE experience store — file-backed cross-run "distilled experience".

Each run distils a compact :class:`ExperienceLesson` from the critic /
adversarial feedback of phases that needed revision (see
:mod:`spine.agents.experience`). Those lessons are persisted here and replayed
into the matching phase's prompt on future runs, so a defect the critic caught
once is guarded against on every subsequent run.

Design choices (mirroring :class:`spine.persistence.project_store.ProjectStore`):

- **File-backed, not in-memory.** Each ``spine run`` is its own process; the
  cross-work ``InMemoryStore`` in :mod:`spine.agents.backend` does not survive
  process exit. Lessons must persist across processes, so they live on disk at
  ``{experience_path}/lessons.jsonl``.
- **Bounded.** Lessons are de-duplicated by phase + normalised text and capped
  per phase (highest-salience, most-recent kept). The injected block stays
  small — the whole point versus dumping raw review history.
- **Concurrency-safe.** Add is a read-modify-write under an exclusive
  ``flock`` so a UI submission and a queue worker writing at once don't clobber
  each other (same lost-update guard ProjectStore uses).
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from spine.models.types import ExperienceLesson

logger = logging.getLogger(__name__)

try:
    import fcntl  # POSIX advisory file locking
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]

# Keep at most this many lessons per phase. Beyond the cap, the lowest-salience
# (then oldest) lessons are dropped so the on-disk store — and the per-phase
# injected block — never grows without bound.
_MAX_LESSONS_PER_PHASE = 12


class ExperienceStore:
    """Persists and retrieves :class:`ExperienceLesson` records on disk."""

    def __init__(self, base_path: str = ".spine/experience") -> None:
        self._base = Path(base_path)
        self._file = self._base / "lessons.jsonl"

    # ── Locking ──────────────────────────────────────────────────────────
    @contextmanager
    def _lock(self) -> Iterator[None]:
        """Hold an exclusive inter-process lock for the whole store.

        Add is a read-modify-write (load all → merge/dedup/cap → rewrite); the
        lock serialises that cycle across processes. Degrades to a no-op
        (logged once) on a platform without ``fcntl``.
        """
        self._base.mkdir(parents=True, exist_ok=True)
        lock_path = self._base / "lessons.lock"
        if fcntl is None:
            logger.warning(
                "fcntl unavailable; experience-store writes are not lock-protected "
                "against concurrent writers."
            )
            yield
            return
        with open(lock_path, "w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    # ── Read ─────────────────────────────────────────────────────────────
    def all(self) -> list[ExperienceLesson]:
        """Return every stored lesson (skips any unparseable line)."""
        if not self._file.exists():
            return []
        lessons: list[ExperienceLesson] = []
        try:
            text = self._file.read_text(encoding="utf-8")
        except OSError:
            return []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                lessons.append(ExperienceLesson.model_validate_json(line))
            except ValueError:
                # A corrupt line must never take down recall — skip it.
                logger.debug("Skipping unparseable experience line", exc_info=True)
        return lessons

    def for_phase(
        self,
        phase: str,
        *,
        category: str | None = None,
        limit: int = 5,
    ) -> list[ExperienceLesson]:
        """Return the most relevant lessons for ``phase``.

        Ranking: lessons whose ``category`` matches the current task category
        rank first (when a category is given), then by descending salience,
        then most recent. Capped to ``limit``.
        """
        candidates = [le for le in self.all() if le.phase == phase]
        if not candidates:
            return []

        def sort_key(le: ExperienceLesson) -> tuple:
            category_match = 1 if (category and le.category == category) else 0
            return (category_match, le.salience, le.created_at)

        candidates.sort(key=sort_key, reverse=True)
        return candidates[: max(0, limit)]

    # ── Write ────────────────────────────────────────────────────────────
    def add_many(self, lessons: list[ExperienceLesson]) -> int:
        """Merge ``lessons`` into the store, de-duplicating and capping per phase.

        Returns the number of genuinely-new lessons written. Existing lessons
        with the same :meth:`ExperienceLesson.dedup_key` are not duplicated; the
        higher-salience copy wins so a recurring defect's priority can only rise.
        """
        new_lessons = [le for le in lessons if le and (le.lesson or "").strip()]
        if not new_lessons:
            return 0

        with self._lock():
            existing = self.all()
            by_key: dict[str, ExperienceLesson] = {}
            for le in existing:
                by_key[le.dedup_key()] = le

            newly_added_keys: set[str] = set()
            for le in new_lessons:
                key = le.dedup_key()
                prior = by_key.get(key)
                if prior is None:
                    by_key[key] = le
                    newly_added_keys.add(key)
                elif le.salience > prior.salience:
                    # Same lesson seen again, costlier this time — keep the
                    # higher salience (and the newer provenance) but don't
                    # count it as new.
                    by_key[key] = le

            if not newly_added_keys:
                # Salience bumps still need persisting, but skip the rewrite when
                # nothing changed at all.
                if all(by_key[le.dedup_key()] is le for le in existing):
                    return 0

            merged = self._cap_per_phase(list(by_key.values()))
            self._rewrite(merged)
            # Report only lessons that actually survived the per-phase cap — a
            # capped-out, lowest-salience new lesson is dropped before write and
            # would never be recallable, so it must not inflate the count.
            surviving_keys = {le.dedup_key() for le in merged}
            return len(newly_added_keys & surviving_keys)

    def delete(self, lesson_id: str) -> bool:
        """Remove a single lesson by id. Returns True if one was removed."""
        if not lesson_id:
            return False
        with self._lock():
            existing = self.all()
            kept = [le for le in existing if le.id != lesson_id]
            if len(kept) == len(existing):
                return False
            self._rewrite(kept)
            return True

    def clear(self, *, phase: str | None = None) -> int:
        """Remove all lessons, or only those for ``phase``. Returns count removed."""
        with self._lock():
            existing = self.all()
            if phase is None:
                removed = len(existing)
                if removed:
                    self._rewrite([])
                return removed
            kept = [le for le in existing if le.phase != phase]
            removed = len(existing) - len(kept)
            if removed:
                self._rewrite(kept)
            return removed

    # ── Internals ────────────────────────────────────────────────────────
    @staticmethod
    def _cap_per_phase(lessons: list[ExperienceLesson]) -> list[ExperienceLesson]:
        """Drop the weakest lessons so each phase keeps at most the cap.

        Within a phase, keep the highest-salience then most-recent lessons.
        """
        by_phase: dict[str, list[ExperienceLesson]] = {}
        for le in lessons:
            by_phase.setdefault(le.phase, []).append(le)
        kept: list[ExperienceLesson] = []
        for phase_lessons in by_phase.values():
            phase_lessons.sort(key=lambda le: (le.salience, le.created_at), reverse=True)
            kept.extend(phase_lessons[:_MAX_LESSONS_PER_PHASE])
        return kept

    def _rewrite(self, lessons: list[ExperienceLesson]) -> None:
        """Atomically rewrite the whole store (tmp file → ``os.replace``)."""
        self._base.mkdir(parents=True, exist_ok=True)
        body = "\n".join(le.model_dump_json() for le in lessons)
        if body:
            body += "\n"
        tmp = self._file.with_suffix(".jsonl.tmp")
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, self._file)
