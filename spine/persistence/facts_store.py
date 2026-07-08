"""SPINE project-facts store — the intent log for the CAM memory organ.

Every fact spine writes (or attempts to write) to the CAM serving plane is
recorded here as a :class:`spine.models.types.ProjectFact`. The CAM store's
value banks are delta-compressed tensors that cannot be enumerated, so this
JSONL side index is the authoritative record of *what was intended*: it is
what makes eviction reconciliation, replay/rebuild after a server reset, and
audit possible.

Mirrors :class:`spine.persistence.experience_store.ExperienceStore` (file-backed
at ``{experience_path}/facts.jsonl``, ``flock``-guarded read-modify-write,
atomic rewrite) with one semantic difference: facts are **one value per
subject per namespace** — the CAM store's own semantics — so a re-add of the
same subject replaces the prior record rather than accumulating.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from spine.models.types import ProjectFact

logger = logging.getLogger(__name__)

try:
    import fcntl  # POSIX advisory file locking
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]


class FactsStore:
    """Persists and retrieves :class:`ProjectFact` records on disk."""

    def __init__(self, base_path: str = ".spine/experience") -> None:
        self._base = Path(base_path)
        self._file = self._base / "facts.jsonl"

    # ── Locking ──────────────────────────────────────────────────────────
    @contextmanager
    def _lock(self) -> Iterator[None]:
        self._base.mkdir(parents=True, exist_ok=True)
        lock_path = self._base / "facts.lock"
        if fcntl is None:
            logger.warning(
                "fcntl unavailable; facts-store writes are not lock-protected "
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
    def all(self) -> list[ProjectFact]:
        """Return every recorded fact (skips any unparseable line)."""
        if not self._file.exists():
            return []
        facts: list[ProjectFact] = []
        try:
            text = self._file.read_text(encoding="utf-8")
        except OSError:
            return []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                facts.append(ProjectFact.model_validate_json(line))
            except ValueError:
                logger.debug("Skipping unparseable facts line", exc_info=True)
        return facts

    def stored(self, *, namespace: str | None = None) -> list[ProjectFact]:
        """Facts the write gate actually accepted (optionally one namespace).

        This is the replay set for ``/cam/rebuild`` reconciliation — gate-skipped
        attempts are on record but were never in the server store.
        """
        return [
            f
            for f in self.all()
            if f.stored and (namespace is None or f.namespace == namespace)
        ]

    # ── Write ────────────────────────────────────────────────────────────
    def add_many(self, facts: list[ProjectFact]) -> int:
        """Merge ``facts``; same subject+namespace replaces (one value/subject).

        Returns the number of records that were genuinely new subjects.
        """
        new_facts = [f for f in facts if f and f.subject.strip()]
        if not new_facts:
            return 0
        with self._lock():
            by_key: dict[str, ProjectFact] = {
                f.dedup_key(): f for f in self.all()
            }
            new_keys = 0
            for f in new_facts:
                key = f.dedup_key()
                if key not in by_key:
                    new_keys += 1
                by_key[key] = f  # an update replaces — store semantics
            self._rewrite(list(by_key.values()))
            return new_keys

    def delete(self, subject: str, *, namespace: str | None = None) -> bool:
        """Remove a subject's record. Returns True if one was removed."""
        if not subject.strip():
            return False
        norm = " ".join(subject.lower().split())
        with self._lock():
            existing = self.all()
            kept = [
                f
                for f in existing
                if not (
                    " ".join(f.subject.lower().split()) == norm
                    and (namespace is None or f.namespace == namespace)
                )
            ]
            if len(kept) == len(existing):
                return False
            self._rewrite(kept)
            return True

    def clear(self, *, namespace: str | None = None) -> int:
        """Remove all facts, or only one namespace's. Returns count removed."""
        with self._lock():
            existing = self.all()
            if namespace is None:
                if existing:
                    self._rewrite([])
                return len(existing)
            kept = [f for f in existing if f.namespace != namespace]
            removed = len(existing) - len(kept)
            if removed:
                self._rewrite(kept)
            return removed

    # ── Internals ────────────────────────────────────────────────────────
    def _rewrite(self, facts: list[ProjectFact]) -> None:
        """Atomically rewrite the whole store (tmp file → ``os.replace``)."""
        self._base.mkdir(parents=True, exist_ok=True)
        body = "\n".join(f.model_dump_json() for f in facts)
        if body:
            body += "\n"
        tmp = self._file.with_suffix(".jsonl.tmp")
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, self._file)
