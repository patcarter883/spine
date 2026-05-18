"""SPINE agent backend — creates backends with workspace root and cross-work memory.

All Deep Agents in SPINE phases need a backend with root_dir pointing to the
project workspace. This module provides the shared factory so every agent
builder uses the same backend configuration with cross-work memory.

**Architecture:**

The ``CompositeBackend`` routes paths to different backends:

- Default: ``LocalShellBackend`` with ``virtual_mode=True`` for file operations
- ``/memories/``: ``StoreBackend`` for durable cross-work persistence

Agents can write project knowledge to ``/memories/`` and it persists across
work items, enabling project-wide learning and context.
"""

from __future__ import annotations

import logging
from pathlib import Path

from deepagents.backends.local_shell import LocalShellBackend
from langgraph.store.memory import InMemoryStore

logger = logging.getLogger(__name__)

# Path prefix reserved for cross-work memory
MEMORY_PATH = "/memories/"

# Singleton store — shared across all work items for cross-work memory
_store: InMemoryStore | None = None


class _NormalizingLocalShellBackend(LocalShellBackend):
    """LocalShellBackend that strips accidental absolute paths before resolution.

    Under virtual_mode=True, paths starting with ``/`` are treated as
    virtual paths relative to ``root_dir``. If the model constructs a
    full absolute path like ``/home/user/project/.spine/...``, it gets
    double-nested (``root_dir + /home/user/project/.spine/...``).

    This subclass detects when a path starts with our own ``root_dir``
    and strips the prefix before delegating to the parent ``_resolve_path``.
    """

    def _resolve_path(self, key: str) -> Path:
        if self.virtual_mode and key.startswith(str(self.cwd)):
            key = key[len(str(self.cwd)):].lstrip("/")
        return super()._resolve_path(key)


def _seed_store(store: InMemoryStore, root: Path) -> None:
    """Pre-seed the InMemoryStore with project knowledge.

    Reads ``AGENTS.md`` from the project root (if it exists) and stores it
    at namespace ``/memories/``, key ``conventions`` so agents don't need
    to read_file it every time.

    Args:
        store: The singleton InMemoryStore to seed.
        root: Absolute path to the project workspace root.
    """
    # ── Seed AGENTS.md as conventions ────────────────────────────
    agents_md = root / "AGENTS.md"
    if agents_md.is_file():
        content = agents_md.read_text(encoding="utf-8")
        store.put(
            ("memories",),  # namespace tuple
            "conventions",  # key
            {"content": content},  # value dict
        )
        logger.info(
            "Seeded memory store with AGENTS.md (%d chars) as 'conventions'",
            len(content),
        )
    else:
        logger.debug("No AGENTS.md found at %s — skipping conventions seed", agents_md)

    # ── Seed project structure summary if available ───────────────
    structure_files = [
        root / ".spine" / "codebase-map.md",
        root / ".spine" / "structure.md",
    ]
    for sf in structure_files:
        if sf.is_file():
            content = sf.read_text(encoding="utf-8")
            key = sf.stem  # e.g. "codebase-map" or "structure"
            store.put(
                ("memories",),
                key,
                {"content": content},
            )
            logger.info(
                "Seeded memory store with %s (%d chars) as %r",
                sf.name,
                len(content),
                key,
            )


def _get_store(root: Path | None = None) -> InMemoryStore:
    """Get or create the singleton InMemoryStore for cross-work memory.

    On first creation, seeds the store with project knowledge from AGENTS.md
    and any available structure summaries under ``.spine/``.

    Args:
        root: Optional project root Path. If provided on the first call,
            the store will be seeded with project knowledge. Subsequent
            calls with different roots will not re-seed.
    """
    global _store
    if _store is None:
        _store = InMemoryStore()
        logger.debug("Created InMemoryStore for cross-work memory")
        if root is not None:
            _seed_store(_store, root)
    return _store


def build_backend(workspace_root: str) -> LocalShellBackend | object:
    """Create a backend for the given workspace root with cross-work memory.

    Returns a ``CompositeBackend`` that routes ``/memories/`` to a
    ``StoreBackend`` for durable cross-work persistence, with a
    ``LocalShellBackend`` (virtual_mode=True) as the default for all other paths.

    Args:
        workspace_root: Absolute path to the project directory.

    Returns:
        A CompositeBackend with cross-work memory support, or a plain
        LocalShellBackend if CompositeBackend is unavailable.
    """
    root = Path(workspace_root).resolve()

    # First try to create a CompositeBackend with cross-work memory
    try:
        from deepagents.backends import CompositeBackend, StoreBackend

        local_backend = _NormalizingLocalShellBackend(
            root_dir=str(root),
            virtual_mode=True,
            timeout=120,
        )

        store = _get_store(root)
        memory_backend = StoreBackend(store=store)

        composite = CompositeBackend(
            default=local_backend,
            routes={MEMORY_PATH: memory_backend},
        )

        logger.debug("Created CompositeBackend with cross-work memory at %s", MEMORY_PATH)
        return composite

    except ImportError:
        # Fall back to LocalShellBackend if CompositeBackend unavailable
        logger.info(
            "CompositeBackend not available (requires deepagents >= 0.6.0), "
            "falling back to LocalShellBackend without cross-work memory"
        )
        return _NormalizingLocalShellBackend(
            root_dir=str(root),
            virtual_mode=True,
            timeout=120,
        )