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


def _get_store() -> InMemoryStore:
    """Get or create the singleton InMemoryStore for cross-work memory."""
    global _store
    if _store is None:
        _store = InMemoryStore()
        logger.debug("Created InMemoryStore for cross-work memory")
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

        local_backend = LocalShellBackend(
            root_dir=str(root),
            virtual_mode=True,
            timeout=120,
        )

        store = _get_store()
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
        return LocalShellBackend(
            root_dir=str(root),
            virtual_mode=True,
            timeout=120,
        )