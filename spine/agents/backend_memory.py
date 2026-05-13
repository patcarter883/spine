"""SPINE backend factory — creates backends with optional cross-work memory.

The standard ``LocalShellBackend`` scopes the virtual filesystem to a single
thread.  This means agents can't learn from previous work items — every work
item starts from zero context about the project.

DA's ``CompositeBackend`` lets us route specific paths to a ``StoreBackend``
backed by a LangGraph Store, providing durable cross-thread persistence.
Agents can write project knowledge to ``/memories/`` and it persists across
work items.

This module provides:

- :func:`build_backend_with_memory` — creates a CompositeBackend when a
  store is configured, falling back to a plain LocalShellBackend otherwise.
- :func:`get_or_create_store` — singleton store instance.

The ``/memories/`` path is reserved for cross-work memory.  Agents are
instructed (via the base prompt) to save project conventions, discovered
patterns, and other durable knowledge there.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from deepagents.backends.local_shell import LocalShellBackend

logger = logging.getLogger(__name__)

# Path prefix reserved for cross-work memory
MEMORY_PATH = "/memories/"

# Singleton store — shared across all work items
_store: Any = None


def get_or_create_store() -> Any:
    """Get or create the singleton LangGraph store for cross-work memory.

    Uses InMemoryStore by default.  Can be replaced with a persistent
    store (e.g. LangGraph Store with a database backend) via configuration.

    Returns:
        A LangGraph BaseStore instance.
    """
    global _store
    if _store is None:
        try:
            from langgraph.store.memory import InMemoryStore

            _store = InMemoryStore()
            logger.debug("Created InMemoryStore for cross-work memory")
        except ImportError:
            logger.warning(
                "langgraph.store not available — cross-work memory disabled"
            )
            return None
    return _store


def build_backend_with_memory(
    workspace_root: str,
    *,
    enable_cross_work_memory: bool = False,
) -> Any:
    """Create a backend with optional cross-work memory support.

    When ``enable_cross_work_memory`` is True (and a store is available),
    returns a ``CompositeBackend`` that routes ``/memories/`` to a
    ``StoreBackend`` for durable cross-thread persistence, with the
    default ``LocalShellBackend`` for everything else.

    When disabled (or when the store is unavailable), returns a plain
    ``LocalShellBackend`` — backward compatible with existing behavior.

    Args:
        workspace_root: Absolute path to the project directory.
        enable_cross_work_memory: Whether to enable cross-work memory.

    Returns:
        A backend instance (CompositeBackend or LocalShellBackend).
    """
    root = Path(workspace_root).resolve()
    local_backend = LocalShellBackend(
        root_dir=str(root),
        virtual_mode=False,
        timeout=120,
    )

    if not enable_cross_work_memory:
        return local_backend

    store = get_or_create_store()
    if store is None:
        logger.info(
            "Cross-work memory requested but store unavailable, "
            "falling back to LocalShellBackend"
        )
        return local_backend

    try:
        from deepagents.backends import CompositeBackend, StoreBackend

        def make_composite(runtime: Any) -> CompositeBackend:
            return CompositeBackend(
                default=local_backend,
                routes={MEMORY_PATH: StoreBackend(runtime)},
            )

        logger.info("Created CompositeBackend with cross-work memory at %s", MEMORY_PATH)
        return make_composite
    except ImportError:
        logger.info(
            "CompositeBackend not available (requires deepagents >= 0.6.0), "
            "falling back to LocalShellBackend"
        )
        return local_backend
