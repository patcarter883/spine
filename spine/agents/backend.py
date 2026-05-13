"""SPINE agent backend — creates backends with workspace root and optional memory.

All Deep Agents in SPINE phases need a backend with root_dir pointing to the
project workspace.  This module provides the shared factory so every agent
builder uses the same backend configuration.

Two modes:

1. **Standard** — ``LocalShellBackend`` with ``virtual_mode=False``, so
   files written by one phase appear on disk for the next.
2. **With memory** — ``CompositeBackend`` that routes ``/memories/`` to a
   ``StoreBackend`` for durable cross-work persistence, with
   ``LocalShellBackend`` as the default.

The mode is determined by the ``enable_cross_work_memory`` setting in
``.spine/config.yaml`` (default: disabled for backward compatibility).
"""

from __future__ import annotations

import logging
from pathlib import Path

from deepagents.backends.local_shell import LocalShellBackend

logger = logging.getLogger(__name__)


def build_backend(workspace_root: str, *, enable_memory: bool = False) -> LocalShellBackend:
    """Create a backend for the given workspace root.

    When ``enable_memory`` is True and CompositeBackend is available,
    returns a backend factory that creates a CompositeBackend with
    cross-work memory support at ``/memories/``.

    Otherwise returns a plain LocalShellBackend (backward compatible).

    Args:
        workspace_root: Absolute path to the project directory.
        enable_memory: Whether to enable cross-work memory persistence.

    Returns:
        A LocalShellBackend or a CompositeBackend factory.
    """
    if enable_memory:
        from spine.agents.backend_memory import build_backend_with_memory
        return build_backend_with_memory(workspace_root, enable_cross_work_memory=True)

    root = Path(workspace_root).resolve()
    return LocalShellBackend(
        root_dir=str(root),
        virtual_mode=False,
        timeout=120,
    )
