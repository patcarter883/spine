"""SPINE agent backend — creates LocalShellBackend with the correct workspace root.

All Deep Agents in SPINE phases need a LocalShellBackend with root_dir
pointing to the project workspace. This module provides a shared factory
so every agent builder uses the same backend configuration.

Without this, the default StateBackend keeps files in-memory only —
so when the implement agent writes /test_file.py, the file never
appears on disk, and the verify agent can't find it.
"""

from __future__ import annotations

from pathlib import Path

from deepagents.backends.local_shell import LocalShellBackend


def build_backend(workspace_root: str) -> LocalShellBackend:
    """Create a LocalShellBackend for the given workspace root.

    The backend provides:
    - Filesystem tools (read_file, write_file, ls, glob, grep) rooted
      at workspace_root
    - Shell execution (execute tool) with workspace_root as cwd

    Args:
        workspace_root: Absolute path to the project directory.

    Returns:
        A LocalShellBackend configured for the workspace.
    """
    root = Path(workspace_root).resolve()
    return LocalShellBackend(
        root_dir=str(root),
        virtual_mode=False,
        timeout=120,
    )
