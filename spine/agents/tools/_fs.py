"""Shared filesystem utilities for SPINE tools."""

from __future__ import annotations

import os
from pathlib import Path


def _atomic_write(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically.

    Writes to ``path + ".tmp"`` first, then renames into place via
    :func:`os.replace`. Replacing through a rename guarantees that
    readers never observe a half-written file — they see either the
    old version or the new version, never a partial one.

    Args:
        path: Destination path. The parent directory must exist.
        content: Text to write (encoded as UTF-8).
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)
