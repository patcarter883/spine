"""Idempotent greenfield project scaffolding for the onboarding engine.

Slice2 of the onboarding engine: when ``mode == "greenfield"`` the engine has no
existing repo to analyse, so it bootstraps a minimal-but-valid SPINE project
layout (``.spine/skills/``, ``.spine/config.yaml``, ``src/``, ``tests/``) before
slice3 synthesises the four onboarding markdown artifacts.

All writes are **idempotent** — re-running ``scaffold_project`` overwrites cleanly
and never raises ``FileExistsError``.  This mirrors the
``mkdir(parents=True, exist_ok=True)`` + ``write_text`` style used by
:class:`spine.persistence.artifacts.ArtifactStore`.  Only :mod:`pathlib` is used;
no ``shutil.rmtree`` ever touches user directories.
"""

from __future__ import annotations

from pathlib import Path

from spine.work.onboarding.templates import baseline_config_yaml, default_dir_layout

# Repo-relative path of the scaffolded config — rendered separately from
# default_dir_layout() because its content depends on the caller's tech_stack.
_CONFIG_REL_PATH = ".spine/config.yaml"


def write_text_idempotent(path: Path, content: str) -> bool:
    """Write *content* to *path*, creating parent dirs, overwriting cleanly.

    Idempotent: if the file already exists with identical content, no write is
    performed and ``False`` is returned.  Otherwise the file is created or
    overwritten and ``True`` is returned.  Parent directories are created with
    ``exist_ok=True`` so a second call never raises ``FileExistsError``.

    Args:
        path: Destination file path.
        content: Text content to write (UTF-8).

    Returns:
        ``True`` if the file was created or its content changed; ``False`` if the
        on-disk content already matched (no-op write).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_file():
        try:
            if path.read_text(encoding="utf-8") == content:
                return False
        except (OSError, UnicodeDecodeError):
            # Unreadable/binary existing file — fall through and overwrite.
            pass
    path.write_text(content, encoding="utf-8")
    return True


def scaffold_project(
    workspace_root: str,
    tech_stack: list[str],
    *,
    force: bool = False,
) -> list[str]:
    """Scaffold a minimal greenfield SPINE project under *workspace_root*.

    Creates the standard layout — ``.spine/skills/``, ``.spine/artifacts/``,
    ``.spine/config.yaml``, ``src/``, ``tests/`` — writing ``.gitkeep`` seeds for
    otherwise-empty directories and a baseline ``config.yaml`` derived from
    ``tech_stack``.  The generated config parses via ``yaml.safe_load`` and
    ``SpineConfig.load()`` against *workspace_root*.

    Idempotent: calling twice in a row succeeds with no exception and yields an
    identical tree.  The returned list always reports every managed path (so the
    onboarding engine can record what the scaffold owns), regardless of whether a
    given file was physically rewritten on this call.

    Args:
        workspace_root: Absolute (or relative) path to the project root to
            scaffold.  Created if it does not exist.
        tech_stack: Technologies seeding the project, e.g.
            ``["python", "langgraph"]``.  Recorded in the config header.
        force: When ``True``, rewrite every managed file even if its content is
            unchanged.  When ``False`` (default), unchanged files are left
            untouched (no-op writes) while still being reported in the result.

    Returns:
        A sorted list of repo-relative paths managed by the scaffold (the four
        layout entries plus ``.spine/config.yaml``).
    """
    root = Path(workspace_root)
    root.mkdir(parents=True, exist_ok=True)

    managed: set[str] = set()

    # Collect every (rel_path, content) the scaffold owns. config.yaml is
    # tech-stack dependent so it is rendered separately from the static layout.
    files: dict[str, str] = dict(default_dir_layout())
    files[_CONFIG_REL_PATH] = baseline_config_yaml(tech_stack)

    for rel_path, content in files.items():
        target = root / rel_path
        if force:
            # Unconditional rewrite: still safe/idempotent (same bytes), but
            # guarantees the on-disk file matches the freshly rendered content.
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        else:
            write_text_idempotent(target, content)
        managed.add(rel_path)

    return sorted(managed)
