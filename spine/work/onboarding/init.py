"""Brownfield ``spine init`` workspace bootstrapping.

A trimmed counterpart to :func:`spine.work.onboarding.scaffold.scaffold_project`
that only writes inside ``.spine/`` — it never touches ``src/`` or ``tests/``,
making it safe to run inside an existing project that already has its own
source layout.

The function reuses :func:`scaffold.write_text_idempotent` and
:func:`templates.baseline_config_yaml` so the generated config is byte-identical
to the greenfield path's ``.spine/config.yaml``.
"""

from __future__ import annotations

from pathlib import Path

from spine.work.onboarding.scaffold import write_text_idempotent
from spine.work.onboarding.templates import baseline_config_yaml

_MANAGED_DIR_SEEDS: tuple[str, ...] = (
    ".spine/skills/.gitkeep",
    ".spine/artifacts/.gitkeep",
)
_CONFIG_REL_PATH = ".spine/config.yaml"


def init_workspace(
    workspace_root: str,
    tech_stack: list[str],
    *,
    force: bool = False,
) -> tuple[list[str], list[str]]:
    """Scaffold the ``.spine/`` directory inside an existing project.

    Creates ``.spine/skills/.gitkeep``, ``.spine/artifacts/.gitkeep``, and
    ``.spine/config.yaml`` under *workspace_root*.  Does not create or modify
    ``src/`` or ``tests/`` — callers who want the full greenfield layout
    should use :func:`spine.work.onboarding.scaffold.scaffold_project`.

    An existing ``.spine/config.yaml`` is preserved unless ``force=True``: the
    file's path is still returned in the ``managed`` list, but the second
    element of the return tuple records which managed paths were left
    untouched so the CLI can warn the user.

    Args:
        workspace_root: Project root under which to write ``.spine/``.  Created
            if it does not exist.
        tech_stack: Technology tags recorded in the config header comment.
        force: When ``True``, overwrite ``.spine/config.yaml`` even if it
            already exists with different content.

    Returns:
        ``(managed, preserved)`` where ``managed`` is the sorted list of
        repo-relative paths the scaffolder owns and ``preserved`` is the subset
        that already existed with different content and were left in place
        (always empty when ``force=True``).
    """
    root = Path(workspace_root)
    root.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {seed: "" for seed in _MANAGED_DIR_SEEDS}
    files[_CONFIG_REL_PATH] = baseline_config_yaml(tech_stack)

    preserved: list[str] = []
    for rel_path, content in files.items():
        target = root / rel_path
        if (
            not force
            and rel_path == _CONFIG_REL_PATH
            and target.exists()
            and target.is_file()
            and target.read_text(encoding="utf-8") != content
        ):
            preserved.append(rel_path)
            continue
        write_text_idempotent(target, content)

    return sorted(files), sorted(preserved)
