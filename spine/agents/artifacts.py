"""SPINE artifact materializer — writes phase artifacts to the filesystem.

Prior phase artifacts were historically inlined into the user message,
causing massive context bloat on later phases.  DA's context engineering
docs recommend: "Use the filesystem — Persist large outputs to files so the
active context stays small; the model can pull in fragments with read_file
and grep when it needs details."

This module provides:

- :func:`materialize_artifacts` — write all prior phase artifacts to disk
  under ``.spine/artifacts/{work_id}/{phase}/`` so the agent can read them
  on demand.  Each work item gets its own isolated subfolder.
- :func:`build_artifact_prompt` — generate a compact reference section that
  tells the agent WHERE to find each artifact, instead of inlining the full
  content.

The LocalShellBackend (used by all SPINE agents) roots the filesystem at
``workspace_root``, so artifact paths are relative to that root.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from spine.models.enums import PhaseName

logger = logging.getLogger(__name__)

# ── Artifact directory structure ─────────────────────────────────────────
# All artifacts live under .spine/artifacts/{work_id}/ relative to the
# workspace root.  Each work item gets its own isolated subfolder, and within
# that, each phase gets its own subdirectory.  The key filename from the
# artifacts dict becomes the file name.

ARTIFACTS_DIR = ".spine/artifacts"


def _artifact_path(work_id: str, phase: str) -> str:
    """Build a work_id-scoped artifact directory path (relative to workspace).

    Args:
        work_id: Unique work item identifier.
        phase: The phase name (e.g. ``"specify"``).

    Returns:
        A relative path like ``.spine/artifacts/{work_id}/{phase}`` when
        ``work_id`` is provided, or ``.spine/artifacts/{phase}`` otherwise.
    """
    if work_id:
        return f"{ARTIFACTS_DIR}/{work_id}/{phase}"
    else:
        return f"{ARTIFACTS_DIR}/{phase}"


def materialize_phase_artifacts(
    phase: str,
    phase_artifacts: dict[str, str],
    workspace_root: str,
    work_id: str = "",
) -> None:
    """Write a single phase's artifacts to the filesystem immediately.

    Unlike :func:`materialize_artifacts` which reads from workflow state, this
    takes the artifacts dict directly so a phase can persist its own output
    right after producing it — without waiting for the next phase to call
    ``materialize_artifacts(state, ...)`` at its start.

    Artifacts are written under ``.spine/artifacts/{work_id}/{phase}/`` when
    ``work_id`` is provided, ensuring isolation between work items.

    Idempotent — rewrites files on each call.

    Args:
        phase: The phase name (e.g. ``"specify"``).
        phase_artifacts: Dict of ``{filename: content}`` for this phase.
        workspace_root: Absolute path to the project workspace.
        work_id: Unique work item identifier. When provided, artifacts are
            scoped under a work_id subfolder.
    """
    if not phase_artifacts:
        return
    root = Path(workspace_root)
    if work_id:
        phase_dir = root / ARTIFACTS_DIR / work_id / phase
    else:
        phase_dir = root / ARTIFACTS_DIR / phase
    phase_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in phase_artifacts.items():
        if not content:
            continue
        file_path = phase_dir / filename
        file_path.write_text(str(content), encoding="utf-8")
        logger.debug("Materialized phase artifact: %s", file_path)


def materialize_artifacts(
    state: dict[str, Any],
    workspace_root: str,
    work_id: str = "",
) -> dict[str, str]:
    """Write all prior phase artifacts to the filesystem.

    Creates ``.spine/artifacts/{work_id}/{phase}/{filename}`` for each
    artifact in the workflow state (when ``work_id`` is provided).  Returns
    a mapping of ``{phase: path}`` for use in the agent prompt.

    Idempotent — rewrites files on each call so the latest content is on
    disk.  This is safe because phases only run once per work item (except
    during rework, when the artifact is updated).

    Args:
        state: The current workflow state (must have ``artifacts`` key).
        workspace_root: Absolute path to the project workspace.
        work_id: Unique work item identifier. When provided, artifacts are
            scoped under a work_id subfolder.

    Returns:
        Dict mapping phase name to the artifact directory path
        (relative to workspace_root).
    """
    artifacts = state.get("artifacts", {})
    root = Path(workspace_root)
    paths: dict[str, str] = {}

    for phase_key, phase_artifacts in artifacts.items():
        if not phase_artifacts or not isinstance(phase_artifacts, dict):
            continue

        phase_dir = root / _artifact_path(work_id, phase_key)
        phase_dir.mkdir(parents=True, exist_ok=True)

        for filename, content in phase_artifacts.items():
            if not content:
                continue
            file_path = phase_dir / filename
            file_path.write_text(str(content), encoding="utf-8")
            logger.debug("Materialized artifact: %s", file_path)

        paths[phase_key] = _artifact_path(work_id, phase_key)

    return paths


def build_artifact_prompt(
    artifacts: dict[str, Any],
    current_phase: str,
    work_id: str = "",
) -> str:
    """Build a compact prompt section referencing artifacts by path.

    Instead of inlining the full content of each artifact, this generates
    a section that tells the agent where each artifact lives on disk.  The
    agent can use ``read_file``, ``grep``, etc. to pull in only the parts
    it needs — saving thousands of tokens on every turn.

    When ``work_id`` is provided, paths include the work_id subfolder
    (``.spine/artifacts/{work_id}/{phase}/{file}``).

    Accepts the raw artifacts dict from WorkflowState and computes paths
    automatically.  Only lists phases that have non-empty artifacts and
    are not the current phase.

    Args:
        artifacts: The artifacts dict from WorkflowState
            (``{phase: {filename: content}}``).
        current_phase: The phase being executed (to skip self-reference).
        work_id: Unique work item identifier. When provided, artifact paths
            include the work_id subfolder for isolation.

    Returns:
        A markdown-formatted string listing artifact locations, or empty
        string if no artifacts are available.
    """
    if not artifacts:
        return ""

    # Order artifacts by workflow phase sequence
    phase_order = [
        PhaseName.SPECIFY.value,
        PhaseName.PLAN.value,
        PhaseName.TASKS.value,
        PhaseName.IMPLEMENT.value,
    ]

    lines: list[str] = ["## Prior Artifacts (on disk)"]
    lines.append(
        "The following artifacts from prior phases are available on disk. "
        "Use `read_file` and `grep` to read only what you need — do NOT "
        "load everything into context at once."
    )

    for phase in phase_order:
        if phase == current_phase:
            continue
        phase_artifacts = artifacts.get(phase)
        if phase_artifacts and isinstance(phase_artifacts, dict):
            path = _artifact_path(work_id, phase)
            phase_label = phase.upper()
            # List individual files in the phase directory
            filenames = list(phase_artifacts.keys())
            file_list = ", ".join(f"`{path}/{f}`" for f in filenames)
            lines.append(f"- **{phase_label}**: {file_list}")

    return "\n".join(lines) + "\n\n"


def build_inline_artifact_prompt(
    state: dict[str, Any],
    current_phase: str,
    max_inline_chars: int = 500,
    work_id: str = "",
) -> str:
    """Build a prompt section with inline artifact summaries for the critic.

    The critic agent needs to see artifact content to review it, but
    full inlining is still wasteful.  This provides a short preview
    (first N chars) of each artifact so the critic can decide whether
    to read the full file.

    Args:
        state: The current workflow state.
        current_phase: The phase being reviewed.
        max_inline_chars: Maximum characters to inline per artifact.
        work_id: Unique work item identifier. When provided, artifact paths
            include the work_id subfolder for isolation.

    Returns:
        A markdown-formatted string with artifact previews.
    """
    artifacts = state.get("artifacts", {})
    phase_artifacts = artifacts.get(current_phase, {})

    if not phase_artifacts:
        return ""

    lines: list[str] = ["## Artifacts Under Review"]
    base_path = _artifact_path(work_id, current_phase)
    for name, content in phase_artifacts.items():
        content_str = str(content)
        if len(content_str) > max_inline_chars:
            preview = content_str[:max_inline_chars] + "..."
            lines.append(f"### {name}")
            lines.append(f"```\n{preview}\n```")
            lines.append(
                f"Full content available at `{base_path}/{name}`"
            )
        else:
            lines.append(f"### {name}")
            lines.append(f"```\n{content_str}\n```")

    return "\n".join(lines) + "\n\n"
