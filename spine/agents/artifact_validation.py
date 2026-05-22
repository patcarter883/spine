"""Artifact path validation helpers for SPINE subgraphs.

After each subgraph phase completes, validate that artifacts exist at the
expected paths. This catches agents that write files to wrong directories.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_artifact_dir(workspace_root: str, work_id: str, phase: str) -> bool:
    """Return True if artifacts exist at the expected path.

    Expected path: ``.spine/artifacts/{work_id}/{phase}/``

    Args:
        workspace_root: Project root directory.
        work_id: Work item ID.
        phase: Phase name.

    Returns:
        True if at least one file exists in the expected directory.
    """
    expected = Path(workspace_root) / ".spine" / "artifacts" / work_id / phase
    if not expected.exists():
        logger.warning(
            "Artifact validation failed: directory does not exist: %s",
            expected,
        )
        return False

    files = list(expected.glob("*"))
    if not files:
        logger.warning(
            "Artifact validation failed: no files in %s",
            expected,
        )
        return False

    logger.info(
        "Validated %d artifact(s) at %s",
        len(files),
        expected,
    )
    return True


def validate_artifacts_for_work(
    workspace_root: str, work_id: str, phases: list[str]
) -> dict[str, bool]:
    """Validate artifacts for multiple phases.

    Returns a dict mapping phase name to validation result.
    """
    return {phase: validate_artifact_dir(workspace_root, work_id, phase) for phase in phases}
