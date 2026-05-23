"""SPINE agents package — one agent builder per file.

Context engineering modules:
- ``context`` — SpineContext dataclass for per-run runtime context
- ``artifacts`` — Materialize artifacts to disk, reference by path
- ``factory`` — Shared build_phase_agent() with memory, skills, context
- ``backend`` — Single backend factory with CompositeBackend + cross-work memory
- ``skills_resolver`` — Locate skill directories for progressive disclosure
- ``profile`` — SPINE HarnessProfile (replaces DA BASE_AGENT_PROMPT)
"""

from __future__ import annotations

# Activate SPINE HarnessProfiles on first import so that any subsequent
# create_deep_agent() call picks up our base prompt instead of the DA default.
from spine.agents.profile import ensure_spine_profiles

ensure_spine_profiles()

# Public API exports for artifact handling (used across phases, workflow, tools)
from spine.agents.artifacts import (
    ARTIFACTS_DIR,
    artifact_path,
    build_artifact_prompt,
    build_current_phase_write_prompt,
    build_inline_artifact_prompt,
    list_slice_files,
    materialize_artifacts,
    materialize_phase_artifacts,
    scan_artifact_dir,
    validate_artifact_dir,
)

__all__ = [
    "ARTIFACTS_DIR",
    "artifact_path",
    "build_artifact_prompt",
    "build_current_phase_write_prompt",
    "build_inline_artifact_prompt",
    "list_slice_files",
    "materialize_artifacts",
    "materialize_phase_artifacts",
    "scan_artifact_dir",
    "validate_artifact_dir",
]
