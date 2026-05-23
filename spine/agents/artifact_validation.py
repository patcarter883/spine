"""Artifact path validation helpers for SPINE subgraphs.

After each subgraph phase completes, validate that artifacts exist at the
expected paths. This catches agents that write files to wrong directories.
"""

from __future__ import annotations

# Re-export the canonical implementation from artifacts.py to avoid
# code duplication.
from spine.agents.artifacts import validate_artifact_dir

__all__ = ["validate_artifact_dir"]
