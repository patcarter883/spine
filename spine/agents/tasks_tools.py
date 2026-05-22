"""Custom tools for the TASKS phase agent.

.. deprecated::
    The TASKS phase tools are deprecated. New workflows should use the PLAN
    phase tools for decomposition instead. This module is retained for
    backward compatibility and will be removed in a future release.
    ``SliceDefinition`` is kept for downstream consumers that reference it.

Replaces generic filesystem tools with purpose-built tools that enforce
the tasks agent's role: research the codebase, then write a complete
structured decomposition in one atomic call. Nothing else.

The key insight from trace analysis (019e4483): with generic filesystem
tools the agent spent 80+ minutes making 87 read_file calls, re-reading
the same files dozens of times, and continued reading even AFTER writing
its artifacts. It never dispatched researcher subagents despite explicit
instructions to do so.

Tools:
- ``read_prior_artifacts`` (reused from plan_tools) — loads spec/plan
  artifacts in one call for spec-workflow tasks runs.
- ``search_codebase`` (reused from plan_tools) — multi-query targeted
  file search; replaces ls/glob/grep/read_file for codebase exploration.
- ``write_tasks_artifacts`` — the ONLY write surface. Accepts all required
  artifacts (slices, tasks.md, codebase-map.md) as structured arguments
  and writes them atomically. The agent cannot write partial output or
  continue after calling this tool — the phase is done.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field

# Re-export shared tools so callers only need to import from tasks_tools
from spine.agents.plan_tools import ReadPriorArtifactsTool, SearchCodebaseTool  # noqa: F401

logger = logging.getLogger(__name__)


# ── SliceDefinition ───────────────────────────────────────────────────────


class SliceDefinition(BaseModel):
    """One feature slice within a task decomposition."""

    name: str = Field(
        description=(
            "Slug name for the slice, e.g. 'update-status-labels'. "
            "Used as the filename: slice-<name>.md"
        )
    )
    description: str = Field(description="What this slice implements and why.")
    files_to_modify: list[str] = Field(
        description=(
            "Existing workspace files that will be changed. "
            "Each path MUST exist in the workspace (confirmed via search_codebase)."
        ),
        default_factory=list,
    )
    files_to_create: list[str] = Field(
        description=(
            "New files to create. Each path's parent directory MUST exist "
            "or be created as part of this slice."
        ),
        default_factory=list,
    )
    dependencies: list[str] = Field(
        description="Names of other slices this depends on (empty if none).",
        default_factory=list,
    )
    acceptance_criteria: list[str] = Field(
        description="Measurable, verifiable criteria for this slice to be complete.",
        min_length=1,
    )
    complexity: str = Field(
        description="One of: small (1pt), medium (2pt), large (3pt).",
        default="small",
    )
    modification_targets: str = Field(
        description=(
            "For each file to modify: a 3-5 line code snippet around the "
            "exact change site with its line range (e.g. '# file.py [L42-47]\\ncode...'). "
            "Subagents use these to locate insertion points without re-exploring."
        ),
        default="",
    )


# ── write_tasks_artifacts ─────────────────────────────────────────────────


class _WriteTasksArtifactsInput(BaseModel):
    slices: list[SliceDefinition] = Field(
        description=(
            "All feature slices for this work item. Each slice becomes a "
            "separate slice-<name>.md file."
        ),
        min_length=1,
    )
    overview: str = Field(
        description=(
            "Brief summary (2-4 sentences) of what needs to be built and "
            "why the slices are structured as they are."
        )
    )
    dependency_waves: str = Field(
        description=(
            "Ordered implementation waves showing which slices can run in "
            "parallel vs. sequentially. E.g. 'Wave 1: slice-a, slice-b (parallel). "
            "Wave 2: slice-c (depends on both).'"
        )
    )
    codebase_map: str = Field(
        description=(
            "Structured codebase map with ALL required sections:\n"
            "1. Files — path → 1-line description → line count\n"
            "2. Key Functions — name(args) → return_type [L<start>-<end>] — desc\n"
            "3. Import Chains — which modules import which\n"
            "4. Conventions — naming patterns, error handling style\n"
            "5. Modification Targets — 3-5 line code snippets at each change site\n\n"
            "This is the primary artifact downstream IMPLEMENT/VERIFY subagents "
            "use to locate code. If it's vague, subagents waste turns re-exploring."
        )
    )


class WriteTasksArtifactsTool(BaseTool):
    """Write all TASKS phase artifacts atomically in one call.

    .. deprecated::
        The TASKS phase is deprecated. This tool is retained for backward
        compatibility only.

    This is the ONLY write tool available to the tasks agent.
    Accepts all required artifacts as structured arguments and writes:
    - One ``slice-<name>.md`` per slice definition
    - ``tasks.md`` — index with dependency waves and file change matrix
    - ``codebase-map.md`` — exploration map for downstream phases

    Once called, the tasks phase is COMPLETE. There is nothing left to do.
    The agent MUST NOT make any more tool calls after this one.
    """

    name: str = "write_tasks_artifacts"
    description: str = (
        "Write ALL tasks phase artifacts atomically: slice files, tasks.md, "
        "and codebase-map.md. Call this ONCE after research is complete. "
        "This is the ONLY write tool. After calling this, the phase is DONE — "
        "make no further tool calls."
    )
    args_schema: Optional[ArgsSchema] = _WriteTasksArtifactsInput

    workspace_root: str = ""
    tasks_dir: str = ""

    def _run(
        self,
        slices: list[dict[str, Any]],
        overview: str,
        dependency_waves: str,
        codebase_map: str,
    ) -> str:
        # Coerce dicts to SliceDefinition objects (pydantic may pass raw dicts)
        parsed_slices = [SliceDefinition(**s) if isinstance(s, dict) else s for s in slices]

        tasks_path = Path(self.workspace_root) / self.tasks_dir
        tasks_path.mkdir(parents=True, exist_ok=True)

        written: list[str] = []

        # ── Write individual slice files ──────────────────────────────
        for sl in parsed_slices:
            lines = [
                f"# Slice: {sl.name}\n\n",
                f"## Description\n{sl.description.strip()}\n\n",
            ]
            if sl.files_to_modify:
                lines.append("## Files to Modify\n")
                lines.extend(f"- `{f}`\n" for f in sl.files_to_modify)
                lines.append("\n")
            if sl.files_to_create:
                lines.append("## Files to Create\n")
                lines.extend(f"- `{f}`\n" for f in sl.files_to_create)
                lines.append("\n")
            if sl.dependencies:
                lines.append("## Dependencies\n")
                lines.extend(f"- {d}\n" for d in sl.dependencies)
                lines.append("\n")
            else:
                lines.append("## Dependencies\nNone\n\n")
            lines.append("## Acceptance Criteria\n")
            lines.extend(f"- {c}\n" for c in sl.acceptance_criteria)
            lines.append(f"\n## Complexity\n{sl.complexity}\n")
            if sl.modification_targets.strip():
                lines.append(f"\n## Modification Targets\n{sl.modification_targets.strip()}\n")

            fname = f"slice-{sl.name}.md"
            content = "".join(lines)
            (tasks_path / fname).write_text(content, encoding="utf-8")
            written.append(fname)

        # ── Write tasks.md ────────────────────────────────────────────
        task_lines = [
            "# Task Breakdown Summary\n\n",
            f"## Overview\n{overview.strip()}\n\n",
            "## Slices\n",
        ]
        for sl in parsed_slices:
            deps = ", ".join(sl.dependencies) if sl.dependencies else "none"
            task_lines.append(
                f"### {sl.name} ({sl.complexity})\n"
                f"{sl.description.strip()}\n"
                f"- Files: {', '.join(sl.files_to_modify + sl.files_to_create) or 'TBD'}\n"
                f"- Depends on: {deps}\n\n"
            )
        task_lines.append(f"## Implementation Waves\n{dependency_waves.strip()}\n\n")

        # File change matrix
        all_files: dict[str, list[str]] = {}
        for sl in parsed_slices:
            for f in sl.files_to_modify + sl.files_to_create:
                all_files.setdefault(f, []).append(sl.name)
        if all_files:
            task_lines.append("## File Change Matrix\n| File | Slices |\n|------|--------|\n")
            for fpath, slice_names in sorted(all_files.items()):
                task_lines.append(f"| `{fpath}` | {', '.join(slice_names)} |\n")

        (tasks_path / "tasks.md").write_text("".join(task_lines), encoding="utf-8")
        written.append("tasks.md")

        # ── Write codebase-map.md ──────────────────────────────────────
        (tasks_path / "codebase-map.md").write_text(
            f"# Codebase Map\n\n{codebase_map.strip()}\n",
            encoding="utf-8",
        )
        written.append("codebase-map.md")

        total = sum((tasks_path / f).stat().st_size for f in written if (tasks_path / f).exists())
        return (
            f"Tasks artifacts written to {self.tasks_dir}/: {', '.join(written)}. "
            f"Total: {len(written)} files, {total:,} bytes. "
            f"PHASE COMPLETE — no further tool calls needed."
        )

    async def _arun(self, **kwargs: Any) -> str:
        return self._run(**kwargs)


# ── Factory ───────────────────────────────────────────────────────────────


def build_tasks_agent_tools(
    workspace_root: str,
    work_id: str,
    prior_phase_dirs: dict[str, str] | None = None,
    description: str = "",
    work_type: str = "",
    feedback: list[str] | None = None,
) -> list[BaseTool]:
    """Build the custom tool set for the tasks agent.

    .. deprecated::
        The TASKS phase tools are deprecated. Use the PLAN phase tools
        for decomposition instead.

    Returns three tools:
    - ``read_prior_artifacts`` — loads spec/plan artifacts (spec workflows)
    - ``search_codebase`` — multi-query targeted file search
    - ``write_tasks_artifacts`` — atomic write of all tasks artifacts

    Together with ``task`` (SubAgentMiddleware) and ``eval``
    (CodeInterpreterMiddleware), these are the complete tool surface.
    No generic filesystem tools are exposed.

    Args:
        workspace_root: Absolute path to the project workspace root.
        work_id: The current work item ID.
        prior_phase_dirs: Phase → artifact dir mapping for prior phases.
        description: Work description (for read_prior_artifacts context).
        work_type: Work type string.
        feedback: Prior rework feedback strings.

    Returns:
        List of three BaseTool instances.
    """
    tasks_dir = f".spine/artifacts/{work_id}/tasks"

    return [
        ReadPriorArtifactsTool(
            workspace_root=workspace_root,
            work_id=work_id,
            work_type=work_type,
            description=description,
            feedback=feedback or [],
            plan_dir=tasks_dir,  # repurposed: tasks dir is the output dir
            prior_phase_dirs=prior_phase_dirs or {},
        ),
        SearchCodebaseTool(
            workspace_root=workspace_root,
        ),
        WriteTasksArtifactsTool(
            workspace_root=workspace_root,
            tasks_dir=tasks_dir,
        ),
    ]
