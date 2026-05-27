"""Custom tools for the IMPLEMENT phase orchestrator.

Replaces generic filesystem tools (ls, read_file, glob, grep, write_file)
with purpose-built tools that enforce the orchestrator's dispatch-only role.

Design rationale:
- The orchestrator's job is exactly two things: (1) read slice definitions
  and hand them to subagents, (2) write implementation artifacts (implementation.md
  and implementation.json) from the results.
- Giving it generic filesystem tools lets a weak model fall back to doing
  the implementation itself (read file → edit file → read file → ...).
- ``read_slice_files`` bundles everything the orchestrator needs in one call,
  eliminating multi-turn exploration entirely.
- ``write_implementation_report`` is the only write surface: it only accepts
  a structured report dict and writes to the fixed implementation.md and
  implementation.json paths. The orchestrator cannot write source files even
  if it tries.

Both tools are LangChain ``BaseTool`` subclasses so they slot directly into
``create_agent(tools=[...])`` or any middleware tool list.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field

from spine.agents.artifacts import artifact_path
from spine.agents.tools._fs import _atomic_write

logger = logging.getLogger(__name__)


# ── read_slice_files ──────────────────────────────────────────────────────


class _ReadSliceFilesInput(BaseModel):
    """No inputs needed — paths are fully determined by work_id at build time."""

    model_config = {"extra": "forbid"}


class _SliceResultItem(BaseModel):
    """Per-slice result for implementation reporting."""

    slice_name: str = Field(min_length=1, description="Slice identifier")
    status: str = Field(
        pattern="^(implemented|partial|blocked)$",
        description="Completion status: implemented, partial, or blocked",
    )
    files_modified: list[str] = Field(
        default_factory=list, description="Files that were modified"
    )
    files_created: list[str] = Field(
        default_factory=list, description="Files that were created"
    )
    test_results: str = Field(
        default="", description="Summary of test/lint outcomes for this slice"
    )
    issues: list[str] = Field(
        default_factory=list, description="Unresolved issues or blockers"
    )


class _WriteImplementationReportInput(BaseModel):
    slice_results: list[_SliceResultItem] = Field(
        min_length=1,
        description="List of result objects, one per slice. Each must have slice_name, status, files_modified, files_created, test_results, issues.",
    )
    summary: str = Field(
        min_length=10,
        description="Overall implementation summary: what was accomplished, any cross-slice issues, and overall readiness for verification.",
    )


class ReadSliceFilesTool(BaseTool):
    """Read all feature slice definitions and the codebase map in one call.

    Reads ``plan.json`` from the plan artifact directory and extracts the
    ``feature_slices`` array as per-slice definitions, plus the embedded
    ``codebase_map`` field.

    Returns a JSON object containing:
    - ``slices``: mapping of slice id → full feature-slice dict (id, title,
      target_files, execution_requirements, dependencies, acceptance_criteria,
      complexity)
    - ``codebase_map``: content of the codebase_map field from plan.json
      (empty string if missing)
    - ``slice_count``: number of slices found
    - ``plan_dir``: the plan artifact directory path

    This is the only read tool the orchestrator needs. It eliminates
    multi-turn ls/glob/read_file exploration — everything is loaded at once.
    """

    name: str = "read_slice_files"
    description: str = (
        "Read all feature slice definitions and the codebase map for this work item. "
        "Loads plan.json from the plan directory and returns all feature_slices "
        "plus the codebase map in one call. "
        "Call this FIRST — it gives you everything you need to dispatch subagents. "
        "No arguments required."
    )
    args_schema: Optional[ArgsSchema] = _ReadSliceFilesInput

    # Injected at build time — not part of the input schema
    plan_dir: str = ""
    workspace_root: str = ""

    def _run(self, **kwargs: Any) -> str:  # noqa: ARG002
        plan_path = Path(self.workspace_root) / self.plan_dir
        result: dict[str, Any] = {
            "plan_dir": self.plan_dir,
            "slice_count": 0,
            "slices": {},
            "codebase_map": "",
        }

        if not plan_path.exists():
            result["error"] = f"Plan directory not found: {self.plan_dir}"
            return json.dumps(result)

        # Read plan.json
        json_path = plan_path / "plan.json"
        if not json_path.exists():
            result["error"] = f"plan.json not found in {self.plan_dir}"
            return json.dumps(result)

        try:
            plan_data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            result["error"] = f"Could not parse plan.json: {exc}"
            return json.dumps(result)

        # Extract codebase_map
        codebase_map = plan_data.get("codebase_map", "")
        if isinstance(codebase_map, dict):
            # Serialize dicts back to a string so the return shape is consistent
            codebase_map = json.dumps(codebase_map, indent=2, ensure_ascii=False)
        result["codebase_map"] = codebase_map or ""

        # Extract feature_slices and index by slice id
        feature_slices = plan_data.get("feature_slices", [])
        if not isinstance(feature_slices, list):
            result["warning"] = "plan.json has no feature_slices array"
            return json.dumps(result)

        for sl in feature_slices:
            if not isinstance(sl, dict):
                continue
            slice_id = sl.get("id")
            if not slice_id:
                continue
            result["slices"][slice_id] = sl

        result["slice_count"] = len(result["slices"])

        if result["slice_count"] == 0:
            result["warning"] = (
                "plan.json has no feature_slices (or they lack 'id' fields). "
                "Check that the PLAN phase completed successfully."
            )

        return json.dumps(result, ensure_ascii=False)

    async def _arun(self, **kwargs: Any) -> str:
        return self._run(**kwargs)


# ── write_implementation_report ───────────────────────────────────────────


class WriteImplementationReportTool(BaseTool):
    """Write the implementation.md synthesis report and implementation.json data.

    This is the ONLY write tool available to the orchestrator.
    It accepts structured results from slice-implementer subagents and
    writes them to the fixed implementation.md and implementation.json paths.
    The orchestrator cannot write source files — this tool only touches
    implementation artifacts (implementation.md, implementation.json).
    """

    name: str = "write_implementation_report"
    description: str = (
        "Write the implementation.md synthesis report and implementation.json data "
        "from slice-implementer results. "
        "This is the ONLY write tool available to you. "
        "Call this after all subagents have completed to produce the phase artifact. "
        "Requires: slice_results (list of per-slice result dicts with slice_name, "
        "status, files_modified, files_created, test_results, issues) and summary (str, min 10 chars)."
    )
    args_schema: Optional[ArgsSchema] = _WriteImplementationReportInput

    # Injected at build time
    impl_dir: str = ""
    workspace_root: str = ""

    def _run(self, slice_results: list[_SliceResultItem], summary: str) -> str:
        impl_path = Path(self.workspace_root) / self.impl_dir
        impl_path.mkdir(parents=True, exist_ok=True)

        report_path = impl_path / "implementation.md"

        lines: list[str] = [
            "# Implementation Report\n",
            f"## Summary\n\n{summary}\n",
            "## Slice Results\n",
        ]

        all_modified: list[str] = []
        all_created: list[str] = []
        blocked: list[str] = []
        partial: list[str] = []
        json_slices: list[dict[str, Any]] = []

        for sr in slice_results:
            # Support both dict (for testing) and _SliceResultItem (validated input)
            if isinstance(sr, dict):
                name = sr.get("slice_name", "unknown")
                status = sr.get("status", "unknown")
                modified = sr.get("files_modified", [])
                created = sr.get("files_created", [])
                test_res = sr.get("test_results", "")
                issues = sr.get("issues", [])
            else:
                name = sr.slice_name
                status = sr.status
                modified = sr.files_modified
                created = sr.files_created
                test_res = sr.test_results
                issues = sr.issues

            status_icon = {"implemented": "✅", "partial": "⚠️", "blocked": "❌"}.get(status, "❓")
            lines.append(f"### {status_icon} {name} — {status}\n")

            if modified:
                lines.append("**Files modified:**\n")
                lines.extend(f"- `{f}`\n" for f in modified)
                all_modified.extend(modified)
            if created:
                lines.append("**Files created:**\n")
                lines.extend(f"- `{f}`\n" for f in created)
                all_created.extend(created)
            if test_res:
                lines.append(f"**Tests:** {test_res}\n")
            if issues:
                lines.append("**Issues:**\n")
                lines.extend(f"- {i}\n" for i in issues)
            lines.append("\n")

            if status == "blocked":
                blocked.append(name)
            elif status == "partial":
                partial.append(name)

            # Collect for implementation.json
            json_slices.append({
                "slice_name": name,
                "status": status,
                "files_modified": modified,
                "files_created": created,
                "test_results": test_res,
                "issues": issues,
            })

        # Aggregated file list
        if all_modified or all_created:
            lines.append("## Files Changed\n")
            if all_modified:
                lines.append("**Modified:**\n")
                lines.extend(f"- `{f}`\n" for f in sorted(set(all_modified)))
            if all_created:
                lines.append("**Created:**\n")
                lines.extend(f"- `{f}`\n" for f in sorted(set(all_created)))
            lines.append("\n")

        # Status summary
        total = len(slice_results)
        implemented = total - len(blocked) - len(partial)
        lines.append("## Status\n")
        lines.append(f"- Total slices: {total}\n")
        lines.append(f"- Implemented: {implemented}\n")
        if partial:
            lines.append(f"- Partial: {len(partial)} ({', '.join(partial)})\n")
        if blocked:
            lines.append(f"- Blocked: {len(blocked)} ({', '.join(blocked)})\n")

        # Write implementation.json with the same structured data
        json_data: dict[str, Any] = {
            "summary": summary,
            "slice_results": json_slices,
        }
        json_path = impl_path / "implementation.json"
        json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
        try:
            _atomic_write(json_path, json_str)
        except OSError as exc:
            return f"ERROR: Could not write implementation.json: {exc}"

        content = "".join(lines)
        try:
            _atomic_write(report_path, content)
        except OSError as exc:
            return f"ERROR: Could not write implementation.md: {exc}"

        return (
            f"implementation.md written to {self.impl_dir}/implementation.md "
            f"({len(content)} chars, {total} slices, {implemented} implemented, "
            f"{len(partial)} partial, {len(blocked)} blocked) "
            f"+ implementation.json ({len(json_str)} bytes)."
        )

    async def _arun(self, slice_results: list[_SliceResultItem] | list[dict[str, Any]], summary: str) -> str:
        # Support both dict (for testing) and _SliceResultItem (validated input)
        typed_results: list[_SliceResultItem] = []
        for sr in slice_results:
            if isinstance(sr, dict):
                typed_results.append(_SliceResultItem(
                    slice_name=sr.get("slice_name", "unknown"),
                    status=sr.get("status", "implemented"),
                    files_modified=sr.get("files_modified", []),
                    files_created=sr.get("files_created", []),
                    test_results=sr.get("test_results", ""),
                    issues=sr.get("issues", []),
                ))
            else:
                typed_results.append(sr)
        return self._run(slice_results=typed_results, summary=summary)


# ── Factory ───────────────────────────────────────────────────────────────


def build_implement_orchestrator_tools(
    workspace_root: str,
    work_id: str,
) -> list[BaseTool]:
    """Build the custom tool set for the implement orchestrator.

    Returns exactly two tools:
    - ``read_slice_files``: loads all slice definitions + codebase map in one call
    - ``write_implementation_report``: writes the implementation.md artifact

    These are the complete tool surface for the IMPLEMENT orchestrator.
    No generic filesystem tools are exposed — the orchestrator physically
    cannot read arbitrary files or write source code. Slice-implementer
    subagents are dispatched per slice by the implement subgraph router
    via the LangGraph Send API.

    Args:
        workspace_root: Absolute path to the project workspace root.
        work_id: The current work item ID (e.g. ``"30f58813"``).

    Returns:
        List of two BaseTool instances ready for use.
    """
    plan_dir = artifact_path(work_id, "plan")
    impl_dir = artifact_path(work_id, "implement")

    return [
        ReadSliceFilesTool(
            workspace_root=workspace_root,
            plan_dir=plan_dir,
        ),
        WriteImplementationReportTool(
            workspace_root=workspace_root,
            impl_dir=impl_dir,
        ),
    ]


# ── Module-level utility (for subgraph synthesis nodes) ────────────────────


def write_implementation_files(
    slice_results: list[dict[str, Any]],
    summary: str,
    workspace_root: str,
    impl_dir: str,
) -> str:
    """Write implementation.md and implementation.json to disk.

    This is the module-level utility that synthesis nodes call directly
    without instantiating the ``BaseTool`` subclass.  Used by the
    ``_synthesize_implementation_node`` in the implement subgraph
    instead of routing through a LangChain tool call.

    Args:
        slice_results: List of per-slice result dicts (slice_name, status,
            files_modified, files_created, test_results, issues).
        summary: Overall implementation summary string.
        workspace_root: Absolute path to the project workspace root.
        impl_dir: Relative artifact directory (e.g. ``".spine/artifacts/<id>/implement"``).

    Returns:
        Status string from ``WriteImplementationReportTool._run()``.
    """
    tool = WriteImplementationReportTool(
        workspace_root=workspace_root,
        impl_dir=impl_dir,
    )
    # Convert dicts to _SliceResultItem for validated processing
    typed_results: list[_SliceResultItem] = []
    for sr in slice_results:
        typed_results.append(_SliceResultItem(
            slice_name=sr.get("slice_name", "unknown"),
            status=sr.get("status", "implemented"),
            files_modified=sr.get("files_modified", []),
            files_created=sr.get("files_created", []),
            test_results=sr.get("test_results", ""),
            issues=sr.get("issues", []),
        ))
    return tool._run(slice_results=typed_results, summary=summary)
