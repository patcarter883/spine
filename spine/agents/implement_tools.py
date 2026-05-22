"""Custom tools for the IMPLEMENT phase orchestrator.

Replaces generic filesystem tools (ls, read_file, glob, grep, write_file)
with purpose-built tools that enforce the orchestrator's dispatch-only role.

Design rationale:
- The orchestrator's job is exactly two things: (1) read slice definitions
  and hand them to subagents, (2) write implementation.md from the results.
- Giving it generic filesystem tools lets a weak model fall back to doing
  the implementation itself (read file → edit file → read file → ...).
- ``read_slice_files`` bundles everything the orchestrator needs in one call,
  eliminating multi-turn exploration entirely.
- ``write_implementation_report`` is the only write surface: it only accepts
  a structured report dict and writes to the fixed implementation.md path.
  The orchestrator cannot write source files even if it tries.

Both tools are LangChain ``BaseTool`` subclasses so they slot directly into
``create_agent(tools=[...])`` or any middleware tool list.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Type

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── read_slice_files ──────────────────────────────────────────────────────


class _ReadSliceFilesInput(BaseModel):
    """No inputs needed — paths are fully determined by work_id at build time."""


class ReadSliceFilesTool(BaseTool):
    """Read all slice definitions and the codebase map in one call.

    Returns a JSON object containing:
    - ``slices``: mapping of slice filename → full markdown content
    - ``codebase_map``: content of codebase-map.md (empty string if missing)
    - ``slice_count``: number of slices found
    - ``tasks_dir``: the tasks artifact directory path

    This is the only read tool the orchestrator needs. It eliminates
    multi-turn ls/glob/read_file exploration — everything is loaded at once.
    """

    name: str = "read_slice_files"
    description: str = (
        "Read all feature slice definitions and the codebase map for this work item. "
        "Returns all slice-*.md files and codebase-map.md in one call. "
        "Call this FIRST — it gives you everything you need to dispatch subagents. "
        "No arguments required."
    )
    args_schema: Optional[ArgsSchema] = _ReadSliceFilesInput

    # Injected at build time — not part of the input schema
    tasks_dir: str = ""
    workspace_root: str = ""

    def _run(self, **kwargs: Any) -> str:  # noqa: ARG002
        tasks_path = Path(self.workspace_root) / self.tasks_dir
        result: dict[str, Any] = {
            "tasks_dir": self.tasks_dir,
            "slice_count": 0,
            "slices": {},
            "codebase_map": "",
        }

        if not tasks_path.exists():
            result["error"] = f"Tasks directory not found: {self.tasks_dir}"
            return json.dumps(result)

        # Load codebase map
        map_path = tasks_path / "codebase-map.md"
        if map_path.exists():
            try:
                result["codebase_map"] = map_path.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning("Could not read codebase-map.md: %s", exc)

        # Load all slice files
        try:
            slice_files = sorted(
                p.name
                for p in tasks_path.iterdir()
                if p.is_file() and p.name.startswith("slice-") and p.name.endswith(".md")
            )
        except OSError as exc:
            result["error"] = f"Could not list tasks directory: {exc}"
            return json.dumps(result)

        for fname in slice_files:
            fpath = tasks_path / fname
            try:
                result["slices"][fname] = fpath.read_text(encoding="utf-8")
            except OSError as exc:
                result["slices"][fname] = f"[ERROR reading file: {exc}]"
                logger.warning("Could not read slice file %s: %s", fname, exc)

        result["slice_count"] = len(result["slices"])

        if result["slice_count"] == 0:
            result["warning"] = (
                "No slice-*.md files found in tasks directory. "
                "Check that the TASKS phase completed successfully."
            )

        return json.dumps(result, ensure_ascii=False)

    async def _arun(self, **kwargs: Any) -> str:
        return self._run(**kwargs)


# ── write_implementation_report ───────────────────────────────────────────


class _WriteImplementationReportInput(BaseModel):
    slice_results: list[dict[str, Any]] = Field(
        description=(
            "List of result objects, one per slice. Each must have: "
            "'slice_name' (str), 'status' (implemented|partial|blocked), "
            "'files_modified' (list[str]), 'files_created' (list[str]), "
            "'test_results' (str), 'issues' (list[str])."
        )
    )
    summary: str = Field(
        description=(
            "Overall implementation summary: what was accomplished, "
            "any cross-slice issues, and overall readiness for verification."
        )
    )


class WriteImplementationReportTool(BaseTool):
    """Write the implementation.md synthesis report.

    This is the ONLY write tool available to the orchestrator.
    It accepts structured results from slice-implementer subagents and
    writes them to the fixed implementation.md path. The orchestrator
    cannot write source files — this tool only touches implementation.md.
    """

    name: str = "write_implementation_report"
    description: str = (
        "Write the implementation.md synthesis report from slice-implementer results. "
        "This is the ONLY write tool available to you. "
        "Call this after all subagents have completed to produce the phase artifact. "
        "Requires: slice_results (list of per-slice result dicts) and summary (str)."
    )
    args_schema: Optional[ArgsSchema] = _WriteImplementationReportInput

    # Injected at build time
    impl_dir: str = ""
    workspace_root: str = ""

    def _run(self, slice_results: list[dict[str, Any]], summary: str) -> str:
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

        for sr in slice_results:
            name = sr.get("slice_name", "unknown")
            status = sr.get("status", "unknown")
            modified = sr.get("files_modified", [])
            created = sr.get("files_created", [])
            test_res = sr.get("test_results", "")
            issues = sr.get("issues", [])

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

        content = "".join(lines)
        try:
            report_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            return f"ERROR: Could not write implementation.md: {exc}"

        return (
            f"implementation.md written to {self.impl_dir}/implementation.md "
            f"({len(content)} chars, {total} slices, {implemented} implemented, "
            f"{len(partial)} partial, {len(blocked)} blocked)."
        )

    async def _arun(self, slice_results: list[dict[str, Any]], summary: str) -> str:
        return self._run(slice_results=slice_results, summary=summary)


# ── Factory ───────────────────────────────────────────────────────────────


def build_implement_orchestrator_tools(
    workspace_root: str,
    work_id: str,
) -> list[BaseTool]:
    """Build the custom tool set for the implement orchestrator.

    Returns exactly two tools:
    - ``read_slice_files``: loads all slice definitions + codebase map in one call
    - ``write_implementation_report``: writes the implementation.md artifact

    Together with ``task`` (from SubAgentMiddleware) and ``eval`` (from
    CodeInterpreterMiddleware), these are all the tools the orchestrator needs.
    No generic filesystem tools are exposed — the orchestrator physically
    cannot read arbitrary files or write source code.

    Args:
        workspace_root: Absolute path to the project workspace root.
        work_id: The current work item ID (e.g. ``"30f58813"``).

    Returns:
        List of two BaseTool instances ready for use.
    """
    tasks_dir = f".spine/artifacts/{work_id}/tasks"
    impl_dir = f".spine/artifacts/{work_id}/implement"

    return [
        ReadSliceFilesTool(
            workspace_root=workspace_root,
            tasks_dir=tasks_dir,
        ),
        WriteImplementationReportTool(
            workspace_root=workspace_root,
            impl_dir=impl_dir,
        ),
    ]
