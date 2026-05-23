"""Custom tools for the VERIFY phase orchestrator.

Replaces generic filesystem tools (ls, read_file, glob, grep, write_file)
with purpose-built tools that enforce the orchestrator's dispatch-only role.

Design rationale:
- The orchestrator's job is exactly two things: (1) read verification inputs
  (slices, plan, implementation) and hand them to subagents, (2) write
  verification.md from the subagent results.
- Giving it generic filesystem tools lets a weak model fall back to doing
  the verification itself (read file → run tests inline → ...).
- `read_verify_context` bundles everything the orchestrator needs in one call,
  eliminating multi-turn exploration entirely.
- `write_verification_report` is the only write surface: it only accepts
  structured verification results and writes to the fixed verification.md path.
  The orchestrator cannot write source files even if it tries.

Both tools are LangChain `BaseTool` subclasses so they slot directly into
`create_agent(tools=[...])` or any middleware tool list.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field

from spine.agents.artifacts import artifact_path, list_slice_files

logger = logging.getLogger(__name__)


# ── read_verify_context ───────────────────────────────────────────────────


class _ReadVerifyContextInput(BaseModel):
    """No inputs needed — paths are fully determined by work_id at build time."""


class ReadVerifyContextTool(BaseTool):
    """Read all verification inputs: slices, plan, codebase map, and implementation.

    Reads from multiple artifact directories and returns a unified JSON object:
    - ``slices``: mapping of slice filename → full slice content from tasks/
    - ``plan_dir``: the plan artifact directory path
    - ``codebase_map``: content of codebase_map field from plan.json (empty if missing)
    - ``implementation``: full content of implementation.md (empty string if missing)
    - ``impl_dir``: the implementation artifact directory path
    - ``slice_count``: number of slice files found
    - ``verify_dir``: the verify artifact directory path

    This is the only read tool the orchestrator needs. It eliminates
    multi-turn ls/glob/read_file exploration — everything is loaded at once.
    """

    name: str = "read_verify_context"
    description: str = (
        "Read all verification inputs: slice definitions, codebase map, and "
        "implementation report in one call. Loads plan.json from the plan "
        "directory, slice files from tasks/, and implementation.md from implement/. "
        "Call this FIRST — it gives you everything you need to dispatch subagents. "
        "No arguments required."
    )
    args_schema: Optional[ArgsSchema] = _ReadVerifyContextInput

    # Injected at build time — not part of the input schema
    plan_dir: str = ""
    tasks_dir: str = ""
    impl_dir: str = ""
    verify_dir: str = ""
    workspace_root: str = ""

    def _run(self, **kwargs: Any) -> str:  # noqa: ARG002
        workspace = Path(self.workspace_root)
        result: dict[str, Any] = {
            "plan_dir": self.plan_dir,
            "tasks_dir": self.tasks_dir,
            "impl_dir": self.impl_dir,
            "verify_dir": self.verify_dir,
            "slice_count": 0,
            "slices": {},
            "codebase_map": "",
            "implementation": "",
        }

        # Load plan.json for codebase_map
        plan_path = workspace / self.plan_dir / "plan.json"
        if plan_path.exists():
            try:
                plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
                codebase_map = plan_data.get("codebase_map", "")
                if isinstance(codebase_map, dict):
                    codebase_map = json.dumps(codebase_map, indent=2, ensure_ascii=False)
                result["codebase_map"] = codebase_map or ""
            except (json.JSONDecodeError, OSError) as exc:
                result["plan_error"] = f"Could not parse plan.json: {exc}"
        else:
            result["plan_error"] = f"plan.json not found in {self.plan_dir}"

        # Load implementation.md
        impl_path = workspace / self.impl_dir / "implementation.md"
        if impl_path.exists():
            try:
                result["implementation"] = impl_path.read_text(encoding="utf-8")
            except OSError as exc:
                result["impl_error"] = f"Could not read implementation.md: {exc}"
        else:
            result["impl_error"] = f"implementation.md not found in {self.impl_dir}"

        # Load slice files from tasks directory
        tasks_path = workspace / self.tasks_dir
        slice_files = list_slice_files(self.workspace_root, "")
        # Filter to just the filenames for the tasks_dir (work_id already in path)
        if tasks_path.exists():
            try:
                for slice_file in sorted(tasks_path.glob("slice-*.md")):
                    if slice_file.is_file():
                        result["slices"][slice_file.name] = slice_file.read_text(encoding="utf-8")
            except OSError as exc:
                result["slices_error"] = f"Could not list slice files: {exc}"

        result["slice_count"] = len(result["slices"])

        if result["slice_count"] == 0:
            result["warning"] = (
                "No slice-*.md files found in tasks/ directory. "
                "Check that the TASKS phase completed successfully."
            )

        return json.dumps(result, ensure_ascii=False)

    async def _arun(self, **kwargs: Any) -> str:
        return self._run(**kwargs)


# ── read_slice_files (verify variant) ─────────────────────────────────────


class _ReadSliceFilesVerifyInput(BaseModel):
    """No inputs needed — paths are fully determined by work_id at build time."""


class ReadSliceFilesVerifyTool(BaseTool):
    """Read all feature slice definitions for the VERIFY phase.

    Similar to implement's ReadSliceFilesTool but reads slice-*.md files
    directly from the tasks artifact directory instead of plan.json.
    This is useful when the orchestrator needs just the slices without
    implementation or plan context.

    Returns a JSON object containing:
    - ``slices``: mapping of slice filename → full slice content
    - ``slice_count``: number of slices found
    - ``tasks_dir``: the tasks artifact directory path
    """

    name: str = "read_slice_files_verify"
    description: str = (
        "Read all slice definitions from the tasks directory for verification. "
        "Loads each slice-*.md file and returns their contents. "
        "No arguments required."
    )
    args_schema: Optional[ArgsSchema] = _ReadSliceFilesVerifyInput

    # Injected at build time
    tasks_dir: str = ""
    workspace_root: str = ""

    def _run(self, **kwargs: Any) -> str:  # noqa: ARG002
        result: dict[str, Any] = {
            "tasks_dir": self.tasks_dir,
            "slice_count": 0,
            "slices": {},
        }

        tasks_path = Path(self.workspace_root) / self.tasks_dir
        if not tasks_path.exists():
            result["error"] = f"Tasks directory not found: {self.tasks_dir}"
            return json.dumps(result)

        try:
            for slice_file in sorted(tasks_path.glob("slice-*.md")):
                if slice_file.is_file():
                    result["slices"][slice_file.name] = slice_file.read_text(encoding="utf-8")
        except OSError as exc:
            result["error"] = f"Could not list slice files: {exc}"

        result["slice_count"] = len(result["slices"])

        return json.dumps(result, ensure_ascii=False)

    async def _arun(self, **kwargs: Any) -> str:
        return self._run(**kwargs)


# ── write_verification_report ─────────────────────────────────────────────


class _VerificationResult(BaseModel):
    """A single slice verification result from a subagent."""

    slice_name: str = Field(description="The slice filename (e.g., 'slice-foo.md')")
    verdict: str = Field(description="One of: VERIFIED, NOT_VERIFIED")
    checklist: list[dict[str, Any]] = Field(
        description="List of check items with criterion, passed, detail"
    )
    gaps: list[str] = Field(description="Gaps or missing items found")
    recommendations: list[str] = Field(description="Suggested improvements")


class _WriteVerificationReportInput(BaseModel):
    verification_results: list[_VerificationResult] = Field(
        description=(
            "List of verification result objects, one per slice. Each must have: "
            "'slice_name' (str), 'verdict' (VERIFIED|NOT_VERIFIED), 'checklist' "
            "(list of {criterion, passed, detail}), 'gaps' (list[str]), "
            "'recommendations' (list[str])."
        )
    )
    summary: str = Field(
        description=(
            "Overall verification summary: what was verified, any cross-slice "
            "issues found, and overall status."
        )
    )


class WriteVerificationReportTool(BaseTool):
    """Write the verification.md synthesis report.

    This is the ONLY write tool available to the orchestrator.
    It accepts structured results from slice-verifier subagents and
    writes them to the fixed verification.md path. The orchestrator
    cannot write source files — this tool only touches verification.md.
    """

    name: str = "write_verification_report"
    description: str = (
        "Write the verification.md synthesis report from slice-verifier results. "
        "This is the ONLY write tool available — you cannot write other files. "
        "Provide verification_results (list of per-slice verdict dicts) and summary. "
        "Call this after all subagents have completed to produce the phase artifact."
    )
    args_schema: Optional[ArgsSchema] = _WriteVerificationReportInput

    # Injected at build time
    verify_dir: str = ""
    workspace_root: str = ""

    def _run(
        self, verification_results: list[dict[str, Any]], summary: str
    ) -> str:
        verify_path = Path(self.workspace_root) / self.verify_dir
        verify_path.mkdir(parents=True, exist_ok=True)

        report_path = verify_path / "verification.md"

        # Determine overall status
        all_verified = all(r.get("verdict") == "VERIFIED" for r in verification_results)
        overall_status = "VERIFIED" if all_verified else "FAILED"

        lines: list[str] = [
            f"# Verification Report — {overall_status}\n",
            f"## Summary\n\n{summary}\n",
            "## Slice Verification Results\n",
        ]

        verified_count = 0
        not_verified_count = 0
        all_gaps: list[str] = []
        all_recommendations: list[str] = []

        for vr in verification_results:
            name = vr.get("slice_name", "unknown")
            verdict = vr.get("verdict", "unknown")
            checklist = vr.get("checklist", [])
            gaps = vr.get("gaps", [])
            recommendations = vr.get("recommendations", [])

            status_icon = "✅" if verdict == "VERIFIED" else "❌"
            lines.append(f"### {status_icon} {name} — {verdict}\n")

            # Checklist items
            for item in checklist:
                criterion = item.get("criterion", "unknown")
                passed = item.get("passed", False)
                detail = item.get("detail", "")
                check_icon = "✓" if passed else "✗"
                lines.append(f"- {check_icon} **{criterion}**: {detail}\n")

            # Gaps
            if gaps:
                lines.append("\n**Gaps:**\n")
                for gap in gaps:
                    lines.append(f"  - {gap}\n")
                all_gaps.extend(f"{name}: {gap}" for gap in gaps)

            # Recommendations
            if recommendations:
                lines.append("\n**Recommendations:**\n")
                for rec in recommendations:
                    lines.append(f"  - {rec}\n")
                all_recommendations.extend(recommendations)

            lines.append("\n")

            if verdict == "VERIFIED":
                verified_count += 1
            else:
                not_verified_count += 1

        # Status summary
        total = len(verification_results)
        lines.append("## Status Summary\n")
        lines.append(f"- Total slices: {total}\n")
        lines.append(f"- Verified: {verified_count}\n")
        if not_verified_count:
            lines.append(f"- Not Verified: {not_verified_count}\n")

        # Aggregated gaps
        if all_gaps:
            lines.append("\n## Aggregated Gaps\n")
            for gap in all_gaps:
                lines.append(f"- {gap}\n")

        # Aggregated recommendations
        if all_recommendations:
            lines.append("\n## Recommendations\n")
            for rec in all_recommendations:
                lines.append(f"- {rec}\n")

        content = "".join(lines)

        try:
            report_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            return f"ERROR: Could not write verification.md: {exc}"

        return (
            f"verification.md written to {self.verify_dir}/verification.md "
            f"({len(content)} chars, {total} slices, {verified_count} verified, "
            f"{not_verified_count} not verified)."
        )

    async def _arun(
        self, verification_results: list[dict[str, Any]], summary: str
    ) -> str:
        return self._run(verification_results, summary)


# ── Factory ───────────────────────────────────────────────────────────────


def build_verify_orchestrator_tools(
    workspace_root: str,
    work_id: str,
) -> list[BaseTool]:
    """Build the custom tool set for the verify orchestrator.

    Returns exactly two tools:
    - ``read_verify_context``: loads all verification inputs in one call
    - ``write_verification_report``: writes the verification.md artifact

    Together with ``task`` (from SubAgentMiddleware) and ``eval`` (from
    CodeInterpreterMiddleware), these are all the tools the orchestrator needs.
    No generic filesystem tools are exposed — the orchestrator physically
    cannot read arbitrary files or write source code.

    Args:
        workspace_root: Absolute path to the project workspace root.
        work_id: The current work item ID (e.g. ``"30f58813"``).

    Returns:
        List of BaseTool instances ready for use.
    """
    plan_dir = artifact_path(work_id, "plan")
    tasks_dir = artifact_path(work_id, "tasks")
    impl_dir = artifact_path(work_id, "implement")
    verify_dir = artifact_path(work_id, "verify")

    return [
        ReadVerifyContextTool(
            workspace_root=workspace_root,
            plan_dir=plan_dir,
            tasks_dir=tasks_dir,
            impl_dir=impl_dir,
            verify_dir=verify_dir,
        ),
        WriteVerificationReportTool(
            workspace_root=workspace_root,
            verify_dir=verify_dir,
        ),
    ]