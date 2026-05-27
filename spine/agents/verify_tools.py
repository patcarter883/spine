"""Custom tools for the VERIFY phase orchestrator.

The orchestrator's job is exactly two things: (1) read verification inputs
(slices, plan, implementation) and hand them to subagents, (2) write
verification.md and verification.json from the subagent results.

Two purpose-built tools enforce dispatch-only behavior:
- ``read_verify_context`` — loads everything in one call, eliminating multi-turn exploration
- ``write_verification_report`` — only write surface; orchestrator cannot write source files

Both tools are LangChain ``BaseTool`` subclasses that slot directly into
``create_agent(tools=[...])``.
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

logger = logging.getLogger(__name__)


# ── read_verify_context ───────────────────────────────────────────────────


class _ReadVerifyContextInput(BaseModel):
    """No inputs needed — paths are fully determined by work_id at build time."""


class ReadVerifyContextTool(BaseTool):
    """Read all verification inputs: slices, plan, codebase map, and implementation."""

    name: str = "read_verify_context"
    description: str = (
        "Read all verification inputs: structured slice definitions from plan.json, "
        "codebase map, and structured implementation results from implementation.json "
        "in one call. Call this FIRST — it gives you everything you need to dispatch "
        "subagents. No arguments required."
    )
    args_schema: Optional[ArgsSchema] = _ReadVerifyContextInput

    # Injected at build time — not part of the input schema
    plan_dir: str = ""
    impl_dir: str = ""
    verify_dir: str = ""
    workspace_root: str = ""

    def _run(self, **kwargs: Any) -> str:  # noqa: ARG002
        workspace = Path(self.workspace_root)
        result: dict[str, Any] = {
            "plan_dir": self.plan_dir,
            "impl_dir": self.impl_dir,
            "verify_dir": self.verify_dir,
            "slice_count": 0,
            "slices": {},
            "codebase_map": "",
            "implementation": {},
        }

        # Load plan.json for structured slice definitions + codebase_map
        plan_path = workspace / self.plan_dir / "plan.json"
        if plan_path.exists():
            try:
                plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
                # Store structured slices (dicts, not markdown strings)
                for sl in plan_data.get("feature_slices", []):
                    if isinstance(sl, dict) and sl.get("id"):
                        result["slices"][sl["id"]] = sl  # structured dict, not markdown
                # Store codebase_map
                result["codebase_map"] = plan_data.get("codebase_map", "")
            except (json.JSONDecodeError, OSError) as exc:
                result["plan_error"] = str(exc)
        else:
            result["plan_error"] = f"plan.json not found in {self.plan_dir}"

        # Load implementation.json for structured results
        impl_json_path = workspace / self.impl_dir / "implementation.json"
        if impl_json_path.exists():
            try:
                result["implementation"] = json.loads(impl_json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                result["impl_error"] = str(exc)
        else:
            result["impl_error"] = f"implementation.json not found in {self.impl_dir}"

        result["slice_count"] = len(result["slices"])

        if result["slice_count"] == 0:
            result["warning"] = (
                "No feature_slices found in plan.json. "
                "Check that the PLAN phase completed successfully."
            )

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
    """Write verification.md and verification.json synthesis reports."""

    name: str = "write_verification_report"
    description: str = (
        "Write verification.md and verification.json synthesis reports from "
        "slice-verifier results. Provide verification_results (list of per-slice "
        "result dicts) and summary. Both files are written to the verify directory. "
        "Call this after all subagents have completed."
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
        slice_data: list[dict[str, Any]] = []

        for vr in verification_results:
            name = vr.get("slice_name", "unknown")
            verdict = vr.get("verdict", "unknown")
            checklist = vr.get("checklist", [])
            gaps = vr.get("gaps", [])
            recommendations = vr.get("recommendations", [])

            slice_data.append({
                "slice_name": name,
                "verdict": verdict,
                "checklist": checklist,
                "gaps": gaps,
                "recommendations": recommendations,
            })

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

        # Write verification.json
        verify_json_path = verify_path / "verification.json"
        json_data = {
            "summary": summary,
            "overall_status": overall_status,
            "verification_results": slice_data,
        }
        try:
            verify_json_path.write_text(
                json.dumps(json_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            return f"ERROR: Could not write verification.json: {exc}"

        json_bytes = verify_json_path.stat().st_size

        return (
            f"verification.md written to {self.verify_dir}/verification.md "
            f"({len(content)} chars, {total} slices, {verified_count} verified, "
            f"{not_verified_count} not verified) + verification.json ({json_bytes} bytes)."
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
    - ``write_verification_report``: writes the verification.md and verification.json artifacts

    These are the complete tool surface for the VERIFY orchestrator.
    No generic filesystem tools are exposed. Slice-verifier subagents are
    dispatched per slice by the verify subgraph router via the LangGraph
    Send API.

    Args:
        workspace_root: Absolute path to the project workspace root.
        work_id: The current work item ID (e.g. ``"30f58813"``).

    Returns:
        List of BaseTool instances ready for use.
    """
    plan_dir = artifact_path(work_id, "plan")
    impl_dir = artifact_path(work_id, "implement")
    verify_dir = artifact_path(work_id, "verify")

    return [
        ReadVerifyContextTool(
            workspace_root=workspace_root,
            plan_dir=plan_dir,
            impl_dir=impl_dir,
            verify_dir=verify_dir,
        ),
        WriteVerificationReportTool(
            workspace_root=workspace_root,
            verify_dir=verify_dir,
        ),
    ]


# ── Module-level utility (for subgraph synthesis nodes) ────────────────────


def write_verification_files(
    verification_results: list[dict[str, Any]],
    summary: str,
    workspace_root: str,
    verify_dir: str,
) -> str:
    """Write verification.md and verification.json to disk.

    Module-level utility that synthesis nodes call directly without
    instantiating the ``BaseTool`` subclass.  Used by the
    ``_synthesize_verification_node`` in the verify subgraph.

    Args:
        verification_results: List of per-slice verdict dicts (slice_name,
            verdict, checklist, gaps, recommendations).
        summary: Overall verification summary string.
        workspace_root: Absolute path to the project workspace root.
        verify_dir: Relative artifact directory (e.g. ``".spine/artifacts/<id>/verify"``).

    Returns:
        Status string from ``WriteVerificationReportTool._run()``.
    """
    tool = WriteVerificationReportTool(
        workspace_root=workspace_root,
        verify_dir=verify_dir,
    )
    return tool._run(verification_results=verification_results, summary=summary)