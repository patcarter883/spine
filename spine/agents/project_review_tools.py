"""Custom tools for the project-level adversarial review agent.

The review agent red-teams the completed project holistically: integration
gaps, requirements not truly implemented, hidden design assumptions, and
cross-member contradictions.  Two purpose-built tools enforce the workflow:

- ``read_project_evidence`` — loads the full project evidence (spec,
  aggregator coverage, member verification reports, project_verification.json
  if present) in one call.
- ``write_project_review`` — writes the structured adversarial review result.

File-system read tools (ls/read_file/glob/grep) are available via
FilesystemMiddleware so the agent can inspect implementation files.
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


# ── read_project_evidence ─────────────────────────────────────────────────


class _ReadProjectEvidenceInput(BaseModel):
    """No arguments — all paths determined at build time."""


class ReadProjectEvidenceTool(BaseTool):
    """Load all project-level evidence for the adversarial review in one call.

    Returns a JSON payload with:
    - project spec (requirements, roadmap, objectives)
    - aggregator coverage (per-requirement status, phase status)
    - project_verification.json (if a verification pipeline was run)
    - each member's verification.json (summary view)
    - cross-member file overlap analysis (files written by multiple members)
    """

    name: str = "read_project_evidence"
    description: str = (
        "Load all project evidence for adversarial review: project spec, "
        "aggregator coverage, project verification result (if available), and "
        "member verification reports.  Call this FIRST — it gives you the full "
        "picture needed to red-team the project.  No arguments."
    )
    args_schema: Optional[ArgsSchema] = _ReadProjectEvidenceInput

    # Injected at build time
    spec_dict: dict = Field(default_factory=dict)
    coverage: dict = Field(default_factory=dict)
    member_states: dict = Field(default_factory=dict)
    project_verification: dict | None = Field(default=None)
    workspace_root: str = "."

    def _run(self, **kwargs: Any) -> str:  # noqa: ARG002
        workspace = Path(self.workspace_root)
        spec = self.spec_dict
        member_work_ids: list[str] = spec.get("member_work_ids", [])

        result: dict[str, Any] = {
            "project": {
                "id": spec.get("id"),
                "title": spec.get("title"),
                "summary": spec.get("summary"),
                "objectives": spec.get("objectives", []),
                "requirements": spec.get("requirements", []),
                "constraints": spec.get("constraints", []),
                "hard_boundaries": spec.get("hard_boundaries", []),
                "roadmap": spec.get("roadmap", {}),
                "member_count": len(member_work_ids),
            },
            "aggregator_coverage": self.coverage,
            "project_verification": self.project_verification,
            "members": [],
            "file_overlap": {},
        }

        files_by_member: dict[str, list[str]] = {}
        for wid in member_work_ids:
            member: dict[str, Any] = {
                "work_id": wid,
                "verification_summary": None,
                "files_written": [],
                "verification_passed": False,
            }

            # Load verification.json artifact (summary only — limit size)
            verify_json_path = workspace / artifact_path(wid, "verify") / "verification.json"
            if verify_json_path.exists():
                try:
                    vdata = json.loads(verify_json_path.read_text(encoding="utf-8"))
                    member["verification_summary"] = {
                        "overall_status": vdata.get("overall_status"),
                        "summary": vdata.get("summary"),
                        "slice_count": len(vdata.get("verification_results", [])),
                        "failed_slices": [
                            r.get("slice_name")
                            for r in vdata.get("verification_results", [])
                            if r.get("verdict") not in ("VERIFIED", "passed")
                        ],
                    }
                except (OSError, json.JSONDecodeError) as exc:
                    member["verification_error"] = str(exc)

            # Load checkpoint state
            state = self.member_states.get(wid)
            if state:
                files_written = state.get("files_written") or []
                member["files_written"] = files_written
                member["verification_passed"] = bool(state.get("verification_passed"))
                files_by_member[wid] = files_written

            result["members"].append(member)

        # Compute file overlap (files written by multiple members)
        from collections import defaultdict
        file_owners: dict[str, list[str]] = defaultdict(list)
        for wid, files in files_by_member.items():
            for f in files:
                file_owners[f].append(wid)
        result["file_overlap"] = {
            f: owners for f, owners in file_owners.items() if len(owners) > 1
        }

        return json.dumps(result, ensure_ascii=False, indent=2)

    async def _arun(self, **kwargs: Any) -> str:
        return self._run(**kwargs)


# ── write_project_review ──────────────────────────────────────────────────


class _FindingInput(BaseModel):
    severity: str = Field(
        description="CRITICAL, HIGH, MEDIUM, or LOW."
    )
    category: str = Field(
        description=(
            "Category: integration_gap, unimplemented_requirement, "
            "design_contradiction, security_concern, performance_concern, or other."
        )
    )
    requirement_id: str = Field(
        default="",
        description="Related requirement ID (e.g. 'R-002'), or empty string.",
    )
    description: str = Field(
        description="Clear description of the issue found."
    )
    evidence: str = Field(
        description="Specific evidence or reasoning supporting this finding."
    )
    recommendation: str = Field(
        description="Concrete action to address this finding."
    )


class _WriteProjectReviewInput(BaseModel):
    verdict: str = Field(
        description=(
            "Overall verdict: PASSED (no blocking issues), "
            "NEEDS_REVISION (fixable autonomously), or "
            "NEEDS_REVIEW (requires human judgment)."
        )
    )
    findings: list[_FindingInput] = Field(
        default_factory=list,
        description="List of findings from the adversarial review.",
    )
    summary: str = Field(
        description="One to three paragraph narrative summary of the review."
    )


class WriteProjectReviewTool(BaseTool):
    """Write the structured project adversarial review result.

    This is the ONLY write tool for the project review agent.  Call after
    all evidence has been reviewed and all source files spot-checked.
    """

    name: str = "write_project_review"
    description: str = (
        "Write the structured adversarial review result (verdict, findings list, "
        "summary).  Verdict: PASSED (no blocking issues), NEEDS_REVISION (fixable "
        "autonomously), NEEDS_REVIEW (requires human judgment).  Each finding must "
        "have severity (CRITICAL/HIGH/MEDIUM/LOW), category, description, evidence, "
        "and recommendation.  Call LAST, after reviewing all evidence."
    )
    args_schema: Optional[ArgsSchema] = _WriteProjectReviewInput

    project_id: str = ""
    output_dir: str = ""
    workspace_root: str = "."

    def _run(
        self,
        verdict: str,
        findings: list[dict],
        summary: str,
    ) -> str:
        valid_verdicts = {"PASSED", "NEEDS_REVISION", "NEEDS_REVIEW"}
        if verdict.upper() not in valid_verdicts:
            return f"ERROR: verdict must be one of {valid_verdicts}, got '{verdict}'"

        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        sorted_findings = sorted(
            findings,
            key=lambda f: severity_order.get(str(f.get("severity", "LOW")).upper(), 4),
        )

        data: dict[str, Any] = {
            "project_id": self.project_id,
            "verdict": verdict.upper(),
            "summary": summary.strip(),
            "findings": [
                {
                    "severity": str(f.get("severity", "LOW")).upper(),
                    "category": f.get("category", "other"),
                    "requirement_id": f.get("requirement_id", ""),
                    "description": f.get("description", ""),
                    "evidence": f.get("evidence", ""),
                    "recommendation": f.get("recommendation", ""),
                }
                for f in sorted_findings
            ],
            "finding_counts": {
                "critical": sum(
                    1 for f in sorted_findings
                    if str(f.get("severity", "")).upper() == "CRITICAL"
                ),
                "high": sum(
                    1 for f in sorted_findings
                    if str(f.get("severity", "")).upper() == "HIGH"
                ),
                "medium": sum(
                    1 for f in sorted_findings
                    if str(f.get("severity", "")).upper() == "MEDIUM"
                ),
                "low": sum(
                    1 for f in sorted_findings
                    if str(f.get("severity", "")).upper() == "LOW"
                ),
            },
        }

        output_path = Path(self.workspace_root) / self.output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / "project_review.json"
        try:
            file_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            return f"ERROR: Could not write project review: {exc}"

        n = len(sorted_findings)
        counts = data["finding_counts"]
        return (
            f"Project review written: {self.output_dir}/project_review.json. "
            f"Verdict={data['verdict']}, {n} finding(s) "
            f"(C:{counts['critical']} H:{counts['high']} M:{counts['medium']} L:{counts['low']})."
        )

    async def _arun(
        self,
        verdict: str,
        findings: list[dict],
        summary: str,
    ) -> str:
        return self._run(verdict, findings, summary)


# ── Factory ───────────────────────────────────────────────────────────────


def build_project_review_tools(
    *,
    project_id: str,
    spec_dict: dict,
    coverage: dict,
    member_states: dict,
    project_verification: dict | None,
    workspace_root: str,
) -> list[BaseTool]:
    """Build the tool set for the project adversarial review agent."""
    output_dir = f".spine/project/{project_id}"

    return [
        ReadProjectEvidenceTool(
            spec_dict=spec_dict,
            coverage=coverage,
            member_states=member_states,
            project_verification=project_verification,
            workspace_root=workspace_root,
        ),
        WriteProjectReviewTool(
            project_id=project_id,
            output_dir=output_dir,
            workspace_root=workspace_root,
        ),
    ]
