"""Custom tools for the project-level phase-verification agent.

Each roadmap phase gets a parallel agent that reads evidence from its
member work items and writes a structured PhaseVerificationResult.  The
tool surface is deliberately narrow:

- ``read_phase_evidence`` — loads member verification.json artifacts,
  specification_json from checkpoint state, and phase requirements in one
  call.  Eliminates multi-turn exploration.
- ``write_phase_verification`` — writes the structured result so the
  synthesis node can aggregate across phases.

File-system read tools (ls/read_file/glob/grep) are allowed via
FilesystemMiddleware so the agent can spot-check implementation files.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field

from spine.agents.artifacts import artifact_path, ARTIFACTS_DIR

logger = logging.getLogger(__name__)

_VERIFY_RESULT_FILE = "project_verify_phase_{phase_id}.json"


# ── read_phase_evidence ──────────────────────────────────────────────────


class _ReadPhaseEvidenceInput(BaseModel):
    """No arguments — paths are fully determined at build time."""


class ReadPhaseEvidenceTool(BaseTool):
    """Load all evidence for one roadmap phase in a single call.

    Returns a JSON payload with:
    - phase metadata (id, title, requirements)
    - aggregator coverage for this phase
    - each member's verification.json (if present)
    - each member's specification_json from checkpoint state (if present)
    - files_written by each member (from checkpoint state)
    """

    name: str = "read_phase_evidence"
    description: str = (
        "Load all evidence for the roadmap phase being verified: phase metadata, "
        "requirement coverage, member verification reports, member specifications, "
        "and files written. Call this FIRST — it gives you everything needed to "
        "assess whether the phase requirements are truly satisfied. No arguments."
    )
    args_schema: Optional[ArgsSchema] = _ReadPhaseEvidenceInput

    # Injected at build time
    phase_dict: dict = Field(default_factory=dict)
    coverage_phase: dict = Field(default_factory=dict)
    member_states: dict = Field(default_factory=dict)  # work_id → checkpoint state
    workspace_root: str = "."

    def _run(self, **kwargs: Any) -> str:  # noqa: ARG002
        workspace = Path(self.workspace_root)
        phase_id = self.phase_dict.get("id", "unknown")
        member_work_ids: list[str] = self.phase_dict.get("member_work_ids", [])

        result: dict[str, Any] = {
            "phase_id": phase_id,
            "phase_title": self.phase_dict.get("title", ""),
            "phase_description": self.phase_dict.get("description", ""),
            "requirement_ids": self.phase_dict.get("requirement_ids", []),
            "member_work_ids": member_work_ids,
            "aggregator_phase_status": self.coverage_phase.get("status", "unknown"),
            "members": [],
        }

        for wid in member_work_ids:
            member: dict[str, Any] = {
                "work_id": wid,
                "verification": None,
                "specification_json": None,
                "files_written": [],
                "verification_passed": False,
            }

            # Load verification.json artifact
            verify_dir = workspace / artifact_path(wid, "verify")
            verify_json_path = verify_dir / "verification.json"
            if verify_json_path.exists():
                try:
                    member["verification"] = json.loads(
                        verify_json_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    member["verification_error"] = str(exc)
            else:
                member["verification_error"] = f"verification.json not found in {verify_dir}"

            # Load checkpoint state fields
            state = self.member_states.get(wid)
            if state:
                spec_raw = state.get("specification_json")
                if isinstance(spec_raw, dict):
                    member["specification_json"] = spec_raw
                elif isinstance(spec_raw, str) and spec_raw.strip():
                    try:
                        member["specification_json"] = json.loads(spec_raw)
                    except (json.JSONDecodeError, TypeError):
                        member["specification_json"] = spec_raw
                member["files_written"] = state.get("files_written") or []
                member["verification_passed"] = bool(state.get("verification_passed"))

            result["members"].append(member)

        return json.dumps(result, ensure_ascii=False, indent=2)

    async def _arun(self, **kwargs: Any) -> str:
        return self._run(**kwargs)


# ── write_phase_verification ─────────────────────────────────────────────


class _RequirementResultInput(BaseModel):
    requirement_id: str = Field(description="The requirement ID (e.g. 'R-001').")
    satisfied: bool = Field(description="Whether this requirement is satisfied.")
    evidence: str = Field(description="Evidence or reasoning for the verdict.")
    gaps: list[str] = Field(
        default_factory=list,
        description="Specific gaps or issues found (empty list if satisfied).",
    )


class _WritePhaseVerificationInput(BaseModel):
    verdict: str = Field(
        description="Overall phase verdict: VERIFIED, PARTIAL, or FAILED."
    )
    requirement_results: list[_RequirementResultInput] = Field(
        description="Per-requirement assessment.",
        min_length=0,
        default_factory=list,
    )
    integration_gaps: list[str] = Field(
        default_factory=list,
        description=(
            "Cross-member integration issues not visible in per-member verification "
            "(e.g. two members implement overlapping features differently, or "
            "member A's output is not consumed by member B)."
        ),
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Concrete next steps for any gaps or partial verdicts.",
    )
    summary: str = Field(description="One-paragraph summary of the phase verification.")


class WritePhaseVerificationTool(BaseTool):
    """Write the structured phase verification result.

    This is the ONLY write tool for the phase verifier agent.  Call after
    all evidence has been reviewed.
    """

    name: str = "write_phase_verification"
    description: str = (
        "Write the structured phase verification result (verdict, per-requirement "
        "results, integration gaps, recommendations). Verdict must be one of: "
        "VERIFIED (all requirements satisfied), PARTIAL (some satisfied), "
        "FAILED (none satisfied or critical gaps). Call this LAST, after analysis."
    )
    args_schema: Optional[ArgsSchema] = _WritePhaseVerificationInput

    phase_id: str = ""
    output_dir: str = ""
    workspace_root: str = "."

    def _run(
        self,
        verdict: str,
        requirement_results: list[dict],
        integration_gaps: list[str],
        recommendations: list[str],
        summary: str,
    ) -> str:
        valid_verdicts = {"VERIFIED", "PARTIAL", "FAILED"}
        if verdict.upper() not in valid_verdicts:
            return f"ERROR: verdict must be one of {valid_verdicts}, got '{verdict}'"

        data: dict[str, Any] = {
            "phase_id": self.phase_id,
            "verdict": verdict.upper(),
            "summary": summary.strip(),
            "requirement_results": [
                {
                    "requirement_id": r.get("requirement_id", ""),
                    "satisfied": bool(r.get("satisfied")),
                    "evidence": r.get("evidence", ""),
                    "gaps": r.get("gaps", []),
                }
                for r in requirement_results
            ],
            "integration_gaps": [g for g in integration_gaps if g],
            "recommendations": [r for r in recommendations if r],
        }

        output_path = Path(self.workspace_root) / self.output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        filename = _VERIFY_RESULT_FILE.format(phase_id=self.phase_id)
        file_path = output_path / filename
        try:
            file_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            return f"ERROR: Could not write verification result: {exc}"

        n_gaps = len(integration_gaps) + sum(len(r.get("gaps", [])) for r in requirement_results)
        return (
            f"Phase verification result written: {self.output_dir}/{filename}. "
            f"Verdict={data['verdict']}, "
            f"{len(requirement_results)} requirement(s), "
            f"{n_gaps} gap(s) found."
        )

    async def _arun(
        self,
        verdict: str,
        requirement_results: list[dict],
        integration_gaps: list[str],
        recommendations: list[str],
        summary: str,
    ) -> str:
        return self._run(verdict, requirement_results, integration_gaps, recommendations, summary)


# ── Factory ───────────────────────────────────────────────────────────────


def build_project_verify_tools(
    *,
    project_id: str,
    phase_dict: dict,
    coverage_phase: dict,
    member_states: dict,
    workspace_root: str,
) -> list[BaseTool]:
    """Build the custom tool set for one phase verifier agent.

    Returns:
    - ``read_phase_evidence``: loads all evidence in one call
    - ``write_phase_verification``: writes structured result to disk
    """
    phase_id = phase_dict.get("id", "unknown")
    output_dir = f".spine/project/{project_id}/verify"

    return [
        ReadPhaseEvidenceTool(
            phase_dict=phase_dict,
            coverage_phase=coverage_phase,
            member_states=member_states,
            workspace_root=workspace_root,
        ),
        WritePhaseVerificationTool(
            phase_id=phase_id,
            output_dir=output_dir,
            workspace_root=workspace_root,
        ),
    ]


def load_phase_verification_result(
    project_id: str,
    phase_id: str,
    workspace_root: str,
) -> dict | None:
    """Read a phase verification result written by WritePhaseVerificationTool."""
    filename = _VERIFY_RESULT_FILE.format(phase_id=phase_id)
    path = Path(workspace_root) / ".spine" / "project" / project_id / "verify" / filename
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
