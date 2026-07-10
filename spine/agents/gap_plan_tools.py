"""Custom tools for the GAP_PLAN phase orchestrator.

The gap_plan orchestrator reads verification feedback and produces a targeted
gap remediation plan. Two purpose-built tools enforce this dispatch-only behavior:

- ``read_verification_findings`` — loads all verification inputs in one call
- ``write_structured_gap_plan`` — writes gap_plan.md + gap_plan.json with structured fixes

Both tools are LangChain ``BaseTool`` subclasses that slot directly into
``create_agent(tools=[...])`` with ``skip_filesystem_middleware=True``.
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

# ── read_verification_findings ───────────────────────────────────────────


class _ReadVerificationFindingsInput(BaseModel):
    """No inputs needed — paths are fully determined by work_id at build time."""


class ReadVerificationFindingsTool(BaseTool):
    """Read all verification inputs for the GAP_PLAN phase.

    Loads the verification report with failed slices, original plan,
    codebase map, tasks/slices, and implementation summary in one call.
    This eliminates multi-turn exploration for the gap_plan agent.
    """

    name: str = "read_verification_findings"
    description: str = (
        "Read all verification inputs: verification report, plan, codebase map, "
        "tasks/slices, and implementation summary in one call. Loads them from "
        "the respective artifacts directories. Call this FIRST — it gives you everything "
        "you need to produce the gap plan. No arguments required."
    )
    args_schema: Optional[ArgsSchema] = _ReadVerificationFindingsInput

    # Injected at build time
    verify_dir: str = ""
    plan_dir: str = ""
    tasks_dir: str = ""
    impl_dir: str = ""
    workspace_root: str = ""

    def _run(self, **kwargs: Any) -> str:  # noqa: ARG002
        workspace = Path(self.workspace_root)
        result: dict[str, Any] = {
            "verify_dir": self.verify_dir,
            "plan_dir": self.plan_dir,
            "tasks_dir": self.tasks_dir,
            "impl_dir": self.impl_dir,
            "verification": None,
            "plan": None,
            "codebase_map": None,
            "tasks": None,
            "implementation": None,
            "failed_slices": [],
        }

        # Load verification.md
        verify_path = workspace / self.verify_dir / "verification.md"
        if verify_path.exists():
            try:
                result["verification"] = verify_path.read_text(encoding="utf-8")
                # Extract failed slices from verification report
                content = result["verification"]
                if "NOT_VERIFIED" in content:
                    # Extract slice names that have NOT_VERIFIED status
                    import re

                    pattern = r"### [❌]?\s*(slice-[^\s]+)\s*—\s*NOT_VERIFIED"
                    matches = re.findall(pattern, content)
                    result["failed_slices"] = matches
            except OSError as exc:
                result["verify_error"] = f"Could not read verification.md: {exc}"
        else:
            result["verify_error"] = f"verification.md not found in {self.verify_dir}"

        # Load plan.md
        plan_path = workspace / self.plan_dir / "plan.md"
        if plan_path.exists():
            try:
                result["plan"] = plan_path.read_text(encoding="utf-8")
            except OSError as exc:
                result["plan_error"] = f"Could not read plan.md: {exc}"
        else:
            result["plan_error"] = f"plan.md not found in {self.plan_dir}"

        # Load codebase-map.md from tasks directory
        codebase_map_path = workspace / self.tasks_dir / "codebase-map.md"
        if codebase_map_path.exists():
            try:
                result["codebase_map"] = codebase_map_path.read_text(encoding="utf-8")
            except OSError as exc:
                result["codebase_map_error"] = f"Could not read codebase-map.md: {exc}"

        # Load tasks.md
        tasks_path = workspace / self.tasks_dir / "tasks.md"
        if tasks_path.exists():
            try:
                result["tasks"] = tasks_path.read_text(encoding="utf-8")
            except OSError as exc:
                result["tasks_error"] = f"Could not read tasks.md: {exc}"

        # Load implementation.md
        impl_path = workspace / self.impl_dir / "implementation.md"
        if impl_path.exists():
            try:
                result["implementation"] = impl_path.read_text(encoding="utf-8")
            except OSError as exc:
                result["impl_error"] = f"Could not read implementation.md: {exc}"

        # Live CURRENT source of every feature target file. The reports above
        # are prose ABOUT the code; diagnosing from prose alone made the
        # planner cross-slice blind: probe 21 (run ad237d70) prescribed a
        # RefreshDatabase trait for an Undefined-table crash whose actual
        # cause — the model missing a $table override for the migration's
        # table name — was plainly visible in two sibling files it never saw.
        result["target_file_sources"] = self._target_file_sources(workspace)

        return json.dumps(result, ensure_ascii=False)

    _MAX_SOURCE_FILES = 12
    _MAX_FILE_CHARS = 6000

    def _target_file_sources(self, workspace: Path) -> dict[str, str]:
        """{rel_path: current content} for the plan's target files, bounded."""
        plan_json = workspace / self.plan_dir / "plan.json"
        try:
            plan = json.loads(plan_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        files: list[str] = []
        for sl in plan.get("feature_slices") or []:
            if not isinstance(sl, dict):
                continue
            for f in sl.get("target_files") or []:
                f = str(f).strip()
                if f and f not in files:
                    files.append(f)
        sources: dict[str, str] = {}
        for rel in files[: self._MAX_SOURCE_FILES]:
            path = workspace / rel
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                sources[rel] = "(file does not exist)"
                continue
            if len(text) > self._MAX_FILE_CHARS:
                text = text[: self._MAX_FILE_CHARS] + "\n… (truncated)"
            sources[rel] = text
        if len(files) > self._MAX_SOURCE_FILES:
            sources["(note)"] = (
                f"{len(files) - self._MAX_SOURCE_FILES} more target file(s) omitted"
            )
        return sources

    async def _arun(self, **kwargs: Any) -> str:
        return self._run(**kwargs)


# ── write_structured_gap_plan ───────────────────────────────────────────────


def _coerce_to_dict(obj: Any) -> dict[str, Any] | None:
    """Normalize a tool-call item to a plain dict, or None if it isn't an object.

    LangChain validates tool args through ``args_schema`` (``_StructuredGapPlanInput``),
    so each ``remediation_items`` entry reaches ``_run`` as a ``_GapRemediationInput``
    **Pydantic model**, not a dict — and under some guided-decoding paths an entry can
    instead arrive as a JSON string. The previous validator did ``isinstance(item, dict)``
    and rejected both shapes as "not a dict," false-failing well-formed plans and driving
    a forced-tool re-generation spiral where the model degenerated to placeholder garbage
    (the gap_plan recurrence of the PLAN-synthesizer mismatch, DECISION_LOG D12/D16).
    Accept model / dict / JSON-string uniformly; return None only for genuinely
    non-object input.
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
        except (json.JSONDecodeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


class _GapFixInput(BaseModel):
    """A single file fix within a gap remediation item."""

    file_path: str = Field(
        description="The file to modify (relative path from workspace root)."
    )
    issue_description: str = Field(
        description="What is wrong or missing in this file."
    )
    suggested_fix: str = Field(
        description="Specific code change or addition needed."
    )
    acceptance_criteria: list[str] = Field(
        description="Measurable criteria to verify the fix works.",
        min_length=1,
    )


class _GapRemediationInput(BaseModel):
    """A single gap remediation item for a failed slice."""

    slice_id: str = Field(
        description="The slice identifier (e.g., 'slice-add-user-auth')."
    )
    failures: list[str] = Field(
        description="List of specific failure descriptions from verification."
    )
    root_cause: str = Field(
        description="Analysis of why this failure occurred."
    )
    fixes: list[_GapFixInput] = Field(
        description="List of file-specific fixes needed for this slice.",
        min_length=1,
    )
    priority: str = Field(
        description="One of: critical, high, medium, low.",
        default="medium",
    )


class _StructuredGapPlanInput(BaseModel):
    """Input schema for write_structured_gap_plan tool."""

    remediation_items: list[_GapRemediationInput] = Field(
        description=(
            "List of gap remediation items, one per failed or partially-verified slice. "
            "Each must specify the slice_id, failures, root_cause, fixes (with file_path, "
            "issue_description, suggested_fix, acceptance_criteria), and priority."
        ),
        min_length=1,
    )
    summary: str = Field(
        description=(
            "Overall gap remediation summary: what needs to be fixed, "
            "estimated effort, and approach for implementation."
        )
    )


class WriteStructuredGapPlanTool(BaseTool):
    """Write gap_plan.md and gap_plan.json artifacts.

    Accepts structured remediation items with file-specific fix instructions
    and writes both a human-readable gap_plan.md and a machine-readable
    gap_plan.json for downstream consumption.
    """

    name: str = "write_structured_gap_plan"
    description: str = (
        "Write the gap_plan.md and gap_plan.json artifacts with structured "
        "remediation items. Provide remediation_items (list of per-slice gap fixes "
        "with file paths, issue descriptions, and acceptance criteria) and summary. "
        "This is the ONLY write tool for the gap plan agent. Call after analysis is complete."
    )
    args_schema: Optional[ArgsSchema] = _StructuredGapPlanInput

    # Injected at build time
    gap_plan_dir: str = ""
    workspace_root: str = ""

    def _run(
        self,
        remediation_items: list[Any],
        summary: str,
    ) -> str:
        gap_plan_path = Path(self.workspace_root) / self.gap_plan_dir
        gap_plan_path.mkdir(parents=True, exist_ok=True)

        # Validate remediation items
        validated_items: list[dict[str, Any]] = []
        errors: list[str] = []

        for idx, raw_item in enumerate(remediation_items):
            # Items arrive as Pydantic models (args_schema coercion), dicts, or
            # JSON strings depending on the decoding path — normalize before
            # validating so well-formed input is never false-rejected.
            item = _coerce_to_dict(raw_item)
            if item is None:
                errors.append(
                    f"Item at index {idx} is not an object "
                    f"(got {type(raw_item).__name__})."
                )
                continue

            # Validate required fields. ``priority`` is intentionally excluded —
            # it carries a schema default ("medium") and is read with a fallback
            # below, so requiring it here would false-reject an otherwise valid
            # item that merely omitted it.
            required = {"slice_id", "failures", "root_cause", "fixes"}
            missing = required - set(item.keys())
            if missing:
                errors.append(
                    f"Item at index {idx} is missing keys: {', '.join(sorted(missing))}."
                )
                continue

            # Validate fixes
            validated_fixes = []
            for fix_idx, raw_fix in enumerate(item.get("fixes", [])):
                fix = _coerce_to_dict(raw_fix)
                if fix is None:
                    errors.append(
                        f"Fix at index {fix_idx} in item {idx} is not an object "
                        f"(got {type(raw_fix).__name__})."
                    )
                    continue
                fix_required = {"file_path", "issue_description", "suggested_fix", "acceptance_criteria"}
                fix_missing = fix_required - set(fix.keys())
                if fix_missing:
                    errors.append(
                        f"Fix at index {fix_idx} in item {idx} is missing keys: "
                        f"{', '.join(sorted(fix_missing))}."
                    )
                    continue
                validated_fixes.append(fix)

            if validated_fixes:
                item["fixes"] = validated_fixes
                validated_items.append(item)

        if errors:
            return "ERROR: Invalid remediation_items:\n" + "\n".join(errors)

        # Build gap_plan.md (narrative)
        lines = [
            "# Gap Remediation Plan\n",
            f"## Summary\n\n{summary.strip()}\n",
        ]

        total_fixes = 0
        for item in validated_items:
            slice_id = item.get("slice_id", "unknown")
            failures = item.get("failures", [])
            root_cause = item.get("root_cause", "")
            fixes = item.get("fixes", [])
            priority = item.get("priority", "medium")

            lines.append(f"\n## {slice_id}\n")
            lines.append(f"**Priority:** {priority}\n\n")
            lines.append(f"**Root Cause:** {root_cause}\n\n")
            lines.append("**Failures:**\n")
            for f in failures:
                lines.append(f"- {f}\n")

            lines.append("\n**Fixes Required:**\n")
            for fix in fixes:
                total_fixes += 1
                lines.append(f"\n### {fix.get('file_path', 'unknown')}\n")
                lines.append(f"**Issue:** {fix.get('issue_description', '')}\n\n")
                lines.append(f"**Suggested Fix:**\n```\n{fix.get('suggested_fix', '')}\n```\n\n")
                lines.append("**Acceptance Criteria:**\n")
                for ac in fix.get("acceptance_criteria", []):
                    lines.append(f"- {ac}\n")

        md_content = "".join(lines)

        # Build gap_plan.json (structured)
        json_data: dict[str, Any] = {
            "summary": summary.strip(),
            "remediation_items": validated_items,
            "total_fixes": sum(len(item.get("fixes", [])) for item in validated_items),
        }
        json_content = json.dumps(json_data, indent=2, ensure_ascii=False)

        # Write files
        md_path = gap_plan_path / "gap_plan.md"
        json_path = gap_plan_path / "gap_plan.json"

        try:
            md_path.write_text(md_content, encoding="utf-8")
        except OSError as exc:
            return f"ERROR: Could not write gap_plan.md: {exc}"

        try:
            json_path.write_text(json_content, encoding="utf-8")
        except OSError as exc:
            return f"ERROR: Could not write gap_plan.json: {exc}"

        return (
            f"Gap plan artifacts written: {self.gap_plan_dir}/gap_plan.md "
            f"({len(md_content)} chars), {self.gap_plan_dir}/gap_plan.json "
            f"({len(json_content)} chars). "
            f"{len(validated_items)} slice(s), {total_fixes} fix(es) included."
        )

    async def _arun(
        self,
        remediation_items: list[Any],
        summary: str,
    ) -> str:
        return self._run(remediation_items, summary)


# ── Factory ───────────────────────────────────────────────────────────────


def build_gap_plan_tools(
    workspace_root: str,
    work_id: str,
) -> list[BaseTool]:
    """Build the custom tool set for the gap_plan agent.

    Returns two tools:
    - ``read_verification_findings``: loads all verification inputs in one call
    - ``write_structured_gap_plan``: writes gap_plan.md + gap_plan.json

    These are the complete tool surface for the gap_plan agent. No generic
    filesystem tools are exposed.

    Args:
        workspace_root: Absolute path to the project workspace root.
        work_id: The current work item ID (e.g. ``"30f58813"``).

    Returns:
        List of BaseTool instances ready for use.
    """
    verify_dir = artifact_path(work_id, "verify")
    plan_dir = artifact_path(work_id, "plan")
    tasks_dir = artifact_path(work_id, "tasks")
    impl_dir = artifact_path(work_id, "implement")
    gap_plan_dir = artifact_path(work_id, "gap_plan")

    return [
        ReadVerificationFindingsTool(
            workspace_root=workspace_root,
            verify_dir=verify_dir,
            plan_dir=plan_dir,
            tasks_dir=tasks_dir,
            impl_dir=impl_dir,
        ),
        WriteStructuredGapPlanTool(
            workspace_root=workspace_root,
            gap_plan_dir=gap_plan_dir,
        ),
    ]