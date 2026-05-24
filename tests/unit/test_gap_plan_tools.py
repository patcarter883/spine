"""Unit tests for gap_plan_tools — ReadVerificationFindingsTool and WriteStructuredGapPlanTool."""

from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spine.agents.gap_plan_tools import (
    ReadVerificationFindingsTool,
    WriteStructuredGapPlanTool,
    build_gap_plan_tools,
)


# ── ReadVerificationFindingsTool ─────────────────────────────────────────────


class TestReadVerificationFindingsTool:
    def _make_tool(
        self,
        tmp_path: Path,
        work_id: str = "wk-gap-01",
    ) -> ReadVerificationFindingsTool:
        return ReadVerificationFindingsTool(
            workspace_root=str(tmp_path),
            verify_dir=f".spine/artifacts/{work_id}/verify",
            plan_dir=f".spine/artifacts/{work_id}/plan",
            tasks_dir=f".spine/artifacts/{work_id}/tasks",
            impl_dir=f".spine/artifacts/{work_id}/implement",
        )

    def _setup_verify_dir(
        self,
        tmp_path: Path,
        work_id: str = "wk-gap-01",
        verification_content: str | None = None,
    ) -> None:
        verify_dir = tmp_path / ".spine" / "artifacts" / work_id / "verify"
        verify_dir.mkdir(parents=True, exist_ok=True)
        if verification_content:
            (verify_dir / "verification.md").write_text(verification_content)

    def _setup_plan_dir(
        self,
        tmp_path: Path,
        work_id: str = "wk-gap-01",
        plan_content: str | None = None,
    ) -> None:
        plan_dir = tmp_path / ".spine" / "artifacts" / work_id / "plan"
        plan_dir.mkdir(parents=True, exist_ok=True)
        if plan_content:
            (plan_dir / "plan.md").write_text(plan_content)

    def _setup_tasks_dir(
        self,
        tmp_path: Path,
        work_id: str = "wk-gap-01",
        codebase_map: str | None = None,
        tasks_content: str | None = None,
    ) -> None:
        tasks_dir = tmp_path / ".spine" / "artifacts" / work_id / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        if codebase_map:
            (tasks_dir / "codebase-map.md").write_text(codebase_map)
        if tasks_content:
            (tasks_dir / "tasks.md").write_text(tasks_content)

    def _setup_impl_dir(
        self,
        tmp_path: Path,
        work_id: str = "wk-gap-01",
        impl_content: str | None = None,
    ) -> None:
        impl_dir = tmp_path / ".spine" / "artifacts" / work_id / "implement"
        impl_dir.mkdir(parents=True, exist_ok=True)
        if impl_content:
            (impl_dir / "implementation.md").write_text(impl_content)

    def test_returns_verification_content(self, tmp_path):
        work_id = "wk-gap-01"
        verification = """# Verification Report

### slice-auth — NOT_VERIFIED
- Tests failed: missing import
"""
        self._setup_verify_dir(tmp_path, work_id, verification)

        tool = self._make_tool(tmp_path, work_id)
        result = json.loads(tool._run())

        assert "verification" in result
        assert "Verification Report" in result["verification"]
        assert result["verify_dir"].endswith("verify")

    def test_extracts_failed_slices_from_verification(self, tmp_path):
        work_id = "wk-gap-01"
        verification = """# Verification Report

### slice-auth — NOT_VERIFIED
Some failure

### slice-api — VERIFIED
All good

### ❌ slice-db — NOT_VERIFIED
Database connection issue
"""
        self._setup_verify_dir(tmp_path, work_id, verification)

        tool = self._make_tool(tmp_path, work_id)
        result = json.loads(tool._run())

        assert len(result["failed_slices"]) == 2
        assert "slice-auth" in result["failed_slices"]
        assert "slice-db" in result["failed_slices"]
        assert "slice-api" not in result["failed_slices"]  # Not NOT_VERIFIED

    def test_handles_verified_slices_correctly(self, tmp_path):
        work_id = "wk-gap-01"
        verification = """# Verification Report

### slice-foo — VERIFIED
All good

### slice-bar — PASSED
All tests pass
"""
        self._setup_verify_dir(tmp_path, work_id, verification)

        tool = self._make_tool(tmp_path, work_id)
        result = json.loads(tool._run())

        assert len(result["failed_slices"]) == 0

    def test_returns_verification_error_when_missing(self, tmp_path):
        work_id = "wk-gap-02"
        self._setup_verify_dir(tmp_path, work_id, None)  # No verification.md

        tool = self._make_tool(tmp_path, work_id)
        result = json.loads(tool._run())

        assert "verify_error" in result
        assert "not found" in result["verify_error"].lower()

    def test_returns_plan_content(self, tmp_path):
        work_id = "wk-gap-01"
        self._setup_plan_dir(
            tmp_path,
            work_id,
            "# Plan\nArchitecture overview here.",
        )

        tool = self._make_tool(tmp_path, work_id)
        result = json.loads(tool._run())

        assert result["plan"] is not None
        assert "Architecture overview" in result["plan"]

    def test_returns_plan_error_when_missing(self, tmp_path):
        work_id = "wk-gap-03"
        # No plan directory created

        tool = self._make_tool(tmp_path, work_id)
        result = json.loads(tool._run())

        assert "plan_error" in result

    def test_returns_codebase_map_content(self, tmp_path):
        work_id = "wk-gap-01"
        self._setup_tasks_dir(
            tmp_path,
            work_id,
            codebase_map="# Codebase Map\n- spine/models.py: state definitions",
        )

        tool = self._make_tool(tmp_path, work_id)
        result = json.loads(tool._run())

        assert result["codebase_map"] is not None
        assert "state definitions" in result["codebase_map"]

    def test_returns_codebase_map_error_when_missing(self, tmp_path):
        work_id = "wk-gap-04"
        # No tasks directory created

        tool = self._make_tool(tmp_path, work_id)
        result = json.loads(tool._run())

        assert result["codebase_map"] is None

    def test_returns_tasks_content(self, tmp_path):
        work_id = "wk-gap-01"
        self._setup_tasks_dir(
            tmp_path,
            work_id,
            tasks_content="# Tasks\n- slice-auth: implement auth\n- slice-api: implement api",
        )

        tool = self._make_tool(tmp_path, work_id)
        result = json.loads(tool._run())

        assert result["tasks"] is not None
        assert "slice-auth" in result["tasks"]

    def test_returns_implementation_content(self, tmp_path):
        work_id = "wk-gap-01"
        self._setup_impl_dir(
            tmp_path,
            work_id,
            impl_content="# Implementation\n## slice-auth\nImplemented auth.",
        )

        tool = self._make_tool(tmp_path, work_id)
        result = json.loads(tool._run())

        assert result["implementation"] is not None
        assert "Implemented auth" in result["implementation"]

    def test_returns_all_dirs_in_result(self, tmp_path):
        work_id = "wk-gap-01"
        self._setup_verify_dir(tmp_path, work_id, "# Verification")

        tool = self._make_tool(tmp_path, work_id)
        result = json.loads(tool._run())

        assert "verify_dir" in result
        assert "plan_dir" in result
        assert "tasks_dir" in result
        assert "impl_dir" in result
        assert result["verify_dir"].endswith("verify")
        assert result["plan_dir"].endswith("plan")
        assert result["tasks_dir"].endswith("tasks")
        assert result["impl_dir"].endswith("implement")


# ── WriteStructuredGapPlanTool ───────────────────────────────────────────────


class TestWriteStructuredGapPlanTool:
    def _make_tool(
        self,
        tmp_path: Path,
        work_id: str = "wk-gap-01",
    ) -> WriteStructuredGapPlanTool:
        return WriteStructuredGapPlanTool(
            workspace_root=str(tmp_path),
            gap_plan_dir=f".spine/artifacts/{work_id}/gap_plan",
        )

    def _valid_remediation_item(self, slice_id: str = "slice-test") -> dict:
        return {
            "slice_id": slice_id,
            "failures": ["Test failure 1", "Test failure 2"],
            "root_cause": "Missing dependency injection",
            "fixes": [
                {
                    "file_path": "spine/agents/auth.py",
                    "issue_description": "Auth module not imported",
                    "suggested_fix": "Add import for AuthModule",
                    "acceptance_criteria": ["Tests pass", "Auth works"],
                },
            ],
            "priority": "high",
        }

    def test_writes_both_artifacts(self, tmp_path):
        tool = self._make_tool(tmp_path)
        result = tool._run(
            remediation_items=[self._valid_remediation_item()],
            summary="Fix the auth slice with import issues.",
        )

        assert "gap_plan.md" in result
        assert "gap_plan.json" in result

        base = tmp_path / ".spine/artifacts/wk-gap-01/gap_plan"
        assert (base / "gap_plan.md").exists()
        assert (base / "gap_plan.json").exists()

    def test_gap_plan_md_has_markdown_format(self, tmp_path):
        tool = self._make_tool(tmp_path)
        tool._run(
            remediation_items=[self._valid_remediation_item("slice-auth")],
            summary="Fix auth.",
        )

        content = (
            tmp_path / ".spine/artifacts/wk-gap-01/gap_plan/gap_plan.md"
        ).read_text()

        assert "# Gap Remediation Plan" in content
        assert "## Summary" in content
        assert "## slice-auth" in content
        assert "Fix auth." in content
        assert "Missing dependency injection" in content
        assert "### spine/agents/auth.py" in content
        assert "Suggested Fix:" in content
        assert "Acceptance Criteria:" in content

    def test_gap_plan_json_has_structured_format(self, tmp_path):
        tool = self._make_tool(tmp_path)
        tool._run(
            remediation_items=[self._valid_remediation_item()],
            summary="Test summary.",
        )

        content = (
            tmp_path / ".spine/artifacts/wk-gap-01/gap_plan/gap_plan.json"
        ).read_text()

        data = json.loads(content)
        assert "summary" in data
        assert "remediation_items" in data
        assert "total_fixes" in data
        assert data["summary"] == "Test summary."
        assert data["total_fixes"] == 1
        assert len(data["remediation_items"]) == 1

    def test_writes_multiple_remediation_items(self, tmp_path):
        tool = self._make_tool(tmp_path)
        tool._run(
            remediation_items=[
                self._valid_remediation_item("slice-auth"),
                self._valid_remediation_item("slice-api"),
            ],
            summary="Fix two slices.",
        )

        content = (
            tmp_path / ".spine/artifacts/wk-gap-01/gap_plan/gap_plan.md"
        ).read_text()

        assert "## slice-auth" in content
        assert "## slice-api" in content

        json_content = (
            tmp_path / ".spine/artifacts/wk-gap-01/gap_plan/gap_plan.json"
        ).read_text()
        data = json.loads(json_content)
        assert len(data["remediation_items"]) == 2

    def test_creates_directory_if_missing(self, tmp_path):
        tool = self._make_tool(tmp_path, work_id="new-work")
        tool._run(
            remediation_items=[self._valid_remediation_item()],
            summary="Test.",
        )

        assert (tmp_path / ".spine/artifacts/new-work/gap_plan").is_dir()

    def test_returns_error_for_missing_required_fields(self, tmp_path):
        tool = self._make_tool(tmp_path)
        result = tool._run(
            remediation_items=[{"slice_id": "test"}],  # Missing required fields
            summary="Test.",
        )

        assert "ERROR" in result
        assert "missing keys" in result.lower()

    def test_returns_error_for_non_dict_item(self, tmp_path):
        tool = self._make_tool(tmp_path)
        result = tool._run(
            remediation_items=["not a dict"],  # Invalid type
            summary="Test.",
        )

        assert "ERROR" in result
        assert "not a dict" in result

    def test_returns_error_for_missing_fix_keys(self, tmp_path):
        tool = self._make_tool(tmp_path)
        result = tool._run(
            remediation_items=[
                {
                    "slice_id": "test",
                    "failures": ["issue"],
                    "root_cause": "cause",
                    "fixes": [{"file_path": "test.py"}],  # Missing fix keys
                    "priority": "medium",
                },
            ],
            summary="Test.",
        )

        assert "ERROR" in result
        assert "fix at index 0" in result.lower()

    def test_returns_error_when_any_fix_missing_keys(self, tmp_path):
        """If ANY fix has missing keys, the whole operation returns an error."""
        tool = self._make_tool(tmp_path)
        result = tool._run(
            remediation_items=[
                {
                    "slice_id": "slice-valid",
                    "failures": ["issue"],
                    "root_cause": "cause",
                    "fixes": [
                        {
                            "file_path": "valid.py",
                            "issue_description": "issue",
                            "suggested_fix": "fix",
                            "acceptance_criteria": ["works"],
                        },
                        {
                            "file_path": "invalid.py",
                            # Missing other required keys
                        },
                    ],
                    "priority": "medium",
                },
            ],
            summary="Test.",
        )

        # Should return error because one fix is invalid
        assert "ERROR" in result
        assert "fix at index 1" in result.lower()
        assert "missing keys" in result.lower()

    def test_priority_field_used_in_output(self, tmp_path):
        """Priority field is required and should appear in output."""
        tool = self._make_tool(tmp_path)
        tool._run(
            remediation_items=[
                {
                    "slice_id": "test",
                    "failures": ["issue"],
                    "root_cause": "cause",
                    "fixes": [
                        {
                            "file_path": "test.py",
                            "issue_description": "issue",
                            "suggested_fix": "fix",
                            "acceptance_criteria": ["works"],
                        },
                    ],
                    "priority": "critical",
                },
            ],
            summary="Test.",
        )

        content = (
            tmp_path / ".spine/artifacts/wk-gap-01/gap_plan/gap_plan.md"
        ).read_text()

        assert "**Priority:** critical" in content

    def test_includes_all_acceptance_criteria(self, tmp_path):
        tool = self._make_tool(tmp_path)
        tool._run(
            remediation_items=[
                {
                    "slice_id": "slice-multi-criteria",
                    "failures": ["issue"],
                    "root_cause": "cause",
                    "fixes": [
                        {
                            "file_path": "test.py",
                            "issue_description": "issue",
                            "suggested_fix": "fix",
                            "acceptance_criteria": [
                                "First criterion",
                                "Second criterion",
                                "Third criterion",
                            ],
                        },
                    ],
                    "priority": "critical",
                },
            ],
            summary="Test.",
        )

        content = (
            tmp_path / ".spine/artifacts/wk-gap-01/gap_plan/gap_plan.md"
        ).read_text()

        assert "First criterion" in content
        assert "Second criterion" in content
        assert "Third criterion" in content

    def test_all_failures_listed(self, tmp_path):
        tool = self._make_tool(tmp_path)
        tool._run(
            remediation_items=[
                {
                    "slice_id": "slice-many-failures",
                    "failures": [
                        "Failure one",
                        "Failure two",
                        "Failure three",
                    ],
                    "root_cause": "cause",
                    "fixes": [
                        {
                            "file_path": "test.py",
                            "issue_description": "issue",
                            "suggested_fix": "fix",
                            "acceptance_criteria": ["works"],
                        },
                    ],
                    "priority": "low",
                },
            ],
            summary="Test.",
        )

        content = (
            tmp_path / ".spine/artifacts/wk-gap-01/gap_plan/gap_plan.md"
        ).read_text()

        assert "Failure one" in content
        assert "Failure two" in content
        assert "Failure three" in content

    def test_result_includes_char_counts(self, tmp_path):
        tool = self._make_tool(tmp_path)
        result = tool._run(
            remediation_items=[self._valid_remediation_item()],
            summary="Test summary for char count.",
        )

        assert "chars" in result.lower()
        assert "slice(s)" in result.lower()
        assert "fix(es)" in result.lower()


# ── build_gap_plan_tools ─────────────────────────────────────────────────────


class TestBuildGapPlanTools:
    def test_returns_two_tools(self, tmp_path):
        tools = build_gap_plan_tools(
            workspace_root=str(tmp_path),
            work_id="abc123",
        )

        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "read_verification_findings" in names
        assert "write_structured_gap_plan" in names

    def test_tools_have_correct_types(self, tmp_path):
        tools = build_gap_plan_tools(
            workspace_root=str(tmp_path),
            work_id="xyz",
        )

        read_tool = next(t for t in tools if t.name == "read_verification_findings")
        write_tool = next(t for t in tools if t.name == "write_structured_gap_plan")

        assert isinstance(read_tool, ReadVerificationFindingsTool)
        assert isinstance(write_tool, WriteStructuredGapPlanTool)

    def test_read_tool_has_correct_directories(self, tmp_path):
        tools = build_gap_plan_tools(
            workspace_root=str(tmp_path),
            work_id="test-work",
        )

        read_tool = next(t for t in tools if t.name == "read_verification_findings")

        assert read_tool.verify_dir == ".spine/artifacts/test-work/verify"
        assert read_tool.plan_dir == ".spine/artifacts/test-work/plan"
        assert read_tool.tasks_dir == ".spine/artifacts/test-work/tasks"
        assert read_tool.impl_dir == ".spine/artifacts/test-work/implement"
        assert read_tool.workspace_root == str(tmp_path)

    def test_write_tool_has_correct_directory(self, tmp_path):
        tools = build_gap_plan_tools(
            workspace_root=str(tmp_path),
            work_id="write-test",
        )

        write_tool = next(t for t in tools if t.name == "write_structured_gap_plan")

        assert write_tool.gap_plan_dir == ".spine/artifacts/write-test/gap_plan"
        assert write_tool.workspace_root == str(tmp_path)

    def test_tools_are_functional(self, tmp_path):
        """Integration test: created tools should work together."""
        tools = build_gap_plan_tools(
            workspace_root=str(tmp_path),
            work_id="functional",
        )

        # Setup verification.md for read tool
        verify_dir = tmp_path / ".spine/artifacts/functional/verify"
        verify_dir.mkdir(parents=True)
        (verify_dir / "verification.md").write_text(
            "### slice-test — NOT_VERIFIED\nFailed.",
        )

        read_tool = next(t for t in tools if t.name == "read_verification_findings")
        write_tool = next(t for t in tools if t.name == "write_structured_gap_plan")

        # Read verification findings
        read_result = json.loads(read_tool._run())
        assert "verification" in read_result

        # Write gap plan
        write_result = write_tool._run(
            remediation_items=[
                {
                    "slice_id": "slice-test",
                    "failures": ["Verification failed"],
                    "root_cause": "Missing implementation",
                    "fixes": [
                        {
                            "file_path": "spine/test.py",
                            "issue_description": "Test file missing",
                            "suggested_fix": "Add test.py",
                            "acceptance_criteria": ["Tests pass"],
                        },
                    ],
                    "priority": "high",
                },
            ],
            summary="Fix after verification failure.",
        )
        assert "Gap plan artifacts written" in write_result