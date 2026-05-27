"""Tests for ``_synthesize_implementation_node`` — the implement-subgraph
synthesiser. Focus: the honesty guard added after trace 019e6974, which
demotes phase_status to needs_review when slice-implementers claim success
without reporting any modified files."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from spine.workflow.subgraphs.implement_subgraph import _synthesize_implementation_node


def _state(tmp_path: Path, completed: list[dict], failed: list[dict] | None = None) -> dict:
    return {
        "phase": "implement",
        "work_id": "test-wk",
        "work_type": "feature",
        "workspace_root": str(tmp_path),
        "completed_slices": completed,
        "failed_slices": failed or [],
    }


def _completed_slice(*, files_modified=None, files_created=None, status="implemented"):
    return {
        "slice_name": "slice-a",
        "status": status,
        "files_modified": files_modified or [],
        "files_created": files_created or [],
        "test_results": "",
        "issues": [],
    }


def test_demotes_to_needs_review_when_all_slices_report_no_files(tmp_path):
    """The trace 019e6974 case: slice_implementer claims "implemented" but
    files_modified is empty across every slice. Synth must report
    needs_review so verify and downstream gates don't trust the report."""
    state = _state(tmp_path, completed=[
        _completed_slice(files_modified=[], files_created=[]),
        _completed_slice(files_modified=[], files_created=[], status="partial"),
    ])
    result = asyncio.run(_synthesize_implementation_node(state, None))
    assert result["phase_status"] == "needs_review"
    summary = result["artifacts_output"].get("implementation.md", "")
    assert "WARNING" in summary
    assert "files_modified/files_created are empty" in summary


def test_keeps_success_when_at_least_one_slice_reports_files(tmp_path):
    """If any non-failed slice reports activity, trust the report and
    leave phase_status at success."""
    state = _state(tmp_path, completed=[
        _completed_slice(files_modified=[], files_created=[]),
        _completed_slice(files_modified=["spine/cli/__init__.py"], files_created=[]),
    ])
    result = asyncio.run(_synthesize_implementation_node(state, None))
    assert result["phase_status"] == "success"
    summary = result["artifacts_output"].get("implementation.md", "")
    assert "WARNING" not in summary


def test_files_created_alone_is_enough(tmp_path):
    """files_created should count as touched — creating a new file is
    legitimate implementation activity."""
    state = _state(tmp_path, completed=[
        _completed_slice(files_modified=[], files_created=["docs/new.md"]),
    ])
    result = asyncio.run(_synthesize_implementation_node(state, None))
    assert result["phase_status"] == "success"


def test_all_blocked_slices_dont_trigger_no_files_warning(tmp_path):
    """Blocked slices already signal failure via their status — the guard
    only fires on non-failed slices that claim success without showing it.
    All-blocked is a separate failure mode handled by the existing summary."""
    state = _state(tmp_path, completed=[
        _completed_slice(files_modified=[], files_created=[], status="blocked"),
        _completed_slice(files_modified=[], files_created=[], status="blocked"),
    ])
    result = asyncio.run(_synthesize_implementation_node(state, None))
    # All blocked → no non-failed slices → guard doesn't fire.
    # Phase status remains success (the existing all-blocked-via-failed path
    # is exercised separately when failed_slices is non-empty).
    assert result["phase_status"] == "success"
    summary = result["artifacts_output"].get("implementation.md", "")
    assert "WARNING" not in summary


def test_empty_results_already_returns_needs_review(tmp_path):
    """Pre-existing behavior: zero results → needs_review. Guard doesn't
    interfere with this."""
    state = _state(tmp_path, completed=[])
    result = asyncio.run(_synthesize_implementation_node(state, None))
    assert result["phase_status"] == "needs_review"
    assert result["slices_dispatched"] is False
