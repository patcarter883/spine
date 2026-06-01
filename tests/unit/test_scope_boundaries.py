"""Tests for the deterministic scope-boundary gate.

Covers ``check_scope_boundaries`` (the computed anti-drift invariant that
enforces ``Specification.hard_boundaries`` against the files IMPLEMENT wrote)
and the ``_collect_files_written`` aggregation helper that feeds it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.workflow.artifact_gate import check_scope_boundaries
from spine.workflow.subgraphs.implement_subgraph import _collect_files_written


def _spec(**fields) -> str:
    """Build a specification_json string with the given fields."""
    base = {
        "title": "t",
        "summary": "s",
        "requirements": ["r"],
    }
    base.update(fields)
    return json.dumps(base)


class TestCheckScopeBoundaries:
    def test_no_specification_passes(self):
        passed, reason = check_scope_boundaries({"files_written": ["a.py"]})
        assert passed is True
        assert reason == ""

    def test_no_hard_boundaries_passes(self):
        state = {
            "specification_json": _spec(hard_boundaries=[]),
            "files_written": ["spine/billing/charge.py"],
        }
        passed, _ = check_scope_boundaries(state)
        assert passed is True

    def test_no_files_written_passes(self):
        state = {
            "specification_json": _spec(hard_boundaries=["spine/billing/*"]),
            "files_written": [],
        }
        passed, _ = check_scope_boundaries(state)
        assert passed is True

    def test_clean_implementation_passes(self):
        state = {
            "specification_json": _spec(hard_boundaries=["spine/billing/*"]),
            "files_written": ["spine/agents/foo.py", "tests/unit/test_foo.py"],
            "workspace_root": "/repo",
        }
        passed, _ = check_scope_boundaries(state)
        assert passed is True

    def test_wildcard_violation_detected(self):
        state = {
            "specification_json": _spec(hard_boundaries=["spine/billing/*"]),
            "files_written": ["spine/billing/charge.py"],
            "workspace_root": "/repo",
        }
        passed, reason = check_scope_boundaries(state)
        assert passed is False
        assert "spine/billing/charge.py" in reason
        assert "spine/billing/*" in reason

    def test_wildcard_spans_path_separators(self):
        # ``*`` in fnmatch spans '/', so a single-star glob covers nested files.
        state = {
            "specification_json": _spec(hard_boundaries=["spine/billing/*"]),
            "files_written": ["spine/billing/sub/deep.py"],
            "workspace_root": "/repo",
        }
        passed, _ = check_scope_boundaries(state)
        assert passed is False

    def test_plain_directory_prefix_violation(self):
        # A boundary with no wildcard is treated as a dir/file prefix.
        state = {
            "specification_json": _spec(hard_boundaries=["spine/billing"]),
            "files_written": ["spine/billing/charge.py"],
            "workspace_root": "/repo",
        }
        passed, _ = check_scope_boundaries(state)
        assert passed is False

    def test_plain_prefix_does_not_overmatch_sibling(self):
        # 'spine/bill' must not match 'spine/billing/...'.
        state = {
            "specification_json": _spec(hard_boundaries=["spine/bill"]),
            "files_written": ["spine/billing/charge.py"],
            "workspace_root": "/repo",
        }
        passed, _ = check_scope_boundaries(state)
        assert passed is True

    def test_absolute_path_normalized_to_workspace_relative(self):
        state = {
            "specification_json": _spec(hard_boundaries=["spine/billing/*"]),
            "files_written": ["/repo/spine/billing/charge.py"],
            "workspace_root": "/repo",
        }
        passed, _ = check_scope_boundaries(state)
        assert passed is False

    def test_specification_json_as_dict(self):
        state = {
            "specification_json": {"hard_boundaries": ["spine/billing/*"]},
            "files_written": ["spine/billing/charge.py"],
            "workspace_root": "/repo",
        }
        passed, _ = check_scope_boundaries(state)
        assert passed is False

    def test_malformed_specification_json_passes(self):
        state = {
            "specification_json": "{not valid json",
            "files_written": ["spine/billing/charge.py"],
        }
        passed, _ = check_scope_boundaries(state)
        assert passed is True

    def test_scope_exclusions_are_not_enforced(self):
        # scope_exclusions are advisory prose, left to the critic — only
        # hard_boundaries are enforced deterministically.
        state = {
            "specification_json": _spec(scope_exclusions=["the billing module"]),
            "files_written": ["spine/billing/charge.py"],
            "workspace_root": "/repo",
        }
        passed, _ = check_scope_boundaries(state)
        assert passed is True


class TestCollectFilesWritten:
    def test_aggregates_modified_and_created(self):
        results = [
            {"files_modified": ["a.py"], "files_created": ["b.py"]},
            {"files_modified": ["c.py"], "files_created": []},
        ]
        assert _collect_files_written(results) == ["a.py", "b.py", "c.py"]

    def test_dedupes_and_sorts(self):
        results = [
            {"files_modified": ["z.py", "a.py"], "files_created": ["a.py"]},
            {"files_modified": ["z.py"], "files_created": ["m.py"]},
        ]
        assert _collect_files_written(results) == ["a.py", "m.py", "z.py"]

    def test_skips_non_string_and_blank_entries(self):
        results = [
            {"files_modified": ["good.py", "", "  "], "files_created": [None, 42]},
        ]
        assert _collect_files_written(results) == ["good.py"]

    def test_handles_missing_keys(self):
        assert _collect_files_written([{}]) == []
        assert _collect_files_written([]) == []
