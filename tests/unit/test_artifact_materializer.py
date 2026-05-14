"""Tests for work_id-scoped artifact materialization.

Tests that artifacts.py correctly scopes artifact paths under work_id
subfolders, ensuring isolation between work items.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ── Helper function tests ──


class TestArtifactPath:
    """Tests for the _artifact_path helper."""

    def test_path_without_work_id(self):
        from spine.agents.artifacts import _artifact_path

        result = _artifact_path("", "specify")
        assert result == ".spine/artifacts/specify"

    def test_path_with_work_id(self):
        from spine.agents.artifacts import _artifact_path

        result = _artifact_path("abc123", "specify")
        assert result == ".spine/artifacts/abc123/specify"

    def test_path_different_phases(self):
        from spine.agents.artifacts import _artifact_path

        assert _artifact_path("w1", "plan") == ".spine/artifacts/w1/plan"
        assert _artifact_path("w1", "tasks") == ".spine/artifacts/w1/tasks"
        assert _artifact_path("w2", "plan") == ".spine/artifacts/w2/plan"


# ── materialize_phase_artifacts tests ──


class TestMaterializePhaseArtifacts:
    """Tests for materialize_phase_artifacts with work_id."""

    def test_creates_work_id_scoped_directory(self):
        from spine.agents.artifacts import materialize_phase_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            phase_artifacts = {"test.md": "This is a test artifact content."}
            materialize_phase_artifacts(
                phase="specify",
                phase_artifacts=phase_artifacts,
                workspace_root=tmpdir,
                work_id="abc123",
            )

            # Verify file exists under work_id subfolder
            expected = Path(tmpdir) / ".spine" / "artifacts" / "abc123" / "specify" / "test.md"
            assert expected.exists()
            assert expected.read_text() == "This is a test artifact content."

    def test_skips_empty_content(self):
        from spine.agents.artifacts import materialize_phase_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            phase_artifacts = {"empty.md": "", "filled.md": "Has content here."}
            materialize_phase_artifacts(
                phase="plan",
                phase_artifacts=phase_artifacts,
                workspace_root=tmpdir,
                work_id="w1",
            )

            # Empty file should not be created
            empty_path = Path(tmpdir) / ".spine" / "artifacts" / "w1" / "plan" / "empty.md"
            filled_path = Path(tmpdir) / ".spine" / "artifacts" / "w1" / "plan" / "filled.md"
            assert not empty_path.exists()
            assert filled_path.exists()

    def test_no_work_id_falls_back_to_flat_structure(self):
        from spine.agents.artifacts import materialize_phase_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            phase_artifacts = {"test.md": "Content"}
            materialize_phase_artifacts(
                phase="specify",
                phase_artifacts=phase_artifacts,
                workspace_root=tmpdir,
                work_id="",
            )

            # Without work_id, should be in flat structure
            expected = Path(tmpdir) / ".spine" / "artifacts" / "specify" / "test.md"
            assert expected.exists()

    def test_isolates_different_work_ids(self):
        from spine.agents.artifacts import materialize_phase_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write artifacts for two different work items
            materialize_phase_artifacts(
                phase="specify",
                phase_artifacts={"spec.md": "Work 1 spec"},
                workspace_root=tmpdir,
                work_id="w1",
            )
            materialize_phase_artifacts(
                phase="specify",
                phase_artifacts={"spec.md": "Work 2 spec"},
                workspace_root=tmpdir,
                work_id="w2",
            )

            # Both should exist independently
            w1_spec = Path(tmpdir) / ".spine" / "artifacts" / "w1" / "specify" / "spec.md"
            w2_spec = Path(tmpdir) / ".spine" / "artifacts" / "w2" / "specify" / "spec.md"
            assert w1_spec.read_text() == "Work 1 spec"
            assert w2_spec.read_text() == "Work 2 spec"

    def test_no_op_when_artifacts_empty(self):
        from spine.agents.artifacts import materialize_phase_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            materialize_phase_artifacts(
                phase="specify",
                phase_artifacts={},
                workspace_root=tmpdir,
                work_id="w1",
            )

            # No directories should be created
            artifacts_dir = Path(tmpdir) / ".spine" / "artifacts"
            assert not artifacts_dir.exists()


# ── materialize_artifacts tests ──


class TestMaterializeArtifacts:
    """Tests for materialize_artifacts with work_id."""

    def test_returns_work_id_scoped_paths(self):
        from spine.agents.artifacts import materialize_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            state = {
                "artifacts": {
                    "specify": {"specification.md": "Specification content here."},
                    "plan": {"plan.md": "Plan content here."},
                }
            }
            paths = materialize_artifacts(state, tmpdir, work_id="abc123")

            assert paths["specify"] == ".spine/artifacts/abc123/specify"
            assert paths["plan"] == ".spine/artifacts/abc123/plan"

    def test_writes_files_under_work_id(self):
        from spine.agents.artifacts import materialize_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            state = {
                "artifacts": {
                    "specify": {"specification.md": "Specification content here."},
                }
            }
            materialize_artifacts(state, tmpdir, work_id="w1")

            expected = (
                Path(tmpdir)
                / ".spine"
                / "artifacts"
                / "w1"
                / "specify"
                / "specification.md"
            )
            assert expected.exists()

    def test_returns_flat_paths_without_work_id(self):
        from spine.agents.artifacts import materialize_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            state = {
                "artifacts": {
                    "specify": {"specification.md": "Content here."},
                }
            }
            paths = materialize_artifacts(state, tmpdir, work_id="")

            assert "specify" in paths
            # Without work_id, path should be flat
            assert paths["specify"] == ".spine/artifacts/specify"

    def test_skips_non_dict_artifacts(self):
        from spine.agents.artifacts import materialize_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            state = {
                "artifacts": {
                    "specify": "not a dict",  # should be skipped
                    "plan": {"plan.md": "Plan content here."},
                }
            }
            paths = materialize_artifacts(state, tmpdir, work_id="w1")

            assert "specify" not in paths
            assert "plan" in paths


# ── build_artifact_prompt tests ──


class TestBuildArtifactPrompt:
    """Tests for build_artifact_prompt with work_id."""

    def test_prompt_includes_work_id_in_paths(self):
        from spine.agents.artifacts import build_artifact_prompt

        artifacts = {
            "specify": {
                "specification.md": "Spec content here."
            }
        }
        prompt = build_artifact_prompt(artifacts, "plan", work_id="abc123")

        assert ".spine/artifacts/abc123/specify" in prompt
        assert "specification.md" in prompt

    def test_prompt_flat_paths_without_work_id(self):
        from spine.agents.artifacts import build_artifact_prompt

        artifacts = {
            "specify": {
                "specification.md": "Spec content here."
            }
        }
        prompt = build_artifact_prompt(artifacts, "plan", work_id="")

        # Should contain the flat path (with double slash due to empty work_id)
        assert ".spine/artifacts" in prompt
        assert "specify" in prompt

    def test_prompt_excludes_current_phase(self):
        from spine.agents.artifacts import build_artifact_prompt

        artifacts = {
            "specify": {"spec.md": "Spec content here."},
            "plan": {"plan.md": "Plan content here."},
        }
        prompt = build_artifact_prompt(artifacts, "specify", work_id="w1")

        # Should only show plan, not specify (current phase)
        assert "plan" in prompt.lower()
        assert "specify" not in prompt.lower()

    def test_prompt_empty_for_no_artifacts(self):
        from spine.agents.artifacts import build_artifact_prompt

        assert build_artifact_prompt({}, "verify", work_id="w1") == ""

    def test_prompt_omits_empty_phase_artifacts(self):
        from spine.agents.artifacts import build_artifact_prompt

        artifacts = {
            "specify": {},  # empty
            "plan": {"plan.md": "Plan content here."},
        }
        prompt = build_artifact_prompt(artifacts, "verify", work_id="w1")

        assert "plan" in prompt.lower()
        assert "specify" not in prompt.lower()


# ── build_inline_artifact_prompt tests ──


class TestBuildInlineArtifactPrompt:
    """Tests for build_inline_artifact_prompt with work_id."""

    def test_preview_shows_work_id_path(self):
        from spine.agents.artifacts import build_inline_artifact_prompt

        state = {
            "artifacts": {
                "specify": {
                    "specification.md": "A" * 600  # exceeds default max_inline_chars
                }
            }
        }
        prompt = build_inline_artifact_prompt(state, "specify", work_id="abc123")

        assert ".spine/artifacts/abc123/specify/specification.md" in prompt
        assert "Full content available at" in prompt

    def test_small_content_inlined_without_path(self):
        from spine.agents.artifacts import build_inline_artifact_prompt

        state = {
            "artifacts": {
                "specify": {
                    "specification.md": "Short content."  # under max_inline_chars
                }
            }
        }
        prompt = build_inline_artifact_prompt(state, "specify", work_id="w1")

        assert "Short content." in prompt
        # Should not include the path hint for short content
        assert "Full content available at" not in prompt

    def test_empty_phase_returns_empty(self):
        from spine.agents.artifacts import build_inline_artifact_prompt

        state = {"artifacts": {}}
        prompt = build_inline_artifact_prompt(state, "verify", work_id="w1")
        assert prompt == ""

    def test_no_work_id_flat_path(self):
        from spine.agents.artifacts import build_inline_artifact_prompt

        state = {
            "artifacts": {
                "specify": {
                    "specification.md": "A" * 600
                }
            }
        }
        prompt = build_inline_artifact_prompt(state, "specify", work_id="")

        # Should use flat path structure
        assert ".spine/artifacts" in prompt
        assert "specify/specification.md" in prompt
