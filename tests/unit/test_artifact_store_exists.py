import tempfile
from pathlib import Path
import pytest

from spine.persistence.artifacts import ArtifactStore


class TestArtifactExists:
    """Unit tests for ArtifactStore.artifact_exists method."""

    def test_artifact_exists_returns_true_after_saving(self):
        """Test that artifact_exists returns True after saving an artifact."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = ArtifactStore(base_path=tmp_dir)
            work_id = "work_123"
            phase = "build"
            name = "output.txt"
            content = "Hello, world!"

            # Save the artifact
            path = store.save_artifact(work_id, phase, name, content)

            # Verify that artifact_exists returns True
            assert store.artifact_exists(work_id, phase, name) is True

    def test_artifact_exists_returns_false_for_nonexistent_artifact(self):
        """Test that artifact_exists returns False for an artifact that has not been saved."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = ArtifactStore(base_path=tmp_dir)
            work_id = "work_456"
            phase = "test"
            name = "result.json"

            # Verify that artifact_exists returns False for non-existent artifact
            assert store.artifact_exists(work_id, phase, name) is False

    def test_artifact_exists_with_different_parameters_returns_false(self):
        """Test that artifact_exists returns False when parameters don't match saved artifact."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = ArtifactStore(base_path=tmp_dir)
            work_id = "work_789"
            phase = "deploy"
            name = "config.yaml"
            content = "app:\n  name: test"

            # Save artifact with specific parameters
            store.save_artifact(work_id, phase, name, content)

            # Check with different work_id
            assert store.artifact_exists("different_work", phase, name) is False

            # Check with different phase
            assert store.artifact_exists(work_id, "different_phase", name) is False

            # Check with different name
            assert store.artifact_exists(work_id, phase, "different_name") is False

    def test_artifact_exists_multiple_artifacts_isolated(self):
        """Test that artifact_exists correctly isolates multiple artifacts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = ArtifactStore(base_path=tmp_dir)

            # Save first artifact
            store.save_artifact("work_a", "phase1", "file1.txt", "content1")

            # Save second artifact with different parameters
            store.save_artifact("work_b", "phase2", "file2.txt", "content2")

            # Verify first artifact exists
            assert store.artifact_exists("work_a", "phase1", "file1.txt") is True

            # Verify second artifact exists
            assert store.artifact_exists("work_b", "phase2", "file2.txt") is True

            # Verify cross-parameters don't match
            assert store.artifact_exists("work_a", "phase2", "file2.txt") is False
            assert store.artifact_exists("work_b", "phase1", "file1.txt") is False
