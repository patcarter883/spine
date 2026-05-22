"""Tests for the extended tasks-quality checks in the artifact gate.

Covers:
- codebase-map.md existence check
- Slice path grounding check (hallucinated paths → needs_review)
- Slice path grounding check (real workspace paths → proceed)
- gate_node end-to-end for tasks→implement with quality failures
- Quality check exceptions are swallowed (gate never crashes)
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Isolate imports from the heavy spine.workflow package ────────────────
# spine/workflow/__init__.py re-exports compose.py which in turn imports all
# phase modules and deepagents/langchain.  To keep tests fast and dependency-
# free we import the module file directly (bypassing __init__).

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _import_artifact_gate():
    """Import spine.workflow.artifact_gate directly, avoiding __init__ side-effects."""
    import importlib.util

    module_path = (
        Path(__file__).resolve().parent.parent.parent / "spine" / "workflow" / "artifact_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "spine.workflow.artifact_gate",
        module_path,
    )
    assert spec is not None, f"Could not locate artifact_gate at {module_path}"
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register under the canonical name so PhaseName import inside the module works
    sys.modules.setdefault("spine.workflow.artifact_gate", mod)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# Load once at module level so all tests share the same module object
_ag = _import_artifact_gate()
_check_tasks_quality = _ag._check_tasks_quality
make_artifact_gate_node = _ag.make_artifact_gate_node


# ── _check_tasks_quality unit tests ────────────────────────────────────


class TestCheckTasksQuality:
    """Direct unit tests for _check_tasks_quality."""

    def _make_tasks_dir(self, tmp_path: Path) -> Path:
        """Create the canonical .spine/artifacts/<work_id>/tasks/ structure."""
        tasks_dir = tmp_path / ".spine" / "artifacts" / "abc123" / "tasks"
        tasks_dir.mkdir(parents=True)
        return tasks_dir

    def test_no_tasks_dir_passes(self, tmp_path: Path) -> None:
        """If the tasks dir doesn't exist the quality check passes (let main gate handle it)."""
        ok, reason = _check_tasks_quality(str(tmp_path), "abc123")
        assert ok is True
        assert reason == ""

    def test_missing_codebase_map_fails(self, tmp_path: Path) -> None:
        """Missing codebase-map.md → quality check fails with clear message."""
        tasks_dir = self._make_tasks_dir(tmp_path)
        # Write slices but no codebase-map.md
        (tasks_dir / "slice-foo.md").write_text("# Slice\n- spine/foo.py\n")
        (tasks_dir / "tasks.md").write_text("# Tasks\n")

        ok, reason = _check_tasks_quality(str(tmp_path), "abc123")
        assert ok is False
        assert "codebase-map.md" in reason

    def test_present_codebase_map_with_real_paths_passes(self, tmp_path: Path) -> None:
        """codebase-map.md present + slice paths exist in workspace → passes."""
        tasks_dir = self._make_tasks_dir(tmp_path)
        (tasks_dir / "codebase-map.md").write_text("# Codebase Map\n")

        # Create a real file in the workspace that the slice references
        real_module = tmp_path / "spine" / "work" / "dispatcher.py"
        real_module.parent.mkdir(parents=True)
        real_module.write_text("# dispatcher\n")

        (tasks_dir / "slice-alpha.md").write_text(
            "## Files to Modify\n- spine/work/dispatcher.py\n"
        )

        ok, reason = _check_tasks_quality(str(tmp_path), "abc123")
        assert ok is True
        assert reason == ""

    def test_hallucinated_paths_fail(self, tmp_path: Path) -> None:
        """All slice paths point to non-existent files → quality check fails."""
        tasks_dir = self._make_tasks_dir(tmp_path)
        (tasks_dir / "codebase-map.md").write_text("# Codebase Map\n")

        # Generic hallucinated paths — none of these exist in tmp_path
        (tasks_dir / "slice-a.md").write_text(
            "## Files to Modify\n- src/main.py\n- api/routes.py\n"
        )
        (tasks_dir / "slice-b.md").write_text(
            "## Files to Create\n- web/components/feature_ui.js\n"
        )

        ok, reason = _check_tasks_quality(str(tmp_path), "abc123")
        assert ok is False
        assert "hallucinated" in reason.lower() or "do not exist" in reason.lower()
        assert "src/main.py" in reason or "api/routes.py" in reason or "web/components" in reason

    def test_mixed_paths_one_real_passes(self, tmp_path: Path) -> None:
        """At least one real path among hallucinated ones → passes."""
        tasks_dir = self._make_tasks_dir(tmp_path)
        (tasks_dir / "codebase-map.md").write_text("# Codebase Map\n")

        # One real file
        real = tmp_path / "spine" / "models" / "enums.py"
        real.parent.mkdir(parents=True)
        real.write_text("# enums\n")

        (tasks_dir / "slice-x.md").write_text(
            "## Files to Modify\n"
            "- src/main.py\n"  # hallucinated
            "- spine/models/enums.py\n"  # real
        )

        ok, reason = _check_tasks_quality(str(tmp_path), "abc123")
        assert ok is True

    def test_no_slices_passes(self, tmp_path: Path) -> None:
        """No slice files → path check is skipped (main gate will catch empty artifacts)."""
        tasks_dir = self._make_tasks_dir(tmp_path)
        (tasks_dir / "codebase-map.md").write_text("# Codebase Map\n")
        # No slice-*.md files

        ok, reason = _check_tasks_quality(str(tmp_path), "abc123")
        assert ok is True

    def test_slices_with_no_extractable_paths_passes(self, tmp_path: Path) -> None:
        """Slices that contain no recognisable path tokens → check skipped (no false-positive)."""
        tasks_dir = self._make_tasks_dir(tmp_path)
        (tasks_dir / "codebase-map.md").write_text("# Codebase Map\n")
        (tasks_dir / "slice-ui.md").write_text(
            "# UI Slice\nAdd a stop button to the queue page.\nAcceptance: button is visible.\n"
        )

        ok, reason = _check_tasks_quality(str(tmp_path), "abc123")
        assert ok is True


# ── gate_node integration tests ─────────────────────────────────────────


class TestArtifactGateQualityIntegration:
    """End-to-end tests for the gate_node with quality checks active."""

    def _base_state(self, workspace_root: str) -> dict:
        return {
            "work_id": "abc123",
            "workspace_root": workspace_root,
            "artifacts": {
                "tasks": {
                    "tasks.md": "# Tasks\n" + "x" * 100,
                    "slice-alpha.md": "# Slice\n" + "x" * 100,
                }
            },
            "status": "running",
            "feedback": [],
            "prompt_request": None,
            "current_phase": "",
        }

    def test_gate_passes_with_good_tasks(self, tmp_path: Path) -> None:
        """Gate passes when codebase-map.md exists and slice paths are real."""
        # Populate workspace
        tasks_dir = tmp_path / ".spine" / "artifacts" / "abc123" / "tasks"
        tasks_dir.mkdir(parents=True)
        (tasks_dir / "codebase-map.md").write_text("# Codebase Map\n")

        real = tmp_path / "spine" / "work" / "dispatcher.py"
        real.parent.mkdir(parents=True)
        real.write_text("# dispatcher\n")

        (tasks_dir / "slice-alpha.md").write_text(
            "# Slice\n" + "x" * 100 + "\n## Files\n- spine/work/dispatcher.py\n"
        )

        state = self._base_state(str(tmp_path))
        state["artifacts"]["tasks"]["slice-alpha.md"] = (
            "# Slice\n" + "x" * 100 + "\n## Files\n- spine/work/dispatcher.py\n"
        )

        gate = make_artifact_gate_node("tasks", "implement")
        result = gate(state)

        assert result["status"] == "running"

    def test_gate_fails_missing_codebase_map(self, tmp_path: Path) -> None:
        """Gate routes to needs_review when codebase-map.md is absent."""
        tasks_dir = tmp_path / ".spine" / "artifacts" / "abc123" / "tasks"
        tasks_dir.mkdir(parents=True)
        # No codebase-map.md
        (tasks_dir / "slice-alpha.md").write_text("# Slice\n" + "x" * 100)

        state = self._base_state(str(tmp_path))
        gate = make_artifact_gate_node("tasks", "implement")
        result = gate(state)

        assert result["status"] == "needs_review"
        assert len(result["feedback"]) == 1
        assert "codebase-map.md" in result["feedback"][0]["reason"]

    def test_gate_fails_hallucinated_paths(self, tmp_path: Path) -> None:
        """Gate routes to needs_review when all slice paths are hallucinated."""
        tasks_dir = tmp_path / ".spine" / "artifacts" / "abc123" / "tasks"
        tasks_dir.mkdir(parents=True)
        (tasks_dir / "codebase-map.md").write_text("# Codebase Map\n")
        (tasks_dir / "slice-alpha.md").write_text(
            "# Slice\n" + "x" * 100 + "\n## Files\n- src/main.py\n- api/routes.py\n"
        )

        state = self._base_state(str(tmp_path))
        gate = make_artifact_gate_node("tasks", "implement")
        result = gate(state)

        assert result["status"] == "needs_review"
        feedback_reason = result["feedback"][0]["reason"]
        assert (
            "hallucinated" in feedback_reason.lower() or "do not exist" in feedback_reason.lower()
        )

    def test_quality_check_exception_does_not_crash_gate(self, tmp_path: Path) -> None:
        """Exceptions inside the quality check are caught; gate proceeds normally."""
        tasks_dir = tmp_path / ".spine" / "artifacts" / "abc123" / "tasks"
        tasks_dir.mkdir(parents=True)
        (tasks_dir / "codebase-map.md").write_text("# Codebase Map\n")
        (tasks_dir / "slice-alpha.md").write_text("# Slice\n" + "x" * 100)

        state = self._base_state(str(tmp_path))

        # Patch on the module object directly (bypasses dotted-name resolution)
        original = getattr(_ag, "_check_tasks_quality")

        def boom(*args, **kwargs):
            raise RuntimeError("disk error")

        try:
            setattr(_ag, "_check_tasks_quality", boom)
            gate = make_artifact_gate_node("tasks", "implement")
            result = gate(state)
        finally:
            setattr(_ag, "_check_tasks_quality", original)

        # Gate must pass — quality check errors are non-fatal
        assert result["status"] == "running"

    def test_quality_check_only_runs_for_tasks_phase(self, tmp_path: Path) -> None:
        """_check_tasks_quality is NOT called for non-tasks required_phase values."""
        state = {
            "work_id": "abc123",
            "workspace_root": str(tmp_path),
            "artifacts": {
                "implement": {"implementation.md": "# Impl\n" + "x" * 100},
            },
            "status": "running",
            "feedback": [],
            "prompt_request": None,
            "current_phase": "",
        }

        calls: list = []
        original = getattr(_ag, "_check_tasks_quality")

        def tracking_check(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)

        setattr(_ag, "_check_tasks_quality", tracking_check)
        try:
            gate = make_artifact_gate_node("implement", "verify")
            gate(state)
        finally:
            setattr(_ag, "_check_tasks_quality", original)

        assert calls == [], "Quality check must not run for non-tasks phases"
