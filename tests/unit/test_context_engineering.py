"""Tests for SPINE context engineering modules.

Verifies:
1. SpineContext dataclass construction and build_context helper
2. Artifact materialization to filesystem
3. Artifact prompt generation (references by path, not inlined)
4. Skills resolver for phase-specific skills
5. Memory resolver for AGENTS.md files
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestSpineContext:
    """Tests for SpineContext dataclass and build_context helper."""

    def test_default_values(self) -> None:
        """SpineContext should have sensible defaults."""
        from spine.agents.context import SpineContext

        ctx = SpineContext()
        assert ctx.work_id == ""
        assert ctx.phase == ""
        assert ctx.workspace_root == "."
        assert ctx.retry_count == 0
        assert ctx.is_rework is False
        assert ctx.critic_feedback == []
        assert ctx.artifact_paths == {}

    def test_build_context_basic(self) -> None:
        """build_context should extract values from state."""
        from spine.agents.context import build_context

        state = {
            "work_id": "work-123",
            "workspace_root": "/home/user/project",
            "retry_count": {"specify": 0, "implement": 2},
            "feedback": [
                {"tier": "agent", "reason": "Missing tests"},
                {"tier": "structural", "reason": "Too short"},
            ],
            "artifacts": {
                "specify": {"specification.md": "# Spec"},
                "plan": {"plan.md": "# Plan"},
            },
        }

        ctx = build_context(state, "implement")
        assert ctx.work_id == "work-123"
        assert ctx.phase == "implement"
        assert ctx.workspace_root == "/home/user/project"
        assert ctx.retry_count == 2
        assert ctx.is_rework is True
        assert len(ctx.critic_feedback) == 2
        assert "[agent]" in ctx.critic_feedback[0]

    def test_build_context_no_rework(self) -> None:
        """is_rework should be False when retry_count is 0."""
        from spine.agents.context import build_context

        state = {
            "work_id": "work-456",
            "workspace_root": "/tmp",
            "retry_count": {"specify": 0},
            "feedback": [],
            "artifacts": {},
        }
        ctx = build_context(state, "specify")
        assert ctx.is_rework is False

    def test_build_context_with_phase_enum(self) -> None:
        """build_context should accept PhaseName enum."""
        from spine.agents.context import build_context
        from spine.models.enums import PhaseName

        state = {"work_id": "w1", "retry_count": {}, "feedback": [], "artifacts": {}}
        ctx = build_context(state, PhaseName.SPECIFY)
        assert ctx.phase == "specify"


class TestArtifactMaterializer:
    """Tests for artifact materialization to filesystem."""

    def test_materialize_writes_files(self) -> None:
        """materialize_artifacts should write artifact content to disk."""
        from spine.agents.artifacts import materialize_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            state = {
                "artifacts": {
                    "specify": {"specification.md": "# My Spec"},
                    "plan": {"plan.md": "# My Plan"},
                }
            }
            paths = materialize_artifacts(state, tmpdir)

            assert "specify" in paths
            assert "plan" in paths

            spec_path = Path(tmpdir) / ".spine/artifacts/specify/specification.md"
            assert spec_path.exists()
            assert spec_path.read_text() == "# My Spec"

            plan_path = Path(tmpdir) / ".spine/artifacts/plan/plan.md"
            assert plan_path.exists()
            assert plan_path.read_text() == "# My Plan"

    def test_materialize_empty_artifacts(self) -> None:
        """materialize_artifacts should handle empty artifacts gracefully."""
        from spine.agents.artifacts import materialize_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = materialize_artifacts({"artifacts": {}}, tmpdir)
            assert paths == {}

    def test_materialize_skips_empty_content(self) -> None:
        """materialize_artifacts should skip artifacts with empty content."""
        from spine.agents.artifacts import materialize_artifacts

        with tempfile.TemporaryDirectory() as tmpdir:
            state = {
                "artifacts": {
                    "specify": {"empty.md": "", "has_content.md": "Hello"},
                }
            }
            materialize_artifacts(state, tmpdir)

            # has_content.md should exist
            assert (Path(tmpdir) / ".spine/artifacts/specify/has_content.md").exists()
            # empty.md should NOT be written (empty content)
            # (the directory still gets created but the file is skipped)
            assert not (Path(tmpdir) / ".spine/artifacts/specify/empty.md").exists()


class TestBuildArtifactPrompt:
    """Tests for the artifact prompt builder."""

    def test_no_artifacts_returns_empty(self) -> None:
        """build_artifact_prompt should return empty string for no artifacts."""
        from spine.agents.artifacts import build_artifact_prompt

        result = build_artifact_prompt({}, "implement")
        assert result == ""

    def test_references_by_path(self) -> None:
        """build_artifact_prompt should reference artifacts by path, not content."""
        from spine.agents.artifacts import build_artifact_prompt

        artifacts = {
            "specify": {"specification.md": "# " + "Long spec\n" * 1000},
            "plan": {"plan.md": "# " + "Long plan\n" * 1000},
        }

        result = build_artifact_prompt(artifacts, "implement")
        assert ".spine/artifacts/specify" in result
        assert ".spine/artifacts/plan" in result
        # Content should NOT be inlined
        assert "Long spec" not in result
        assert "Long plan" not in result

    def test_skips_current_phase(self) -> None:
        """build_artifact_prompt should not reference the current phase."""
        from spine.agents.artifacts import build_artifact_prompt

        artifacts = {
            "specify": {"specification.md": "# Spec"},
            "plan": {"plan.md": "# Plan"},
        }

        result = build_artifact_prompt(artifacts, "specify")
        # Should reference plan but NOT specify (that's the current phase)
        assert "specify" not in result.lower().replace("prior artifacts", "")
        assert ".spine/artifacts/plan" in result

    def test_lists_individual_files(self) -> None:
        """build_artifact_prompt should list individual file names."""
        from spine.agents.artifacts import build_artifact_prompt

        artifacts = {
            "specify": {"specification.md": "# Spec", "diagram.md": "# Diagram"},
        }

        result = build_artifact_prompt(artifacts, "implement")
        assert "specification.md" in result
        assert "diagram.md" in result


class TestSkillsResolver:
    """Tests for the skills resolver."""

    def test_specify_gets_spec_writing_skill(self) -> None:
        """SPECIFY phase should get spec-writing skill."""
        from spine.agents.skills_resolver import resolve_skills

        skills = resolve_skills("specify")
        skill_names = [s.split("/")[-1] for s in skills]
        assert "spec-writing" in skill_names

    def test_tasks_gets_decomposition_skill(self) -> None:
        """TASKS phase should get feature-slice-decomposition skill."""
        from spine.agents.skills_resolver import resolve_skills

        skills = resolve_skills("tasks")
        skill_names = [s.split("/")[-1] for s in skills]
        assert "feature-slice-decomposition" in skill_names

    def test_verify_gets_code_review_skill(self) -> None:
        """VERIFY phase should get code-review skill."""
        from spine.agents.skills_resolver import resolve_skills

        skills = resolve_skills("verify")
        skill_names = [s.split("/")[-1] for s in skills]
        assert "code-review" in skill_names

    def test_rlm_skill_included_when_requested(self) -> None:
        """RLM pattern skill should be included when include_rlm=True."""
        from spine.agents.skills_resolver import resolve_skills

        skills = resolve_skills("specify", include_rlm=True)
        skill_names = [s.split("/")[-1] for s in skills]
        assert "rlm-pattern" in skill_names

    def test_rlm_skill_excluded_by_default(self) -> None:
        """RLM pattern skill should NOT be included by default."""
        from spine.agents.skills_resolver import resolve_skills

        skills = resolve_skills("specify", include_rlm=False)
        skill_names = [s.split("/")[-1] for s in skills]
        assert "rlm-pattern" not in skill_names

    def test_skill_paths_are_absolute(self) -> None:
        """All skill paths should be absolute."""
        from spine.agents.skills_resolver import resolve_skills

        skills = resolve_skills("specify")
        for path in skills:
            assert Path(path).is_absolute()

    def test_workspace_skills_discovered(self) -> None:
        """Skills in .spine/skills/ should be discovered."""
        from spine.agents.skills_resolver import resolve_skills

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a workspace skill
            skill_dir = Path(tmpdir) / ".spine" / "skills" / "my-skill"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("---\nname: my-skill\n---\nMy skill")

            skills = resolve_skills("specify", workspace_root=tmpdir)
            skill_names = [s.split("/")[-1] for s in skills]
            assert "my-skill" in skill_names


class TestMemoryResolver:
    """Tests for the memory resolver."""

    def test_no_workspace_returns_empty(self) -> None:
        """resolve_memory with no workspace should return empty list."""
        from spine.agents.skills_resolver import resolve_memory

        result = resolve_memory(None)
        assert result == []

    def test_finds_agents_md(self) -> None:
        """resolve_memory should find AGENTS.md at workspace root."""
        from spine.agents.skills_resolver import resolve_memory

        with tempfile.TemporaryDirectory() as tmpdir:
            agents_md = Path(tmpdir) / "AGENTS.md"
            agents_md.write_text("# Project conventions")

            result = resolve_memory(tmpdir)
            assert len(result) == 1
            assert "AGENTS.md" in result[0]

    def test_finds_spine_agents_md(self) -> None:
        """resolve_memory should find .spine/AGENTS.md."""
        from spine.agents.skills_resolver import resolve_memory

        with tempfile.TemporaryDirectory() as tmpdir:
            spine_dir = Path(tmpdir) / ".spine"
            spine_dir.mkdir()
            (spine_dir / "AGENTS.md").write_text("# SPINE conventions")

            result = resolve_memory(tmpdir)
            assert len(result) == 1

    def test_finds_both_agents_files(self) -> None:
        """resolve_memory should find both AGENTS.md files."""
        from spine.agents.skills_resolver import resolve_memory

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "AGENTS.md").write_text("# Project conventions")
            spine_dir = Path(tmpdir) / ".spine"
            spine_dir.mkdir()
            (spine_dir / "AGENTS.md").write_text("# SPINE conventions")

            result = resolve_memory(tmpdir)
            assert len(result) == 2
