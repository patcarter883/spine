"""Unit tests for `spine init` and the brownfield init_workspace helper."""

from __future__ import annotations

from pathlib import Path

import yaml
from click.testing import CliRunner

from spine.cli import main
from spine.config import SpineConfig
from spine.work.onboarding.init import init_workspace


class TestInitWorkspace:
    """Tests for the brownfield-safe init_workspace() helper."""

    def test_creates_only_dot_spine(self, tmp_path: Path) -> None:
        managed, preserved = init_workspace(str(tmp_path), ["python"])

        assert (tmp_path / ".spine" / "config.yaml").is_file()
        assert (tmp_path / ".spine" / "skills" / ".gitkeep").is_file()
        assert (tmp_path / ".spine" / "artifacts" / ".gitkeep").is_file()
        # Brownfield contract: never touch src/ or tests/.
        assert not (tmp_path / "src").exists()
        assert not (tmp_path / "tests").exists()

        assert set(managed) == {
            ".spine/config.yaml",
            ".spine/skills/.gitkeep",
            ".spine/artifacts/.gitkeep",
        }
        assert preserved == []

    def test_generated_config_parses(self, tmp_path: Path) -> None:
        init_workspace(str(tmp_path), ["python", "langgraph"])
        cfg_path = tmp_path / ".spine" / "config.yaml"

        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        assert "spine" in data
        assert data["providers"] == {"llm": [], "embedding": []}

        config = SpineConfig.load(str(cfg_path))
        assert config.checkpoint_path == SpineConfig().checkpoint_path
        assert config.artifact_path == SpineConfig().artifact_path

    def test_config_contains_commented_provider_stubs(self, tmp_path: Path) -> None:
        init_workspace(str(tmp_path), ["python"])
        text = (tmp_path / ".spine" / "config.yaml").read_text(encoding="utf-8")

        assert "Example providers" in text
        assert "# providers:" in text
        assert "openrouter-default" in text
        assert "local-embeddings" in text
        # Every stub line must remain a YAML comment so the document still
        # parses with providers.llm/embedding as empty arrays.
        data = yaml.safe_load(text)
        assert data["providers"] == {"llm": [], "embedding": []}

    def test_idempotent_second_run(self, tmp_path: Path) -> None:
        first_managed, _ = init_workspace(str(tmp_path), ["python"])
        snapshot = _tree_snapshot(tmp_path)

        second_managed, preserved = init_workspace(str(tmp_path), ["python"])

        assert first_managed == second_managed
        assert preserved == []
        assert _tree_snapshot(tmp_path) == snapshot

    def test_preserves_modified_config_without_force(self, tmp_path: Path) -> None:
        init_workspace(str(tmp_path), ["python"])
        cfg_path = tmp_path / ".spine" / "config.yaml"
        cfg_path.write_text("spine:\n  checkpoint_path: custom.db\n", encoding="utf-8")

        _, preserved = init_workspace(str(tmp_path), ["python"])

        assert preserved == [".spine/config.yaml"]
        # Existing content untouched.
        assert "custom.db" in cfg_path.read_text(encoding="utf-8")

    def test_force_overwrites_modified_config(self, tmp_path: Path) -> None:
        init_workspace(str(tmp_path), ["python"])
        cfg_path = tmp_path / ".spine" / "config.yaml"
        cfg_path.write_text("spine:\n  checkpoint_path: custom.db\n", encoding="utf-8")

        _, preserved = init_workspace(str(tmp_path), ["python"], force=True)

        assert preserved == []
        assert "custom.db" not in cfg_path.read_text(encoding="utf-8")
        assert "Example providers" in cfg_path.read_text(encoding="utf-8")

    def test_creates_root_when_missing(self, tmp_path: Path) -> None:
        target = tmp_path / "fresh"
        assert not target.exists()
        init_workspace(str(target), ["python"])
        assert (target / ".spine" / "config.yaml").is_file()


class TestInitCliCommand:
    """End-to-end tests for the `spine init` Click command."""

    def test_init_succeeds_in_empty_dir(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            main, ["init", str(tmp_path), "--tech-stack", "python"]
        )
        assert result.exit_code == 0, result.output
        assert "SPINE initialized" in result.output
        assert (tmp_path / ".spine" / "config.yaml").is_file()
        # Brownfield-safe: no src/ or tests/ from `spine init`.
        assert not (tmp_path / "src").exists()
        assert not (tmp_path / "tests").exists()

    def test_init_idempotent(self, tmp_path: Path) -> None:
        runner = CliRunner()
        first = runner.invoke(main, ["init", str(tmp_path), "--tech-stack", "python"])
        assert first.exit_code == 0

        second = runner.invoke(main, ["init", str(tmp_path), "--tech-stack", "python"])
        assert second.exit_code == 0
        # No "preserved" warning when content is byte-identical.
        assert "Re-run with --force" not in second.output

    def test_init_warns_on_preserved_modified_config(self, tmp_path: Path) -> None:
        runner = CliRunner()
        runner.invoke(main, ["init", str(tmp_path), "--tech-stack", "python"])
        cfg_path = tmp_path / ".spine" / "config.yaml"
        cfg_path.write_text("spine:\n  checkpoint_path: edited.db\n", encoding="utf-8")

        result = runner.invoke(main, ["init", str(tmp_path), "--tech-stack", "python"])
        assert result.exit_code == 0
        assert "Preserved" in result.output
        assert "edited.db" in cfg_path.read_text(encoding="utf-8")

    def test_init_force_overwrites(self, tmp_path: Path) -> None:
        runner = CliRunner()
        runner.invoke(main, ["init", str(tmp_path), "--tech-stack", "python"])
        cfg_path = tmp_path / ".spine" / "config.yaml"
        cfg_path.write_text("spine:\n  checkpoint_path: edited.db\n", encoding="utf-8")

        result = runner.invoke(
            main, ["init", str(tmp_path), "--tech-stack", "python", "--force"]
        )
        assert result.exit_code == 0
        assert "edited.db" not in cfg_path.read_text(encoding="utf-8")

    def test_init_listed_in_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "init" in result.output


def _tree_snapshot(root: Path) -> dict[str, str]:
    """Map repo-relative file path -> content for every file under *root*."""
    snapshot: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = str(path.relative_to(root))
            snapshot[rel] = path.read_text(encoding="utf-8")
    return snapshot
