"""Unit tests for the greenfield onboarding scaffolder (slice2)."""

from __future__ import annotations

from pathlib import Path

import yaml

from spine.config import SpineConfig
from spine.work.onboarding.scaffold import scaffold_project, write_text_idempotent
from spine.work.onboarding.templates import baseline_config_yaml, default_dir_layout


class TestWriteTextIdempotent:
    """Tests for the low-level idempotent writer."""

    def test_creates_file_and_parents(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "c.txt"
        changed = write_text_idempotent(target, "hello")
        assert changed is True
        assert target.read_text(encoding="utf-8") == "hello"

    def test_noop_second_write_returns_false(self, tmp_path: Path) -> None:
        target = tmp_path / "x.txt"
        assert write_text_idempotent(target, "same") is True
        # Identical content on the second write -> no-op.
        assert write_text_idempotent(target, "same") is False

    def test_changed_content_returns_true(self, tmp_path: Path) -> None:
        target = tmp_path / "x.txt"
        write_text_idempotent(target, "one")
        assert write_text_idempotent(target, "two") is True
        assert target.read_text(encoding="utf-8") == "two"

    def test_overwrite_existing_never_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "dir" / "f.txt"
        target.parent.mkdir(parents=True)
        target.write_text("old", encoding="utf-8")
        # Must not raise FileExistsError on overwrite.
        assert write_text_idempotent(target, "new") is True


class TestDefaultDirLayout:
    """Tests for the static layout template."""

    def test_layout_keys_are_file_paths(self) -> None:
        layout = default_dir_layout()
        assert ".spine/skills/.gitkeep" in layout
        assert "src/.gitkeep" in layout
        assert "tests/.gitkeep" in layout

    def test_config_not_in_static_layout(self) -> None:
        # config.yaml is rendered separately (tech-stack dependent).
        assert ".spine/config.yaml" not in default_dir_layout()


class TestBaselineConfigYaml:
    """Tests for the generated config content."""

    def test_parses_via_safe_load(self) -> None:
        text = baseline_config_yaml(["python", "langgraph"])
        data = yaml.safe_load(text)
        assert isinstance(data, dict)
        assert "spine" in data
        spine = data["spine"]
        assert spine["checkpoint_path"] == SpineConfig().checkpoint_path
        assert spine["artifact_path"] == SpineConfig().artifact_path
        assert spine["queue_path"] == SpineConfig().queue_path

    def test_tech_stack_recorded_in_header(self) -> None:
        text = baseline_config_yaml(["python", "streamlit"])
        assert "python, streamlit" in text

    def test_empty_tech_stack_does_not_break(self) -> None:
        text = baseline_config_yaml([])
        assert yaml.safe_load(text) is not None

    def test_scaffolds_codebase_index_mcp_server(self) -> None:
        # Without an active mcp_servers block, a freshly scaffolded project
        # logs "mcp_codebase-index_list_files tool not available" on
        # `spine index`. The codebase-index server ships as a hard spine
        # dependency, so the baseline config wires it up by default.
        data = yaml.safe_load(baseline_config_yaml(["python"]))
        server = data["mcp_servers"]["codebase-index"]
        assert server["command"] == "mcp-codebase-index"
        assert server["transport"] == "stdio"
        # PROJECT_ROOT is injected at load time from workspace_root, so it
        # must NOT be pinned in the scaffolded config.
        assert "PROJECT_ROOT" not in server.get("env", {})


class TestScaffoldProject:
    """End-to-end scaffolding behaviour and acceptance criteria."""

    def test_creates_expected_layout(self, tmp_path: Path) -> None:
        created = scaffold_project(str(tmp_path), ["python"])

        assert (tmp_path / ".spine" / "skills").is_dir()
        assert (tmp_path / ".spine" / "config.yaml").is_file()
        assert (tmp_path / "src").is_dir()
        assert (tmp_path / "tests").is_dir()

        assert ".spine/config.yaml" in created
        assert ".spine/skills/.gitkeep" in created
        assert "src/.gitkeep" in created
        assert "tests/.gitkeep" in created

    def test_returns_repo_relative_paths(self, tmp_path: Path) -> None:
        created = scaffold_project(str(tmp_path), ["python"])
        for rel in created:
            assert not Path(rel).is_absolute()
            assert (tmp_path / rel).exists()

    def test_idempotent_no_exception_identical_tree(self, tmp_path: Path) -> None:
        first = scaffold_project(str(tmp_path), ["python", "langgraph"])
        snapshot_1 = _tree_snapshot(tmp_path)

        # Second run must not raise and must produce an identical tree.
        second = scaffold_project(str(tmp_path), ["python", "langgraph"])
        snapshot_2 = _tree_snapshot(tmp_path)

        assert first == second
        assert snapshot_1 == snapshot_2

    def test_force_rewrites_without_error(self, tmp_path: Path) -> None:
        scaffold_project(str(tmp_path), ["python"])
        before = _tree_snapshot(tmp_path)
        # force=True must still be idempotent in terms of resulting bytes.
        scaffold_project(str(tmp_path), ["python"], force=True)
        assert _tree_snapshot(tmp_path) == before

    def test_generated_config_loads_via_spineconfig(self, tmp_path: Path) -> None:
        scaffold_project(str(tmp_path), ["python", "langgraph"])
        config_path = tmp_path / ".spine" / "config.yaml"

        # Parses as YAML.
        with config_path.open() as f:
            assert yaml.safe_load(f) is not None

        # SpineConfig.load against the scaffolded config succeeds and reads
        # back the baseline keys.  (workspace_root is intentionally left empty
        # in the generated config so SpineConfig auto-detects it at load time;
        # we therefore assert on the explicitly-written keys instead.)
        config = SpineConfig.load(str(config_path))
        assert config.checkpoint_path == SpineConfig().checkpoint_path
        assert config.artifact_path == SpineConfig().artifact_path
        assert config.queue_path == SpineConfig().queue_path

    def test_creates_root_when_missing(self, tmp_path: Path) -> None:
        target = tmp_path / "new_project"
        assert not target.exists()
        scaffold_project(str(target), ["python"])
        assert (target / ".spine" / "config.yaml").is_file()


def _tree_snapshot(root: Path) -> dict[str, str]:
    """Map repo-relative file path -> content for every file under *root*."""
    snapshot: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = str(path.relative_to(root))
            snapshot[rel] = path.read_text(encoding="utf-8")
    return snapshot
