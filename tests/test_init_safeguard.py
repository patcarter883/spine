"""Tests for `spine init` safeguard behaviour.

Covers:
  - Fresh init (no .spine/ exists)
  - Already initialised + --abort (exit 0)
  - Already initialised + --force (full reinit)
  - Already initialised + --keep-config (preserve config)
  - Already initialised + non-interactive (exit 10)
  - Backup creation before destructive operations
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from spine.cli import cli


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


# ---------------------------------------------------------------------------
# Fresh init
# ---------------------------------------------------------------------------

class TestFreshInit:
    def test_creates_spine_dir(self, runner: CliRunner):
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0
            assert Path(".spine").is_dir()

    def test_creates_config(self, runner: CliRunner):
        with runner.isolated_filesystem():
            runner.invoke(cli, ["init"])
            cfg = Path(".spine/config.yaml")
            assert cfg.exists()
            content = cfg.read_text()
            assert "qwen3:32b" in content

    def test_creates_subdirs(self, runner: CliRunner):
        with runner.isolated_filesystem():
            runner.invoke(cli, ["init"])
            expected = [
                ".spine/spec",
                ".spine/state",
                ".spine/knowledge",
                ".spine/events",
                ".spine/artifacts",
            ]
            for d in expected:
                assert Path(d).is_dir(), f"{d} should exist"


# ---------------------------------------------------------------------------
# Abort flag
# ---------------------------------------------------------------------------

class TestAbortFlag:
    def test_exits_cleanly(self, runner: CliRunner):
        with runner.isolated_filesystem():
            runner.invoke(cli, ["init"])  # first init
            result = runner.invoke(cli, ["init", "--abort"])
            assert result.exit_code == 0
            assert "No changes made" in result.output

    def test_preserves_existing_state(self, runner: CliRunner):
        with runner.isolated_filesystem():
            runner.invoke(cli, ["init"])
            # Add a marker file
            Path(".spine/custom.txt").write_text("keep-me")
            runner.invoke(cli, ["init", "--abort"])
            assert Path(".spine/custom.txt").read_text() == "keep-me"


# ---------------------------------------------------------------------------
# Force flag (full reinit)
# ---------------------------------------------------------------------------

class TestForceFlag:
    def test_wipes_existing_data(self, runner: CliRunner):
        with runner.isolated_filesystem():
            runner.invoke(cli, ["init"])
            Path(".spine/custom_marker.txt").write_text("wiped")
            result = runner.invoke(cli, ["init", "--force"])
            assert result.exit_code == 0
            assert not Path(".spine/custom_marker.txt").exists()
            assert "completely reinitialised" in result.output

    def test_recreates_default_config(self, runner: CliRunner):
        with runner.isolated_filesystem():
            runner.invoke(cli, ["init"])
            # Overwrite config with custom content
            Path(".spine/config.yaml").write_text("custom: true")
            runner.invoke(cli, ["init", "--force"])
            content = Path(".spine/config.yaml").read_text()
            assert "custom: true" not in content
            assert "qwen3:32b" in content

    def test_creates_backup(self, runner: CliRunner):
        with runner.isolated_filesystem():
            runner.invoke(cli, ["init"])
            Path(".spine/custom_marker.txt").write_text("backup-me")
            result = runner.invoke(cli, ["init", "--force"])
            # Should have created a backup directory
            backup_dirs = [d for d in os.listdir(".") if d.startswith(".spine_backup_")]
            assert len(backup_dirs) >= 1, "Expected a backup directory"
            # Backup should contain the custom marker
            backup = backup_dirs[0]
            assert Path(backup + "/custom_marker.txt").read_text() == "backup-me"


# ---------------------------------------------------------------------------
# Keep-config flag
# ---------------------------------------------------------------------------

class TestKeepConfigFlag:
    def test_preserves_custom_config(self, runner: CliRunner):
        with runner.isolated_filesystem():
            runner.invoke(cli, ["init"])
            # Customise config
            cfg = Path(".spine/config.yaml")
            cfg.write_text(cfg.read_text().replace("qwen3:32b", "my-model:v1"))
            result = runner.invoke(cli, ["init", "--keep-config"])
            assert result.exit_code == 0
            content = cfg.read_text()
            assert "my-model:v1" in content
            assert "Preserved existing configuration" in result.output

    def test_wipes_non_config_files(self, runner: CliRunner):
        with runner.isolated_filesystem():
            runner.invoke(cli, ["init"])
            Path(".spine/custom_marker.txt").write_text("wiped")
            runner.invoke(cli, ["init", "--keep-config"])
            assert not Path(".spine/custom_marker.txt").exists()

    def test_creates_backup(self, runner: CliRunner):
        with runner.isolated_filesystem():
            runner.invoke(cli, ["init"])
            result = runner.invoke(cli, ["init", "--keep-config"])
            backup_dirs = [d for d in os.listdir(".") if d.startswith(".spine_backup_")]
            assert len(backup_dirs) >= 1


# ---------------------------------------------------------------------------
# Non-interactive (no TTY, no flags)
# ---------------------------------------------------------------------------

class TestNonInteractive:
    def test_exits_with_code_10(self, runner: CliRunner):
        with runner.isolated_filesystem():
            runner.invoke(cli, ["init"])  # first init
            # CliRunner doesn't provide a TTY, so this should trigger the
            # non-interactive path
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 10
            assert "not a TTY" in result.output


# ---------------------------------------------------------------------------
# Interactive prompt (mock stdin.isatty)
# ---------------------------------------------------------------------------

class TestInteractivePrompt:
    def test_option_1_via_force_flag(self, runner: CliRunner):
        """The interactive 'option 1' code path is exercised by --force."""
        with runner.isolated_filesystem():
            runner.invoke(cli, ["init"])
            Path(".spine/custom_marker.txt").write_text("wiped")
            result = runner.invoke(cli, ["init", "--force"])
            assert result.exit_code == 0
            assert not Path(".spine/custom_marker.txt").exists()
            assert "completely reinitialised" in result.output

    def test_option_3_do_nothing(self, runner: CliRunner):
        """Verify the 'do nothing' path via --abort flag (same code path)."""
        with runner.isolated_filesystem():
            runner.invoke(cli, ["init"])
            result = runner.invoke(cli, ["init", "--abort"])
            assert result.exit_code == 0
            assert "No changes made" in result.output


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_partial_init_counts_as_initialised(self, runner: CliRunner):
        """If .spine/ exists but is incomplete, it's still detected."""
        with runner.isolated_filesystem():
            os.makedirs(".spine", exist_ok=True)
            # No config.yaml, no subdirs — but .spine/ exists
            result = runner.invoke(cli, ["init"])
            # Should NOT do a fresh init; should hit safeguard
            assert result.exit_code == 10  # non-interactive fallback
            assert "already initialised" in result.output

    def test_double_fresh_init_idempotent(self, runner: CliRunner):
        """Running init twice should NOT silently overwrite."""
        with runner.isolated_filesystem():
            runner.invoke(cli, ["init"])
            Path(".spine/config.yaml").write_text("custom: true")
            # Second init without flags should trigger safeguard
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 10
            # Config should NOT have been overwritten
            assert Path(".spine/config.yaml").read_text() == "custom: true"
