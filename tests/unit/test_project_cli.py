"""Smoke tests for the `spine project` CLI command group."""

from __future__ import annotations

import sys
from pathlib import Path

from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.cli import main


def _write_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "spine:\n"
        f"  project_path: {tmp_path}/.spine/project\n"
        f"  checkpoint_path: {tmp_path}/.spine/spine.db\n"
        f"  artifact_path: {tmp_path}/.spine/artifacts\n"
        f"  queue_path: {tmp_path}/.spine/queue.db\n"
        f"  workspace_root: {tmp_path}\n",
        encoding="utf-8",
    )
    return cfg


def test_project_lifecycle_smoke(tmp_path: Path) -> None:
    runner = CliRunner()
    cfg = str(_write_config(tmp_path))

    create = runner.invoke(
        main, ["project", "create", "demo", "--title", "Demo", "--config", cfg]
    )
    assert create.exit_code == 0, create.output
    assert "Project Created" in create.output

    add = runner.invoke(main, ["project", "add", "demo", "w1", "w2", "--config", cfg])
    assert add.exit_code == 0, add.output
    assert "2 member" in add.output

    show = runner.invoke(main, ["project", "show", "demo", "--config", cfg])
    assert show.exit_code == 0, show.output
    assert "Project: demo" in show.output

    listing = runner.invoke(main, ["project", "list", "--config", cfg])
    assert listing.exit_code == 0, listing.output
    assert "demo" in listing.output


def test_project_create_from_json(tmp_path: Path) -> None:
    runner = CliRunner()
    cfg = str(_write_config(tmp_path))
    spec_json = tmp_path / "spec.json"
    spec_json.write_text(
        '{"title": "Imported", "requirements": [{"id": "R-001", "text": "Do X"}]}',
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        ["project", "create", "imp", "--from-json", str(spec_json), "--config", cfg],
    )
    assert result.exit_code == 0, result.output
    assert "Requirements: 1" in result.output


def test_project_create_rejects_duplicate(tmp_path: Path) -> None:
    runner = CliRunner()
    cfg = str(_write_config(tmp_path))
    runner.invoke(main, ["project", "create", "demo", "--config", cfg])
    dup = runner.invoke(main, ["project", "create", "demo", "--config", cfg])
    assert dup.exit_code == 1
    assert "already exists" in dup.output


def test_project_add_unknown_fails(tmp_path: Path) -> None:
    runner = CliRunner()
    cfg = str(_write_config(tmp_path))
    result = runner.invoke(main, ["project", "add", "ghost", "w1", "--config", cfg])
    assert result.exit_code == 1
    assert "not found" in result.output
