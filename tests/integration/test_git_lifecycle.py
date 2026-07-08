"""End-to-end git-lifecycle tests against a real throwaway repo in tmp_path."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from spine.config import SpineConfig
from spine.git.orchestrator import SpineGitOrchestrator

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git is not available on PATH"
)


def _git(args: list[str], cwd: Path) -> str:
    """Run a git command in ``cwd`` and return stripped stdout."""
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return proc.stdout.strip()


@pytest.fixture
def git_repo(tmp_path):
    """Initialise a throwaway git repo on ``main`` with one commit.

    Yields ``(repo_dir, SpineConfig)`` where the config points at the repo.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["init"], repo)
    _git(["config", "user.email", "test@example.com"], repo)
    _git(["config", "user.name", "Test User"], repo)
    _git(["config", "commit.gpgsign", "false"], repo)
    # Force a deterministic main branch regardless of git defaults.
    _git(["checkout", "-b", "main"], repo)
    (repo / "README.md").write_text("initial\n", encoding="utf-8")
    _git(["add", "."], repo)
    _git(["commit", "-m", "initial commit"], repo)

    config = SpineConfig()
    config.workspace_root = str(repo)
    config.checkpoint_path = str(repo / ".spine" / "spine.db")
    config.queue_path = str(repo / ".spine" / "queue.db")

    yield repo, config


def _build_orchestrator(repo: Path, config: SpineConfig, tmp_path: Path):
    """Construct an orchestrator anchored to ``repo`` with a trivial pipeline."""
    orch = SpineGitOrchestrator(
        config_path=str(tmp_path / "no-gate.yaml"),
        base_config=config,
    )
    # The orchestrator records cwd at init; force it to the throwaway repo so
    # every git command targets the real test repo (never the spine repo).
    orch.master_dir = str(repo)
    orch.main_branch = "main"
    orch.strategy = "worktree"
    orch.sandbox_dir_base = str(tmp_path / "sandbox")
    # Trivial always-passing validation gate.
    orch.gate_config["validation_pipeline"] = {
        "noop": {"command": "true", "timeout_seconds": 10},
    }
    return orch


def test_prepare_and_merge_round_trip(git_repo, tmp_path):
    repo, config = git_repo
    orch = _build_orchestrator(repo, config, tmp_path)

    sandbox = orch.prepare_sandbox()
    sandbox_dir = Path(sandbox)
    assert sandbox_dir.is_dir()

    # Simulate a "workflow" producing a new file in the sandbox.
    (sandbox_dir / "feature.txt").write_text("hello from sandbox\n", encoding="utf-8")

    validation = orch.run_validation_pipeline()
    assert validation == {"success": True}

    merge = orch.commit_and_merge()
    assert merge["success"] is True
    assert merge["merged"] is True

    # The file landed on main in the master tree.
    merged_file = repo / "feature.txt"
    assert merged_file.is_file()
    assert merged_file.read_text(encoding="utf-8") == "hello from sandbox\n"

    # Current branch is main.
    assert _git(["rev-parse", "--abbrev-ref", "HEAD"], repo) == "main"

    # The worktree directory was removed.
    assert not sandbox_dir.exists()

    # The worktree registration was cleaned up (no lingering worktree).
    worktrees = _git(["worktree", "list"], repo)
    assert str(sandbox_dir) not in worktrees

    # Working tree is clean.
    assert _git(["status", "--porcelain"], repo) == ""


def test_rollback_leaves_main_pristine(git_repo, tmp_path):
    repo, config = git_repo
    orch = _build_orchestrator(repo, config, tmp_path)
    # Make the single gate fail so validation rejects the patch.
    orch.gate_config["validation_pipeline"] = {
        "fail": {"command": "false", "timeout_seconds": 10},
    }

    head_before = _git(["rev-parse", "HEAD"], repo)

    sandbox = orch.prepare_sandbox()
    sandbox_dir = Path(sandbox)
    # Stray file in the sandbox that must NOT reach main.
    (sandbox_dir / "stray.txt").write_text("should not survive\n", encoding="utf-8")

    validation = orch.run_validation_pipeline()
    assert validation["success"] is False

    result = orch.rollback_workspace()
    assert result == {"rolled_back": True}

    # main is unchanged: same HEAD, clean tree, stray file absent.
    assert _git(["rev-parse", "--abbrev-ref", "HEAD"], repo) == "main"
    assert _git(["rev-parse", "HEAD"], repo) == head_before
    assert _git(["status", "--porcelain"], repo) == ""
    assert not (repo / "stray.txt").exists()
    assert not sandbox_dir.exists()

    # The patch branch was deleted.
    branches = _git(["branch", "--list", orch.patch_branch], repo)
    assert orch.patch_branch not in branches


def test_prepare_sandbox_rejects_dirty_tree(git_repo, tmp_path):
    from spine.exceptions import SandboxPreparationError

    repo, config = git_repo
    orch = _build_orchestrator(repo, config, tmp_path)

    # Dirty the master tree.
    (repo / "dirty.txt").write_text("uncommitted\n", encoding="utf-8")

    with pytest.raises(SandboxPreparationError):
        orch.prepare_sandbox()


def test_merge_lands_when_main_advanced_mid_run(git_repo, tmp_path):
    """A verified patch must land even when main moved during the run.

    Regression: commit_and_merge used a bare --ff-only merge, so ANY commit
    that reached main while the sandbox was in flight (another work item, a
    human push) failed the merge and discarded the verified patch. The patch
    branch is now rebased onto main first.
    """
    repo, config = git_repo
    orch = _build_orchestrator(repo, config, tmp_path)

    sandbox = orch.prepare_sandbox()
    sandbox_dir = Path(sandbox)
    (sandbox_dir / "feature.txt").write_text("hello from sandbox\n", encoding="utf-8")

    # Simulate main advancing while the run is in flight (disjoint file).
    (repo / "other.txt").write_text("landed by someone else\n", encoding="utf-8")
    _git(["add", "."], repo)
    _git(["commit", "-m", "mid-run commit on main"], repo)

    merge = orch.commit_and_merge()
    assert merge["success"] is True
    assert merge["merged"] is True

    # Both the mid-run commit and the sandbox patch are on main.
    assert (repo / "other.txt").is_file()
    assert (repo / "feature.txt").read_text(encoding="utf-8") == "hello from sandbox\n"
    assert _git(["rev-parse", "--abbrev-ref", "HEAD"], repo) == "main"


def test_merge_conflict_with_advanced_main_raises(git_repo, tmp_path):
    """A genuine overlap between the patch and mid-run main commits must
    surface as MergeError (triggering the caller's rollback), not land a
    mangled tree."""
    from spine.git.orchestrator import MergeError

    repo, config = git_repo
    orch = _build_orchestrator(repo, config, tmp_path)

    sandbox = orch.prepare_sandbox()
    sandbox_dir = Path(sandbox)
    (sandbox_dir / "README.md").write_text("sandbox version\n", encoding="utf-8")

    # Conflicting mid-run commit on main touching the same file.
    (repo / "README.md").write_text("main version\n", encoding="utf-8")
    _git(["add", "."], repo)
    _git(["commit", "-m", "conflicting mid-run commit"], repo)

    with pytest.raises(MergeError):
        orch.commit_and_merge()
