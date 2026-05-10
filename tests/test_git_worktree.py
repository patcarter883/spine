"""Tests for spine.git - parallel worktree management and PR automation."""

import os
import sys
import json
import subprocess
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, call

# Ensure spine package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spine.git.worktree_manager import (
    WorktreeManager,
    WorktreeInfo,
    WorktreeCreationError,
    WorktreeCleanupError,
)
from spine.git.pr_handler import (
    PRHandler,
    PRInfo,
    PRCreationError,
    PRStatusError,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_git_dir():
    """Create a temporary directory with a git repo for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "test-repo"
        repo_dir.mkdir()
        # Initialize a git repo
        import subprocess
        subprocess.run(["git", "init", str(repo_dir)], check=True, capture_output=True)
        # Configure git user for commits
        subprocess.run(
            ["git", "-C", str(repo_dir), "config", "user.email", "test@test.com"],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "config", "user.name", "Test User"],
            check=True, capture_output=True,
        )
        # Create an initial commit so we have a branch to work from
        (repo_dir / "README.md").write_text("# Test Repo")
        subprocess.run(
            ["git", "-C", str(repo_dir), "add", "README.md"],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "commit", "-m", "initial commit"],
            check=True, capture_output=True,
        )
        yield str(repo_dir)


@pytest.fixture
def worktree_manager(temp_git_dir):
    """Create a WorktreeManager with a temp git repo."""
    return WorktreeManager(repo_path=temp_git_dir)


@pytest.fixture
def pr_handler(temp_git_dir):
    """Create a PRHandler with a temp repo."""
    return PRHandler(repo_path=temp_git_dir)


# ============================================================================
# WorktreeInfo tests
# ============================================================================

class TestWorktreeInfo:
    """Tests for the WorktreeInfo dataclass."""

    def test_worktree_info_fields(self):
        """WorktreeInfo should have the expected fields."""
        info = WorktreeInfo(
            task_id="task-001",
            path="/tmp/worktrees/task-001",
            branch="feature/task-001",
            base_branch="main",
        )
        assert info.task_id == "task-001"
        assert info.path == "/tmp/worktrees/task-001"
        assert info.branch == "feature/task-001"
        assert info.base_branch == "main"

    def test_worktree_info_defaults(self):
        """WorktreeInfo should have sensible defaults."""
        info = WorktreeInfo(
            task_id="task-002",
            path="/some/path",
            branch="feature/task-002",
        )
        assert info.base_branch == "main"
        assert info.task_id == "task-002"

    def test_worktree_info_to_dict(self):
        """WorktreeInfo.to_dict should return correct dict."""
        info = WorktreeInfo(
            task_id="task-003",
            path="/wt/task-003",
            branch="feature/task-003",
            base_branch="develop",
        )
        d = info.to_dict()
        assert d["task_id"] == "task-003"
        assert d["path"] == "/wt/task-003"
        assert d["branch"] == "feature/task-003"
        assert d["base_branch"] == "develop"


# ============================================================================
# WorktreeManager initialization tests
# ============================================================================

class TestWorktreeManagerInit:
    """Tests for WorktreeManager initialization."""

    def test_init_with_repo_path(self, temp_git_dir):
        """WorktreeManager should initialize with a repo path."""
        wm = WorktreeManager(repo_path=temp_git_dir)
        assert wm.repo_path == temp_git_dir
        assert wm.worktree_base == os.path.join(temp_git_dir, ".spine", "worktrees")

    def test_init_worktree_base_created(self, temp_git_dir):
        """WorktreeManager should create worktree_base directory."""
        wm = WorktreeManager(repo_path=temp_git_dir)
        assert os.path.isdir(wm.worktree_base)

    def test_init_default_worktree_base(self, temp_git_dir):
        """Default worktree_base should be .spine/worktrees under repo."""
        wm = WorktreeManager(repo_path=temp_git_dir)
        assert wm.worktree_base.endswith(".spine/worktrees")


# ============================================================================
# WorktreeManager.create_worktree tests
# ============================================================================

class TestWorktreeManagerCreate:
    """Tests for WorktreeManager.create_worktree()."""

    def test_create_worktree_basic(self, worktree_manager):
        """create_worktree should create a new worktree."""
        info = worktree_manager.create_worktree("task-001")
        assert info.task_id == "task-001"
        assert os.path.isdir(info.path)
        assert info.branch.startswith("spine/task-001")

    def test_create_worktree_returns_worktree_info(self, worktree_manager):
        """create_worktree should return a WorktreeInfo instance."""
        info = worktree_manager.create_worktree("task-002")
        assert isinstance(info, WorktreeInfo)
        assert info.task_id == "task-002"

    def test_create_worktree_sets_correct_branch(self, worktree_manager):
        """create_worktree should create a branch named spine/<task_id>-<short_hash>."""
        info = worktree_manager.create_worktree("my-feature")
        assert info.branch.startswith("spine/my-feature")

    def test_create_worktree_with_base_branch(self, worktree_manager):
        """create_worktree should accept a custom base branch."""
        info = worktree_manager.create_worktree("task-003", base_branch="main")
        assert info.base_branch == "main"

    def test_create_worktree_unique_paths(self, worktree_manager):
        """create_worktree should generate unique paths per task."""
        info1 = worktree_manager.create_worktree("task-a")
        info2 = worktree_manager.create_worktree("task-b")
        assert info1.path != info2.path

    def test_create_worktree_raises_on_invalid_repo(self):
        """create_worktree should raise WorktreeCreationError for non-git dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            wm = WorktreeManager(repo_path=tmpdir)
            with pytest.raises(WorktreeCreationError):
                wm.create_worktree("task-fail")

    def test_create_worktree_sanitizes_task_id(self, worktree_manager):
        """create_worktree should sanitize task IDs with special characters."""
        info = worktree_manager.create_worktree("task/with spaces & stuff!")
        assert "/" not in info.branch.split("/", 1)[1] if "/" in info.branch else True
        assert " " not in info.branch


# ============================================================================
# WorktreeManager.list_worktrees tests
# ============================================================================

class TestWorktreeManagerList:
    """Tests for WorktreeManager.list_worktrees()."""

    def test_list_worktrees_empty(self, worktree_manager):
        """list_worktrees should return empty list when no worktrees exist."""
        worktrees = worktree_manager.list_worktrees()
        assert worktrees == []

    def test_list_worktrees_after_create(self, worktree_manager):
        """list_worktrees should return created worktrees."""
        info = worktree_manager.create_worktree("task-list-test")
        worktrees = worktree_manager.list_worktrees()
        assert len(worktrees) >= 1
        paths = [w.path for w in worktrees]
        assert info.path in paths

    def test_list_worktrees_multiple(self, worktree_manager):
        """list_worktrees should return all worktrees."""
        info1 = worktree_manager.create_worktree("task-m1")
        info2 = worktree_manager.create_worktree("task-m2")
        worktrees = worktree_manager.list_worktrees()
        paths = [w.path for w in worktrees]
        assert info1.path in paths
        assert info2.path in paths


# ============================================================================
# WorktreeManager.cleanup_worktree tests
# ============================================================================

class TestWorktreeManagerCleanup:
    """Tests for WorktreeManager.cleanup_worktree()."""

    def test_cleanup_worktree_removes_directory(self, worktree_manager):
        """cleanup_worktree should remove the worktree directory."""
        info = worktree_manager.create_worktree("task-cleanup")
        assert os.path.isdir(info.path)
        worktree_manager.cleanup_worktree("task-cleanup")
        assert not os.path.exists(info.path)

    def test_cleanup_nonexistent_task(self, worktree_manager):
        """cleanup_worktree should handle nonexistent task IDs gracefully."""
        # Should not raise
        worktree_manager.cleanup_worktree("nonexistent-task")

    def test_cleanup_removes_branch(self, worktree_manager):
        """cleanup_worktree should remove the associated branch."""
        info = worktree_manager.create_worktree("task-branch-cleanup")
        branch = info.branch
        worktree_manager.cleanup_worktree("task-branch-cleanup")
        # Verify branch is gone
        import subprocess
        result = subprocess.run(
            ["git", "-C", worktree_manager.repo_path, "branch", "--list", branch],
            capture_output=True, text=True,
        )
        assert branch not in result.stdout

    def test_cleanup_all_worktrees(self, worktree_manager):
        """cleanup_all should remove all managed worktrees."""
        wm = worktree_manager
        info1 = wm.create_worktree("task-all-1")
        info2 = wm.create_worktree("task-all-2")
        assert os.path.isdir(info1.path)
        assert os.path.isdir(info2.path)
        cleaned = wm.cleanup_all()
        assert cleaned >= 2
        assert not os.path.exists(info1.path)
        assert not os.path.exists(info2.path)


# ============================================================================
# WorktreeManager.get_worktree_path tests
# ============================================================================

class TestGetWorktreePath:
    """Tests for WorktreeManager.get_worktree_path()."""

    def test_get_existing_worktree_path(self, worktree_manager):
        """get_worktree_path should return path for existing worktree."""
        info = worktree_manager.create_worktree("task-gwp")
        path = worktree_manager.get_worktree_path("task-gwp")
        assert path == info.path

    def test_get_nonexistent_worktree_path(self, worktree_manager):
        """get_worktree_path should return None for nonexistent task."""
        path = worktree_manager.get_worktree_path("no-such-task")
        assert path is None


# ============================================================================
# WorktreeManager subprocess error handling tests
# ============================================================================

class TestWorktreeManagerErrors:
    """Tests for WorktreeManager error handling."""

    def test_create_worktree_raises_on_git_failure(self, temp_git_dir):
        """create_worktree should raise on git command failure."""
        wm = WorktreeManager(repo_path=temp_git_dir)
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("git failure")
            with pytest.raises(WorktreeCreationError, match="git failure"):
                wm.create_worktree("task-err")

    def test_list_worktrees_handles_git_list_failure(self, worktree_manager):
        """list_worktrees should handle git worktree list failure gracefully."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("git list failed")
            worktrees = worktree_manager.list_worktrees()
            assert worktrees == []

    def test_cleanup_raises_on_git_failure(self, worktree_manager):
        """cleanup_worktree should raise on git command failure during cleanup."""
        info = worktree_manager.create_worktree("task-cleanup-err")
        with patch.object(worktree_manager, "_run_git") as mock_git:
            mock_git.side_effect = Exception("git remove failed")
            with pytest.raises(WorktreeCleanupError, match="git remove failed"):
                worktree_manager.cleanup_worktree("task-cleanup-err")


# ============================================================================
# PRInfo tests
# ============================================================================

class TestPRInfo:
    """Tests for the PRInfo dataclass."""

    def test_pr_info_fields(self):
        """PRInfo should have the expected fields."""
        info = PRInfo(
            pr_number=42,
            title="Add feature X",
            branch="feature/x",
            base_branch="main",
            url="https://github.com/org/repo/pull/42",
            status="open",
        )
        assert info.pr_number == 42
        assert info.title == "Add feature X"
        assert info.branch == "feature/x"
        assert info.base_branch == "main"
        assert info.url == "https://github.com/org/repo/pull/42"
        assert info.status == "open"

    def test_pr_info_to_dict(self):
        """PRInfo.to_dict should return correct dict."""
        info = PRInfo(
            pr_number=1,
            title="Test PR",
            branch="feature/test",
            base_branch="main",
            url="https://github.com/org/repo/pull/1",
            status="merged",
        )
        d = info.to_dict()
        assert d["pr_number"] == 1
        assert d["title"] == "Test PR"
        assert d["status"] == "merged"


# ============================================================================
# PRHandler initialization tests
# ============================================================================

class TestPRHandlerInit:
    """Tests for PRHandler initialization."""

    def test_init_with_repo_path(self, temp_git_dir):
        """PRHandler should initialize with a repo path."""
        handler = PRHandler(repo_path=temp_git_dir)
        assert handler.repo_path == temp_git_dir
        assert handler.github_token is None

    def test_init_with_github_token(self, temp_git_dir):
        """PRHandler should accept an optional github token."""
        handler = PRHandler(repo_path=temp_git_dir, github_token="ghp_test123")
        assert handler.github_token == "ghp_test123"

    def test_init_detect_gh_cli(self, temp_git_dir):
        """PRHandler should detect if gh CLI is available."""
        handler = PRHandler(repo_path=temp_git_dir)
        assert isinstance(handler._gh_available, bool)


# ============================================================================
# PRHandler.create_pr tests (via gh CLI)
# ============================================================================

class TestPRHandlerCreatePR:
    """Tests for PRHandler.create_pr()."""

    def test_create_pr_via_gh_cli(self, pr_handler):
        """create_pr should call gh CLI when available."""
        with patch.object(pr_handler, "_gh_available", True):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout="https://github.com/org/repo/pull/1\n",
                    stderr="",
                )
                result = pr_handler.create_pr(
                    branch="feature/test-pr",
                    title="Test PR",
                    body="PR body here",
                    base_branch="main",
                )

        assert result.pr_number == 1
        assert result.title == "Test PR"
        assert result.url == "https://github.com/org/repo/pull/1"

    def test_create_pr_parses_url(self, pr_handler):
        """create_pr should parse pr number from URL template."""
        with patch.object(pr_handler, "_gh_available", True):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout="https://github.com/myorg/myrepo/pull/99\n",
                    stderr="",
                    text="https://github.com/myorg/myrepo/pull/99\n",
                )
                result = pr_handler.create_pr(
                    branch="feature/test",
                    title="My PR",
                    body="Body",
                )
        assert result.pr_number == 99
        assert result.url == "https://github.com/myorg/myrepo/pull/99"

    def test_create_pr_failure_raises(self, pr_handler):
        """create_pr should raise PRCreationError on failure."""
        with patch.object(pr_handler, "_gh_available", True):
            # The first subprocess.run is the git push (check=False, should succeed).
            # The second is the gh CLI call via _run_gh (check=True, should fail).
            push_result = MagicMock(returncode=0, stdout="", stderr="")
            gh_failure = subprocess.CalledProcessError(
                1, "gh", stderr="PR creation failed: duplicate branch"
            )
            with patch("subprocess.run", side_effect=[push_result, gh_failure]):
                with pytest.raises(PRCreationError, match="PR creation failed"):
                    pr_handler.create_pr(
                        branch="feature/fail",
                        title="Will Fail",
                        body="Body",
                    )

    def test_create_pr_with_labels(self, pr_handler):
        """create_pr should support labels."""
        with patch.object(pr_handler, "_gh_available", True):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout="https://github.com/org/repo/pull/3\n",
                    stderr="",
                )
                result = pr_handler.create_pr(
                    branch="feature/labeled",
                    title="Labeled PR",
                    body="Has labels",
                    labels=["enhancement", "priority-high"],
                )
        assert result.pr_number == 3

    def test_create_pr_with_reviewers(self, pr_handler):
        """create_pr should support reviewer assignment."""
        with patch.object(pr_handler, "_gh_available", True):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout="https://github.com/org/repo/pull/4\n",
                    stderr="",
                )
                result = pr_handler.create_pr(
                    branch="feature/reviewed",
                    title="Reviewed PR",
                    body="Has reviewers",
                    reviewers=["alice", "bob"],
                )
        assert result.pr_number == 4


# ============================================================================
# PRHandler.create_pr_via_api tests
# ============================================================================

class TestPRHandlerCreateViaAPI:
    """Tests for PRHandler.create_pr_via_api()."""

    def test_create_pr_via_api(self, pr_handler):
        """create_pr_via_api should make HTTP request to GitHub API."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            raw_json = json.dumps({
                "number": 123,
                "html_url": "https://github.com/org/repo/pull/123",
                "state": "open",
                "title": "Test PR via API",
            })
            encoded = raw_json.encode("utf-8")
            mock_response.read.return_value = encoded
            mock_response.status = 201
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = pr_handler.create_pr_via_api(
                token="ghp_fake_token",
                repo="org/repo",
                title="Test PR via API",
                body="Created via API",
                head="feature/api-test",
                base="main",
            )

        assert result.pr_number == 123
        assert result.url == "https://github.com/org/repo/pull/123"
        assert result.status == "open"

    def test_create_pr_via_api_requires_token(self, pr_handler):
        """create_pr_via_api should raise if token is empty."""
        with pytest.raises(PRCreationError, match="GitHub token"):
            pr_handler.create_pr_via_api(
                token="",
                repo="org/repo",
                title="PR",
                body="Body",
                head="feature/x",
                base="main",
            )

    def test_create_pr_via_api_handles_http_error(self, pr_handler):
        """create_pr_via_api should handle HTTP errors gracefully."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            from urllib.error import HTTPError
            mock_urlopen.side_effect = HTTPError(
                "https://api.github.com/repos/org/repo/pulls",
                422, "Unprocessable Entity", {}, None
            )
            with pytest.raises(PRCreationError, match="GitHub API"):
                pr_handler.create_pr_via_api(
                    token="ghp_test",
                    repo="org/repo",
                    title="PR",
                    body="Body",
                    head="feature/x",
                    base="main",
                )


# ============================================================================
# PRHandler.check_pr_status tests
# ============================================================================

class TestPRHandlerCheckStatus:
    """Tests for PRHandler.check_pr_status()."""

    def test_check_pr_status_via_gh_cli(self, pr_handler):
        """check_pr_status should use gh CLI to check PR status."""
        with patch.object(pr_handler, "_gh_available", True):
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = [
                    # First call: gh pr view
                    MagicMock(
                        returncode=0,
                        stdout=json.dumps({
                            "number": 42,
                            "state": "MERGED",
                            "title": "Done PR",
                            "url": "https://github.com/org/repo/pull/42",
                        }),
                        stderr="",
                    ),
                    # Second call: gh pr checks
                    MagicMock(
                        returncode=0,
                        stdout=json.dumps([
                            {"name": "CI", "conclusion": "SUCCESS"},
                            {"name": "Lint", "conclusion": "SUCCESS"},
                        ]),
                        stderr="",
                    ),
                ]
                result = pr_handler.check_pr_status(42)
        assert result["number"] == 42
        assert result["state"] == "MERGED"
        assert len(result["checks"]) == 2

    def test_check_pr_status_nonexistent(self, pr_handler):
        """check_pr_status should raise PRStatusError for nonexistent PR."""
        with patch.object(pr_handler, "_gh_available", True):
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = Exception("no pull requests found")
                with pytest.raises(PRStatusError, match="no pull requests found"):
                    pr_handler.check_pr_status(99999)


# ============================================================================
# Integration tests
# ============================================================================

class TestWorktreePRIntegration:
    """Integration tests for WorktreeManager + PRHandler."""

    def test_full_workflow_create_and_pr(self, temp_git_dir):
        """End-to-end workflow: create worktree, make changes, create PR."""
        wm = WorktreeManager(repo_path=temp_git_dir)
        prh = PRHandler(repo_path=temp_git_dir)

        # Create worktree for a task
        info = wm.create_worktree("feature-test")
        assert os.path.isdir(info.path), f"Worktree not created at {info.path}"

        # Make a change in the worktree
        feat_file = Path(info.path) / "feature.py"
        feat_file.write_text("# Feature implementation\n")
        import subprocess
        subprocess.run(
            ["git", "-C", info.path, "add", "feature.py"],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", info.path, "commit", "-m", "Add feature"],
            check=True, capture_output=True,
        )

        # Verify branch exists
        result = subprocess.run(
            ["git", "-C", temp_git_dir, "branch", "--list", info.branch],
            capture_output=True, text=True,
        )
        assert info.branch in result.stdout

        # Mock PR creation (gh CLI may not be available)
        with patch.object(prh, "_gh_available", True):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout="https://github.com/org/repo/pull/10\n",
                    stderr="",
                )
                pr_info = prh.create_pr(
                    branch=info.branch,
                    title="Add feature via worktree",
                    body="Automated PR from SPINE worktree manager",
                )
                assert pr_info.pr_number == 10

        # Cleanup
        wm.cleanup_worktree("feature-test")
        assert not os.path.exists(info.path), f"Worktree not cleaned up: {info.path}"

    def test_list_worktrees_after_multiple_creates(self, temp_git_dir):
        """list_worktrees should correctly track multiple worktrees."""
        wm = WorktreeManager(repo_path=temp_git_dir)
        infos = []
        for i in range(3):
            info = wm.create_worktree(f"multi-task-{i}")
            infos.append(info)

        worktrees = wm.list_worktrees()
        paths = [w.path for w in worktrees]
        for info in infos:
            assert info.path in paths

        # Cleanup all
        wm.cleanup_all()

    def test_worktree_persistence_across_instances(self, temp_git_dir):
        """WorktreeManager should track worktrees across instances (via git list)."""
        wm1 = WorktreeManager(repo_path=temp_git_dir)
        info = wm1.create_worktree("persist-task")

        # New instance should see the same worktree
        wm2 = WorktreeManager(repo_path=temp_git_dir)
        worktrees = wm2.list_worktrees()
        paths = [w.path for w in worktrees]
        assert info.path in paths

        # Cleanup
        wm2.cleanup_worktree("persist-task")
