"""PR handler for automated pull request creation.

Creates and manages GitHub pull requests via the GitHub CLI (gh) or
GitHub REST API. Integrates with SPINE's swarm supervisor for
automated PR creation upon task/phase completion.
"""

import json
import os
import re
import subprocess
import urllib.request
import urllib.error
from typing import Optional, Any
from dataclasses import dataclass, field, asdict

from .worktree_manager import WorktreeManager, WorktreeInfo


@dataclass
class PRInfo:
    """Metadata about a created or queried pull request."""

    pr_number: int
    title: str
    branch: str
    base_branch: str = "main"
    url: Optional[str] = None
    status: str = "open"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PRCreationError(Exception):
    """Raised when PR creation fails."""


class PRStatusError(Exception):
    """Raised when PR status query fails."""


class PRHandler:
    """Handles creation and management of GitHub pull requests.

    Supports both ``gh`` CLI (preferred) and GitHub REST API (fallback).
    Integrates with WorktreeManager for branch-to-PR lifecycle.

    Usage::

        handler = PRHandler(repo_path="/path/to/repo")
        pr = handler.create_pr(
            branch="feature/my-feature",
            title="Add feature",
            body="Detailed description",
        )
    """

    _PR_URL_RE = re.compile(r"github\.com/([^/]+/[^/]+)/pull/(\d+)")

    def __init__(
        self,
        repo_path: str,
        github_token: Optional[str] = None,
        worktree_manager: Optional[WorktreeManager] = None,
    ):
        """Initialize the PR handler.

        Args:
            repo_path: Path to the git repository.
            github_token: GitHub personal access token for API operations.
            worktree_manager: Optional WorktreeManager for branch lifecycle integration.
        """
        self.repo_path = os.path.abspath(repo_path)
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")
        self._worktree_manager = worktree_manager
        self._gh_available = self._detect_gh_cli()

    def _detect_gh_cli(self) -> bool:
        """Check if the GitHub CLI (gh) is available."""
        try:
            result = subprocess.run(
                ["gh", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _run_gh(self, args: list[str], **kwargs) -> subprocess.CompletedProcess:
        """Run a gh CLI command.

        Kwargs are forwarded to subprocess.run.
        """
        return subprocess.run(
            ["gh"] + args,
            capture_output=True,
            text=True,
            timeout=kwargs.pop("timeout", 30),
            cwd=kwargs.pop("cwd", self.repo_path),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # PR Creation (gh CLI)
    # ------------------------------------------------------------------

    def create_pr(
        self,
        branch: str,
        title: str,
        body: str = "",
        base_branch: str = "main",
        labels: Optional[list[str]] = None,
        reviewers: Optional[list[str]] = None,
        draft: bool = False,
    ) -> PRInfo:
        """Create a pull request using the GitHub CLI.

        Args:
            branch: The head branch to merge from.
            title: PR title.
            body: PR description body.
            base_branch: Target branch (default: main).
            labels: Optional list of labels to apply.
            reviewers: Optional list of reviewer usernames.
            draft: If True, create as draft PR.

        Returns:
            PRInfo with details about the created PR.

        Raises:
            PRCreationError: If the gh CLI is unavailable or creation fails.
        """
        if not self._gh_available:
            raise PRCreationError(
                "GitHub CLI (gh) is not available. Install it or use create_pr_via_api()."
            )

        args = [
            "pr", "create",
            "--head", branch,
            "--base", base_branch,
            "--title", title,
        ]

        if body:
            args.extend(["--body", body])
        if draft:
            args.append("--draft")

        if labels:
            for label in labels:
                args.extend(["--label", label])

        if reviewers:
            for reviewer in reviewers:
                args.extend(["--reviewer", reviewer])

        try:
            # Push the branch first
            subprocess.run(
                ["git", "-C", self.repo_path, "push", "-u", "origin", branch],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,  # Branch may already exist on remote
            )

            result = self._run_gh(args, check=True)
            url = result.stdout.strip()

            pr_number = self._extract_pr_number(url)
            return PRInfo(
                pr_number=pr_number,
                title=title,
                branch=branch,
                base_branch=base_branch,
                url=url,
                status="open",
            )

        except Exception as e:
            raise PRCreationError(
                f"PR creation failed for branch '{branch}': {e}"
            ) from e

    def _extract_pr_number(self, url: str) -> int:
        """Extract PR number from a GitHub PR URL."""
        match = self._PR_URL_RE.search(url)
        if match:
            return int(match.group(2))
        # Try extracting just a number (fallback)
        numbers = re.findall(r"\d+", url)
        if numbers:
            return int(numbers[-1])
        return 0

    # ------------------------------------------------------------------
    # PR Creation (REST API)
    # ------------------------------------------------------------------

    def create_pr_via_api(
        self,
        token: str,
        repo: str,
        title: str,
        body: str,
        head: str,
        base: str = "main",
        draft: bool = False,
    ) -> PRInfo:
        """Create a PR via the GitHub REST API.

        Args:
            token: GitHub personal access token.
            repo: Repository in 'owner/repo' format.
            title: PR title.
            body: PR description.
            head: Head branch name.
            base: Base branch name.
            draft: If True, create as draft PR.

        Returns:
            PRInfo with details about the created PR.

        Raises:
            PRCreationError: If API request fails.
        """
        if not token:
            raise PRCreationError("GitHub token is required for API-based PR creation.")

        url = f"https://api.github.com/repos/{repo}/pulls"
        payload = json.dumps({
            "title": title,
            "body": body,
            "head": head,
            "base": base,
            "draft": draft,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                raw_data = response.read()
                data = json.loads(raw_data.decode("utf-8"))

            return PRInfo(
                pr_number=data.get("number", 0),
                title=data.get("title", title),
                branch=head,
                base_branch=base,
                url=data.get("html_url", ""),
                status=data.get("state", "open"),
            )

        except urllib.error.HTTPError as e:
            raise PRCreationError(
                f"GitHub API returned HTTP {e.code}: {e.reason}"
            ) from e
        except Exception as e:
            raise PRCreationError(f"PR creation via API failed: {e}") from e

    # ------------------------------------------------------------------
    # PR Status
    # ------------------------------------------------------------------

    def check_pr_status(self, pr_number: int) -> dict[str, Any]:
        """Check the status of a pull request.

        Uses ``gh pr view`` to get PR details and ``gh pr checks`` to get
        CI check statuses.

        Args:
            pr_number: The PR number to check.

        Returns:
            Dict with keys: number, state, title, url, checks.

        Raises:
            PRStatusError: If status query fails.
        """
        if not self._gh_available:
            raise PRStatusError("GitHub CLI (gh) is not available.")

        try:
            # Get PR details
            view_result = self._run_gh(
                ["pr", "view", str(pr_number), "--json", "number,state,title,url"],
                check=True,
            )
            pr_data = json.loads(view_result.stdout)

            # Get checks
            checks_result = self._run_gh(
                ["pr", "checks", str(pr_number), "--json", "name,conclusion"],
                check=False,
            )
            checks = []
            if checks_result.returncode == 0:
                try:
                    checks = json.loads(checks_result.stdout)
                except json.JSONDecodeError:
                    pass

            return {
                "number": pr_data.get("number"),
                "state": pr_data.get("state"),
                "title": pr_data.get("title"),
                "url": pr_data.get("url"),
                "checks": checks,
            }

        except Exception as e:
            raise PRStatusError(
                f"Failed to check PR #{pr_number} status: {e}"
            ) from e

    # ------------------------------------------------------------------
    # Worktree + PR integration
    # ------------------------------------------------------------------

    def create_pr_from_worktree(
        self,
        task_id: str,
        title: str,
        body: str = "",
        labels: Optional[list[str]] = None,
        reviewers: Optional[list[str]] = None,
    ) -> Optional[PRInfo]:
        """Create a PR from a worktree-managed branch.

        Looks up the worktree for ``task_id`` and creates a PR from its
        branch.

        Args:
            task_id: The task/cell identifier.
            title: PR title.
            body: PR description.
            labels: Optional labels.
            reviewers: Optional reviewers.

        Returns:
            PRInfo if successful, None if worktree not found.

        Raises:
            PRCreationError: If PR creation fails.
        """
        if not self._worktree_manager:
            raise PRCreationError("WorktreeManager is required for create_pr_from_worktree().")

        wt_path = self._worktree_manager.get_worktree_path(task_id)
        if not wt_path:
            return None

        # Get the branch from manifest or git
        manifest = self._worktree_manager._manifest or {}  # noqa: SLF001
        entry = manifest.get(task_id, {})
        branch = entry.get("branch", "")

        if not branch:
            # Try to get current branch from the worktree
            try:
                result = subprocess.run(
                    ["git", "-C", wt_path, "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    branch = result.stdout.strip()
            except Exception:
                pass

        if not branch:
            raise PRCreationError(f"Could not determine branch for task '{task_id}'")

        return self.create_pr(
            branch=branch,
            title=title,
            body=body,
            labels=labels,
            reviewers=reviewers,
        )
