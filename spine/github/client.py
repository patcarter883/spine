"""GitHub REST API client for the SPINE harness.

Uses stdlib urllib for zero-dependency HTTP operations.
Supports GitHub token-based auth via GITHUB_TOKEN env var.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Iterator


class GitHubError(Exception):
    """Base exception for GitHub API errors."""

    def __init__(self, message: str, status_code: int = 0, response_body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class GitHubClient:
    """GitHub REST API client.

    Communicates with the GitHub v3 REST API using urllib.
    Supports token authentication, pagination, and issue CRUD.

    Usage:
        client = GitHubClient(token="ghp_xxx", owner="my-org", repo="my-repo")
        issues = client.list_issues(state="open", labels=["bug"])
        issue = client.get_issue(42)
    """

    API_BASE = "https://api.github.com"
    DEFAULT_PER_PAGE = 30
    MAX_PER_PAGE = 100

    def __init__(
        self,
        token: Optional[str] = None,
        owner: str = "",
        repo: str = "",
        base_url: str = API_BASE,
    ):
        """Initialize the GitHub client.

        Args:
            token: GitHub personal access token. If None, reads GITHUB_TOKEN env var.
            owner: Repository owner (username or org).
            repo: Repository name.
            base_url: Override for GitHub Enterprise or testing.
        """
        self._token = token or os.environ.get("GITHUB_TOKEN", "")
        self._owner = owner
        self._repo = repo
        self._base_url = base_url.rstrip("/")
        self._session_headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "SPINE-harness/0.1.0",
        }
        if self._token:
            self._session_headers["Authorization"] = f"token {self._token}"

    @property
    def owner(self) -> str:
        return self._owner

    @owner.setter
    def owner(self, value: str) -> None:
        self._owner = value

    @property
    def repo(self) -> str:
        return self._repo

    @repo.setter
    def repo(self, value: str) -> None:
        self._repo = value

    @property
    def repo_path(self) -> str:
        """Return 'owner/repo' path segment."""
        if not self._owner or not self._repo:
            return ""
        return f"{self._owner}/{self._repo}"

    # ── HTTP Helpers ───────────────────────────────────────────

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an HTTP request to the GitHub API.

        Args:
            method: HTTP method (GET, POST, PATCH, etc.)
            endpoint: API path relative to base URL (e.g., "/repos/owner/repo/issues").
            data: Optional JSON body for POST/PATCH.
            params: Optional query parameters.

        Returns:
            Parsed JSON response as dict.

        Raises:
            GitHubError: On non-2xx responses or network errors.
        """
        url = self._base_url + endpoint

        if params:
            encoded = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
            url = url + "?" + encoded

        body_bytes: Optional[bytes] = None
        if data is not None:
            body_bytes = json.dumps(data).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=body_bytes,
            headers=self._session_headers,
            method=method,
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                response_data = resp.read().decode("utf-8")
                return json.loads(response_data) if response_data else {}
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8") if e.fp else ""
            raise GitHubError(
                f"GitHub API {method} {endpoint} failed: {e.code} {e.reason}",
                status_code=e.code,
                response_body=body,
            )
        except urllib.error.URLError as e:
            raise GitHubError(f"Network error: {e.reason}")

    def _paginated(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        per_page: int = DEFAULT_PER_PAGE,
    ) -> Iterator[Dict[str, Any]]:
        """Paginate through GitHub API results.

        Yields individual items from paginated list endpoints.
        Automatically follows 'next' links from Link headers.

        Args:
            method: HTTP method.
            endpoint: API endpoint.
            params: Query parameters (page and per_page are managed automatically).
            per_page: Items per page (1-100).

        Yields:
            Individual response items.
        """
        params = dict(params or {})
        params["per_page"] = min(per_page, self.MAX_PER_PAGE)
        params["page"] = 1

        while True:
            response = self._request(method, endpoint, params=params)
            if not isinstance(response, list):
                # Single-object response — yield and stop
                yield response
                break

            if not response:
                break

            for item in response:
                yield item

            # GitHub caps at 100 per page; if we got fewer items than requested,
            # we're likely on the last page.
            if len(response) < params["per_page"]:
                break

            params["page"] += 1

    # ── Issue Operations ───────────────────────────────────────

    def list_issues(
        self,
        state: str = "open",
        labels: Optional[List[str]] = None,
        milestone: Optional[int] = None,
        assignee: Optional[str] = None,
        since: Optional[str] = None,
        per_page: int = DEFAULT_PER_PAGE,
    ) -> List[Dict[str, Any]]:
        """List issues for the configured repository.

        Args:
            state: "open", "closed", or "all".
            labels: Filter by label names (comma-separated in the API).
            milestone: Filter by milestone number.
            assignee: Filter by assignee login.
            since: ISO 8601 timestamp for issues updated after this date.
            per_page: Items per page.

        Returns:
            List of issue dicts as returned by the GitHub API.
        """
        if not self.repo_path:
            raise GitHubError("owner and repo must be set before listing issues")

        params: Dict[str, Any] = {"state": state}
        if labels:
            params["labels"] = ",".join(labels)
        if milestone is not None:
            params["milestone"] = str(milestone)
        if assignee:
            params["assignee"] = assignee
        if since:
            params["since"] = since

        endpoint = f"/repos/{self.repo_path}/issues"
        result: List[Dict[str, Any]] = []
        for item in self._paginated("GET", endpoint, params=params, per_page=per_page):
            result.append(item)
        return result

    def get_issue(self, issue_number: int) -> Dict[str, Any]:
        """Get a single issue by number.

        Args:
            issue_number: Issue number (not ID).

        Returns:
            Issue dict.

        Raises:
            GitHubError: If the issue does not exist (404) or other errors.
        """
        if not self.repo_path:
            raise GitHubError("owner and repo must be set before getting an issue")

        endpoint = f"/repos/{self.repo_path}/issues/{issue_number}"
        result = self._request("GET", endpoint)
        return result

    def search_issues(
        self,
        query: str,
        per_page: int = DEFAULT_PER_PAGE,
    ) -> List[Dict[str, Any]]:
        """Search issues across GitHub using a query string.

        Uses the GitHub Search Issues API.
        Query syntax: https://docs.github.com/en/search-github/searching-on-github/searching-issues-and-pull-requests

        Args:
            query: Search query string (e.g., "is:issue is:open label:bug repo:owner/repo").
            per_page: Items per page.

        Returns:
            List of issue items from search results.
        """
        endpoint = "/search/issues"
        params: Dict[str, Any] = {"q": query}

        result: List[Dict[str, Any]] = []
        for item in self._paginated("GET", endpoint, params=params, per_page=per_page):
            result.append(item)
        return result

    def create_issue(
        self,
        title: str,
        body: str = "",
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None,
        milestone: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create a new issue in the configured repository.

        Args:
            title: Issue title.
            body: Issue body (Markdown).
            labels: Label names to apply.
            assignees: Usernames to assign.
            milestone: Milestone number.

        Returns:
            Created issue dict.
        """
        if not self.repo_path:
            raise GitHubError("owner and repo must be set before creating issues")

        data: Dict[str, Any] = {"title": title, "body": body}
        if labels:
            data["labels"] = labels
        if assignees:
            data["assignees"] = assignees
        if milestone is not None:
            data["milestone"] = milestone

        endpoint = f"/repos/{self.repo_path}/issues"
        return self._request("POST", endpoint, data=data)

    def update_issue(
        self,
        issue_number: int,
        title: Optional[str] = None,
        body: Optional[str] = None,
        state: Optional[str] = None,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Update an existing issue.

        Args:
            issue_number: Issue number to update.
            title: New title (if provided).
            body: New body (if provided).
            state: "open" or "closed" (if provided).
            labels: New labels (if provided, replaces all existing).

        Returns:
            Updated issue dict.
        """
        if not self.repo_path:
            raise GitHubError("owner and repo must be set before updating issues")

        data: Dict[str, Any] = {}
        if title is not None:
            data["title"] = title
        if body is not None:
            data["body"] = body
        if state is not None:
            data["state"] = state
        if labels is not None:
            data["labels"] = labels

        endpoint = f"/repos/{self.repo_path}/issues/{issue_number}"
        return self._request("PATCH", endpoint, data=data)

    def create_comment(
        self,
        issue_number: int,
        body: str,
    ) -> Dict[str, Any]:
        """Add a comment to an issue.

        Args:
            issue_number: Issue number.
            body: Comment body (Markdown).

        Returns:
            Created comment dict.
        """
        if not self.repo_path:
            raise GitHubError("owner and repo must be set before commenting")

        endpoint = f"/repos/{self.repo_path}/issues/{issue_number}/comments"
        return self._request("POST", endpoint, data={"body": body})

    def list_comments(
        self,
        issue_number: int,
        per_page: int = DEFAULT_PER_PAGE,
    ) -> List[Dict[str, Any]]:
        """List comments on an issue.

        Args:
            issue_number: Issue number.
            per_page: Items per page.

        Returns:
            List of comment dicts.
        """
        if not self.repo_path:
            raise GitHubError("owner and repo must be set before listing comments")

        endpoint = f"/repos/{self.repo_path}/issues/{issue_number}/comments"
        result: List[Dict[str, Any]] = []
        for item in self._paginated("GET", endpoint, per_page=per_page):
            result.append(item)
        return result

    # ── Repository Operations ──────────────────────────────────

    def get_repo(self) -> Dict[str, Any]:
        """Get repository information.

        Returns:
            Repository dict.
        """
        if not self.repo_path:
            raise GitHubError("owner and repo must be set")

        endpoint = f"/repos/{self.repo_path}"
        return self._request("GET", endpoint)

    def list_labels(self) -> List[Dict[str, Any]]:
        """List all labels in the repository.

        Returns:
            List of label dicts.
        """
        if not self.repo_path:
            raise GitHubError("owner and repo must be set")

        endpoint = f"/repos/{self.repo_path}/labels"
        return self._request("GET", endpoint)
