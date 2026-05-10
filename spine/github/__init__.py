"""GitHub Issue Integration Service for SPINE.

Provides:
- GitHubClient: REST API client for GitHub issue operations
- IssueResolver: AI-powered issue analysis and hierarchy mapping
- GitHubIssue: Dataclass for normalized issue representation
"""

from .client import GitHubClient, GitHubError
from .issue_resolver import IssueResolver, GitHubIssue

__all__ = [
    "GitHubClient",
    "GitHubError",
    "IssueResolver",
    "GitHubIssue",
]
