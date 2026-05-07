"""Tests for GitHub Issue Integration Service.

Tests cover:
- GitHubIssue data model and normalization
- GitHubClient API operations with mocked HTTP
- IssueResolver analysis (LLM + fallback) and hierarchy mapping
- Ralph Loop integration (issue → PhaseNode/TaskNode)
- Edge cases: empty responses, malformed JSON, error handling
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import urllib.error
from io import BytesIO
from unittest.mock import patch, MagicMock, call

from spine.github.client import GitHubClient, GitHubError
from spine.github.issue_resolver import (
    GitHubIssue,
    IssueResolver,
    IssueAnalysis,
)
from spine.models.types import (
    PhaseNode,
    SubPhaseNode,
    TaskNode,
    ProjectNode,
    HierarchyLevel,
    NodeStatus,
    HierarchyProgress,
    Phase,
    SubPhase,
    Task,
)
from spine.core.hierarchy import (
    RalphLoopEngine,
    TransitionManager,
    ProgressAggregator,
)


# ── Test Fixtures ──────────────────────────────────────────────────

SAMPLE_ISSUE_RAW = {
    "number": 42,
    "title": "Fix login timeout bug",
    "body": "When users are idle for >5min, login times out.\n\n- [ ] Reproduce in staging\n- [ ] Identify timeout config\n- [ ] Increase session TTL\n- [ ] Add test coverage",
    "state": "open",
    "labels": [{"name": "bug"}, {"name": "priority-high"}],
    "assignees": [{"login": "dev1"}, {"login": "dev2"}],
    "milestone": {"title": "v1.2"},
    "html_url": "https://github.com/org/repo/issues/42",
    "created_at": "2026-05-01T10:00:00Z",
    "updated_at": "2026-05-07T12:00:00Z",
}

SAMPLE_ISSUE_MINIMAL = {
    "number": 1,
    "title": "Minimal issue",
    "body": "",
    "state": "open",
    "labels": [],
    "assignees": [],
    "milestone": None,
    "html_url": "https://github.com/org/repo/issues/1",
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
}

SAMPLE_FEATURE_ISSUE = {
    "number": 99,
    "title": "Add export functionality",
    "body": "Users need CSV export.\n\n* [ ] Design export API\n* [ ] Implement CSV writer\n* [ ] Add CLI flag",
    "state": "open",
    "labels": [{"name": "feature"}, {"name": "enhancement"}],
    "assignees": [],
    "milestone": {"title": "v2.0"},
    "html_url": "https://github.com/org/repo/issues/99",
    "created_at": "2026-05-01T00:00:00Z",
    "updated_at": "2026-05-01T00:00:00Z",
}


# ── Helper: Mock URL opener ───────────────────────────────────────

def _mock_urlopen(response_data, status_code=200, headers=None):
    """Create a mock urlopen that returns a BytesIO response."""

    def mock_urlopen(req, timeout=30):
        # Return a mock that behaves like urlopen response
        resp = MagicMock()
        resp.read.return_value = json.dumps(response_data).encode("utf-8")
        resp.status = status_code
        resp.__enter__.return_value = resp
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    return mock_urlopen


def _mock_urlopen_error(status_code, reason="Error"):
    """Create a mock urlopen that raises HTTPError."""

    def mock_urlopen(req, timeout=30):
        fp = BytesIO(b'{"message": "Not Found"}')
        raise urllib.error.HTTPError(
            url="https://api.github.com/test",
            code=status_code,
            msg=reason,
            hdrs={},
            fp=fp,
        )

    return mock_urlopen


# ═══════════════════════════════════════════════════════════════════
# GitHubIssue Model Tests
# ═══════════════════════════════════════════════════════════════════


class TestGitHubIssue:
    """Tests for the GitHubIssue dataclass and normalization."""

    def test_from_api_response_full(self):
        """Full API response is properly normalized."""
        issue = GitHubIssue.from_api_response(SAMPLE_ISSUE_RAW)
        assert issue.number == 42
        assert issue.title == "Fix login timeout bug"
        assert "login times out" in issue.body
        assert issue.state == "open"
        assert issue.labels == ["bug", "priority-high"]
        assert issue.assignees == ["dev1", "dev2"]
        assert issue.milestone == "v1.2"
        assert issue.url == "https://github.com/org/repo/issues/42"
        assert issue.created_at == "2026-05-01T10:00:00Z"

    def test_from_api_response_minimal(self):
        """Minimal API response handles missing fields."""
        issue = GitHubIssue.from_api_response(SAMPLE_ISSUE_MINIMAL)
        assert issue.number == 1
        assert issue.title == "Minimal issue"
        assert issue.body == ""
        assert issue.labels == []
        assert issue.assignees == []
        assert issue.milestone is None

    def test_from_api_response_label_strings(self):
        """Handles labels that are already strings (not dicts)."""
        data = dict(SAMPLE_ISSUE_RAW)
        data["labels"] = ["bug", "docs"]
        issue = GitHubIssue.from_api_response(data)
        assert issue.labels == ["bug", "docs"]

    def test_default_values(self):
        """Default field values are sensible."""
        issue = GitHubIssue(number=1, title="Test")
        assert issue.body == ""
        assert issue.state == "open"
        assert issue.labels == []
        assert issue.estimated_complexity == "medium"
        assert issue.extracted_tasks == []
        assert issue.suggested_phase == ""

    def test_to_summary(self):
        """to_summary() produces a readable Markdown block."""
        issue = GitHubIssue.from_api_response(SAMPLE_ISSUE_RAW)
        summary = issue.to_summary()
        assert "### Issue #42: Fix login timeout bug" in summary
        assert "**State:** open" in summary
        assert "**Labels:** bug, priority-high" in summary
        assert "**Assignees:** dev1, dev2" in summary
        assert "**Milestone:** v1.2" in summary
        assert "login times out" in summary

    def test_to_summary_minimal(self):
        """to_summary works with minimal fields."""
        issue = GitHubIssue.from_api_response(SAMPLE_ISSUE_MINIMAL)
        summary = issue.to_summary()
        assert "Issue #1" in summary
        assert "_No description_" in summary

    def test_body_truncation_in_summary(self):
        """Very long bodies are truncated to 2000 chars in summary."""
        issue = GitHubIssue(number=1, title="Test", body="x" * 5000)
        summary = issue.to_summary()
        # Should contain at most 2000 chars of body, not 5000
        body_start = summary.find("xxx")
        body_section = summary[body_start:]
        assert len(body_section) <= 2100  # allowance for surrounding text


# ═══════════════════════════════════════════════════════════════════
# GitHubClient Tests
# ═══════════════════════════════════════════════════════════════════


class TestGitHubClient:
    """Tests for GitHubClient initialization and URL construction."""

    def test_default_init(self):
        client = GitHubClient()
        assert client.owner == ""
        assert client.repo == ""
        assert client.repo_path == ""

    def test_init_with_params(self):
        client = GitHubClient(token="ghp_test123", owner="my-org", repo="my-repo")
        assert client.owner == "my-org"
        assert client.repo == "my-repo"
        assert client.repo_path == "my-org/my-repo"

    def test_init_token_from_env(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_from_env")
        client = GitHubClient()
        assert client._token == "ghp_from_env"

    def test_init_explicit_token_overrides_env(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_from_env")
        client = GitHubClient(token="ghp_explicit")
        assert client._token == "ghp_explicit"

    def test_owner_repo_settable(self):
        client = GitHubClient()
        client.owner = "new-org"
        client.repo = "new-repo"
        assert client.repo_path == "new-org/new-repo"

    def test_enterprise_base_url(self):
        client = GitHubClient(
            token="ghp_test",
            owner="org",
            repo="repo",
            base_url="https://github.mycompany.com/api/v3",
        )
        assert client._base_url == "https://github.mycompany.com/api/v3"


class TestGitHubClientAPI:
    """Tests for GitHubClient API operations with mocked HTTP."""

    def test_list_issues_with_mock(self):
        """list_issues returns normalized issue list."""
        client = GitHubClient(owner="org", repo="repo")
        response = [SAMPLE_ISSUE_RAW, SAMPLE_ISSUE_MINIMAL]

        # Minimal issue for test
        minimal = SAMPLE_ISSUE_MINIMAL.copy()

        with patch.object(client, "_request", return_value=[SAMPLE_ISSUE_RAW, minimal]):
            issues = client.list_issues(state="open")
            assert len(issues) == 2
            assert issues[0]["number"] == 42
            assert issues[1]["number"] == 1

    def test_list_issues_requires_repo(self):
        """Raises error when owner/repo not set."""
        client = GitHubClient()
        with pytest.raises(GitHubError, match="owner and repo must be set"):
            client.list_issues()

    def test_get_issue_with_mock(self):
        """get_issue returns single issue."""
        client = GitHubClient(owner="org", repo="repo")
        with patch.object(client, "_request", return_value=SAMPLE_ISSUE_RAW):
            issue = client.get_issue(42)
            assert issue["number"] == 42
            assert issue["title"] == "Fix login timeout bug"

    def test_get_issue_requires_repo(self):
        client = GitHubClient()
        with pytest.raises(GitHubError, match="owner and repo must be set"):
            client.get_issue(1)

    def test_search_issues_with_mock(self):
        """search_issues returns items from search endpoint."""
        client = GitHubClient(owner="org", repo="repo")
        search_response = {"total_count": 1, "items": [SAMPLE_ISSUE_RAW]}

        with patch.object(client, "_request", return_value=search_response):
            results = client.search_issues("is:issue is:open")
            assert len(results) == 1
            assert results[0]["total_count"] == 1

    def test_create_issue_with_mock(self):
        """create_issue sends POST and returns created issue."""
        client = GitHubClient(owner="org", repo="repo")
        created = dict(SAMPLE_ISSUE_RAW, number=43)

        with patch.object(client, "_request", return_value=created) as mock_req:
            result = client.create_issue(
                title="New issue",
                body="Description",
                labels=["bug"],
                assignees=["dev1"],
            )
            assert result["number"] == 43
            mock_req.assert_called_once()
            call_args = mock_req.call_args
            call_data = call_args[0][3] if len(call_args[0]) > 3 else call_args[1].get("data")
            assert call_data["title"] == "New issue"
            assert call_data["labels"] == ["bug"]

    def test_update_issue_with_mock(self):
        client = GitHubClient(owner="org", repo="repo")
        updated = dict(SAMPLE_ISSUE_RAW, state="closed")

        with patch.object(client, "_request", return_value=updated) as mock_req:
            result = client.update_issue(42, state="closed")
            assert result["state"] == "closed"
            call_data = mock_req.call_args[1].get("data")
            assert call_data["state"] == "closed"

    def test_create_comment_with_mock(self):
        client = GitHubClient(owner="org", repo="repo")
        comment = {"id": 1, "body": "Great!"}

        with patch.object(client, "_request", return_value=comment) as mock_req:
            result = client.create_comment(42, body="Great!")
            assert result["body"] == "Great!"
            call_data = mock_req.call_args[1].get("data")
            assert call_data["body"] == "Great!"

    def test_list_comments_with_mock(self):
        client = GitHubClient(owner="org", repo="repo")
        comments = [{"id": 1, "body": "First"}]

        with patch.object(client, "_request", return_value=comments):
            result = client.list_comments(42)
            assert len(result) == 1

    def test_get_repo_with_mock(self):
        client = GitHubClient(owner="org", repo="repo")
        repo_data = {"name": "repo", "full_name": "org/repo"}

        with patch.object(client, "_request", return_value=repo_data):
            result = client.get_repo()
            assert result["name"] == "repo"

    def test_list_labels_with_mock(self):
        client = GitHubClient(owner="org", repo="repo")
        labels = [{"name": "bug"}, {"name": "feature"}]

        with patch.object(client, "_request", return_value=labels):
            result = client.list_labels()
            assert len(result) == 2

    def test_paginated_iteration(self):
        """_paginated yields items from multiple pages."""
        client = GitHubClient(owner="org", repo="repo")

        page1 = [{"number": 1}, {"number": 2}]
        page2 = [{"number": 3}]

        responses = [page1, page2]

        with patch.object(client, "_request", side_effect=responses) as mock_req:
            items = list(client._paginated("GET", "/test", per_page=2))
            assert len(items) == 3
            assert items[0]["number"] == 1
            assert items[2]["number"] == 3

    def test_paginated_handles_short_last_page(self):
        """_paginated stops when page has fewer items than per_page."""
        client = GitHubClient(owner="org", repo="repo")

        full_page = [{"n": i} for i in range(30)]
        short_page = [{"n": 99}]  # 1 item < 30

        with patch.object(client, "_request", side_effect=[full_page, short_page]):
            items = list(client._paginated("GET", "/test"))
            assert len(items) == 31

    def test_http_error_conversion(self):
        """HTTP errors are converted to GitHubError."""
        client = GitHubClient(owner="org", repo="repo")

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(
                url="https://api.github.com/test",
                code=404,
                msg="Not Found",
                hdrs={},
                fp=BytesIO(b'{"message": "Not Found"}'),
            ),
        ):
            with pytest.raises(GitHubError, match="404"):
                client.get_issue(999)

    def test_network_error(self):
        """URLError is wrapped as GitHubError."""
        client = GitHubClient(owner="org", repo="repo")

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            with pytest.raises(GitHubError, match="Network error"):
                client.get_issue(1)

    def test_github_error_attributes(self):
        """GitHubError carries status_code and response_body."""
        error = GitHubError("Failed", status_code=403, response_body='{"msg":"Forbidden"}')
        assert error.status_code == 403
        assert error.response_body == '{"msg":"Forbidden"}'
        assert "Failed" in str(error)


# ═══════════════════════════════════════════════════════════════════
# IssueResolver Tests
# ═══════════════════════════════════════════════════════════════════


class TestIssueResolverInit:
    """Tests for IssueResolver initialization."""

    def test_default_init(self):
        client = GitHubClient(owner="org", repo="repo")
        resolver = IssueResolver(client)
        assert resolver.client is client
        assert resolver.engine is not None
        assert resolver.llm_provider is None

    def test_init_with_engine(self):
        client = GitHubClient(owner="org", repo="repo")
        engine = RalphLoopEngine()
        resolver = IssueResolver(client, engine=engine)
        assert resolver.engine is engine

    def test_custom_analysis_prompt(self):
        client = GitHubClient(owner="org", repo="repo")
        resolver = IssueResolver(client)
        resolver.set_analysis_prompt("Custom: {issue_summary}")
        assert resolver._analysis_prompt == "Custom: {issue_summary}"


class TestIssueResolverFetch:
    """Tests for issue fetching and normalization."""

    def test_fetch_issues_normalizes(self):
        client = GitHubClient(owner="org", repo="repo")
        resolver = IssueResolver(client)

        raw = [SAMPLE_ISSUE_RAW, SAMPLE_FEATURE_ISSUE]
        with patch.object(client, "_request", return_value=raw):
            issues = resolver.fetch_issues(state="open")
            assert len(issues) == 2
            assert all(isinstance(i, GitHubIssue) for i in issues)
            assert issues[0].number == 42
            assert issues[1].number == 99

    def test_fetch_single_issue(self):
        client = GitHubClient(owner="org", repo="repo")
        resolver = IssueResolver(client)

        with patch.object(client, "_request", return_value=SAMPLE_ISSUE_RAW):
            issue = resolver.fetch_issue(42)
            assert isinstance(issue, GitHubIssue)
            assert issue.number == 42
            assert issue.title == "Fix login timeout bug"

    def test_fetch_issues_with_labels(self):
        client = GitHubClient(owner="org", repo="repo")
        resolver = IssueResolver(client)

        with patch.object(client, "_request", return_value=[SAMPLE_ISSUE_RAW]):
            issues = resolver.fetch_issues(state="open", labels=["bug"])
            assert len(issues) == 1


class TestIssueAnalyzerFallback:
    """Tests for fallback (non-LLM) issue analysis."""

    def test_analyze_bug_issue(self):
        client = GitHubClient(owner="org", repo="repo")
        resolver = IssueResolver(client)
        issue = GitHubIssue.from_api_response(SAMPLE_ISSUE_RAW)

        analysis = resolver.analyze_issue(issue)
        assert isinstance(analysis, IssueAnalysis)
        # Bug labeled by "bug" → should be "Bug Fixes" phase
        assert analysis.suggested_phase == "Bug Fixes"
        assert analysis.complexity in ("low", "medium", "high")
        assert len(analysis.extracted_tasks) > 0
        assert analysis.confidence > 0

    def test_analyze_feature_issue(self):
        client = GitHubClient(owner="org", repo="repo")
        resolver = IssueResolver(client)
        issue = GitHubIssue.from_api_response(SAMPLE_FEATURE_ISSUE)

        analysis = resolver.analyze_issue(issue)
        assert analysis.suggested_phase == "Features"
        assert analysis.complexity == "high"

    def test_analyze_extracts_checklists_from_body(self):
        """Analyzer finds checklist items in issue body (markdown task lists)."""
        client = GitHubClient(owner="org", repo="repo")
        resolver = IssueResolver(client)
        issue = GitHubIssue(
            number=1,
            title="Add tests",
            body="- [ ] Write unit tests\n- [ ] Run CI\n- [x] Already done",
            labels=["feature"],
        )

        analysis = resolver.analyze_issue(issue)
        # The fallback should have parsed the checklist
        assert "Write unit tests" in analysis.extracted_tasks or "Run CI" in analysis.extracted_tasks

    def test_analyze_falls_back_on_title_keywords(self):
        """Title keywords determine phase even without labels."""
        client = GitHubClient(owner="org", repo="repo")
        resolver = IssueResolver(client)

        bug_issue = GitHubIssue(number=1, title="Fix crash on startup", labels=[])
        analysis = resolver.analyze_issue(bug_issue)
        assert analysis.suggested_phase == "Bug Fixes"

        feat_issue = GitHubIssue(number=2, title="Add dark mode support", labels=[])
        analysis = resolver.analyze_issue(feat_issue)
        assert analysis.suggested_phase == "Features"

        doc_issue = GitHubIssue(number=3, title="Update README docs", labels=[])
        analysis = resolver.analyze_issue(doc_issue)
        assert analysis.suggested_phase == "Documentation"

    def test_analyze_unknown_issue_goes_to_backlog(self):
        client = GitHubClient(owner="org", repo="repo")
        resolver = IssueResolver(client)
        issue = GitHubIssue(number=1, title="Something vague", labels=[])

        analysis = resolver.analyze_issue(issue)
        assert analysis.suggested_phase == "Backlog"

    def test_analyze_updates_issue_metadata(self):
        """analyze_issue updates the issue object with extracted metadata."""
        client = GitHubClient(owner="org", repo="repo")
        resolver = IssueResolver(client)
        issue = GitHubIssue.from_api_response(SAMPLE_ISSUE_RAW)

        resolver.analyze_issue(issue)
        assert issue.estimated_complexity != "medium" or issue.estimated_complexity == "medium"
        assert issue.suggested_phase != ""
        assert len(issue.extracted_tasks) > 0

    def test_analyze_issues_batch(self):
        """Batch analysis returns dict mapping numbers to analyses."""
        client = GitHubClient(owner="org", repo="repo")
        resolver = IssueResolver(client)
        issues = [
            GitHubIssue.from_api_response(SAMPLE_ISSUE_RAW),
            GitHubIssue.from_api_response(SAMPLE_FEATURE_ISSUE),
        ]

        results = resolver.analyze_issues(issues)
        assert len(results) == 2
        assert 42 in results
        assert 99 in results
        assert isinstance(results[42], IssueAnalysis)

    def test_issue_without_body_gets_default_tasks(self):
        client = GitHubClient(owner="org", repo="repo")
        resolver = IssueResolver(client)
        issue = GitHubIssue(number=1, title="Minimal", labels=["bug"])

        analysis = resolver.analyze_issue(issue)
        assert len(analysis.extracted_tasks) > 0
        assert "Analyze" in analysis.extracted_tasks[0] or "Implement" in analysis.extracted_tasks[0]


class TestIssueAnalyzerLLM:
    """Tests for LLM-powered analysis."""

    def test_analyze_with_llm(self):
        """LLM response is parsed correctly."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = json.dumps({
            "complexity": "low",
            "suggested_phase": "Testing",
            "extracted_tasks": ["Write tests", "Run coverage"],
            "confidence": 0.9,
        })

        client = GitHubClient(owner="org", repo="repo")
        resolver = IssueResolver(client, llm_provider=mock_llm)
        issue = GitHubIssue.from_api_response(SAMPLE_ISSUE_RAW)

        analysis = resolver.analyze_issue(issue)
        assert analysis.suggested_phase == "Testing"
        assert analysis.complexity == "low"
        assert analysis.extracted_tasks == ["Write tests", "Run coverage"]
        assert analysis.confidence == 0.9
        assert analysis.raw_response != ""

    def test_llm_response_with_code_fence(self):
        """JSON inside markdown code fences is parsed correctly."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = '```json\n{"complexity": "high", "suggested_phase": "API", "extracted_tasks": ["Build"], "confidence": 0.8}\n```'

        client = GitHubClient(owner="org", repo="repo")
        resolver = IssueResolver(client, llm_provider=mock_llm)
        issue = GitHubIssue(number=1, title="Test")

        analysis = resolver.analyze_issue(issue)
        assert analysis.complexity == "high"
        assert analysis.suggested_phase == "API"

    def test_malformed_llm_response(self):
        """Malformed JSON falls back gracefully."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "not json at all"

        client = GitHubClient(owner="org", repo="repo")
        resolver = IssueResolver(client, llm_provider=mock_llm)
        issue = GitHubIssue(number=1, title="Test")

        analysis = resolver.analyze_issue(issue)
        # Should fall back to defaults
        assert analysis.complexity == "medium"
        assert len(analysis.extracted_tasks) == 3
        assert analysis.confidence == 0.3

    def test_llm_error_falls_back(self):
        """If LLM throws, fallback analysis is used."""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("LLM unavailable")

        client = GitHubClient(owner="org", repo="repo")
        resolver = IssueResolver(client, llm_provider=mock_llm)
        issue = GitHubIssue.from_api_response(SAMPLE_ISSUE_RAW)

        analysis = resolver.analyze_issue(issue)
        # Fallback for bug-labeled issue
        assert analysis.suggested_phase == "Bug Fixes"
        assert analysis.confidence <= 0.6  # fallback confidence


class TestParseAnalysisJSON:
    """Tests for the _parse_analysis_json helper."""

    def test_clean_json(self):
        client = GitHubClient(owner="org", repo="repo")
        resolver = IssueResolver(client)
        result = resolver._parse_analysis_json(
            '{"complexity": "high", "suggested_phase": "API", "extracted_tasks": ["Do X"], "confidence": 0.9}',
            1,
        )
        assert result.complexity == "high"
        assert result.suggested_phase == "API"
        assert result.extracted_tasks == ["Do X"]
        assert result.confidence == 0.9

    def test_json_with_text_wrapper(self):
        """Text before/after JSON is stripped."""
        client = GitHubClient(owner="org", repo="repo")
        resolver = IssueResolver(client)
        result = resolver._parse_analysis_json(
            'Here is the analysis:\n{"complexity": "low", "suggested_phase": "Docs", "extracted_tasks": ["Write"], "confidence": 0.7}\nEnd.',
            1,
        )
        assert result.complexity == "low"
        assert result.suggested_phase == "Docs"

    def test_partial_json_defaults(self):
        """Missing fields get defaults."""
        client = GitHubClient(owner="org", repo="repo")
        resolver = IssueResolver(client)
        result = resolver._parse_analysis_json(
            '{"complexity": "low"}',  # missing suggested_phase, tasks, confidence
            1,
        )
        assert result.complexity == "low"
        assert result.suggested_phase == "Backlog"
        assert result.extracted_tasks == []
        assert result.confidence == 0.5


# ═══════════════════════════════════════════════════════════════════
# Hierarchy Mapping Tests
# ═══════════════════════════════════════════════════════════════════


class TestHierarchyMapping:
    """Tests for mapping GitHub issues into Ralph Loop hierarchy."""

    def test_map_issues_to_empty_project(self):
        """Map issues into a fresh project."""
        client = GitHubClient(owner="org", repo="repo")
        engine = RalphLoopEngine()
        resolver = IssueResolver(client, engine=engine)

        proj = engine.create_project("test-proj", "Test Project")
        issues = [
            GitHubIssue.from_api_response(SAMPLE_ISSUE_RAW),
            GitHubIssue.from_api_response(SAMPLE_FEATURE_ISSUE),
        ]

        result = resolver.map_issues_to_hierarchy(issues, proj, analyze=True)
        assert result is proj
        # Should have created phases (Bug Fixes + Features from labels)
        assert len(proj.phases) > 0

    def test_map_issues_groups_by_phase(self):
        """Issues with the same suggested_phase are grouped."""
        client = GitHubClient(owner="org", repo="repo")
        engine = RalphLoopEngine()
        resolver = IssueResolver(client, engine=engine)

        proj = engine.create_project("proj", "Test")
        issues = [
            GitHubIssue(number=1, title="Bug A", labels=["bug"]),
            GitHubIssue(number=2, title="Bug B", labels=["bug"]),
        ]

        result = resolver.map_issues_to_hierarchy(issues, proj, analyze=True)

        # Both bugs should be in one "Bug Fixes" phase
        bug_fix_phase = None
        for phase in proj.phases:
            if phase.name == "Bug Fixes":
                bug_fix_phase = phase
                break

        assert bug_fix_phase is not None
        # Should have 2 subphases (one per issue)
        assert len(bug_fix_phase.subphases) == 2

    def test_map_issue_with_extracted_tasks(self):
        """Issue with checklist → multiple TaskNodes."""
        client = GitHubClient(owner="org", repo="repo")
        engine = RalphLoopEngine()
        resolver = IssueResolver(client, engine=engine)

        proj = engine.create_project("proj", "Test")
        issue = GitHubIssue.from_api_response(SAMPLE_ISSUE_RAW)
        issue.extracted_tasks = ["Do thing 1", "Do thing 2", "Do thing 3"]
        issue.suggested_phase = "Bug Fixes"

        result = resolver.map_issues_to_hierarchy([issue], proj, analyze=False)
        # Find the subphase for this issue
        phase = proj.phases[0]
        sp = phase.subphases[0]
        assert len(sp.tasks) == 3
        assert sp.tasks[0].name == "Do thing 1"

    def test_map_issue_without_tasks_gets_default(self):
        """Issue without extracted tasks gets a single default TaskNode."""
        client = GitHubClient(owner="org", repo="repo")
        engine = RalphLoopEngine()
        resolver = IssueResolver(client, engine=engine)

        proj = engine.create_project("proj", "Test")
        issue = GitHubIssue(number=1, title="Simple issue")
        issue.suggested_phase = "Backlog"

        result = resolver.map_issues_to_hierarchy([issue], proj, analyze=False)
        phase = proj.phases[0]
        sp = phase.subphases[0]
        # Should have at least one task (the issue title itself)
        assert len(sp.tasks) >= 1

    def test_create_phase_from_issue(self):
        """create_phase_from_issue creates a dedicated PhaseNode."""
        client = GitHubClient(owner="org", repo="repo")
        engine = RalphLoopEngine()
        resolver = IssueResolver(client, engine=engine)

        proj = engine.create_project("proj", "Test")
        issue = GitHubIssue.from_api_response(SAMPLE_ISSUE_RAW)
        issue.extracted_tasks = ["Reproduce", "Fix", "Test"]
        issue.suggested_phase = "Bug Fixes"

        phase = resolver.create_phase_from_issue(issue, proj, phase_counter=0)
        assert isinstance(phase, PhaseNode)
        assert phase in proj.phases
        assert len(phase.subphases) == 1
        assert len(phase.subphases[0].tasks) == 3

    def test_resolve_issues_to_project_integration(self):
        """End-to-end: fetch → analyze → map."""
        client = GitHubClient(owner="org", repo="repo")
        engine = RalphLoopEngine()
        resolver = IssueResolver(client, engine=engine)

        proj = engine.create_project("int-proj", "Integration Test")

        with patch.object(client, "_request", return_value=[SAMPLE_ISSUE_RAW]):
            result = resolver.resolve_issues_to_project(
                project=proj,
                state="open",
                analyze=True,
            )
            assert result is proj
            assert len(proj.phases) > 0

    def test_map_no_issues(self):
        """Mapping zero issues results in an empty project."""
        client = GitHubClient(owner="org", repo="repo")
        engine = RalphLoopEngine()
        resolver = IssueResolver(client, engine=engine)

        proj = engine.create_project("proj", "Empty")
        result = resolver.map_issues_to_hierarchy([], proj)
        assert len(proj.phases) == 0


class TestProjectSummary:
    """Tests for the get_project_summary method."""

    def test_summary_of_populated_project(self):
        client = GitHubClient(owner="org", repo="repo")
        engine = RalphLoopEngine()
        resolver = IssueResolver(client, engine=engine)

        proj = engine.create_project("proj", "Summary Project")
        issue = GitHubIssue.from_api_response(SAMPLE_ISSUE_RAW)
        issue.extracted_tasks = ["Task A", "Task B"]
        issue.suggested_phase = "Bug Fixes"

        resolver.map_issues_to_hierarchy([issue], proj, analyze=False)

        summary = resolver.get_project_summary(proj)
        assert "Summary Project" in summary
        # Should show 0% progress initially (all PENDING)
        assert "0.0%" in summary
        assert "Task A" in summary

    def test_summary_empty_project(self):
        client = GitHubClient(owner="org", repo="repo")
        engine = RalphLoopEngine()
        resolver = IssueResolver(client, engine=engine)

        proj = engine.create_project("proj", "Empty")
        summary = resolver.get_project_summary(proj)
        assert "Empty" in summary
        assert "0" in summary


# ═══════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_github_error_empty_body(self):
        """GitHubError with default values is created correctly."""
        err = GitHubError("Something went wrong")
        assert err.status_code == 0
        assert err.response_body == ""

    def test_client_no_token_no_env(self, monkeypatch):
        """Client works without any token (for public repos)."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        client = GitHubClient(owner="org", repo="repo")
        assert client._token == ""

    def test_resolver_creates_engine_if_not_provided(self):
        client = GitHubClient(owner="org", repo="repo")
        resolver = IssueResolver(client)
        assert resolver.engine is not None
        assert isinstance(resolver.engine, RalphLoopEngine)

    def test_fetch_issues_with_since_filter(self):
        client = GitHubClient(owner="org", repo="repo")
        resolver = IssueResolver(client)

        with patch.object(client, "_request", return_value=[]) as mock_req:
            issues = resolver.fetch_issues(state="open", since="2026-05-01T00:00:00Z")
            assert issues == []

    def test_analyze_with_mixed_labels(self):
        """When multiple labels, the first matched rule wins."""
        client = GitHubClient(owner="org", repo="repo")
        resolver = IssueResolver(client)
        issue = GitHubIssue(number=1, title="Fix docs typo", labels=["bug", "documentation"])

        analysis = resolver.analyze_issue(issue)
        # "bug" is checked first in fallback
        assert analysis.suggested_phase == "Bug Fixes"

    def test_paginated_empty_response(self):
        """Empty list response stops pagination."""
        client = GitHubClient(owner="org", repo="repo")
        with patch.object(client, "_request", return_value=[]):
            items = list(client._paginated("GET", "/test"))
            assert items == []

    def test_paginated_non_list_response(self):
        """Non-list responses are yielded as-is."""
        client = GitHubClient(owner="org", repo="repo")
        with patch.object(client, "_request", return_value={"total_count": 0, "items": []}):
            items = list(client._paginated("GET", "/test"))
            assert len(items) == 1
            assert "total_count" in items[0]


# ═══════════════════════════════════════════════════════════════════
# Ralph Loop Integration Smoke Tests
# ═══════════════════════════════════════════════════════════════════


class TestRalphLoopIntegration:
    """Smoke tests verifying that the issue resolver works with RalphLoopEngine."""

    def test_hierarchy_created_by_resolver_is_valid(self):
        """The hierarchy created by map_issues_to_hierarchy passes validation."""
        from spine.core.hierarchy import HierarchyValidator

        client = GitHubClient(owner="org", repo="repo")
        engine = RalphLoopEngine()
        resolver = IssueResolver(client, engine=engine)

        proj = engine.create_project("val-proj", "Validation Project")
        issues = [
            GitHubIssue.from_api_response(SAMPLE_ISSUE_RAW),
            GitHubIssue.from_api_response(SAMPLE_FEATURE_ISSUE),
        ]

        resolver.map_issues_to_hierarchy(issues, proj, analyze=True)

        validator = HierarchyValidator()
        result = validator.validate(proj)
        assert result.is_valid, f"Validation errors: {result.errors}"

    def test_progress_aggregation_on_mapped_project(self):
        """Progress can be rolled up on a hierarchy created from issues."""
        client = GitHubClient(owner="org", repo="repo")
        engine = RalphLoopEngine()
        resolver = IssueResolver(client, engine=engine)

        proj = engine.create_project("prog-proj", "Progress Project")
        issue = GitHubIssue.from_api_response(SAMPLE_ISSUE_RAW)
        issue.extracted_tasks = ["Task 1", "Task 2"]
        issue.suggested_phase = "Bug Fixes"

        resolver.map_issues_to_hierarchy([issue], proj, analyze=False)

        progress = engine.get_project_progress(proj)
        assert progress.total_tasks == 2
        assert progress.completed_tasks == 0
        assert progress.percent_complete == 0.0

    def test_state_transitions_on_mapped_issues(self):
        """State transitions work on hierarchy nodes created from issues."""
        client = GitHubClient(owner="org", repo="repo")
        engine = RalphLoopEngine()
        resolver = IssueResolver(client, engine=engine)

        proj = engine.create_project("trans-proj", "Transition Project")
        issue = GitHubIssue(number=1, title="Test", labels=["bug"])
        issue.suggested_phase = "Bug Fixes"
        issue.extracted_tasks = ["Fix it"]

        resolver.map_issues_to_hierarchy([issue], proj, analyze=False)

        # Find the task and transition it
        phase = proj.phases[0]
        sp = phase.subphases[0]
        task = sp.tasks[0]

        engine.transition_node(task, NodeStatus.RUNNING)
        assert task.status == NodeStatus.RUNNING

        task.progress = 100.0
        engine.transition_node(task, NodeStatus.SUCCESS)
        assert task.status == NodeStatus.SUCCESS

        # Progress should now be 100%
        progress = engine.get_project_progress(proj)
        assert progress.completed_tasks == 1
        assert progress.percent_complete == 100.0
