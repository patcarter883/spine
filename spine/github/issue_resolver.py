"""GitHub Issue Resolver — AI-powered issue-to-hierarchy mapping.

Integrates GitHub issues with the Ralph Loop hierarchical automation framework.
Uses an LLM provider to analyze issue content and auto-generate PhaseNodes/TaskNodes.

Architecture:
    GitHub API → IssueResolver.fetch_issues() → LLM analysis → RalphLoopEngine tree
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Union

from .client import GitHubClient
from ..models.types import (
    PhaseNode,
    SubPhaseNode,
    TaskNode,
    ProjectNode,
    HierarchyLevel,
    NodeStatus,
    Phase,
    SubPhase,
    Task,
)
from ..core.hierarchy import RalphLoopEngine


# ── Issue Data Model ─────────────────────────────────────────────

@dataclass
class GitHubIssue:
    """Normalized representation of a GitHub issue.

    Strips API-specific fields to a clean, portable structure
    suitable for hierarchy mapping and LLM analysis.
    """

    number: int
    title: str
    body: str = ""
    state: str = "open"
    labels: List[str] = field(default_factory=list)
    assignees: List[str] = field(default_factory=list)
    milestone: Optional[str] = None
    url: str = ""
    created_at: str = ""
    updated_at: str = ""
    # Extracted metadata from analysis
    estimated_complexity: str = "medium"  # low, medium, high
    extracted_tasks: List[str] = field(default_factory=list)
    suggested_phase: str = ""

    @classmethod
    def from_api_response(cls, data: Dict[str, Any]) -> "GitHubIssue":
        """Create a GitHubIssue from a raw API response dict.

        Args:
            data: Raw issue dict from GitHub API (list_issues or get_issue).

        Returns:
            Normalized GitHubIssue.
        """
        labels = [lb["name"] if isinstance(lb, dict) else str(lb) for lb in data.get("labels", [])]
        assignees = [a["login"] if isinstance(a, dict) else str(a) for a in data.get("assignees", [])]

        milestone = None
        ms_data = data.get("milestone")
        if ms_data and isinstance(ms_data, dict):
            milestone = ms_data.get("title", "")

        body = data.get("body") or ""

        return cls(
            number=data.get("number", 0),
            title=data.get("title", ""),
            body=body,
            state=data.get("state", "open"),
            labels=labels,
            assignees=assignees,
            milestone=milestone,
            url=data.get("html_url", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

    def to_summary(self) -> str:
        """Return a compact summary suitable for LLM prompt injection.

        Returns:
            Markdown-formatted summary string.
        """
        parts = [
            f"### Issue #{self.number}: {self.title}",
            f"**State:** {self.state}",
        ]
        if self.labels:
            parts.append(f"**Labels:** {', '.join(self.labels)}")
        if self.assignees:
            parts.append(f"**Assignees:** {', '.join(self.assignees)}")
        if self.milestone:
            parts.append(f"**Milestone:** {self.milestone}")
        parts.append("")
        parts.append(self.body[:2000] if self.body else "_No description_")
        return "\n".join(parts)


# ── Issue Resolver ──────────────────────────────────────────────

@dataclass
class IssueAnalysis:
    """Result of AI-powered issue analysis.

    Contains extracted tasks, complexity estimate, and a suggested
    phase name for hierarchy placement.
    """

    issue_number: int
    complexity: str = "medium"  # low, medium, high
    suggested_phase: str = ""
    extracted_tasks: List[str] = field(default_factory=list)
    confidence: float = 0.5
    raw_response: str = ""


class IssueResolver:
    """Resolves GitHub issues into Ralph Loop hierarchy nodes.

    Fetches issues from GitHub, uses an LLM provider (optional) to
    analyze issue content, and maps them into PhaseNodes and TaskNodes
    for the RalphLoopEngine.

    Usage:
        client = GitHubClient(token=..., owner="org", repo="repo")
        resolver = IssueResolver(client, engine=RalphLoopEngine())
        proj = engine.create_project("project-1", "GitHub Issues")
        resolver.resolve_issues_to_project(state="open", project=proj)
    """

    # Default analysis prompt for LLM-based issue decomposition
    DEFAULT_ANALYSIS_PROMPT = """Analyze this GitHub issue and extract actionable information.

{issue_summary}

Please respond with a JSON object containing:
1. "complexity": one of "low", "medium", "high"
2. "suggested_phase": a suggested phase name (e.g., "Implementation", "Bug Fix", "Documentation")
3. "extracted_tasks": a list of concrete, atomic tasks that would resolve this issue
4. "confidence": your confidence in this analysis (0.0 to 1.0)

Respond with ONLY the JSON object, no markdown or explanation.
Example: {{"complexity": "medium", "suggested_phase": "Bug Fix", "extracted_tasks": ["Reproduce the bug", "Identify root cause", "Implement fix", "Add tests"], "confidence": 0.85}}"""

    # Fallback analysis when no LLM is available
    FALLBACK_ANALYSIS_PROMPT = """Based on labels and title, provide a structured analysis of this issue.

Issue title: {title}
Labels: {labels}

Rules:
- "bug" label → complexity: medium, phase: "Bug Fixes"
- "feature" or "enhancement" label → complexity: high, phase: "Features"
- "documentation" label → complexity: low, phase: "Documentation"
- "good first issue" label → complexity: low
- Default → complexity: medium, phase: "Backlog"

Return a JSON object with complexity, suggested_phase, extracted_tasks, and confidence.
Default extracted_tasks: ["Analyze issue requirements", "Implement solution", "Verify and test"]"""

    def __init__(
        self,
        client: GitHubClient,
        engine: Optional[RalphLoopEngine] = None,
        llm_provider: Any = None,
    ):
        """Initialize the issue resolver.

        Args:
            client: Configured GitHubClient instance.
            engine: RalphLoopEngine for hierarchy construction (created if None).
            llm_provider: Optional LLMProvider for AI-powered issue analysis.
        """
        self.client = client
        self.engine = engine or RalphLoopEngine()
        self.llm_provider = llm_provider
        self._analysis_prompt = self.DEFAULT_ANALYSIS_PROMPT

    def set_analysis_prompt(self, prompt_template: str) -> None:
        """Override the analysis prompt template.

        The template should contain `{issue_summary}` which will be
        replaced with the issue's to_summary() output.

        Args:
            prompt_template: Custom prompt template string.
        """
        self._analysis_prompt = prompt_template

    # ── Issue Fetching ──────────────────────────────────────────

    def fetch_issues(
        self,
        state: str = "open",
        labels: Optional[List[str]] = None,
        since: Optional[str] = None,
    ) -> List[GitHubIssue]:
        """Fetch issues from GitHub and normalize them.

        Args:
            state: "open", "closed", or "all".
            labels: Optional label filter.
            since: ISO 8601 timestamp for updated-after filter.

        Returns:
            List of normalized GitHubIssue objects.
        """
        raw = self.client.list_issues(state=state, labels=labels, since=since)
        return [GitHubIssue.from_api_response(item) for item in raw]

    def fetch_issue(self, issue_number: int) -> GitHubIssue:
        """Fetch a single issue by number.

        Args:
            issue_number: Issue number.

        Returns:
            Normalized GitHubIssue.
        """
        raw = self.client.get_issue(issue_number)
        return GitHubIssue.from_api_response(raw)

    # ── Issue Analysis ──────────────────────────────────────────

    def analyze_issue(self, issue: GitHubIssue) -> IssueAnalysis:
        """Analyze a GitHub issue to extract tasks and estimate complexity.

        Uses the LLM provider if available, otherwise falls back to
        rule-based analysis using labels and keywords.

        Args:
            issue: The GitHubIssue to analyze.

        Returns:
            IssueAnalysis with extracted tasks and metadata.
        """
        if self.llm_provider is not None:
            return self._analyze_with_llm(issue)
        return self._analyze_fallback(issue)

    def analyze_issues(self, issues: List[GitHubIssue]) -> Dict[int, IssueAnalysis]:
        """Analyze multiple issues in batch.

        Analyzes each issue sequentially to avoid overwhelming the LLM.
        Returns a dict mapping issue_number → IssueAnalysis.

        Args:
            issues: List of issues to analyze.

        Returns:
            Dict[int, IssueAnalysis] mapping issue numbers to analyses.
        """
        results: Dict[int, IssueAnalysis] = {}
        for issue in issues:
            results[issue.number] = self.analyze_issue(issue)
        return results

    def _analyze_with_llm(self, issue: GitHubIssue) -> IssueAnalysis:
        """Analyze an issue using the configured LLM provider.

        Args:
            issue: The GitHubIssue to analyze.

        Returns:
            IssueAnalysis with LLM-extracted data.
        """
        prompt = self._analysis_prompt.format(issue_summary=issue.to_summary())
        try:
            raw = self.llm_provider.generate(prompt)
        except Exception:
            return self._analyze_fallback(issue)

        parsed = self._parse_analysis_json(raw, issue.number)
        parsed.raw_response = raw
        return parsed

    def _analyze_fallback(self, issue: GitHubIssue) -> IssueAnalysis:
        """Rule-based fallback analysis when no LLM is available.

        Uses issue labels and title keywords to infer phase, complexity,
        and generate default tasks.

        Args:
            issue: The GitHubIssue to analyze.

        Returns:
            IssueAnalysis with rule-based extraction.
        """
        complexity = "medium"
        suggested_phase = "Backlog"
        label_set = {lb.lower() for lb in issue.labels}

        if "bug" in label_set:
            complexity = "medium"
            suggested_phase = "Bug Fixes"
        elif "feature" in label_set or "enhancement" in label_set:
            complexity = "high"
            suggested_phase = "Features"
        elif "documentation" in label_set or "docs" in label_set:
            complexity = "low"
            suggested_phase = "Documentation"
        elif "good first issue" in label_set:
            complexity = "low"

        # Also check title for hints
        title_lower = issue.title.lower()
        if any(kw in title_lower for kw in ("bug", "fix", "crash", "error", "broken")):
            suggested_phase = "Bug Fixes"
            complexity = "medium"
        elif any(kw in title_lower for kw in ("feature", "add", "implement", "support")):
            suggested_phase = "Features"
            complexity = "high"
        elif any(kw in title_lower for kw in ("doc", "readme", "document")):
            suggested_phase = "Documentation"
            complexity = "low"

        tasks = self._extract_tasks_from_body(issue)
        if not tasks:
            tasks = ["Analyze issue requirements", "Implement solution", "Verify and test"]

        # Update the issue with extracted metadata
        issue.estimated_complexity = complexity
        issue.suggested_phase = suggested_phase
        issue.extracted_tasks = tasks

        return IssueAnalysis(
            issue_number=issue.number,
            complexity=complexity,
            suggested_phase=suggested_phase,
            extracted_tasks=tasks,
            confidence=0.6,
            raw_response="fallback",
        )

    def _parse_analysis_json(self, raw: str, issue_number: int) -> IssueAnalysis:
        """Parse LLM response JSON into an IssueAnalysis.

        Handles markdown code fences and partial JSON gracefully.

        Args:
            raw: Raw LLM response string.
            issue_number: The issue number.

        Returns:
            IssueAnalysis with parsed data.
        """
        # Try to extract JSON from code fences or direct content
        cleaned = raw.strip()

        # Remove markdown code fences if present
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        # Find first { and last }
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return IssueAnalysis(
                issue_number=issue_number,
                complexity="medium",
                suggested_phase="Backlog",
                extracted_tasks=["Analyze issue", "Implement solution", "Verify"],
                confidence=0.3,
                raw_response=raw,
            )

        return IssueAnalysis(
            issue_number=issue_number,
            complexity=data.get("complexity", "medium"),
            suggested_phase=data.get("suggested_phase", "Backlog"),
            extracted_tasks=data.get("extracted_tasks", []),
            confidence=float(data.get("confidence", 0.5)),
            raw_response=raw,
        )

    def _extract_tasks_from_body(self, issue: GitHubIssue) -> List[str]:
        """Extract task items from a Markdown checklist in the issue body.

        Parses lines like '- [ ] Do something' or '* [x] Completed item'.

        Args:
            issue: The GitHubIssue to scan.

        Returns:
            List of task descriptions extracted from checklists.
        """
        if not issue.body:
            return []

        tasks: List[str] = []
        for line in issue.body.split("\n"):
            stripped = line.strip()
            # Match '- [ ] task' or '- [x] task' or '* [ ] task'
            match = re.match(r"[-*]\s+\[[ xX]\]\s+(.+)", stripped)
            if match:
                tasks.append(match.group(1).strip())
        return tasks

    # ── Hierarchy Mapping ───────────────────────────────────────

    def map_issues_to_hierarchy(
        self,
        issues: List[GitHubIssue],
        project: ProjectNode,
        analyze: bool = True,
    ) -> ProjectNode:
        """Map GitHub issues into a Ralph Loop hierarchy tree.

        For each issue:
        1. Optionally analyze with LLM/fallback
        2. Determine if issue should be a PhaseNode or TaskNode
        3. Create appropriate nodes in the project hierarchy
        4. Group issues by suggested phase

        Args:
            issues: List of GitHubIssues to map.
            project: The ProjectNode to attach hierarchy to.
            analyze: If True, run LLM/fallback analysis on each issue first.

        Returns:
            The updated ProjectNode with mapped hierarchy.
        """
        # Group issues by suggested phase for PhaseNode creation
        phase_groups: Dict[str, List[GitHubIssue]] = {}

        for issue in issues:
            if analyze:
                analysis = self.analyze_issue(issue)
                issue.estimated_complexity = analysis.complexity
                issue.suggested_phase = analysis.suggested_phase
                issue.extracted_tasks = analysis.extracted_tasks

            phase_name = issue.suggested_phase or "Backlog"
            if phase_name not in phase_groups:
                phase_groups[phase_name] = []
            phase_groups[phase_name].append(issue)

        # Create PhaseNodes for each group
        phase_counter = 0
        for phase_name, phase_issues in phase_groups.items():
            phase_node = self.engine.create_phase(
                id=f"{project.id}-phase-{phase_counter}",
                name=phase_name,
                parent_project=project,
            )
            phase_counter += 1

            # Create subphases and tasks
            for i, issue in enumerate(phase_issues):
                if issue.extracted_tasks:
                    # Issue with extracted tasks → SubPhaseNode → TaskNodes
                    sp = self.engine.create_subphase(
                        id=f"{project.id}-sp-{phase_counter - 1}-{i}",
                        name=f"#{issue.number}: {issue.title[:60]}",
                        parent_phase=phase_node,
                    )
                    for j, task_desc in enumerate(issue.extracted_tasks):
                        self.engine.create_task(
                            id=f"{project.id}-task-{phase_counter - 1}-{i}-{j}",
                            name=task_desc,
                            parent_subphase=sp,
                        )
                else:
                    # Simple issue → single TaskNode
                    sp = self.engine.create_subphase(
                        id=f"{project.id}-sp-{phase_counter - 1}-{i}",
                        name=f"#{issue.number}: {issue.title[:60]}",
                        parent_phase=phase_node,
                    )
                    self.engine.create_task(
                        id=f"{project.id}-task-{phase_counter - 1}-{i}-0",
                        name=issue.title[:100],
                        parent_subphase=sp,
                    )

        return project

    def create_phase_from_issue(
        self,
        issue: GitHubIssue,
        project: ProjectNode,
        phase_counter: int = 0,
    ) -> PhaseNode:
        """Create a single PhaseNode from a GitHub issue.

        For large/scoped issues that warrant their own phase.

        Args:
            issue: The GitHubIssue to convert.
            project: Parent ProjectNode.
            phase_counter: Index for unique ID generation.

        Returns:
            The created PhaseNode.
        """
        phase = self.engine.create_phase(
            id=f"{project.id}-phase-{phase_counter}",
            name=f"#{issue.number}: {issue.title[:50]}",
            parent_project=project,
        )

        # Create subphase and tasks
        sp = self.engine.create_subphase(
            id=f"{project.id}-sp-{phase_counter}-0",
            name="Implementation",
            parent_phase=phase,
        )

        tasks = issue.extracted_tasks or ["Implement solution"]
        for j, task_desc in enumerate(tasks):
            self.engine.create_task(
                id=f"{project.id}-task-{phase_counter}-0-{j}",
                name=task_desc,
                parent_subphase=sp,
            )

        return phase

    def resolve_issues_to_project(
        self,
        project: ProjectNode,
        state: str = "open",
        labels: Optional[List[str]] = None,
        analyze: bool = True,
    ) -> ProjectNode:
        """High-level convenience: fetch + analyze + map in one call.

        Args:
            project: The ProjectNode to populate.
            state: Issue state filter.
            labels: Optional label filter.
            analyze: Whether to run LLM analysis.

        Returns:
            The populated ProjectNode.
        """
        issues = self.fetch_issues(state=state, labels=labels)
        return self.map_issues_to_hierarchy(issues, project, analyze=analyze)

    def get_project_summary(self, project: ProjectNode) -> str:
        """Generate a human-readable summary of the issue hierarchy.

        Args:
            project: The populated ProjectNode.

        Returns:
            Markdown-formatted summary.
        """
        progress = self.engine.get_project_progress(project)
        lines = [
            f"## Project: {project.name}",
            f"**Total Issues:** {progress.total_tasks}",
            f"**Completed:** {progress.completed_tasks}",
            f"**Failed:** {progress.failed_tasks}",
            f"**Blocked:** {progress.blocked_tasks}",
            f"**Progress:** {progress.percent_complete:.1f}%",
            "",
        ]

        for phase in project.phases:
            p_progress = self.engine.progress_aggregator.aggregate_phase(phase)
            lines.append(f"### Phase: {phase.name} ({p_progress.percent_complete:.0f}%)")
            for sp in phase.subphases:
                sp_progress = self.engine.progress_aggregator.aggregate_from_children(sp.tasks)
                lines.append(f"  - **{sp.name}** [{sp_progress.completed_tasks}/{sp_progress.total_tasks} tasks]")
                for task in sp.tasks:
                    status_icon = {
                        NodeStatus.PENDING: "⏳",
                        NodeStatus.RUNNING: "🔄",
                        NodeStatus.SUCCESS: "✅",
                        NodeStatus.FAILED: "❌",
                        NodeStatus.BLOCKED: "🚫",
                        NodeStatus.REWORKING: "♻️",
                        NodeStatus.CANCELLED: "❎",
                    }.get(task.status, "❓")
                    lines.append(f"    {status_icon} {task.name}")

        return "\n".join(lines)
