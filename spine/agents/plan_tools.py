"""Custom tools for the PLAN phase agent.

Replaces generic filesystem tools with purpose-built tools that enforce
the plan agent's role: read the specification + codebase context, then
write a structured technical plan. Nothing else.

Tools:
- ``read_prior_artifacts`` — loads the specification and any other prior
  artifacts in one call. Eliminates multi-turn read_file hunting.
- ``search_codebase`` — targeted codebase lookup: given a list of topics
  or keywords, finds relevant files and returns their content summaries.
  Combines glob + grep + read_file into one composable call.
- ``write_plan`` — the only write surface. Accepts structured plan sections
  and writes to the fixed plan.md path.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# File extensions worth reading when doing codebase search
_CODE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".yaml", ".yml",
                    ".toml", ".json", ".md", ".txt", ".sql", ".sh"}
_MAX_FILE_PREVIEW = 3000   # chars per file returned in search results
_MAX_SEARCH_FILES = 8      # max files to return per search query


# ── read_prior_artifacts ──────────────────────────────────────────────────


class _ReadPriorArtifactsInput(BaseModel):
    """No inputs — artifact paths are determined by build-time config."""


class ReadPriorArtifactsTool(BaseTool):
    """Load all prior phase artifacts in one call.

    Returns a JSON object with one key per available phase (e.g. "specify"),
    each containing the full text of every artifact file in that phase's
    directory. The plan agent uses this to read the specification without
    needing to know the artifact path or call read_file.

    Also includes:
    - ``work_id``, ``work_type``, ``description`` — basic context
    - ``plan_dir`` — where to write plan.md
    - ``feedback`` — prior rework feedback (empty list on first run)
    """

    name: str = "read_prior_artifacts"
    description: str = (
        "Load all prior phase artifacts (specification, etc.) in one call. "
        "No arguments. Returns the full content of every prior artifact "
        "plus basic work context. Call this FIRST."
    )
    args_schema: Optional[ArgsSchema] = _ReadPriorArtifactsInput

    # Injected at build time
    workspace_root: str = ""
    work_id: str = ""
    work_type: str = ""
    description: str = ""
    feedback: list[str] = Field(default_factory=list)
    plan_dir: str = ""
    prior_phase_dirs: dict[str, str] = Field(default_factory=dict)

    def _run(self, **kwargs: Any) -> str:  # noqa: ARG002
        result: dict[str, Any] = {
            "work_id": self.work_id,
            "work_type": self.work_type,
            "description": self.description,
            "feedback": self.feedback,
            "plan_dir": self.plan_dir,
            "artifacts": {},
        }

        root = Path(self.workspace_root)
        for phase, phase_dir in self.prior_phase_dirs.items():
            phase_path = root / phase_dir
            if not phase_path.exists():
                continue
            phase_content: dict[str, str] = {}
            for fpath in sorted(phase_path.iterdir()):
                if (
                    fpath.is_file()
                    and not fpath.name.endswith(".meta.json")
                    and fpath.suffix in _CODE_EXTENSIONS | {".md", ""}
                ):
                    try:
                        phase_content[fpath.name] = fpath.read_text(encoding="utf-8")
                    except OSError as exc:
                        phase_content[fpath.name] = f"[read error: {exc}]"
            if phase_content:
                result["artifacts"][phase] = phase_content

        if not result["artifacts"]:
            result["warning"] = (
                "No prior artifacts found. This may be a quick workflow "
                "with no specification phase — work directly from the description."
            )

        return json.dumps(result, ensure_ascii=False)

    async def _arun(self, **kwargs: Any) -> str:
        return self._run(**kwargs)


# ── search_codebase ───────────────────────────────────────────────────────


class _SearchCodebaseInput(BaseModel):
    queries: list[str] = Field(
        description=(
            "List of search terms or topic keywords to find in the codebase. "
            "Each query is run as a case-insensitive substring search across "
            "all source files. E.g. ['WorkflowState', 'submit_work', 'UIApi']."
        ),
        min_length=1,
    )
    file_patterns: list[str] = Field(
        default_factory=list,
        description=(
            "Optional glob patterns to restrict the search scope. "
            "E.g. ['spine/agents/*.py', 'spine/workflow/*.py']. "
            "Leave empty to search the entire workspace."
        ),
    )
    max_files: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Maximum number of files to return content for (default 6).",
    )


class SearchCodebaseTool(BaseTool):
    """Search the codebase for relevant files given a list of topic keywords.

    Combines glob + ripgrep + file reading into a single composable call.
    Given a list of queries, finds matching files and returns their content
    summaries. Designed for the plan agent to answer questions like:
    "what files handle workflow state?" or "where is submit_work defined?"
    without needing ls/glob/grep/read_file as separate tools.

    Returns a JSON object:
    - ``results``: list of {file, matches: [{line, text}], preview: str}
    - ``total_files_found``: total matches before the max_files cap
    - ``queries_run``: echo of the queries used
    """

    name: str = "search_codebase"
    description: str = (
        "Search the codebase for files relevant to given topics or keywords. "
        "Pass a list of search terms; get back matching file paths with content "
        "previews. Use this to understand existing code before writing the plan. "
        "Replaces ls + glob + grep + read_file for exploratory research."
    )
    args_schema: Optional[ArgsSchema] = _SearchCodebaseInput

    workspace_root: str = ""

    def _run(
        self,
        queries: list[str],
        file_patterns: list[str] | None = None,
        max_files: int = 6,
    ) -> str:
        root = Path(self.workspace_root)
        file_patterns = file_patterns or []

        # Build candidate file set: pattern-filtered or all source files
        candidates: set[Path] = set()
        if file_patterns:
            for pattern in file_patterns:
                candidates.update(root.glob(pattern))
        else:
            # Walk workspace, skip common non-code dirs
            skip_dirs = {
                ".venv", "venv", ".git", "__pycache__", "node_modules",
                ".mypy_cache", ".ruff_cache", ".pytest_cache", "dist",
                "build", ".spine",
            }
            for fpath in root.rglob("*"):
                if fpath.is_file() and fpath.suffix in _CODE_EXTENSIONS:
                    if not any(part in skip_dirs for part in fpath.parts):
                        candidates.add(fpath)

        # Score files by how many queries match (using ripgrep for speed,
        # fall back to pure Python if rg is unavailable)
        file_scores: dict[Path, int] = {}
        file_match_lines: dict[Path, list[dict[str, Any]]] = {}

        for query in queries:
            matched_files = self._grep_files(root, query, candidates)
            for fpath, lines in matched_files.items():
                file_scores[fpath] = file_scores.get(fpath, 0) + 1
                if fpath not in file_match_lines:
                    file_match_lines[fpath] = []
                file_match_lines[fpath].extend(lines[:3])  # up to 3 lines per query

        # Sort by score desc, then name for stability
        ranked = sorted(file_scores.items(), key=lambda x: (-x[1], str(x[0])))
        total_found = len(ranked)
        top = ranked[: min(max_files, _MAX_SEARCH_FILES)]

        results = []
        for fpath, score in top:
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
                preview = content[:_MAX_FILE_PREVIEW]
                if len(content) > _MAX_FILE_PREVIEW:
                    preview += f"\n... [{len(content) - _MAX_FILE_PREVIEW} chars truncated]"
            except OSError:
                preview = "[could not read file]"

            rel = str(fpath.relative_to(root))
            results.append({
                "file": rel,
                "score": score,
                "match_lines": file_match_lines.get(fpath, [])[:6],
                "preview": preview,
            })

        return json.dumps({
            "queries_run": queries,
            "total_files_found": total_found,
            "results": results,
        }, ensure_ascii=False)

    def _grep_files(
        self,
        root: Path,
        query: str,
        candidates: set[Path],
    ) -> dict[Path, list[dict[str, Any]]]:
        """Return files containing query with matching line info."""
        matched: dict[Path, list[dict[str, Any]]] = {}

        # Try rg first (fast)
        try:
            cmd = ["rg", "-i", "--line-number",
                   "--with-filename", "--max-count=5", query, str(root)]
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10, cwd=str(root)
            )
            for line in out.stdout.splitlines():
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    fpath = Path(parts[0])
                    if fpath in candidates:
                        matched.setdefault(fpath, []).append({
                            "line": parts[1],
                            "text": parts[2][:120],
                        })
            return matched
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass  # rg not available, fall back to Python

        # Pure Python fallback
        q_lower = query.lower()
        for fpath in candidates:
            try:
                lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
                hits = [
                    {"line": str(i + 1), "text": ln[:120]}
                    for i, ln in enumerate(lines)
                    if q_lower in ln.lower()
                ][:5]
                if hits:
                    matched[fpath] = hits
            except OSError:
                continue
        return matched

    async def _arun(
        self,
        queries: list[str],
        file_patterns: list[str] | None = None,
        max_files: int = 6,
    ) -> str:
        return self._run(queries=queries, file_patterns=file_patterns, max_files=max_files)


# ── write_plan ────────────────────────────────────────────────────────────


class _WritePlanInput(BaseModel):
    architecture_overview: str = Field(
        description=(
            "Components, data flow, and interfaces. Include a diagram or "
            "structured breakdown of how the pieces fit together."
        )
    )
    technology_choices: str = Field(
        description="Technology/library choices with rationale."
    )
    module_structure: str = Field(
        description=(
            "File/module layout as a tree or table. Every new file must "
            "include its parent directory (which must exist in the workspace)."
        )
    )
    api_designs: str = Field(
        description=(
            "API endpoints, function signatures, data models, and contracts. "
            "Include types and schemas."
        )
    )
    implementation_order: str = Field(
        description=(
            "Ordered phases or waves of implementation with inter-slice "
            "dependencies. Used by the TASKS phase for decomposition."
        )
    )
    testing_strategy: str = Field(
        description=(
            "Test approach: which tests to add/modify, test file paths, "
            "and how to verify correctness."
        )
    )
    risks: str = Field(
        default="",
        description="Known risks, edge cases, or open questions. Optional.",
    )


class WritePlanTool(BaseTool):
    """Write the plan.md artifact.

    This is the ONLY write tool available to the plan agent.
    Accepts structured plan sections and writes them to the fixed plan.md
    path. Cannot write to any other location.
    """

    name: str = "write_plan"
    description: str = (
        "Write the plan.md artifact. "
        "This is the ONLY write tool — you cannot write other files. "
        "Provide all six required sections. Call this after research is complete."
    )
    args_schema: Optional[ArgsSchema] = _WritePlanInput

    workspace_root: str = ""
    plan_dir: str = ""

    def _run(
        self,
        architecture_overview: str,
        technology_choices: str,
        module_structure: str,
        api_designs: str,
        implementation_order: str,
        testing_strategy: str,
        risks: str = "",
    ) -> str:
        plan_path = Path(self.workspace_root) / self.plan_dir
        plan_path.mkdir(parents=True, exist_ok=True)
        output = plan_path / "plan.md"

        lines = [
            "# Technical Plan\n",
            "## Architecture Overview\n",
            f"{architecture_overview.strip()}\n",
            "\n## Technology Choices\n",
            f"{technology_choices.strip()}\n",
            "\n## Module Structure\n",
            f"{module_structure.strip()}\n",
            "\n## API Designs & Data Models\n",
            f"{api_designs.strip()}\n",
            "\n## Implementation Order\n",
            f"{implementation_order.strip()}\n",
            "\n## Testing Strategy\n",
            f"{testing_strategy.strip()}\n",
        ]
        if risks.strip():
            lines += ["\n## Risks & Open Questions\n", f"{risks.strip()}\n"]

        content = "".join(lines)
        try:
            output.write_text(content, encoding="utf-8")
        except OSError as exc:
            return f"ERROR: Could not write plan.md: {exc}"

        return (
            f"plan.md written to {self.plan_dir}/plan.md "
            f"({len(content)} chars)."
        )

    async def _arun(self, **kwargs: Any) -> str:
        return self._run(**kwargs)


# ── write_structured_plan ─────────────────────────────────────────────────

_REQUIRED_SLICE_KEYS = {
    "id",
    "title",
    "target_files",
    "execution_requirements",
    "dependencies",
    "acceptance_criteria",
    "complexity",
}


class _FeatureSliceInput(BaseModel):
    """Schema for a single feature slice within a structured plan."""

    id: str = Field(description="Unique identifier for the slice, e.g. 'add-user-auth'.")
    title: str = Field(description="Human-readable title for the slice.")
    target_files: list[str] = Field(
        description="Files to modify or create for this slice.",
        default_factory=list,
    )
    execution_requirements: str = Field(
        description="What must be done to implement this slice.",
    )
    dependencies: list[str] = Field(
        description="IDs of other slices this depends on (empty if none).",
        default_factory=list,
    )
    acceptance_criteria: list[str] = Field(
        description="Measurable criteria for this slice to be considered complete.",
        min_length=1,
    )
    complexity: str = Field(
        description="One of: small, medium, large.",
        default="medium",
    )


class _StructuredWritePlanInput(BaseModel):
    architecture_overview: str = Field(
        description=(
            "Components, data flow, and interfaces. Include a diagram or "
            "structured breakdown of how the pieces fit together."
        )
    )
    technology_choices: str = Field(
        description="Technology/library choices with rationale."
    )
    feature_slices: list[dict[str, Any]] = Field(
        description=(
            "Array of feature slices. Each must have: id, title, target_files, "
            "execution_requirements, dependencies, acceptance_criteria, complexity."
        ),
        min_length=1,
    )
    testing_strategy: str = Field(
        description=(
            "Test approach: which tests to add/modify, test file paths, "
            "and how to verify correctness."
        )
    )
    risks: str = Field(
        default="",
        description="Known risks, edge cases, or open questions. Optional.",
    )
    codebase_map: str = Field(
        default="",
        description=(
            "Structured map of relevant codebase files, functions, and conventions "
            "discovered during research."
        ),
    )


class StructuredWritePlanTool(BaseTool):
    """Write both plan.md (narrative) and plan.json (structured) artifacts.

    Extends the plan output with a ``feature_slices`` array that captures
    decomposition, dependencies, and acceptance criteria per slice.  The
    human-readable ``plan.md`` is written alongside a machine-readable
    ``plan.json`` so downstream phases (TASKS, IMPLEMENT) can consume
    structured data without re-parsing markdown.
    """

    name: str = "write_structured_plan"
    description: str = (
        "Write the plan artifacts (plan.md + plan.json) with structured "
        "feature slices. Use this instead of write_plan when you want to "
        "include a slice decomposition with dependencies. Call after research "
        "is complete."
    )
    args_schema: Optional[ArgsSchema] = _StructuredWritePlanInput

    workspace_root: str = ""
    plan_dir: str = ""

    def _run(
        self,
        architecture_overview: str,
        technology_choices: str,
        feature_slices: list[dict[str, Any]],
        testing_strategy: str,
        risks: str = "",
        codebase_map: str = "",
    ) -> str:
        # ── Validate feature_slices structure ─────────────────────────
        validated_slices: list[_FeatureSliceInput] = []
        errors: list[str] = []
        for idx, sl in enumerate(feature_slices):
            if not isinstance(sl, dict):
                errors.append(f"Slice at index {idx} is not a dict.")
                continue
            missing = _REQUIRED_SLICE_KEYS - set(sl.keys())
            if missing:
                errors.append(
                    f"Slice at index {idx} is missing keys: {', '.join(sorted(missing))}."
                )
                continue
            try:
                validated_slices.append(_FeatureSliceInput(**sl))
            except Exception as exc:
                errors.append(f"Slice at index {idx} validation error: {exc}")

        if errors:
            return "ERROR: Invalid feature_slices:\n" + "\n".join(errors)

        # ── Prepare output directory ──────────────────────────────────
        plan_path = Path(self.workspace_root) / self.plan_dir
        plan_path.mkdir(parents=True, exist_ok=True)

        # ── Build plan.md (narrative) ─────────────────────────────────
        lines = [
            "# Technical Plan\n",
            "\n## Architecture Overview\n",
            f"{architecture_overview.strip()}\n",
            "\n## Technology Choices\n",
            f"{technology_choices.strip()}\n",
            "\n## Feature Slices\n",
        ]

        for sl in validated_slices:
            deps = ", ".join(sl.dependencies) if sl.dependencies else "none"
            lines.append(f"\n### {sl.id}: {sl.title}\n")
            lines.append(f"**Complexity:** {sl.complexity}  \n")
            lines.append(f"**Dependencies:** {deps}  \n")
            lines.append(f"**Target files:** {', '.join(sl.target_files) or 'TBD'}\n\n")
            lines.append(f"{sl.execution_requirements.strip()}\n\n")
            lines.append("**Acceptance Criteria:**\n")
            for ac in sl.acceptance_criteria:
                lines.append(f"- {ac}\n")

        lines.append("\n## Testing Strategy\n")
        lines.append(f"{testing_strategy.strip()}\n")

        if risks.strip():
            lines.append("\n## Risks & Open Questions\n")
            lines.append(f"{risks.strip()}\n")

        if codebase_map.strip():
            lines.append("\n## Codebase Map\n")
            lines.append(f"{codebase_map.strip()}\n")

        md_content = "".join(lines)

        # ── Build plan.json (structured) ──────────────────────────────
        json_data: dict[str, Any] = {
            "architecture_overview": architecture_overview.strip(),
            "technology_choices": technology_choices.strip(),
            "feature_slices": [sl.model_dump() for sl in validated_slices],
            "testing_strategy": testing_strategy.strip(),
            "risks": risks.strip(),
            "codebase_map": codebase_map.strip(),
        }
        json_content = json.dumps(json_data, indent=2, ensure_ascii=False)

        # ── Write files ───────────────────────────────────────────────
        md_path = plan_path / "plan.md"
        json_path = plan_path / "plan.json"

        try:
            md_path.write_text(md_content, encoding="utf-8")
        except OSError as exc:
            return f"ERROR: Could not write plan.md: {exc}"

        try:
            json_path.write_text(json_content, encoding="utf-8")
        except OSError as exc:
            return f"ERROR: Could not write plan.json: {exc}"

        slice_count = len(validated_slices)
        return (
            f"Plan artifacts written: {self.plan_dir}/plan.md "
            f"({len(md_content)} chars), {self.plan_dir}/plan.json "
            f"({len(json_content)} chars). "
            f"{slice_count} feature slice(s) included."
        )

    async def _arun(self, **kwargs: Any) -> str:
        return self._run(**kwargs)


# ── Factory ───────────────────────────────────────────────────────────────


def build_plan_agent_tools(
    workspace_root: str,
    work_id: str,
    description: str,
    work_type: str,
    prior_phase_dirs: dict[str, str],
    feedback: list[str] | None = None,
) -> list[BaseTool]:
    """Build the custom tool set for the plan agent.

    Returns four tools:
    - ``read_prior_artifacts``: loads spec + context in one call
    - ``search_codebase``: targeted multi-query file search
    - ``write_plan``: structured write to plan.md (narrative only)
    - ``write_structured_plan``: structured write to plan.md + plan.json
      with feature slices and dependencies

    Together with ``eval`` (CodeInterpreterMiddleware), these replace all
    generic filesystem tools. No ls/glob/grep/read_file/write_file exposed.

    Args:
        workspace_root: Absolute path to the project workspace root.
        work_id: The current work item ID.
        description: The work description from WorkflowState.
        work_type: The work type (quick, spec, etc.).
        prior_phase_dirs: Mapping of phase name → artifact directory
            (relative to workspace_root). E.g. {"specify": ".spine/artifacts/x/specify"}.
        feedback: List of prior feedback strings (for rework passes).

    Returns:
        List of four BaseTool instances.
    """
    plan_dir = f".spine/artifacts/{work_id}/plan"

    return [
        ReadPriorArtifactsTool(
            workspace_root=workspace_root,
            work_id=work_id,
            work_type=work_type,
            description=description,
            feedback=feedback or [],
            plan_dir=plan_dir,
            prior_phase_dirs=prior_phase_dirs,
        ),
        SearchCodebaseTool(
            workspace_root=workspace_root,
        ),
        WritePlanTool(
            workspace_root=workspace_root,
            plan_dir=plan_dir,
        ),
        StructuredWritePlanTool(
            workspace_root=workspace_root,
            plan_dir=plan_dir,
        ),
    ]
