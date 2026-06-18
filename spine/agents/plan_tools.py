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
- ``write_structured_plan`` — the only write surface. Accepts structured
  plan fields (architecture_overview, technology_choices, feature_slices,
  testing_strategy, risks, codebase_map) and writes both plan.md and
  plan.json to the fixed plan dir.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import BaseTool, ToolException
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from spine.agents.artifacts import artifact_path
# Reuse the canonical tool-call-markup token list so search_codebase and
# codebase_query agree on what a spilled <tool_call>/<arg_value> envelope
# looks like (codebase_query.py imports nothing from spine at module level,
# so this does not reintroduce the circular-import hazard noted below).
from spine.agents.tools.codebase_query import _FORBIDDEN_MARKUP_TOKENS
from spine.models.types import FeatureSlice as _SchedFeatureSlice

logger = logging.getLogger(__name__)

# NOTE: ``spine.workflow.slice_scheduler.validate_feature_slices`` is imported
# lazily inside ``write_structured_plan`` (below). Importing it at module level
# pulls in ``spine.workflow`` -> compose -> the phase subgraphs, which re-import
# this module — a circular import that breaks ``import spine.agents.plan_tools``
# when it is the first module loaded.

# File extensions worth reading when doing codebase search
_CODE_EXTENSIONS = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
    ".md",
    ".txt",
    ".sql",
    ".sh",
}
_MAX_FILE_PREVIEW = 3000  # chars per file returned in search results
_MAX_SEARCH_FILES = 8  # max files to return per search query


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
                "No prior artifacts found — work directly from the description."
            )

        return json.dumps(result, ensure_ascii=False)

    async def _arun(self, **kwargs: Any) -> str:
        return self._run(**kwargs)


# ── search_codebase ───────────────────────────────────────────────────────


def _reject_tool_markup(value: str, field: str) -> None:
    """Raise ``ValueError`` if a model-supplied string carries tool-call markup.

    Reuses :data:`codebase_query._FORBIDDEN_MARKUP_TOKENS` so the two
    research tools agree on what a spilled tool-call envelope looks like.
    """
    for token in _FORBIDDEN_MARKUP_TOKENS:
        if token in value:
            raise ValueError(
                f"search_codebase: {field!r} contains tool-call markup "
                f"({token!r}). Pass {field} as a clean JSON array of strings "
                f'(e.g. {field}=["WorkflowState", "submit_work"]); do not echo '
                f"the <tool_call>/<arg_value> envelope or merge other arguments in."
            )


def _clean_str_item(item: Any, field: str) -> str:
    """Stringify, markup-check, and strip a single list element."""
    text = item if isinstance(item, str) else str(item)
    _reject_tool_markup(text, field)
    return text.strip()


def _coerce_str_list(value: Any, field: str) -> Any:
    """Coerce a model-supplied ``list[str]`` tool argument into a real list.

    Local models intermittently serialise list args as a JSON-encoded string
    (``'["a", "b"]'``), a single bare keyword, or a fragment with their own
    tool-call XML spilled in. Trace ``019e72f5`` showed two PLAN
    ``search_codebase`` calls fail with a bare Pydantic ``list_type`` error
    for exactly this reason, and the unhelpful message taught the model
    nothing. Mirror the ``codebase_query`` hardening: reject spilled markup
    with a teaching message, otherwise coerce strings into the expected list
    so a recoverable call succeeds instead of failing. Empty elements are
    dropped (``queries`` then trips its own ``min_length`` guard).

    Non-string / non-list values (``None``, dicts, ints) pass through
    untouched so Pydantic raises its own type error.
    """
    if isinstance(value, list):
        return [s for item in value if (s := _clean_str_item(item, field))]
    if isinstance(value, str):
        _reject_tool_markup(value, field)
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            # Not JSON — treat the whole string as a single search term.
            return [stripped]
        if isinstance(parsed, list):
            return [s for item in parsed if (s := _clean_str_item(item, field))]
        # JSON scalar (string/number) — wrap as a one-element list.
        return [text] if (text := str(parsed).strip()) else []
    return value


class _SearchCodebaseInput(BaseModel):
    queries: list[str] = Field(
        default_factory=list,
        description=(
            "REQUIRED. Non-empty list of search terms or topic keywords. "
            "Each query is run as a case-insensitive substring search across "
            "all source files. E.g. ['WorkflowState', 'submit_work', 'UIApi']. "
            "You MUST pass this as a list of strings — a bare string will be "
            "rejected."
        ),
    )
    file_patterns: list[str] = Field(
        default_factory=list,
        description=(
            "Optional glob patterns to restrict the search scope (default: []). "
            "E.g. ['spine/agents/*.py', 'spine/workflow/*.py']. "
            "Leave empty / omit to search the entire workspace."
        ),
    )
    max_files: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Maximum number of files to return content for (default 6).",
    )

    @field_validator("queries", "file_patterns", mode="before")
    @classmethod
    def _coerce_list_args(cls, value: Any, info: ValidationInfo) -> Any:
        # Runs BEFORE list[str] type validation so a JSON-string / bare-string
        # arg is salvaged instead of 400-ing at the schema layer (trace
        # 019e72f5). Markup-spilled values are rejected with a teaching error.
        return _coerce_str_list(value, info.field_name or "queries")


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
        queries: list[str] | None = None,
        file_patterns: list[str] | None = None,
        max_files: int = 6,
    ) -> str:
        # Empty / missing queries used to 400 at the pydantic schema layer
        # ("queries Field required"), an unrecoverable error the local model
        # re-emitted verbatim (trace 019e77a7: 4× `search_codebase({})`).
        # Raise a recoverable ToolException with a worked example instead —
        # langchain returns it to the model as a ToolMessage it can act on.
        queries = [q for q in (queries or []) if str(q).strip()]
        if not queries:
            raise ToolException(
                "search_codebase: 'queries' is required and must be a non-empty "
                "list of keyword strings. Retry like "
                "queries=['WorkflowState', 'submit_work'] — a list of plain "
                "search terms, not an empty call."
            )
        # Resolve to an absolute path so candidate paths from rglob and
        # rg's output use the same representation. Trace 019e6974
        # showed total_files_found=0 for queries that obviously matched
        # because rglob returned relative paths ("spine/cli/...") while
        # rg returned ones prefixed with "./" — different Path objects,
        # so the `fpath in candidates` membership test below silently
        # dropped every hit.
        root = Path(self.workspace_root or ".").resolve()
        file_patterns = file_patterns or []

        # Build candidate file set: pattern-filtered or all source files
        candidates: set[Path] = set()
        all_patterns_invalid = False
        if file_patterns:
            invalid_patterns: list[str] = []
            for pattern in file_patterns:
                try:
                    candidates.update(root.glob(pattern))
                except ValueError as exc:
                    # pathlib rejects malformed globs — a bare '**' or 'a**b'
                    # raises "'**' can only be an entire path component"
                    # (trace 019e784c crashed the whole search on one such
                    # pattern from the slice-implementer). Skip the bad
                    # pattern instead of crashing; remember it so an all-bad
                    # set falls back to a full walk below.
                    invalid_patterns.append(pattern)
                    logger.warning(
                        "search_codebase: skipping invalid glob pattern %r: %s",
                        pattern, exc,
                    )
            all_patterns_invalid = len(invalid_patterns) == len(file_patterns)

        if not file_patterns or all_patterns_invalid:
            # No usable patterns → walk workspace, skip common non-code dirs.
            skip_dirs = {
                ".venv",
                "venv",
                ".git",
                "__pycache__",
                "node_modules",
                ".mypy_cache",
                ".ruff_cache",
                ".pytest_cache",
                "dist",
                "build",
                ".spine",
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

        # Surface the path-mismatch class of bugs (rg/candidates disagree
        # on canonical form, workspace_root pointing somewhere empty, etc.)
        # — otherwise the agent sees an empty result and assumes "nothing
        # to find", burning research turns.
        if total_found == 0 and candidates:
            logger.warning(
                "search_codebase: 0/%d candidates matched queries=%r in root=%s",
                len(candidates), queries, root,
            )

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
            results.append(
                {
                    "file": rel,
                    "score": score,
                    "match_lines": file_match_lines.get(fpath, [])[:6],
                    "preview": preview,
                }
            )

        return json.dumps(
            {
                "queries_run": queries,
                "total_files_found": total_found,
                "results": results,
            },
            ensure_ascii=False,
        )

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
            cmd = [
                "rg",
                "-i",
                "--line-number",
                "--with-filename",
                "--max-count=5",
                query,
                str(root),
            ]
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10, cwd=str(root))
            for line in out.stdout.splitlines():
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    # Resolve to the same absolute form candidates use so a
                    # leading "./" or symlink discrepancy doesn't kill the
                    # membership check.
                    try:
                        fpath = Path(parts[0]).resolve()
                    except OSError:
                        continue
                    if fpath in candidates:
                        matched.setdefault(fpath, []).append(
                            {
                                "line": parts[1],
                                "text": parts[2][:120],
                            }
                        )
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
        # Run the blocking _run on a worker thread so it doesn't stall
        # the LangGraph event loop (researchers fan out concurrently and
        # a 10s ripgrep timeout × N candidates would otherwise serialise).
        import asyncio

        return await asyncio.to_thread(
            self._run,
            queries=queries,
            file_patterns=file_patterns,
            max_files=max_files,
        )


# ── write_structured_plan ─────────────────────────────────────────────────


# Generous upper bound on feature slices — real plans are well under 20. The
# cap exists to contain local-model repetition collapse (trace 019ed44a, where
# the synthesizer emitted 1 real slice followed by 3561 scalar ``False`` values)
# before the degenerate array can fan out into the IMPLEMENT phase.
_MAX_FEATURE_SLICES = 60


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
    reference_symbols: list[str] = Field(
        default_factory=list,
        description=(
            "Qualified names of EXISTING symbols the implementer must read to "
            "write this slice correctly — the methods/classes its new code "
            "calls, extends, or mimics (e.g. 'UIApi.update_mcp_server', "
            "'SpineConfig'). Populate from your codebase research so the "
            "implementer can read_symbol them directly instead of searching. "
            "Names only — no source."
        ),
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
            "Components, data flow, and interfaces — prose paragraph describing "
            "how the pieces fit together."
        )
    )
    technology_choices: list[str] = Field(
        default_factory=list,
        description=(
            "Technology/library choices as a list of short strings (one item per "
            "choice, include rationale inline)."
        ),
    )
    feature_slices: list[_FeatureSliceInput] = Field(
        description=(
            "Array of feature slices. Each slice is a structured object with: id, "
            "title, target_files, execution_requirements, reference_symbols, "
            "dependencies, acceptance_criteria, complexity. Populate "
            "reference_symbols from your codebase research — the existing "
            "symbols each slice's code calls/extends/mimics — so the "
            "implementer reads exactly those instead of surveying files."
        ),
        min_length=1,
        max_length=_MAX_FEATURE_SLICES,
    )

    @field_validator("feature_slices", mode="before")
    @classmethod
    def _sanitize_feature_slices(cls, value: Any) -> Any:
        """Drop degenerate non-mapping entries before per-element validation.

        Weak local models occasionally collapse into a repetition loop and pad
        the array with scalars (e.g. 3561 ``False`` values after one real slice,
        trace 019ed44a). Left unfiltered, pydantic builds a multi-megabyte
        ValidationError (one error per bad element) that is recorded as the span
        error and can blow the agent's context window. Stripping the non-mapping
        entries here turns that into a single, recoverable validation message
        (the one malformed real slice) and prevents a degenerate-but-valid mega
        array from fanning out downstream. Non-list inputs are passed through so
        pydantic raises its normal type error.
        """
        if not isinstance(value, list):
            return value
        mappings = [v for v in value if isinstance(v, dict)]
        dropped = len(value) - len(mappings)
        if dropped:
            logger.warning(
                "write_structured_plan: dropped %d non-mapping feature_slice "
                "entries (likely model repetition collapse)",
                dropped,
            )
        if len(mappings) > _MAX_FEATURE_SLICES:
            logger.warning(
                "write_structured_plan: truncating %d feature_slices to %d",
                len(mappings),
                _MAX_FEATURE_SLICES,
            )
            mappings = mappings[:_MAX_FEATURE_SLICES]
        return mappings
    testing_strategy: str = Field(
        description=(
            "Test approach — prose paragraph: which tests to add/modify, test file "
            "paths, and how to verify correctness."
        )
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Known risks, edge cases, or open questions — list of short strings.",
    )
    codebase_map: str = Field(
        default="",
        description=(
            "Structured map of relevant codebase files, functions, and conventions "
            "discovered during research. Optional."
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
        "feature_slices. Provide structured fields (architecture_overview, "
        "technology_choices, feature_slices, testing_strategy, risks, "
        "codebase_map). The tool renders markdown and emits JSON for you — "
        "do not call write_file. Call once after research is complete."
    )
    args_schema: Optional[ArgsSchema] = _StructuredWritePlanInput

    workspace_root: str = ""
    plan_dir: str = ""

    def _run(
        self,
        architecture_overview: str,
        feature_slices: list[_FeatureSliceInput | dict[str, Any]],
        testing_strategy: str,
        technology_choices: list[str] | None = None,
        risks: list[str] | None = None,
        codebase_map: str = "",
    ) -> str:
        # Pydantic may pass either model instances or raw dicts depending on
        # the call site — coerce dicts so we can rely on attribute access.
        validated_slices: list[_FeatureSliceInput] = [
            sl if isinstance(sl, _FeatureSliceInput) else _FeatureSliceInput(**sl)
            for sl in feature_slices
        ]
        tech_choices = [c.strip() for c in (technology_choices or []) if c and c.strip()]
        risk_items = [r.strip() for r in (risks or []) if r and r.strip()]

        # Run the same structural validation the downstream scheduler will run
        # (unique IDs, dependency integrity, no cycles). Pulling it upstream
        # lets the synthesizer's tool loop self-correct within the same agent
        # invocation instead of round-tripping through a critic rework cycle.
        scheduler_slices = [
            _SchedFeatureSlice(
                id=sl.id,
                title=sl.title,
                target_files=list(sl.target_files),
                execution_requirements=[sl.execution_requirements]
                if isinstance(sl.execution_requirements, str)
                else list(sl.execution_requirements),
                dependencies=list(sl.dependencies),
                acceptance_criteria=list(sl.acceptance_criteria),
                complexity=sl.complexity,
            )
            for sl in validated_slices
        ]
        # Lazy import to avoid a module-load circular import (see top of file).
        from spine.workflow.slice_scheduler import validate_feature_slices

        try:
            validate_feature_slices(scheduler_slices)
        except ValueError as exc:
            return (
                f"VALIDATION_ERROR: plan rejected before writing.\n{exc}\n"
                "Fix the structural issue and call write_structured_plan again."
            )

        # ── Prepare output directory ──────────────────────────────────
        plan_path = Path(self.workspace_root) / self.plan_dir
        plan_path.mkdir(parents=True, exist_ok=True)

        # ── Build plan.md (narrative) ─────────────────────────────────
        lines = [
            "# Technical Plan\n",
            "\n## Architecture Overview\n",
            f"{architecture_overview.strip()}\n",
        ]

        if tech_choices:
            lines.append("\n## Technology Choices\n")
            lines.extend(f"- {item}\n" for item in tech_choices)

        lines.append("\n## Feature Slices\n")

        for sl in validated_slices:
            deps = ", ".join(sl.dependencies) if sl.dependencies else "none"
            lines.append(f"\n### {sl.id}: {sl.title}\n")
            lines.append(f"**Complexity:** {sl.complexity}  \n")
            lines.append(f"**Dependencies:** {deps}  \n")
            lines.append(f"**Target files:** {', '.join(sl.target_files) or 'TBD'}\n\n")
            if sl.reference_symbols:
                lines.append(
                    f"**Reference symbols:** {', '.join(sl.reference_symbols)}\n\n"
                )
            lines.append(f"{sl.execution_requirements.strip()}\n\n")
            lines.append("**Acceptance Criteria:**\n")
            for ac in sl.acceptance_criteria:
                lines.append(f"- {ac}\n")

        lines.append("\n## Testing Strategy\n")
        lines.append(f"{testing_strategy.strip()}\n")

        if risk_items:
            lines.append("\n## Risks & Open Questions\n")
            lines.extend(f"- {item}\n" for item in risk_items)

        if codebase_map.strip():
            lines.append("\n## Codebase Map\n")
            lines.append(f"{codebase_map.strip()}\n")

        md_content = "".join(lines)

        # ── Build plan.json (structured) ──────────────────────────────
        json_data: dict[str, Any] = {
            "architecture_overview": architecture_overview.strip(),
            "technology_choices": tech_choices,
            "feature_slices": [sl.model_dump() for sl in validated_slices],
            "testing_strategy": testing_strategy.strip(),
            "risks": risk_items,
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
        # ForceToolUntilCalledMiddleware treats any result not starting with
        # VALIDATION_ERROR/ERROR as a successful write — keep failure strings
        # on those prefixes (trace 019eb43f).
        return (
            f"plan.md ({len(md_content)} chars) and plan.json "
            f"({len(json_content)} chars) written to {self.plan_dir}/. "
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

    Returns three tools:
    - ``read_prior_artifacts``: loads spec + context in one call
    - ``search_codebase``: targeted multi-query file search
    - ``write_structured_plan``: structured write to plan.md + plan.json
      with feature slices and dependencies

    These replace all generic filesystem tools. No ls/glob/grep/read_file/write_file
    exposed. Researcher subagents (when needed) are dispatched by the
    exploration subgraph router, not by the agent itself.

    Args:
        workspace_root: Absolute path to the project workspace root.
        work_id: The current work item ID.
        description: The work description from WorkflowState.
        work_type: The work type (quick, spec, etc.).
        prior_phase_dirs: Mapping of phase name → artifact directory
            (relative to workspace_root). E.g. {"specify": ".spine/artifacts/x/specify"}.
        feedback: List of prior feedback strings (for rework passes).

    Returns:
        List of three BaseTool instances.
    """
    plan_dir = artifact_path(work_id, "plan")

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
        StructuredWritePlanTool(
            workspace_root=workspace_root,
            plan_dir=plan_dir,
        ),
    ]
