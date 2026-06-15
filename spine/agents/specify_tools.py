"""Custom tools for the SPECIFY phase agent.

Replaces generic filesystem tools with purpose-built tools that enforce
the specify agent's role: research the codebase via subagents, then write
a structured specification document. Nothing else.

Tools:
- ``read_work_context`` — returns work description, rework feedback, and
  any prior spec (for rework) in a single structured call. Eliminates
  multi-turn prompt/artifact reading.
- ``write_specification`` — the only write surface. Accepts structured
  spec sections and writes to the fixed specification.md path. Cannot
  write anything else.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field

from spine.agents.artifacts import artifact_path
from spine.models.types import Specification

logger = logging.getLogger(__name__)


# ── work context (eager-injected, no tool round-trip) ─────────────────────


def load_prior_spec(workspace_root: str, work_id: str) -> str:
    """Read a prior ``specification.md`` (rework pass) for eager injection.

    The work description, classification, retrieved code and critic feedback
    are already inlined into the SPECIFY agent's prompt, so the only context
    a rework pass needs that is not already present is the prior specification.
    This loads it directly so the agent never has to spend a turn calling
    ``read_work_context`` (trace 019ec965: that round-trip cost ~19K prompt
    tokens for a 29-token no-op tool call). Returns ``""`` when no prior spec
    exists (first pass).
    """
    spec_path = Path(workspace_root) / artifact_path(work_id, "specify") / "specification.md"
    if spec_path.exists():
        try:
            return spec_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not read prior spec at %s: %s", spec_path, exc)
    return ""


# ── read_work_context (legacy tool, retained for salvage/compat) ───────────


class _ReadWorkContextInput(BaseModel):
    """No inputs — all context is injected at build time."""


class ReadWorkContextTool(BaseTool):
    """Load all context the specify agent needs in one call.

    Returns a JSON object containing:
    - ``description``: the original work description
    - ``work_id``: the work item ID
    - ``work_type``: quick / spec / etc.
    - ``feedback``: list of prior critic/human feedback messages (empty on
      first run, populated on rework)
    - ``prior_spec``: content of a prior specification.md if this is a
      rework pass (empty string otherwise)
    - ``spec_dir``: where to write the specification artifact

    Call this FIRST and ONCE. Everything you need to write the spec is here.
    """

    name: str = "read_work_context"
    description: str = (
        "Load all context needed for the SPECIFY phase: work description, "
        "rework feedback (if any), and prior spec artifact (if rework). "
        "No arguments. Call this first — it gives you everything you need "
        "before dispatching researcher subagents."
    )
    args_schema: Optional[ArgsSchema] = _ReadWorkContextInput

    # Injected at build time
    workspace_root: str = ""
    work_id: str = ""
    work_type: str = ""
    description: str = ""
    feedback: list[str] = Field(default_factory=list)
    spec_dir: str = ""

    def _run(self, **kwargs: Any) -> str:  # noqa: ARG002
        result: dict[str, Any] = {
            "work_id": self.work_id,
            "work_type": self.work_type,
            "description": self.description,
            "feedback": self.feedback,
            "spec_dir": self.spec_dir,
            "prior_spec": "",
        }

        # Load prior spec if this is a rework pass
        spec_path = Path(self.workspace_root) / self.spec_dir / "specification.md"
        if spec_path.exists():
            try:
                result["prior_spec"] = spec_path.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning("Could not read prior spec at %s: %s", spec_path, exc)

        return json.dumps(result, ensure_ascii=False)


def build_work_context_block(prior_spec: str) -> str:
    """Render a prior specification as a labeled prompt block for rework.

    Empty (no prior spec) returns ``""`` so the section elides on first pass.
    """
    if not prior_spec.strip():
        return ""
    return (
        "## Prior Specification (revise this)\n\n"
        "A specification already exists from a previous pass. Revise it to "
        "address the feedback above — do not start from scratch.\n\n"
        f"```markdown\n{prior_spec.strip()}\n```"
    )

    async def _arun(self, **kwargs: Any) -> str:
        return self._run(**kwargs)


# ── write_specification ───────────────────────────────────────────────────


class _WriteSpecificationInput(BaseModel):
    """Structured specification fields. Mirrors :class:`Specification`.

    The agent provides structured data; the tool renders markdown and
    emits JSON. The agent does not author markdown.
    """

    title: str = Field(description="Specification title.")
    summary: str = Field(description="Executive summary (2-3 sentences).")
    objectives: list[str] = Field(
        default_factory=list,
        description="High-level goals as a list of short strings.",
    )
    requirements: list[str] = Field(
        description=(
            "Functional requirements as a list of short, measurable strings. "
            "Required — at least one item."
        ),
        min_length=1,
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Non-functional constraints as a list of short strings.",
    )
    scope_inclusions: list[str] = Field(
        default_factory=list,
        description="Areas explicitly in scope as a list of short strings.",
    )
    scope_exclusions: list[str] = Field(
        default_factory=list,
        description="Areas explicitly out of scope as a list of short strings.",
    )
    hard_boundaries: list[str] = Field(
        default_factory=list,
        description=(
            "No-touch file path globs, workspace-root-relative (fnmatch syntax, "
            "e.g. 'spine/billing/**', 'migrations/*'). These are ENFORCED: if the "
            "implementation writes any file matching one of these, the run is "
            "halted for human review. Use for surfaces that must stay untouched "
            "(other teams' modules, generated code, sensitive subsystems). Leave "
            "empty if no path is strictly off-limits — this is stricter than the "
            "prose scope_exclusions."
        ),
    )
    known_risks: list[str] = Field(
        default_factory=list,
        description="Known risks / open questions as a list of short strings.",
    )


def _render_spec_markdown(spec: Specification) -> str:
    """Render a Specification as markdown. Tool owns markdown shape."""
    parts: list[str] = [f"# {spec.title.strip()}\n", "\n## Summary\n", f"{spec.summary.strip()}\n"]

    def _bullets(heading: str, items: list[str]) -> None:
        cleaned = [s.strip() for s in items if s and s.strip()]
        if not cleaned:
            return
        parts.append(f"\n## {heading}\n")
        parts.extend(f"- {item}\n" for item in cleaned)

    def _numbered(heading: str, items: list[str]) -> None:
        cleaned = [s.strip() for s in items if s and s.strip()]
        if not cleaned:
            return
        parts.append(f"\n## {heading}\n")
        parts.extend(f"{i}. {item}\n" for i, item in enumerate(cleaned, start=1))

    _bullets("Objectives", spec.objectives)
    _numbered("Requirements", spec.requirements)
    _bullets("Constraints", spec.constraints)
    _bullets("Scope — Inclusions", spec.scope_inclusions)
    _bullets("Scope — Exclusions", spec.scope_exclusions)
    _bullets("Hard Boundaries (no-touch)", spec.hard_boundaries)
    _bullets("Known Risks", spec.known_risks)

    return "".join(parts)


_NONTRIVIAL_KEYWORDS: tuple[str, ...] = (
    "implement",
    "design",
    "refactor",
    "rebuild",
    "architect",
    "build",
)


def _is_nontrivial_description(description: str) -> bool:
    """Mirror the SPECIFY critic's proportionality heuristic.

    Long descriptions or descriptions that name implementation-class verbs
    require the spec to make scope decisions explicit; trivial flags or
    one-line tweaks do not.
    """
    if not description:
        return False
    if len(description) > 200:
        return True
    desc_lower = description.lower()
    return any(kw in desc_lower for kw in _NONTRIVIAL_KEYWORDS)


class WriteSpecificationTool(BaseTool):
    """Write the specification artifacts (specification.md + specification.json).

    This is the ONLY write tool available to the specify agent. Accepts
    structured fields matching :class:`Specification`; the tool itself
    renders markdown and emits JSON. The agent does not author markdown
    and does not hand-serialize JSON.
    """

    name: str = "write_specification"
    description: str = (
        "Write the specification artifacts (specification.md + specification.json). "
        "Provide structured fields (title, summary, objectives, requirements, "
        "constraints, scope_inclusions, scope_exclusions, hard_boundaries, known_risks). "
        "The tool renders markdown and emits JSON for you — do not call write_file."
    )
    args_schema: Optional[ArgsSchema] = _WriteSpecificationInput

    workspace_root: str = ""
    spec_dir: str = ""
    # Injected at build time so the empty-scope pre-check can apply the
    # proportionality heuristic without re-reading state.
    work_description: str = ""

    def _run(
        self,
        title: str,
        summary: str,
        requirements: list[str],
        objectives: list[str] | None = None,
        constraints: list[str] | None = None,
        scope_inclusions: list[str] | None = None,
        scope_exclusions: list[str] | None = None,
        hard_boundaries: list[str] | None = None,
        known_risks: list[str] | None = None,
    ) -> str:
        # Pre-validate: non-trivial descriptions MUST declare scope. Without
        # scope lists the downstream PLAN critic has nothing to check slice
        # scope-creep against and will reject every plan — better to fail
        # the synthesizer's tool call so it self-corrects in the same loop.
        if _is_nontrivial_description(self.work_description):
            missing: list[str] = []
            inclusions = scope_inclusions or []
            exclusions = scope_exclusions or []
            if not any(s and s.strip() for s in inclusions):
                missing.append("scope_inclusions")
            if not any(s and s.strip() for s in exclusions):
                missing.append("scope_exclusions")
            if missing:
                desc_preview = self.work_description[:80]
                return (
                    f"VALIDATION_ERROR: specification rejected before writing.\n"
                    f"For non-trivial work ('{desc_preview}…'), the "
                    f"following fields MUST be non-empty: {missing}. "
                    f"Without these, the downstream PLAN critic cannot perform "
                    f"scope-creep validation and will reject every plan. "
                    f"Re-call write_specification with these fields populated."
                )

        spec = Specification(
            title=title,
            summary=summary,
            objectives=objectives or [],
            requirements=requirements,
            constraints=constraints or [],
            scope_inclusions=scope_inclusions or [],
            scope_exclusions=scope_exclusions or [],
            hard_boundaries=hard_boundaries or [],
            known_risks=known_risks or [],
        )

        spec_path = Path(self.workspace_root) / self.spec_dir
        spec_path.mkdir(parents=True, exist_ok=True)
        md_output = spec_path / "specification.md"
        json_output = spec_path / "specification.json"

        md_content = _render_spec_markdown(spec)
        json_content = spec.model_dump_json(indent=2)

        try:
            md_output.write_text(md_content, encoding="utf-8")
        except OSError as exc:
            return f"ERROR: Could not write specification.md: {exc}"

        try:
            json_output.write_text(json_content, encoding="utf-8")
        except OSError as exc:
            return f"ERROR: Could not write specification.json: {exc}"

        return (
            f"specification.md ({len(md_content)} chars) and "
            f"specification.json ({len(json_content)} chars) written to {self.spec_dir}/."
        )

    async def _arun(self, **kwargs: Any) -> str:
        return self._run(**kwargs)


# ── Factory ───────────────────────────────────────────────────────────────


def build_specify_orchestrator_tools(
    workspace_root: str,
    work_id: str,
    description: str,
    work_type: str,
    feedback: list[str] | None = None,
    include_read_work_context: bool = True,
) -> list[BaseTool]:
    """Build the custom tool set for the specify agent.

    Returns:
    - ``write_specification``: structured write to specification.md (always)
    - ``read_work_context``: loads description, feedback, and prior spec —
      only when ``include_read_work_context`` is True

    These are the complete tool surface for the SPECIFY agent. No generic
    filesystem tools are exposed. Researcher subagents (when needed) are
    dispatched by the exploration subgraph router, not by the agent itself.

    Args:
        workspace_root: Absolute path to the project workspace root.
        work_id: The current work item ID.
        description: The work description from WorkflowState.
        work_type: The work type (quick, spec, etc.).
        feedback: List of prior feedback strings (for rework passes).
        include_read_work_context: When False, omit ``read_work_context``.
            The specify builders eagerly inline the work context into the
            prompt (see :func:`load_prior_spec`), so the tool — and its
            ~19K-token round-trip — is no longer needed (trace 019ec965).

    Returns:
        List of BaseTool instances (one or two depending on the flag).
    """
    spec_dir = artifact_path(work_id, "specify")

    tools: list[BaseTool] = [
        WriteSpecificationTool(
            workspace_root=workspace_root,
            spec_dir=spec_dir,
            work_description=description,
        ),
    ]
    if include_read_work_context:
        tools.insert(
            0,
            ReadWorkContextTool(
                workspace_root=workspace_root,
                work_id=work_id,
                work_type=work_type,
                description=description,
                feedback=feedback or [],
                spec_dir=spec_dir,
            ),
        )
    return tools


# ── Salvage (recover a spec the model printed as text) ─────────────────────


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of one JSON object from model text.

    Handles a ```json fenced block, a bare ``` fence, or — as a last resort —
    the first balanced top-level ``{...}`` object in the text. Returns the
    first candidate that parses to a dict, or ``None``.
    """
    if not text:
        return None

    candidates: list[str] = [
        m.group(1)
        for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    ]
    if not candidates:
        # Brace-match the first top-level object (ignores braces inside, since
        # we only need the outermost balanced span).
        start = text.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start : i + 1])
                        break

    for raw in candidates:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(data, dict):
            return data
    return None


def _coerce_str_list(value: Any) -> list[str]:
    """Coerce a spec field into a clean list of non-empty strings."""
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    return []


def salvage_specification_from_text(
    text: str, workspace_root: str, work_id: str
) -> bool:
    """Recover a specification from a model that printed JSON instead of calling.

    When the synthesizer emits the spec as a fenced ```json block rather than
    calling :class:`WriteSpecificationTool`, ``specification.json`` never lands
    on disk and the SPECIFY contract hard-fails. This parses that text and, if
    it carries the contract-required fields (``title``, ``summary``,
    ``requirements``), writes ``specification.json`` + ``specification.md``
    exactly as the tool would — turning an unrecoverable failure into a
    salvage. The full spend on the synthesizer's generation is preserved.

    Returns ``True`` when a valid specification was written, ``False`` when
    nothing parseable/sufficient was found (the caller then fails closed).
    """
    data = _extract_json_object(text)
    if not isinstance(data, dict):
        return False

    title = data.get("title")
    summary = data.get("summary")
    requirements = _coerce_str_list(data.get("requirements"))
    if not (isinstance(title, str) and title.strip()):
        return False
    if not (isinstance(summary, str) and summary.strip()):
        return False
    if not requirements:
        return False

    try:
        spec = Specification(
            title=title.strip(),
            summary=summary.strip(),
            objectives=_coerce_str_list(data.get("objectives")),
            requirements=requirements,
            constraints=_coerce_str_list(data.get("constraints")),
            scope_inclusions=_coerce_str_list(data.get("scope_inclusions")),
            scope_exclusions=_coerce_str_list(data.get("scope_exclusions")),
            hard_boundaries=_coerce_str_list(data.get("hard_boundaries")),
            known_risks=_coerce_str_list(data.get("known_risks")),
        )
    except Exception:
        logger.warning(
            "Salvage: parsed JSON did not satisfy the Specification schema",
            exc_info=True,
        )
        return False

    spec_path = Path(workspace_root) / artifact_path(work_id, "specify")
    try:
        spec_path.mkdir(parents=True, exist_ok=True)
        (spec_path / "specification.md").write_text(
            _render_spec_markdown(spec), encoding="utf-8"
        )
        (spec_path / "specification.json").write_text(
            spec.model_dump_json(indent=2), encoding="utf-8"
        )
    except OSError:
        logger.warning("Salvage: failed to write recovered specification", exc_info=True)
        return False

    logger.info(
        "Salvage: recovered specification.json from model text output for %s", work_id
    )
    return True
