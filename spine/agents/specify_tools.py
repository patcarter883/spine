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
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field

from spine.agents.artifacts import artifact_path
from spine.models.types import Specification

logger = logging.getLogger(__name__)


# ── read_work_context ─────────────────────────────────────────────────────


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
    _bullets("Known Risks", spec.known_risks)

    return "".join(parts)


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
        "constraints, scope_inclusions, scope_exclusions, known_risks). "
        "The tool renders markdown and emits JSON for you — do not call write_file."
    )
    args_schema: Optional[ArgsSchema] = _WriteSpecificationInput

    workspace_root: str = ""
    spec_dir: str = ""

    def _run(
        self,
        title: str,
        summary: str,
        requirements: list[str],
        objectives: list[str] | None = None,
        constraints: list[str] | None = None,
        scope_inclusions: list[str] | None = None,
        scope_exclusions: list[str] | None = None,
        known_risks: list[str] | None = None,
    ) -> str:
        spec = Specification(
            title=title,
            summary=summary,
            objectives=objectives or [],
            requirements=requirements,
            constraints=constraints or [],
            scope_inclusions=scope_inclusions or [],
            scope_exclusions=scope_exclusions or [],
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
) -> list[BaseTool]:
    """Build the custom tool set for the specify agent.

    Returns two tools:
    - ``read_work_context``: loads description, feedback, and prior spec
    - ``write_specification``: structured write to specification.md

    These are the complete tool surface for the SPECIFY agent. No generic
    filesystem tools are exposed. Researcher subagents (when needed) are
    dispatched by the exploration subgraph router, not by the agent itself.

    Args:
        workspace_root: Absolute path to the project workspace root.
        work_id: The current work item ID.
        description: The work description from WorkflowState.
        work_type: The work type (quick, spec, etc.).
        feedback: List of prior feedback strings (for rework passes).

    Returns:
        List of two BaseTool instances.
    """
    spec_dir = artifact_path(work_id, "specify")

    return [
        ReadWorkContextTool(
            workspace_root=workspace_root,
            work_id=work_id,
            work_type=work_type,
            description=description,
            feedback=feedback or [],
            spec_dir=spec_dir,
        ),
        WriteSpecificationTool(
            workspace_root=workspace_root,
            spec_dir=spec_dir,
        ),
    ]
