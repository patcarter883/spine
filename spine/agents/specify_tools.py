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
    overview: str = Field(description="Summary of what needs to be built (2-4 paragraphs).")
    requirements: str = Field(
        description=(
            "Functional and non-functional requirements as a markdown list. "
            "Each requirement should be measurable."
        )
    )
    architecture: str = Field(
        description=(
            "High-level design decisions: components, data flow, key patterns. "
            "Include rationale for major choices."
        )
    )
    interfaces: str = Field(
        description=(
            "API endpoints, data models, and contracts. Include types, "
            "signatures, and schemas where applicable."
        )
    )
    success_criteria: str = Field(
        description=(
            "Measurable outcomes that define completion. Each criterion "
            "must be verifiable by the VERIFY phase."
        )
    )
    open_questions: str = Field(
        default="",
        description=(
            "Any open questions or risks discovered during research. Optional — omit if none."
        ),
    )


class WriteSpecificationTool(BaseTool):
    """Write the specification.md artifact.

    This is the ONLY write tool available to the specify agent.
    Accepts structured sections and writes a complete specification
    to the fixed path. Cannot write to any other location.
    """

    name: str = "write_specification"
    description: str = (
        "Write the specification.md artifact. "
        "This is the ONLY write tool available — you cannot write other files. "
        "Provide all five required sections. Call this after researcher subagents "
        "have returned their findings."
    )
    args_schema: Optional[ArgsSchema] = _WriteSpecificationInput

    workspace_root: str = ""
    spec_dir: str = ""

    def _run(
        self,
        overview: str,
        requirements: str,
        architecture: str,
        interfaces: str,
        success_criteria: str,
        open_questions: str = "",
    ) -> str:
        spec_path = Path(self.workspace_root) / self.spec_dir
        spec_path.mkdir(parents=True, exist_ok=True)
        output = spec_path / "specification.md"

        lines = [
            "# Specification\n",
            "## Overview\n",
            f"{overview.strip()}\n",
            "\n## Requirements\n",
            f"{requirements.strip()}\n",
            "\n## Architecture\n",
            f"{architecture.strip()}\n",
            "\n## Interfaces\n",
            f"{interfaces.strip()}\n",
            "\n## Success Criteria\n",
            f"{success_criteria.strip()}\n",
        ]
        if open_questions.strip():
            lines += ["\n## Open Questions\n", f"{open_questions.strip()}\n"]

        content = "".join(lines)
        try:
            output.write_text(content, encoding="utf-8")
        except OSError as exc:
            return f"ERROR: Could not write specification.md: {exc}"

        return (
            f"specification.md written to {self.spec_dir}/specification.md ({len(content)} chars)."
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

    Together with ``task`` (SubAgentMiddleware) and ``eval``
    (CodeInterpreterMiddleware), these are the complete tool surface.
    No generic filesystem tools are exposed.

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
