"""Custom tools for the onboarding artifact-synthesis agent.

Replaces generic filesystem tools with two purpose-built tools that enforce
the synthesis agent's role: read the analysed :class:`RepoManifest`, then
write the four onboarding markdown documents. Nothing else.

Tools:
- ``read_repo_manifest`` — returns the persisted ``repo_manifest.json`` as a
  JSON string in one no-argument call. The agent uses this to understand the
  repository's tech stack, module boundaries, dependency chains, and extracted
  conventions without reading any source files at runtime.
- ``write_onboarding_doc`` — the ONLY write surface. Accepts a fixed document
  name (one of the four onboarding artifacts) and its markdown ``content``;
  writes ``<NAME>.md`` idempotently to the onboarding artifact directory via
  the :class:`ArtifactStore` (phase ``"onboarding"``). Re-running cleanly
  overwrites without filesystem errors.

This mirrors the constrained-tool pattern used by
:mod:`spine.agents.plan_tools` / :mod:`spine.agents.specify_tools`: BaseTool
subclasses with pydantic ``ArgsSchema`` definitions, build-time injection of
paths, and a single write tool so the model cannot escape the curated surface.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal, Optional

from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from pydantic import BaseModel, Field

from spine.persistence.artifacts import ArtifactStore

logger = logging.getLogger(__name__)

# The four onboarding artifacts. The canonical document keys (what the agent
# passes to write_onboarding_doc) map 1:1 to the rendered ``<KEY>.md`` files.
ONBOARDING_DOC_NAMES: tuple[str, ...] = (
    "PROJECT_DEFINITION",
    "CODING_GUIDELINES",
    "ARCHITECTURE_MAP",
    "SPINE_ASSISTANCE_REQUIREMENTS",
)

# Phase under which onboarding artifacts are persisted in the ArtifactStore.
ONBOARDING_PHASE = "onboarding"


# ── read_repo_manifest ─────────────────────────────────────────────────────


class _ReadRepoManifestInput(BaseModel):
    """No inputs — the manifest path is fixed at build time."""


class ReadRepoManifestTool(BaseTool):
    """Load the analysed :class:`RepoManifest` as JSON in one call.

    Returns the contents of ``repo_manifest.json`` (the JSON round-trip of the
    manifest produced by the analysis stage). The synthesis agent calls this
    FIRST to get the repository's tech stack, core domains, module boundaries,
    dependency chains, and extracted patterns — everything it needs to write
    the four onboarding documents without touching any source files.

    No arguments.
    """

    name: str = "read_repo_manifest"
    description: str = (
        "Load the analysed repository manifest (repo_manifest.json) as JSON. "
        "No arguments. Returns the tech stack, core domains, module boundaries, "
        "dependency chains, and extracted coding patterns. Call this FIRST — it "
        "gives you everything you need to write the onboarding documents. Do "
        "NOT read source files; the manifest already summarises the codebase."
    )
    args_schema: Optional[ArgsSchema] = _ReadRepoManifestInput

    # Injected at build time.
    workspace_root: str = ""
    manifest_dir: str = ""
    manifest_name: str = "repo_manifest.json"

    def _manifest_path(self) -> Path:
        # ``manifest_dir`` is workspace-relative (e.g.
        # ".spine/artifacts/<work_id>/onboarding"); resolve against the root.
        base = Path(self.workspace_root) if self.workspace_root else Path(".")
        return base / self.manifest_dir / self.manifest_name

    def _run(self, **kwargs: Any) -> str:  # noqa: ARG002
        path = self._manifest_path()
        if not path.exists():
            return json.dumps(
                {
                    "error": "manifest_not_found",
                    "message": (
                        f"repo_manifest.json was not found at {self.manifest_dir}/"
                        f"{self.manifest_name}. The analysis stage must run before "
                        "synthesis."
                    ),
                },
                ensure_ascii=False,
            )
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not read repo manifest at %s: %s", path, exc)
            return json.dumps(
                {"error": "manifest_read_error", "message": str(exc)},
                ensure_ascii=False,
            )

    async def _arun(self, **kwargs: Any) -> str:
        return self._run(**kwargs)


# ── write_onboarding_doc ────────────────────────────────────────────────────


class _WriteOnboardingDocInput(BaseModel):
    """Schema for a single onboarding-document write."""

    doc: Literal[
        "PROJECT_DEFINITION",
        "CODING_GUIDELINES",
        "ARCHITECTURE_MAP",
        "SPINE_ASSISTANCE_REQUIREMENTS",
    ] = Field(
        description=(
            "Which onboarding document to write. MUST be exactly one of: "
            "PROJECT_DEFINITION, CODING_GUIDELINES, ARCHITECTURE_MAP, "
            "SPINE_ASSISTANCE_REQUIREMENTS. The tool writes the corresponding "
            "<NAME>.md file."
        )
    )
    content: str = Field(
        description=(
            "The full markdown body of the document. Author the complete "
            "markdown — the tool writes it verbatim to <NAME>.md."
        ),
        min_length=1,
    )


class WriteOnboardingDocTool(BaseTool):
    """Write one onboarding markdown document idempotently.

    This is the ONLY write surface for the synthesis agent. Given a fixed
    document name and its markdown content, writes ``<NAME>.md`` to the
    onboarding artifact directory via :meth:`ArtifactStore.save_artifact`
    (phase ``"onboarding"``, ``overwrite_shorter=True`` so re-runs cleanly
    overwrite). Rejects any document name outside the fixed set of four.
    """

    name: str = "write_onboarding_doc"
    description: str = (
        "Write one onboarding document (PROJECT_DEFINITION, CODING_GUIDELINES, "
        "ARCHITECTURE_MAP, or SPINE_ASSISTANCE_REQUIREMENTS) as <NAME>.md. "
        "Provide the document name and its full markdown content — the tool "
        "writes the file for you. Do not call write_file. Call once per "
        "document; all four must be written."
    )
    args_schema: Optional[ArgsSchema] = _WriteOnboardingDocInput

    # Injected at build time.
    workspace_root: str = ""
    work_id: str = ""
    out_dir: str = ""

    def _run(self, doc: str, content: str) -> str:
        if doc not in ONBOARDING_DOC_NAMES:
            return (
                f"VALIDATION_ERROR: unknown document name {doc!r}. "
                f"Must be one of: {', '.join(ONBOARDING_DOC_NAMES)}."
            )
        if not content or not content.strip():
            return (
                f"VALIDATION_ERROR: empty content for {doc}. Provide the full "
                "markdown body."
            )

        filename = f"{doc}.md"
        # ``out_dir`` is the base path passed to ArtifactStore — the store
        # writes to ``out_dir/{work_id}/{phase}/{filename}``. Using
        # overwrite_shorter=True makes re-runs idempotent: a re-synthesised
        # (possibly shorter) document still replaces the prior file cleanly,
        # never raising FileExistsError.
        store = ArtifactStore(base_path=self.out_dir)
        try:
            path = store.save_artifact(
                work_id=self.work_id,
                phase=ONBOARDING_PHASE,
                name=filename,
                content=content,
                overwrite_shorter=True,
            )
        except OSError as exc:
            return f"ERROR: Could not write {filename}: {exc}"

        return f"{filename} ({len(content)} chars) written to {path}."

    async def _arun(self, doc: str, content: str) -> str:
        return self._run(doc=doc, content=content)


# ── Factory ─────────────────────────────────────────────────────────────────


def build_synthesis_tools(
    workspace_root: str,
    work_id: str,
    manifest_dir: str,
    out_dir: str,
) -> list[BaseTool]:
    """Build the custom tool set for the onboarding synthesis agent.

    Returns two tools:
    - ``read_repo_manifest``: loads ``repo_manifest.json`` in one call.
    - ``write_onboarding_doc``: the only write surface, writes the four
      ``<NAME>.md`` documents idempotently.

    These replace all generic filesystem tools — pair with
    ``build_phase_agent(..., extra_tools=..., skip_filesystem_middleware=True)``
    so the model's only write path is :class:`WriteOnboardingDocTool`.

    Args:
        workspace_root: Absolute path to the project workspace root.
        work_id: The current onboarding work item ID.
        manifest_dir: Workspace-relative directory holding ``repo_manifest.json``
            (e.g. ``.spine/artifacts/<work_id>/onboarding``).
        out_dir: Base path for the :class:`ArtifactStore` that receives the
            written documents (e.g. the absolute path to ``.spine/artifacts``).

    Returns:
        List of two :class:`BaseTool` instances.
    """
    return [
        ReadRepoManifestTool(
            workspace_root=workspace_root,
            manifest_dir=manifest_dir,
        ),
        WriteOnboardingDocTool(
            workspace_root=workspace_root,
            work_id=work_id,
            out_dir=out_dir,
        ),
    ]
