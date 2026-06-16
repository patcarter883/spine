"""Onboarding artifact synthesis — thin back-compat shim over the graph.

Historically this module built a single Deep Agent, handed it the **entire
manifest twice** (~27k tokens for spine, on a 60k window), and asked it to
author all four onboarding documents in one agent lifetime. Token-budget
forensics on the first onboarding run showed that framing was unsafe at scale
(design Revision 2, §0).

The orchestration now lives in :mod:`spine.work.onboarding.synthesis_nodes` as
a two-tier *documentation manager → section-worker* hierarchy where **no LLM
ever receives the whole manifest** — a compact index drives planning and
bounded fragments drive writing. :func:`synthesize_artifacts` is retained only
as a thin shim that builds and invokes that synthesis graph and returns the
same ``{doc_name: path}`` mapping, preserving the
``RuntimeError``-on-missing-documents semantics existing callers rely on.
"""

from __future__ import annotations

import logging
from typing import Any

from spine.config import SpineConfig
from spine.models.state import WorkflowState
from spine.work.onboarding.manifest import RepoManifest
from spine.work.onboarding.synthesis_nodes import build_synthesis_graph

logger = logging.getLogger(__name__)


def _coerce_state(
    state: WorkflowState | dict[str, Any] | None,
    workspace_root: str,
    work_id: str,
) -> dict[str, Any]:
    """Return a minimal state carrying at least ``workspace_root`` + ``work_id``.

    Retained from the legacy driver: when the caller passes no state (the
    engine dispatch path), we synthesise a minimal one so model resolution and
    artifact materialisation still work.
    """
    base: dict[str, Any] = dict(state) if state else {}
    base.setdefault("workspace_root", workspace_root)
    base.setdefault("work_id", work_id)
    return base


async def synthesize_artifacts(
    manifest: RepoManifest,
    workspace_root: str,
    work_id: str,
    config: SpineConfig,
    state: WorkflowState | dict[str, Any] | None = None,
) -> dict[str, str]:
    """Synthesise the four onboarding markdown documents from a manifest.

    Builds the synthesis hierarchy graph
    (:func:`spine.work.onboarding.synthesis_nodes.build_synthesis_graph`) with
    the manifest in the initial state, runs Tier A (plan) → Tier B (per-section
    fan-out) → Tier C (deterministic assembly) → aggregate, and returns the
    written documents.

    Args:
        manifest: The analysed (or greenfield-seeded) repository manifest.
        workspace_root: Absolute path to the project workspace root.
        work_id: The onboarding work item ID.
        config: Loaded :class:`SpineConfig` (threaded to the graph for model
            resolution + the section token cap).
        state: Optional workflow state. A minimal one carrying
            ``workspace_root`` + ``work_id`` is synthesised when omitted.

    Returns:
        Mapping of ``{doc_name: absolute_path}`` for each of the four written
        documents (e.g. ``"PROJECT_DEFINITION"`` → its ``.md`` path).

    Raises:
        RetryableSynthesisError: (a ``RuntimeError`` subclass) if one or more
            sections failed *transiently* — the model endpoint was unreachable.
            The sections that completed are already written to disk; a re-run
            fills only the gaps. Callers that cannot distinguish it still see a
            ``RuntimeError`` and treat the run as failed.
        RuntimeError: if any of the four ``<NAME>.md`` is missing afterward, or a
            document carries no synthesised content (placeholder-only). A section
            that failed because the model could not produce usable *content* is
            tolerated (the section is omitted) as long as its document still
            carries real content from other sections.
    """
    _coerce_state(state, workspace_root, work_id)

    graph = build_synthesis_graph()
    compiled = graph.compile()

    initial_state: dict[str, Any] = {
        "work_id": work_id,
        "workspace_root": workspace_root,
        "mode": manifest.mode,
        "tech_stack": list(manifest.tech_stack),
        "manifest": manifest.to_dict(),
    }

    runnable_config: dict[str, Any] = {"configurable": {"spine_config": config}}

    final_state = await compiled.ainvoke(initial_state, config=runnable_config)
    written: dict[str, str] = dict(final_state.get("written", {}) or {})

    logger.info(
        "[%s] synthesize_artifacts: wrote %d onboarding documents",
        work_id,
        len(written),
    )
    return written
