"""Structural slice decomposer — splits a spec or failed slice into
smaller FeatureSlice-shaped dicts via a single LLM call.

Two modes:

- ``PLAN``     — input is the raw specification markdown; output is the
                 initial wave of parallelizable feature slices.
- ``FALLBACK`` — input is one slice that the slice-implementer failed to
                 land plus its captured traceback; output is 2–3 strictly
                 smaller micro-slices, each addressing one aspect of the
                 failure. Used by the IMPLEMENT subgraph's
                 decompose-on-failure loop.

The schema here is intentionally narrower than ``_FeatureSliceInput`` in
``plan_tools`` — we drop ``execution_requirements`` / ``dependencies`` /
``complexity`` because the IMPLEMENT dispatch loop only needs id, title,
description, target_files, and acceptance_criteria.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from spine.agents.helpers import resolve_model

logger = logging.getLogger(__name__)


class FeatureSliceSchema(BaseModel):
    """Minimal slice schema used by the structural decomposer."""

    id: str = Field(description="Unique slug, e.g. 'add-user-auth'.")
    title: str = Field(description="Human-readable title.")
    description: str = Field(description="One-paragraph statement of intent.")
    target_files: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(min_length=1)


class DecompositionResult(BaseModel):
    """Top-level structured output from the decomposer."""

    slices: list[FeatureSliceSchema] = Field(min_length=1)


_PLAN_PROMPT = """\
You are a structural decomposer. Given a specification, break it down
into parallelizable FeatureSlice objects.

Rules:
- Each slice must be self-contained: it can be implemented end-to-end
  without waiting on a sibling slice.
- Slices in the same wave MUST NOT touch the same files. If two pieces
  of work modify the same file, merge them into a single slice.
- Each slice MUST have at least one acceptance criterion that a
  verifier could check against the working tree.
- Slice ids are lowercase slugs (e.g. 'add-token-refresh').
"""

_FALLBACK_PROMPT = """\
You are a structural decomposer operating in FALLBACK mode. A previous
slice-implementer subagent failed to land the slice below. Your job is
to produce **2 to 3 micro-slices** that are strictly smaller in scope
than the original — each addressing one specific aspect of the failure.

Rules:
- Each micro-slice modifies a small subset of the original target_files
  (ideally one file).
- Each micro-slice has tight, locally-checkable acceptance criteria.
- Inherit the parent slice's id with a '-micro-N' suffix (you will be
  reminded of the parent id below).
- Do NOT propose work outside the parent slice's scope; if the failure
  hints at a missing dependency, report it inside an acceptance criterion
  rather than creating a slice for it.
"""


async def run_decomposer(
    *,
    mode: Literal["PLAN", "FALLBACK"],
    spec_markdown: str | None = None,
    failed_slice: dict | None = None,
    error_traceback: str | None = None,
    config: RunnableConfig | None = None,
    session_id: str | None = None,
) -> list[dict]:
    """Run the structural decomposer and return a list of slice dicts.

    Args:
        mode: ``"PLAN"`` for top-level spec breakdown,
              ``"FALLBACK"`` for failure-driven micro-slicing.
        spec_markdown: Specification text (required when mode is PLAN).
        failed_slice: The slice dict that failed (required when mode is
            FALLBACK). Must include ``id``.
        error_traceback: Captured failure detail from the implementer
            (required when mode is FALLBACK).
        config: LangGraph runtime config; carries per-phase model overrides.
        session_id: Work id for OpenRouter session grouping.

    Returns:
        A list of slice dicts ready to be appended to ``pending_slices``.
    """
    if mode == "PLAN":
        if not spec_markdown or not spec_markdown.strip():
            raise ValueError("run_decomposer(mode='PLAN') requires non-empty spec_markdown")
    elif mode == "FALLBACK":
        if not failed_slice or not failed_slice.get("id"):
            raise ValueError(
                "run_decomposer(mode='FALLBACK') requires failed_slice with an 'id'"
            )
        if not error_traceback:
            raise ValueError("run_decomposer(mode='FALLBACK') requires error_traceback")
    else:
        raise ValueError(f"Unknown decomposer mode: {mode!r}")

    phase_path = f"implement/decomposer/{mode.lower()}"
    model = resolve_model(config, session_id=session_id, phase=phase_path)
    if isinstance(model, str):
        from langchain.chat_models import init_chat_model

        model = init_chat_model(model)

    structured = model.with_structured_output(DecompositionResult)

    if mode == "PLAN":
        system_prompt = _PLAN_PROMPT
        human_content = (
            "## Specification\n"
            f"{spec_markdown.strip()}\n\n"
            "Return a DecompositionResult covering the work above."
        )
    else:
        parent_id = failed_slice["id"]
        slice_json = json.dumps(failed_slice, indent=2, ensure_ascii=False, default=str)
        system_prompt = _FALLBACK_PROMPT
        human_content = (
            f"## Parent slice id\n{parent_id}\n\n"
            f"## Failed slice (full JSON)\n```json\n{slice_json}\n```\n\n"
            f"## Failure detail\n```\n{error_traceback.strip()}\n```\n\n"
            "Return a DecompositionResult with 2-3 micro-slices whose ids "
            f"are '{parent_id}-micro-1', '{parent_id}-micro-2', and "
            f"optionally '{parent_id}-micro-3'."
        )

    response: Any = await structured.ainvoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=human_content)]
    )

    if isinstance(response, DecompositionResult):
        parsed = response
    elif hasattr(response, "parsed") and isinstance(response.parsed, DecompositionResult):
        parsed = response.parsed
    else:
        raise ValueError(
            f"Decomposer returned unexpected structured-output type: {type(response).__name__}"
        )

    slices = [s.model_dump() for s in parsed.slices]

    if mode == "FALLBACK":
        parent_id = failed_slice["id"]
        for i, sl in enumerate(slices, start=1):
            expected = f"{parent_id}-micro-{i}"
            if sl.get("id") != expected:
                sl["id"] = expected

    logger.info(
        "Decomposer(%s) produced %d slice(s): %s",
        mode,
        len(slices),
        [s.get("id", "?") for s in slices],
    )
    return slices
