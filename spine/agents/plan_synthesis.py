"""Decomposed PLAN synthesis — manager skeleton → per-slice workers → assemble.

The monolithic path (``build_plan_synthesizer`` + a single forced
``write_structured_plan`` call) asks a local 30B model to emit the ENTIRE plan —
every slice × every field — in one nested structured call. When any part is
malformed the force-tool middleware re-generates the whole plan, which spins
(trace 019edd7c: 24 calls / 1.77M tokens).

This module decomposes it exactly like the onboarding doc synthesis
(``spine/work/onboarding/synthesis_nodes.py``): *no single LLM call produces the
whole plan*.

- **Tier A — manager** (:func:`_run_manager`): ONE structured call that emits
  only the SKELETON — architecture/testing prose + per-slice STUBS (id, title,
  target_files, dependencies, reference_symbols, a one-line summary). Small and
  easy to get valid; this is where cross-slice structure is decided.
- **Tier B — slice workers** (:func:`_run_slice_worker`): ONE structured call
  PER slice, run concurrently, each filling only that slice's
  ``execution_requirements`` + ``acceptance_criteria``. A bad slice retries (or
  degrades) alone — never the whole plan.
- **Assembly** (:func:`synthesize_plan`): deterministic — stitch stubs+details
  into feature_slices and hand them to the existing ``StructuredWritePlanTool``
  for validation, same-file merge, and plan.md/plan.json rendering.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from spine.agents.helpers import (
    ainvoke_structured_with_retry,
    bind_structured_output,
    cap_completion_tokens,
    coerce_structured_output,
    resolve_chat_model,
    suppress_reasoning,
)
from spine.agents.exploration_agents import build_findings_ledger
from spine.agents.prompt_format import Tag, hostage_layout, xml_blocks
from spine.models.enums import PhaseName

logger = logging.getLogger(__name__)

# Per-call completion caps. The skeleton is small (stubs only); each slice
# detail is a paragraph + a few criteria. Keep them tight so a local model can't
# burn the global 30K window on one call (LengthFinishReasonError).
_MANAGER_MAX_COMPLETION_TOKENS = 4096
_SLICE_MAX_COMPLETION_TOKENS = 2048
_MAX_SLICES = 12
_RESEARCH_CHARS = 8000


# ── Schemas ───────────────────────────────────────────────────────────────────
class SliceStub(BaseModel):
    """The skeleton of one feature slice — no implementation detail yet."""

    id: str = Field(description="Unique lowercase slug, e.g. 'add-embedding-ui'.")
    title: str = Field(description="Short human-readable title.")
    target_files: list[str] = Field(
        default_factory=list, description="Files this slice creates or modifies."
    )
    dependencies: list[str] = Field(
        default_factory=list, description="ids of slices that must complete first."
    )
    reference_symbols: list[str] = Field(
        default_factory=list,
        description=(
            "Existing symbols this slice's code calls/extends/mimics (qualified "
            "names, e.g. 'UIApi.update_mcp_server'), taken from the research."
        ),
    )
    complexity: str = Field(default="medium", description="small | medium | large.")
    summary: str = Field(
        description="One sentence: what this slice changes and why (guides the worker)."
    )


class PlanSkeleton(BaseModel):
    """Tier-A output: plan prose + slice stubs (no per-slice detail)."""

    architecture_overview: str = Field(default="")
    technology_choices: list[str] = Field(default_factory=list)
    testing_strategy: str = Field(default="")
    risks: list[str] = Field(default_factory=list)
    slices: list[SliceStub] = Field(min_length=1)


class SliceDetail(BaseModel):
    """Tier-B output: one slice's implementation detail."""

    execution_requirements: str = Field(
        description="Detailed, step-by-step instructions to implement THIS slice."
    )
    acceptance_criteria: list[str] = Field(
        min_length=1,
        description="Measurable checks that prove this slice is complete.",
    )


# ── Inputs ────────────────────────────────────────────────────────────────────
def _research_text(state: dict[str, Any]) -> str:
    """Compact the retrieved research context into a bounded grounding blob."""
    chunks: list[str] = []
    for item in state.get("retrieved_context") or []:
        if not isinstance(item, dict):
            continue
        sym = item.get("symbol_name") or item.get("qualified_name") or ""
        path = item.get("file_path") or ""
        summ = item.get("enriched_summary") or item.get("summary") or ""
        if sym or summ:
            chunks.append(f"- {sym} ({path}): {str(summ)[:240]}")
    text = "\n".join(chunks)
    return text[:_RESEARCH_CHARS]


def _phase(sub: str) -> str:
    return f"{PhaseName.PLAN.value}/{sub}"


# ── Tier A: manager ─────────────────────────────────────────────────────────
_MANAGER_ROLE = (
    "You are the PLAN manager. From the specification and codebase research, "
    "produce only the SKELETON of the technical plan: short architecture and "
    "testing prose, and a list of small, single-purpose feature-slice STUBS. "
    "Do NOT write per-slice execution steps or acceptance criteria — separate "
    "workers fill those in. Keep it small and structurally sound."
)

_MANAGER_RULES = (
    "- Each stub: a unique lowercase-slug id, a title, the target_files it will "
    "touch, dependencies (ids of prerequisite stubs), reference_symbols (existing "
    "symbols its code will call/extend — take these from the research), a "
    "complexity, and a one-sentence summary.\n"
    "- Prefer FEWER, cohesive slices. If two pieces of work touch the same "
    "file, put them in ONE slice (they cannot run in parallel).\n"
    "- reference_symbols must be real qualified names that appear in the "
    "research — do not invent them.\n"
    "- Output the prose + stubs only. No execution_requirements, no "
    "acceptance_criteria."
)


async def _run_manager(
    spec_md: str,
    research: str,
    config: RunnableConfig | None,
    work_id: str,
    feedback: str = "",
    ledger: str = "",
) -> PlanSkeleton:
    """One small structured call → the plan skeleton (stubs)."""
    model = resolve_chat_model(config, session_id=work_id, phase=_phase("manager"))
    model = suppress_reasoning(cap_completion_tokens(model, _MANAGER_MAX_COMPLETION_TOKENS))
    structured = bind_structured_output(model, PlanSkeleton)
    blocks = [
        (Tag.SPECIFICATION, spec_md.strip()),
        # Durable file→role map (A2): full fidelity, never truncated by the
        # _research_text cap that bounds the verbose findings below it.
        (Tag.CODEBASE_LEDGER, ledger),
        (Tag.FINDINGS, research or "(no research context)"),
    ]
    if feedback.strip():
        blocks.append((Tag.CRITIC_FEEDBACK, "Address this prior review feedback:\n" + feedback.strip()))
    human = hostage_layout(
        xml_blocks(*blocks),
        "Return a PlanSkeleton: architecture/testing prose + feature-slice stubs.",
    )
    response = await ainvoke_structured_with_retry(
        structured,
        [SystemMessage(content=f"{_MANAGER_ROLE}\n\n{_MANAGER_RULES}"), HumanMessage(content=human)],
        label="plan-manager",
    )
    skeleton = coerce_structured_output(response, PlanSkeleton)
    if skeleton is None or not skeleton.slices:
        raise ValueError("plan manager returned no usable skeleton")
    skeleton.slices = skeleton.slices[:_MAX_SLICES]
    logger.info(
        "[%s] plan manager: %d slice stub(s): %s",
        work_id, len(skeleton.slices), [s.id for s in skeleton.slices],
    )
    return skeleton


# ── Tier B: per-slice worker ────────────────────────────────────────────────
_WORKER_ROLE = (
    "You implement the detail of ONE feature slice of a technical plan. You are "
    "given the slice stub (what it does, which files, which existing symbols it "
    "builds on), the OTHER slices (handled by separate workers), and the "
    "specification. Produce ONLY this slice's execution_requirements (precise "
    "step-by-step instructions) and acceptance_criteria (measurable checks).\n"
    "CRITICAL SCOPE RULE: describe ONLY changes to THIS slice's target_files. "
    "The other slices listed own their own files — do NOT re-describe their "
    "work, and do NOT implement anything outside your target_files. If your "
    "slice depends on a sibling slice's output, reference it by id; don't "
    "restate its implementation."
)


def _siblings_block(stub: SliceStub, all_stubs: list[SliceStub]) -> str:
    others = [s for s in all_stubs if s.id != stub.id]
    if not others:
        return "(none)"
    return "\n".join(
        f"- {s.id}: {s.title} — owns {', '.join(s.target_files) or '?'} (do NOT describe this slice's work)"
        for s in others
    )


async def _run_slice_worker(
    stub: SliceStub,
    all_stubs: list[SliceStub],
    spec_md: str,
    research: str,
    config: RunnableConfig | None,
    work_id: str,
    ledger: str = "",
) -> SliceDetail:
    """One small structured call → this slice's execution detail.

    Falls back to a minimal-but-valid detail derived from the stub if the model
    fails, so one bad slice never fails the whole plan (mirrors the onboarding
    section worker's omit-on-error).
    """
    model = resolve_chat_model(config, session_id=work_id, phase=_phase("slice-worker"))
    model = suppress_reasoning(cap_completion_tokens(model, _SLICE_MAX_COMPLETION_TOKENS))
    structured = bind_structured_output(model, SliceDetail)
    stub_json = stub.model_dump_json(indent=2)
    human = hostage_layout(
        xml_blocks(
            (Tag.OBJECTIVE, f"YOUR slice (detail ONLY this one):\n```json\n{stub_json}\n```"),
            (Tag.CONSTRAINTS, "Other slices — owned by separate workers, do NOT describe their work:\n"
             + _siblings_block(stub, all_stubs)),
            (Tag.SPECIFICATION, spec_md.strip()[:6000]),
            (Tag.CODEBASE_LEDGER, ledger),
            (Tag.FINDINGS, research or "(no research context)"),
        ),
        f"Return a SliceDetail for slice '{stub.id}': execution_requirements + "
        f"acceptance_criteria for ONLY the changes to {', '.join(stub.target_files) or 'its target_files'}. "
        "Do not describe work that belongs to the other slices listed.",
    )
    try:
        response = await ainvoke_structured_with_retry(
            structured,
            [SystemMessage(content=_WORKER_ROLE), HumanMessage(content=human)],
            label=f"plan-slice-worker:{stub.id}",
        )
        detail = coerce_structured_output(response, SliceDetail)
        if detail and detail.execution_requirements.strip() and detail.acceptance_criteria:
            return detail
        logger.warning("[%s] slice worker %r returned empty detail — degrading", work_id, stub.id)
    except Exception as exc:  # noqa: BLE001 — isolate the slice, don't fail the plan
        logger.warning("[%s] slice worker %r failed (%s) — degrading", work_id, stub.id, exc)
    return SliceDetail(
        execution_requirements=(stub.summary or f"Implement slice {stub.id}."),
        acceptance_criteria=[f"{stub.title} is implemented in {', '.join(stub.target_files) or 'the target files'}."],
    )


# ── Assembly ─────────────────────────────────────────────────────────────────
def _assemble_feature_slices(
    skeleton: PlanSkeleton, details: list[SliceDetail]
) -> list[dict[str, Any]]:
    slices: list[dict[str, Any]] = []
    for stub, detail in zip(skeleton.slices, details):
        slices.append(
            {
                "id": stub.id,
                "title": stub.title,
                "target_files": list(stub.target_files),
                "execution_requirements": detail.execution_requirements,
                "reference_symbols": list(stub.reference_symbols),
                "dependencies": list(stub.dependencies),
                "acceptance_criteria": list(detail.acceptance_criteria),
                "complexity": stub.complexity or "medium",
            }
        )
    return slices


async def synthesize_plan(
    state: dict[str, Any],
    config: Optional[RunnableConfig],
    spec_md: str,
    workspace_root: str,
    plan_dir: str,
    feedback: str = "",
) -> str:
    """Run the decomposed synthesis and write plan.md/plan.json.

    Returns the StructuredWritePlanTool result string (``VALIDATION_ERROR``/
    ``ERROR`` prefix on failure, success message otherwise) — the same contract
    the monolithic path produced, so the caller is unchanged.
    """
    work_id = state.get("work_id", "unknown")
    research = _research_text(state)

    # Durable file→role map (A2). The decomposed manager/workers otherwise see
    # only ``retrieved_context`` (vector recall) via ``_research_text`` — the
    # exploration ``findings`` never reach them. The ledger carries that
    # structural map in at full fidelity, exempt from the 8000-char research
    # cap, so the skeleton's target_files/reference_symbols are grounded in
    # what exploration actually found.
    ledger = build_findings_ledger(
        state.get("findings") or [],
        prior_findings=state.get("prior_phase_findings"),
    )

    skeleton = await _run_manager(
        spec_md, research, config, work_id, feedback=feedback, ledger=ledger,
    )
    details = await asyncio.gather(
        *(
            _run_slice_worker(
                stub, skeleton.slices, spec_md, research, config, work_id,
                ledger=ledger,
            )
            for stub in skeleton.slices
        )
    )
    feature_slices = _assemble_feature_slices(skeleton, details)

    # Reuse the existing tool for validation + same-file merge + rendering, so
    # the on-disk format is identical to the monolithic path.
    from spine.agents.plan_tools import StructuredWritePlanTool

    tool = StructuredWritePlanTool(workspace_root=workspace_root, plan_dir=plan_dir)
    result = tool._run(
        architecture_overview=skeleton.architecture_overview,
        feature_slices=feature_slices,
        testing_strategy=skeleton.testing_strategy,
        technology_choices=skeleton.technology_choices,
        risks=skeleton.risks,
    )
    logger.info("[%s] decomposed plan synthesis: %s", work_id, result[:120])
    return result
