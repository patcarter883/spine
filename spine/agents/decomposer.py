"""Structural slice decomposer — splits a spec or failed slice into
smaller FeatureSlice-shaped dicts via a single LLM call.

Three modes:

- ``PLAN``     — input is the raw specification markdown; output is the
                 initial wave of parallelizable feature slices.
- ``FALLBACK`` — input is one slice that the slice-implementer failed to
                 land plus its captured traceback; output is 2–3 strictly
                 smaller micro-slices, each addressing one aspect of the
                 failure. Used by the IMPLEMENT subgraph's
                 decompose-on-failure loop.
- ``PER_FILE`` — input is one multi-file slice; output is one single-file
                 sub-slice per ``target_files`` entry, emitted in dependency
                 order (a file appears only after the files it imports), with
                 a tailored per-file ``description`` and the parent's full
                 ``acceptance_criteria`` copied onto each. Used proactively by
                 the IMPLEMENT subgraph so each implementer touches one file.

The schema here is intentionally narrower than ``_FeatureSliceInput`` in
``plan_tools`` — we drop ``execution_requirements`` / ``dependencies`` /
``complexity`` because the IMPLEMENT dispatch loop only needs id, title,
description, target_files, and acceptance_criteria.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

try:  # openai is always present when the model is ChatOpenAI; guard for safety.
    from openai import LengthFinishReasonError as _LengthFinishReasonError
except Exception:  # pragma: no cover - openai missing → length salvage is a no-op

    class _LengthFinishReasonError(Exception):  # type: ignore[no-redef]
        """Fallback sentinel — never raised, so the salvage branch stays inert."""

from spine.agents.helpers import (
    ainvoke_structured_with_retry,
    bind_structured_output,
    cap_completion_tokens,
    resolve_chat_model,
    suppress_reasoning,
)
from spine.agents.prompt_format import Tag, hostage_layout, xml_block, xml_blocks

logger = logging.getLogger(__name__)


class EditHint(BaseModel):
    """A planner-provided pointer to ONE concrete edit the implementer makes.

    The targeting work — *where* and *what* — moves here, to the (stronger)
    decomposer, so the slice-implementer can be a thin model that just applies
    the edit. Anchors are by SYMBOL (drift-proof) wherever a named definition
    is involved, so the implementer can call ``read_edit_lint`` with
    ``ast_edit`` and never hunt for line numbers.
    """

    file: str = Field(description="Workspace-relative file to edit or create.")
    symbol: str = Field(
        default="",
        description=(
            "Qualified name of the definition to target (e.g. "
            "'SpineConfig.resolve_model', 'UIApi.set_phase_provider') when the "
            "edit changes or sits adjacent to a named function/method/class. "
            "Leave empty for new files or non-symbol edits (imports, config)."
        ),
    )
    mode: str = Field(
        default="",
        description=(
            "Suggested read_edit_lint mode: 'ast_edit' (replace/insert "
            "by symbol), 'patch' (whitespace-tolerant search/replace), or "
            "'full_replace' (new/small file). Empty = implementer chooses."
        ),
    )
    action: str = Field(
        default="",
        description=(
            "For ast_edit: 'replace' to overwrite the symbol, 'insert_after' "
            "or 'insert_before' to add code adjacent to it. Leave empty for "
            "patch/full_replace or when mode is unset."
        ),
    )
    intent: str = Field(
        description="Precise statement of the change to make at this anchor."
    )


class FeatureSliceSchema(BaseModel):
    """Minimal slice schema used by the structural decomposer."""

    id: str = Field(description="Unique slug, e.g. 'add-user-auth'.")
    title: str = Field(description="Human-readable title.")
    description: str = Field(description="One-paragraph statement of intent.")
    target_files: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(min_length=1)
    reference_symbols: list[str] = Field(
        default_factory=list,
        description=(
            "Qualified names of existing symbols this slice's code calls, "
            "extends, or mimics — so the implementer can read_symbol them for "
            "context instead of surveying files. Carry through any provided on "
            "the parent slice; add ones specific to this file."
        ),
    )
    edit_plan: list[EditHint] = Field(
        default_factory=list,
        description=(
            "Ordered, concrete edits that satisfy this slice — one entry per "
            "change site. Populate when you can name the target symbol or the "
            "exact change; this lets a lightweight implementer apply edits "
            "directly instead of re-discovering where to work. Omit entries "
            "you genuinely cannot anchor."
        ),
    )


class DecompositionResult(BaseModel):
    """Top-level structured output from the decomposer."""

    slices: list[FeatureSliceSchema] = Field(min_length=1)


_PLAN_PROMPT = (
    xml_block(
        Tag.ROLE,
        "You are a structural decomposer. Given a specification, break it "
        "down into parallelizable FeatureSlice objects.",
    )
    + "\n\n"
    + xml_block(
        Tag.CONSTRAINTS,
        "- Each slice must be self-contained: it can be implemented end-to-end "
        "without waiting on a sibling slice.\n"
        "- Slices in the same wave MUST NOT touch the same files. If two "
        "pieces of work modify the same file, merge them into a single slice.\n"
        "- Each slice MUST have at least one acceptance criterion that a "
        "verifier could check against the working tree.\n"
        "- Slice ids are lowercase slugs (e.g. 'add-token-refresh').\n"
        "- Populate `edit_plan` with the concrete edits each slice needs — one "
        "entry per change site. Anchor by `symbol` (qualified name, e.g. "
        "'ClassName.method') whenever the edit touches a named definition, and "
        "suggest a `mode` ('ast_edit' for symbol edits, 'patch' for "
        "snippet-level changes, 'full_replace' for new files). This lets a "
        "lightweight implementer apply edits directly. Omit only what you "
        "genuinely cannot anchor.",
    )
)

_FALLBACK_PROMPT = (
    xml_block(
        Tag.ROLE,
        "You are a structural decomposer operating in FALLBACK mode. A "
        "previous slice-implementer subagent failed to land the slice in the "
        "user message. Your job is to produce 2 to 3 micro-slices that are "
        "strictly smaller in scope than the original — each addressing one "
        "specific aspect of the failure.",
    )
    + "\n\n"
    + xml_block(
        Tag.CONSTRAINTS,
        "- Each micro-slice modifies a small subset of the original "
        "target_files (ideally one file).\n"
        "- Each micro-slice has tight, locally-checkable acceptance criteria.\n"
        "- Inherit the parent slice's id with a '-micro-N' suffix (you will "
        "be reminded of the parent id below).\n"
        "- Do NOT propose work outside the parent slice's scope; if the "
        "failure hints at a missing dependency, report it inside an "
        "acceptance criterion rather than creating a slice for it.",
    )
)

_PER_FILE_PROMPT = (
    xml_block(
        Tag.ROLE,
        "You are a structural decomposer operating in PER_FILE mode. You are "
        "given one feature slice that touches several files. Split it into "
        "single-file sub-slices — exactly one sub-slice per file in the "
        "slice's target_files — so each can be handed to an implementer that "
        "creates or modifies only that one file.",
    )
    + "\n\n"
    + xml_block(
        Tag.CONSTRAINTS,
        "- Emit EXACTLY one sub-slice per file in the parent's target_files, "
        "and each sub-slice's target_files MUST be a list of that single "
        "file path (copied verbatim from the parent).\n"
        "- ORDER MATTERS: emit files in dependency order — a file that "
        "imports from or builds on another file MUST appear AFTER it. Source "
        "modules before the tests that exercise them; base classes before "
        "subclasses.\n"
        "- Each sub-slice's description tailors the parent's intent to that "
        "ONE file: what to create/modify in it, and which already-built "
        "sibling files it may read for context.\n"
        "- Copy the parent's full acceptance_criteria onto every sub-slice "
        "verbatim — do NOT invent or drop criteria. Only the last file will "
        "be expected to satisfy slice-level tests.\n"
        "- Do NOT introduce files that are not in the parent's target_files, "
        "and do NOT merge two files into one sub-slice.\n"
        "- For each sub-slice, populate `edit_plan` with the concrete edits in "
        "that one file — anchor by `symbol` (qualified name) for edits to a "
        "named function/method/class and set `mode` to 'ast_edit', use 'patch' "
        "for snippet edits, 'full_replace' for a new file. This lets a "
        "lightweight implementer apply the edits without re-discovering them.\n"
        "- Sub-slice ids are placeholders; they will be reassigned by the "
        "caller, so any unique slug is fine.",
    )
)


_OVERFLOW_HINT = (
    "IMPORTANT — the parent failed because the implementer's context window "
    "OVERFLOWED, not because of a logic error. Re-attempting the same whole-file "
    "work will overflow again. Each micro-slice MUST therefore narrow the amount "
    "of code that has to be in context at once:\n"
    "- Scope each micro-slice to a SMALL region of the file (a single function, "
    "class, or contiguous block), and say so in its description.\n"
    "- Instruct the implementer (in the description) to read ONLY the relevant "
    "line range with offset/limit, NOT the whole file.\n"
    "- Prefer MORE, SMALLER micro-slices over fewer large ones."
)


async def run_decomposer(
    *,
    mode: Literal["PLAN", "FALLBACK", "PER_FILE"],
    spec_markdown: str | None = None,
    failed_slice: dict | None = None,
    error_traceback: str | None = None,
    source_slice: dict | None = None,
    overflow_hint: bool = False,
    config: RunnableConfig | None = None,
    session_id: str | None = None,
) -> list[dict]:
    """Run the structural decomposer and return a list of slice dicts.

    Args:
        mode: ``"PLAN"`` for top-level spec breakdown,
              ``"FALLBACK"`` for failure-driven micro-slicing,
              ``"PER_FILE"`` for proactive single-file sub-slicing.
        spec_markdown: Specification text (required when mode is PLAN).
        failed_slice: The slice dict that failed (required when mode is
            FALLBACK). Must include ``id``.
        error_traceback: Captured failure detail from the implementer
            (required when mode is FALLBACK).
        source_slice: The multi-file slice to split (required when mode is
            PER_FILE). Must include ``id`` and ≥2 ``target_files``.
        overflow_hint: When True (FALLBACK mode), tells the decomposer the
            parent failed on a context-window overflow rather than a logic
            error, so micro-slices are scoped to narrow file regions read with
            offset/limit instead of repeating the whole-file work that 400'd.
        config: LangGraph runtime config; carries per-phase model overrides.
        session_id: Work id for OpenRouter session grouping.

    Returns:
        A list of slice dicts ready to be appended to ``pending_slices``.
        For PER_FILE the list is ordered (head file first) and every dict's
        ``target_files`` holds exactly one path.
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
    elif mode == "PER_FILE":
        if not source_slice or not source_slice.get("id"):
            raise ValueError(
                "run_decomposer(mode='PER_FILE') requires source_slice with an 'id'"
            )
        if len([f for f in (source_slice.get("target_files") or []) if f]) < 2:
            raise ValueError(
                "run_decomposer(mode='PER_FILE') requires source_slice with ≥2 target_files"
            )
    else:
        raise ValueError(f"Unknown decomposer mode: {mode!r}")

    phase_path = f"implement/decomposer/{mode.lower()}"
    model = resolve_chat_model(config, session_id=session_id, phase=phase_path)
    # Clamp the completion reservation. DecompositionResult is small, but the
    # bare structured call would otherwise inherit the global
    # max_completion_tokens (30K) and ask a finite-window local server to
    # reserve a 30K generation slot — starving KV cache for the prompt and
    # OOM-crashing the backend (trace 019ed360). cap_completion_tokens writes
    # whichever underlying field is set (max_tokens vs its alias).
    from spine.config import SpineConfig

    _spine_cfg = SpineConfig.load()
    decompose_cap = _spine_cfg.decompose_max_completion_tokens
    if decompose_cap and decompose_cap > 0:
        model = cap_completion_tokens(model, decompose_cap)
    # Suppress the reasoning channel on local thinking models (e.g. Qwen3.6).
    # Without this, chain-of-thought consumes the entire decompose_cap before
    # any JSON is emitted, so the structured call dies with
    # LengthFinishReasonError *after* burning the full completion budget — a
    # multi-minute, fully-wasted call on a slow local backend that then drops
    # the slice (trace 019ed3dc: every fallback_decomposer span errored this
    # way). Mirrors the cap+suppress pattern in plan_do/researcher_supervisor.
    # No-op for OpenRouter / real OpenAI models.
    model = suppress_reasoning(model)
    structured = bind_structured_output(model, DecompositionResult)

    if mode == "PLAN":
        system_prompt = _PLAN_PROMPT
        human_content = hostage_layout(
            xml_blocks((Tag.SPECIFICATION, spec_markdown.strip())),
            "Return a DecompositionResult covering the specification above.",
        )
    elif mode == "PER_FILE":
        parent_id = source_slice["id"]
        parent_files = [f for f in (source_slice.get("target_files") or []) if f]
        slice_json = json.dumps(source_slice, indent=2, ensure_ascii=False, default=str)
        system_prompt = _PER_FILE_PROMPT
        # Inject a ground-truth symbol menu so the model never invents names.
        # Each target file's indexed symbols are listed; the decomposer must
        # anchor edit_plan entries and reference_symbols to names from this
        # list — any name not here is a new definition, not an existing anchor.
        known_syms_text = _build_known_symbols_block(
            _spine_cfg.checkpoint_path, parent_files
        )
        known_syms_pairs = (
            [(Tag.RETRIEVED_CODE, known_syms_text)] if known_syms_text else []
        )
        human_content = hostage_layout(
            xml_blocks(
                (Tag.OBJECTIVE, f"Parent slice id: {parent_id}"),
                (Tag.FINDINGS, f"```json\n{slice_json}\n```"),
                *known_syms_pairs,
            ),
            (
                f"Return a DecompositionResult with EXACTLY {len(parent_files)} "
                "single-file sub-slice(s) — one per parent target file — in "
                "dependency order."
            ),
        )
    else:
        parent_id = failed_slice["id"]
        # Slim the failed slice before embedding it: the bulky free-text result
        # fields (test_results / issues / _failure_traceback) are either the same
        # text already carried in error_traceback or noise the decomposer doesn't
        # need — dropping them keeps the prompt from doubling the traceback and
        # truncating the structured call mid-JSON (trace 019ed3dc).
        slim_slice = {
            k: v
            for k, v in failed_slice.items()
            if k not in ("test_results", "issues", "_failure_traceback")
        }
        slice_json = json.dumps(slim_slice, indent=2, ensure_ascii=False, default=str)
        # Bound the traceback, keeping the tail (most informative end).
        tb_cap = _spine_cfg.decompose_max_traceback_chars
        traceback_text = error_traceback.strip()
        if tb_cap and tb_cap > 0 and len(traceback_text) > tb_cap:
            traceback_text = "...[traceback truncated]...\n" + traceback_text[-tb_cap:]
        system_prompt = (
            _FALLBACK_PROMPT + "\n\n" + xml_block(Tag.CONSTRAINTS, _OVERFLOW_HINT)
            if overflow_hint
            else _FALLBACK_PROMPT
        )
        tail = (
            f"Return a DecompositionResult with 2-3 micro-slices whose ids "
            f"are '{parent_id}-micro-1', '{parent_id}-micro-2', and "
            f"optionally '{parent_id}-micro-3'."
        )
        if overflow_hint:
            tail += (
                " The parent OVERFLOWED the context window — make each "
                "micro-slice region-scoped and read-narrow per the constraints."
            )
        human_content = hostage_layout(
            xml_blocks(
                (Tag.OBJECTIVE, f"Parent slice id: {parent_id}"),
                (Tag.FINDINGS, f"```json\n{slice_json}\n```"),
                (Tag.ERRORS, f"```\n{traceback_text}\n```"),
            ),
            tail,
        )

    try:
        response: Any = await ainvoke_structured_with_retry(
            structured,
            [SystemMessage(content=system_prompt), HumanMessage(content=human_content)],
            label="decomposer",
        )
    except _LengthFinishReasonError:
        # The completion hit the length limit before valid JSON closed. For the
        # FALLBACK path this would otherwise drop a recoverable slice (trace
        # 019ed3dc), so salvage once with a hard-shrunk prompt: no traceback at
        # all (just the slice id + objective) so the small completion budget is
        # spent on micro-slice JSON, not echoing context. PLAN/PER_FILE keep the
        # original propagate-immediately behaviour.
        if mode != "FALLBACK":
            raise
        logger.warning(
            "Decomposer(FALLBACK) hit length limit for %r — retrying with a "
            "minimal prompt",
            parent_id,
        )
        salvage_content = hostage_layout(
            xml_blocks(
                (Tag.OBJECTIVE, f"Parent slice id: {parent_id}"),
                (
                    Tag.FINDINGS,
                    f"Parent slice {parent_id!r} failed to implement and must be "
                    "split into 2-3 smaller, region-scoped micro-slices.",
                ),
            ),
            tail,
        )
        response = await ainvoke_structured_with_retry(
            structured,
            [SystemMessage(content=system_prompt), HumanMessage(content=salvage_content)],
            label="decomposer-length-salvage",
        )

    if isinstance(response, DecompositionResult):
        parsed = response
    elif hasattr(response, "parsed") and isinstance(response.parsed, DecompositionResult):
        parsed = response.parsed
        response.parsed = None  # prevent Pydantic serialization warning
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
    elif mode == "PER_FILE":
        slices = _normalize_per_file_slices(source_slice, slices)

    _scrub_phantom_symbols(slices, _spine_cfg.checkpoint_path)

    logger.info(
        "Decomposer(%s) produced %d slice(s): %s",
        mode,
        len(slices),
        [s.get("id", "?") for s in slices],
    )
    return slices


def _build_known_symbols_block(db_path: str, files: list[str]) -> str:
    """Return a text block listing indexed symbols for each file in *files*.

    Used to inject a ground-truth symbol menu into the PER_FILE decomposer
    prompt so it never invents qualified names when anchoring edit_plan entries
    or reference_symbols. Returns '' when the local index has no data.
    """
    try:
        from spine.agents.tools.codebase_query import list_file_symbols
    except ImportError:
        return ""

    lines: list[str] = [
        "Symbols that ALREADY EXIST in the target files "
        "(use only these as edit_plan symbol= values or reference_symbols; "
        "anything not listed here is a NEW definition, not an existing anchor):"
    ]
    found_any = False
    for f in files:
        try:
            syms = list_file_symbols(db_path, f)
        except Exception:
            continue
        if syms:
            found_any = True
            # Cap at 60 names to avoid token waste on huge files.
            cap = 60
            suffix = f" … (+{len(syms) - cap} more)" if len(syms) > cap else ""
            lines.append(f"  {f}: {', '.join(syms[:cap])}{suffix}")
        else:
            lines.append(f"  {f}: (not yet indexed — file may be new)")
    return "\n".join(lines) if found_any else ""


def _scrub_phantom_symbols(slices: list[dict], db_path: str) -> None:
    """Remove hallucinated symbol names from *slices* in-place.

    For each slice:
    - edit_plan entries whose symbol is not in the local index have their
      symbol cleared (and mode demoted from ast_edit to '' so the implementer
      falls back to patch/full_replace instead of a guaranteed-failing ast_edit).
    - reference_symbols that don't exist in the local index are dropped so the
      implementer doesn't waste turns searching for them.

    No-op when db_path is falsy or the local index is unavailable.
    """
    if not db_path:
        return
    try:
        from spine.agents.tools.codebase_query import find_symbol
    except ImportError:
        return

    for sl in slices:
        slice_id = sl.get("id", "?")

        # Scrub edit_plan symbols.
        for hint in sl.get("edit_plan") or []:
            sym = hint.get("symbol", "")
            if not sym:
                continue
            try:
                exists = find_symbol(db_path, sym) is not None
            except Exception:
                continue
            if not exists:
                logger.warning(
                    "decomposer: clearing phantom edit_plan symbol %r in slice %r "
                    "(not found in codebase index — likely a new definition, not an anchor)",
                    sym, slice_id,
                )
                hint["symbol"] = ""
                if hint.get("mode") == "ast_edit":
                    hint["mode"] = ""

        # Scrub reference_symbols.
        refs = sl.get("reference_symbols") or []
        if not refs:
            continue
        good: list[str] = []
        for sym in refs:
            try:
                exists = find_symbol(db_path, sym) is not None
            except Exception:
                good.append(sym)
                continue
            if exists:
                good.append(sym)
            else:
                logger.warning(
                    "decomposer: removing phantom reference_symbol %r from slice %r "
                    "(not in codebase index — may need to be created first)",
                    sym, slice_id,
                )
        sl["reference_symbols"] = good


def _normalize_per_file_slices(
    source_slice: dict,
    raw_slices: list[dict],
) -> list[dict]:
    """Coerce PER_FILE decomposer output into ordered single-file sub-slices.

    Guarantees, regardless of what the model returned:
      - exactly one sub-slice per parent ``target_files`` entry,
      - each sub-slice's ``target_files`` is ``[that single path]``,
      - the parent's full ``acceptance_criteria`` is copied onto each,
      - deterministic ids ``"{parent_id}::{i}-{basename}"`` (1-based, index
        prefix avoids basename collisions across directories).

    The model's emitted order is honoured for files it covered; any parent
    file the model missed is appended at the end so coverage is never lost.
    """
    parent_id = source_slice["id"]
    parent_title = source_slice.get("title", parent_id)
    parent_criteria = list(source_slice.get("acceptance_criteria") or [])
    parent_refs = list(source_slice.get("reference_symbols") or [])
    parent_files = [f for f in (source_slice.get("target_files") or []) if f]

    # Map each model sub-slice to a parent file (first target_files entry that
    # belongs to the parent and is not yet claimed), preserving model order.
    remaining = list(parent_files)
    ordered: list[tuple[str, dict]] = []
    for sl in raw_slices:
        match = next(
            (f for f in (sl.get("target_files") or []) if f in remaining),
            None,
        )
        if match is None:
            continue
        remaining.remove(match)
        ordered.append((match, sl))
    # Append any parent files the model failed to cover, in parent order.
    for f in remaining:
        ordered.append((f, {}))

    normalized: list[dict] = []
    for i, (path, sl) in enumerate(ordered, start=1):
        description = (sl.get("description") or "").strip() or (
            f"Implement the portion of slice {parent_id!r} that lives in "
            f"{path}."
        )
        # Carry only the edit hints that target THIS sub-slice's file (the
        # model may emit a file-scoped plan or none).
        plan = [
            h for h in (sl.get("edit_plan") or [])
            if not h.get("file") or h.get("file") == path
        ]
        # Inherit the parent's reference symbols; merge any the model added for
        # this file (deduped, order-preserving).
        refs = list(parent_refs)
        for r in sl.get("reference_symbols") or []:
            if r not in refs:
                refs.append(r)
        normalized.append(
            {
                "id": f"{parent_id}::{i}-{os.path.basename(path)}",
                "title": f"{parent_title} — {path}",
                "description": description,
                "target_files": [path],
                "acceptance_criteria": list(parent_criteria),
                "reference_symbols": refs,
                "edit_plan": plan,
            }
        )
    return normalized


class _EnrichmentOutput(BaseModel):
    edit_plan: list[EditHint] = Field(default_factory=list)


_ENRICH_PROMPT = (
    xml_block(
        Tag.ROLE,
        "You are a slice enricher. Given one feature slice targeting a single "
        "file and the list of symbols that currently exist in that file, "
        "produce a concrete edit_plan: ordered, targeted edits the implementer "
        "applies directly without re-discovering where to work.",
    )
    + "\n\n"
    + xml_block(
        Tag.CONSTRAINTS,
        "- Each edit_plan entry MUST anchor by a symbol that is in the "
        "provided symbol list. Symbols being CREATED do not exist yet — "
        "they are not valid anchors.\n"
        "- To ADD new methods/functions to a class: use the last existing "
        "method of that class as symbol, set action='insert_after'.\n"
        "- To MODIFY an existing method: set symbol to its qualified name, "
        "action='replace'.\n"
        "- To add module-level code (imports, constants, top-level functions): "
        "set symbol to the nearest existing top-level definition, "
        "action='insert_before' or 'insert_after'.\n"
        "- For a completely new file: one entry, mode='full_replace', "
        "symbol='', action=''.\n"
        "- Do NOT anchor by symbols absent from the provided list.",
    )
)


async def enrich_slice(
    *,
    source_slice: dict,
    config: RunnableConfig | None = None,
    session_id: str | None = None,
) -> dict:
    """Populate edit_plan for a single-file slice from the codebase index.

    Runs the enricher model on *source_slice* to generate a concrete
    edit_plan grounded in the currently-indexed symbols for the target file.
    Returns the slice unchanged on error, empty index, or when edit_plan is
    already populated.
    """
    if source_slice.get("edit_plan"):
        return source_slice

    files = [f for f in (source_slice.get("target_files") or []) if f]
    if not files:
        return source_slice

    from spine.config import SpineConfig

    _spine_cfg = SpineConfig.load()
    known_block = _build_known_symbols_block(_spine_cfg.checkpoint_path, files)
    if not known_block:
        return source_slice

    model = resolve_chat_model(
        config, session_id=session_id, phase="implement/decomposer/enrich"
    )
    decompose_cap = _spine_cfg.decompose_max_completion_tokens
    if decompose_cap and decompose_cap > 0:
        model = cap_completion_tokens(model, decompose_cap)
    model = suppress_reasoning(model)
    structured = bind_structured_output(model, _EnrichmentOutput)

    slice_json = json.dumps(
        {
            k: source_slice.get(k)
            for k in (
                "id",
                "title",
                "description",
                "target_files",
                "acceptance_criteria",
            )
        },
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    human_content = hostage_layout(
        xml_blocks(
            (Tag.SPECIFICATION, slice_json),
            (Tag.FINDINGS, known_block),
        ),
        "Return an _EnrichmentOutput with a concrete edit_plan for this slice.",
    )

    try:
        response: Any = await ainvoke_structured_with_retry(
            structured,
            [SystemMessage(content=_ENRICH_PROMPT), HumanMessage(content=human_content)],
            label="enrich_slice",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "enrich_slice: failed for slice %r: %s — continuing without edit_plan",
            source_slice.get("id"),
            exc,
        )
        return source_slice

    if isinstance(response, _EnrichmentOutput):
        parsed = response
    elif hasattr(response, "parsed") and isinstance(response.parsed, _EnrichmentOutput):
        parsed = response.parsed
        response.parsed = None
    else:
        logger.warning(
            "enrich_slice: unexpected response type %r for slice %r",
            type(response).__name__,
            source_slice.get("id"),
        )
        return source_slice

    if not parsed.edit_plan:
        return source_slice

    enriched = dict(source_slice)
    enriched["edit_plan"] = [h.model_dump() for h in parsed.edit_plan]
    _scrub_phantom_symbols([enriched], _spine_cfg.checkpoint_path)

    logger.info(
        "enrich_slice: slice=%r populated %d edit_plan entry(ies)",
        source_slice.get("id"),
        len(enriched.get("edit_plan") or []),
    )
    return enriched
