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
from pydantic import BaseModel, Field, create_model

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
        description=(
            "Precise statement of the change at this anchor — naming exactly "
            "ONE symbol and its full signature + behaviour. One change site per "
            "entry; never bundle multiple new methods into one intent."
        )
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
        "entry per change site, and ONE entry per new method/function (never a "
        "single 'add all the methods' umbrella entry — a bundled entry forces "
        "the implementer to survey the whole file). Anchor by `symbol` "
        "(qualified name, e.g. 'ClassName.method') whenever the edit touches a "
        "named definition, and suggest a `mode` ('ast_edit' for symbol edits, "
        "'patch' for snippet-level changes, 'full_replace' for new files). This "
        "lets a lightweight implementer apply edits directly. Omit only what "
        "you genuinely cannot anchor.",
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
        "acceptance criterion rather than creating a slice for it.\n"
        "- If you populate a micro-slice's edit_plan: a symbol that does not "
        "exist yet is being CREATED, not edited — never anchor a NEW method "
        "with action='replace' on a DIFFERENT, already-existing method just "
        "because it's a convenient sibling (that deletes the existing method "
        "and replaces it with the new one — 019f1c10: this destroyed "
        "UIApi.add_llm_provider while adding UIApi.add_embedding_provider). "
        "To ADD a new method, anchor on the last existing method of that "
        "class with action='insert_after'. Only use action='replace' when "
        "the anchor symbol IS the exact thing you are rewriting.",
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
        "that one file — one entry per change site, and ONE entry per new "
        "method/function (never a single 'add all the methods' umbrella entry). "
        "Anchor by `symbol` (qualified name) for edits to a "
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


def _bind_capped(base_model: Any, schema: type, cap: int) -> Any:
    """Cap completion + suppress reasoning + bind structured output for *schema*.

    ``base_model`` is the uncapped resolved model; a fresh capped copy is made
    each call so the escalation retry can re-bind at a larger cap.
    """
    m = cap_completion_tokens(base_model, cap) if cap and cap > 0 else base_model
    return bind_structured_output(suppress_reasoning(m), schema)


async def _ainvoke_structured_escalating(
    base_model: Any,
    schema: type,
    messages: list,
    *,
    label: str,
    base_cap: int,
    window: int = 0,
    max_escalations: int = 1,
) -> Any:
    """Invoke a structured model, doubling the completion cap on length-truncation.

    Reasoning-heavy local models (North-Mini-Code via llama.cpp, where
    ``suppress_reasoning``'s vLLM knobs are ignored) burn the whole completion
    budget on chain-of-thought before the JSON closes, raising
    ``LengthFinishReasonError`` — which otherwise silently drops the slice's
    edit_plan. Doubling the cap (bounded by ``max_escalations`` and the model's
    context window) gives the JSON room to land. No-op-ish for cloud models:
    ``base_cap<=0`` means a single invoke with no escalation.
    """
    cap = base_cap if (base_cap and base_cap > 0) else 0
    attempt = 0
    while True:
        try:
            return await ainvoke_structured_with_retry(
                _bind_capped(base_model, schema, cap), messages, label=label
            )
        except _LengthFinishReasonError:
            nxt = cap * 2
            # Stop if uncapped, out of escalations, or the doubled cap would
            # crowd the window (leave ~2K headroom for the prompt).
            if not cap or attempt >= max_escalations or (window and nxt + 2048 >= window):
                raise
            attempt += 1
            logger.warning(
                "%s: length-truncated at cap=%d — escalating to %d", label, cap, nxt
            )
            cap = nxt


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
    # ``model`` stays UNCAPPED here; _bind_capped (inside the escalation helper)
    # applies the cap + suppress_reasoning per attempt, doubling the cap once on
    # LengthFinishReasonError so a reasoning-heavy local model's JSON still lands
    # instead of silently dropping the slice. Window bounds the escalation.
    try:
        _window = int(
            (_spine_cfg.resolve_provider_config(phase=phase_path) or {}).get(
                "context_window"
            )
            or 0
        )
    except Exception:  # noqa: BLE001
        _window = 0

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
        response: Any = await _ainvoke_structured_escalating(
            model,
            DecompositionResult,
            [SystemMessage(content=system_prompt), HumanMessage(content=human_content)],
            label="decomposer",
            base_cap=decompose_cap,
            window=_window,
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
        response = await _ainvoke_structured_escalating(
            model,
            DecompositionResult,
            [SystemMessage(content=system_prompt), HumanMessage(content=salvage_content)],
            label="decomposer-length-salvage",
            base_cap=decompose_cap,
            window=_window,
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


def _build_reference_signatures_block(
    db_path: str, workspace_root: str, refs: list[str]
) -> str:
    """Signatures + module paths of the slice's reference_symbols, or ''.

    Feeds enrich the EXACT names and import paths of the existing symbols its
    code will call/extend (e.g. the ``UIApi`` methods a UI slice persists
    through), grouped by file. Without this the model guesses both the method
    names and the module path — North guessed ``spine.api.ui`` and Qwen
    ``spine.ui.api`` for what is really ``spine/ui_api/api.py``.
    """
    if not refs:
        return ""
    try:
        from spine.agents.tools.codebase_query import get_symbol_signature
    except ImportError:
        return ""
    by_file: dict[str, list[str]] = {}
    imports: dict[str, set[str]] = {}  # module -> {ClassName}
    for r in refs:
        try:
            res = get_symbol_signature(db_path, workspace_root, r)
        except Exception:  # noqa: BLE001
            res = None
        if res:
            fp, head = res
            by_file.setdefault(fp or "(unknown)", []).append(head)
            # Derive the exact import for class-qualified refs (e.g.
            # 'UIApi.set_phase_provider' in spine/ui_api/api.py ->
            # 'from spine.ui_api.api import UIApi'). Skip module-path refs whose
            # leading segment is lowercase (a module/function, not a class).
            cls = r.split(".")[0]
            if fp and cls[:1].isupper():
                module = fp[:-3] if fp.endswith(".py") else fp
                module = module.replace("/", ".").strip(".")
                imports.setdefault(module, set()).add(cls)
    if not by_file:
        return ""
    lines: list[str] = []
    if imports:
        # An explicit, imperative import directive — instruct models under-attend
        # to the file-header hint below and otherwise guess the module path
        # (North->spine.api.ui, Qwen->spine.ui.api for spine/ui_api/api.py).
        lines.append("Import these EXISTING classes EXACTLY as written — do NOT "
                     "invent a module path:")
        for module, classes in sorted(imports.items()):
            lines.append(f"    from {module} import {', '.join(sorted(classes))}")
        lines.append("")
    lines.append(
        "Signatures of EXISTING symbols this slice calls or extends. Use these "
        "EXACT names — do NOT invent a method name:"
    )
    for fp, heads in by_file.items():
        lines.append(f"# {fp}")
        for head in heads:
            lines.append("\n".join("    " + ln for ln in head.splitlines()))
    return "\n".join(lines)


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
        from spine.agents.tools.codebase_query import find_symbol, list_file_symbols
    except ImportError:
        return

    _file_syms: dict[str, set[str]] = {}

    def _syms_for(fp: str) -> set[str]:
        if fp not in _file_syms:
            try:
                _file_syms[fp] = set(list_file_symbols(db_path, fp))
            except Exception:
                _file_syms[fp] = set()
        return _file_syms[fp]

    def _in_files(sym: str, files: list[str]) -> bool:
        """True if *sym* (or its bare tail) is indexed in any of *files*."""
        tail = sym.split(".")[-1]
        for fp in files:
            fs = _syms_for(fp)
            if sym in fs or any(s == tail or s.rsplit(".", 1)[-1] == tail for s in fs):
                return True
        return False

    for sl in slices:
        slice_id = sl.get("id", "?")
        target_files = [f for f in (sl.get("target_files") or []) if f]

        # Scrub edit_plan symbols: clear a symbol that (a) is not in the index at
        # all (a phantom — likely a new definition, not an anchor), or (b) exists
        # but NOT in the file the edit targets (a guaranteed-failing ast_edit
        # anchor — trace 4aa24c6b: SpineConfig.load, which lives in config.py,
        # anchored into an api.py slice). Demote ast_edit so the implementer falls
        # back to patch/full_replace instead.
        for hint in sl.get("edit_plan") or []:
            sym = hint.get("symbol", "")
            if not sym:
                continue
            hint_file = hint.get("file") or ""
            scope_files = [hint_file] if hint_file else target_files

            if scope_files and _in_files(sym, scope_files):
                continue  # resolves in its own target file → valid anchor

            try:
                exists_anywhere = find_symbol(db_path, sym) is not None
            except Exception:
                continue

            if exists_anywhere and not scope_files:
                continue  # real symbol, no target files to scope-check → keep

            reason = (
                "not found in codebase index — likely a new definition, not an anchor"
                if not exists_anywhere
                else f"exists in the index but not in target {scope_files} — a "
                "guaranteed-failing ast_edit anchor"
            )
            logger.warning(
                "decomposer: clearing edit_plan symbol %r in slice %r (%s)",
                sym, slice_id, reason,
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


# Hard cap on the anchor enum injected into the enrichment schema. Beyond this
# the enum is dropped (prompt constraint + scrub still apply) so the JSON-schema
# grammar stays cheap and within provider structured-output limits.
_MAX_ANCHOR_ENUM = 200


def _known_anchor_set(db_path: str, files: list[str]) -> list[str]:
    """Sorted union of indexed symbol names across *files* — the valid anchors."""
    try:
        from spine.agents.tools.codebase_query import list_file_symbols
    except ImportError:
        return []
    names: set[str] = set()
    for f in files:
        try:
            names.update(list_file_symbols(db_path, f))
        except Exception:
            continue
    return sorted(names)


def _enrichment_schema(db_path: str, files: list[str]) -> type[BaseModel]:
    """Enrichment output schema, constraining ``EditHint.symbol`` to live anchors.

    With the json_schema structured-output path (vLLM guided decoding), an
    ``enum`` on ``symbol`` makes a phantom anchor *unrepresentable at decode
    time*: the model can only emit an existing symbol or ``''`` (new file,
    new-definition insert, or non-symbol edit). This is the generation-time
    complement to ``_scrub_phantom_symbols``, which stays as the backstop for
    providers that ignore the enum or fall back off json_schema. Degrades to the
    unconstrained schema when nothing is indexed or the anchor set is too large.
    """
    anchors = _known_anchor_set(db_path, files)
    if not anchors or len(anchors) > _MAX_ANCHOR_ENUM:
        return _EnrichmentOutput
    # "" stays valid — new files, new-definition inserts (anchored via the last
    # existing symbol + action), and non-symbol edits all need an empty slot.
    sym_type = Literal[tuple([""] + anchors)]  # type: ignore[valid-type]
    anchored_hint = create_model(
        "AnchoredEditHint",
        __base__=EditHint,
        symbol=(
            sym_type,
            Field(default="", description=EditHint.model_fields["symbol"].description),
        ),
    )
    return create_model(
        "AnchoredEnrichmentOutput",
        __base__=_EnrichmentOutput,
        edit_plan=(list[anchored_hint], Field(default_factory=list)),
    )


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
        "- ONE entry per change site. Emit a SEPARATE entry for EACH new "
        "method/function you add — never a single umbrella entry like 'add "
        "the embedding, reranker and timeout methods'. If the slice adds six "
        "methods, emit six entries. Each entry's `intent` names exactly ONE "
        "symbol and states its full signature + behaviour (a thin implementer "
        "writes one small edit per entry; a bundled entry forces it to survey "
        "the whole file and is the api.py 89×-read spiral).\n"
        "- To ADD a new method/function to a class: anchor each new symbol on "
        "the last existing method of that class, action='insert_after' (still "
        "one entry per new method — they may share the same anchor).\n"
        "- To MODIFY an existing method: set symbol to its qualified name, "
        "action='replace'.\n"
        "- To add module-level code (imports, constants, top-level functions): "
        "set symbol to the nearest existing top-level definition, "
        "action='insert_before' or 'insert_after'.\n"
        "- A new function/method inserted next to an anchor lands as a SIBLING "
        "at that anchor's own nesting level — it has NO implicit access to any "
        "other function's local variables, and no `self` unless the anchor "
        "itself is a method of the class it joins. If a new helper needs data "
        "that only exists inside another function (e.g. an `api`/`config` "
        "object passed into that function), its `intent` MUST spell out the "
        "helper's full parameter list carrying that data explicitly — never "
        "assume closure capture or a class attribute that doesn't exist.\n"
        "- If new helpers must be CALLED from an existing function for the "
        "slice to do anything, emit a separate entry that anchors on that "
        "existing function with action='replace' and states in `intent` that "
        "its body now calls the new helper(s) — inserting a helper next to a "
        "function does not wire it in; only an explicit replace of the caller "
        "does.\n"
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

    enrich_phase = "implement/decomposer/enrich"
    model = resolve_chat_model(config, session_id=session_id, phase=enrich_phase)
    decompose_cap = _spine_cfg.decompose_max_completion_tokens
    enrich_schema = _enrichment_schema(_spine_cfg.checkpoint_path, files)
    try:
        enrich_window = int(
            (_spine_cfg.resolve_provider_config(phase=enrich_phase) or {}).get(
                "context_window"
            )
            or 0
        )
    except Exception:  # noqa: BLE001
        enrich_window = 0

    slice_json = json.dumps(
        {
            k: source_slice.get(k)
            for k in (
                "id",
                "title",
                "description",
                "target_files",
                "acceptance_criteria",
                "reference_symbols",
            )
        },
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    # Resolve the slice's reference_symbols to their real signatures + module
    # paths so enrich calls them correctly instead of guessing names/imports.
    ref_block = _build_reference_signatures_block(
        _spine_cfg.checkpoint_path,
        os.getcwd(),
        source_slice.get("reference_symbols") or [],
    )
    findings = known_block + ("\n\n" + ref_block if ref_block else "")
    human_content = hostage_layout(
        xml_blocks(
            (Tag.SPECIFICATION, slice_json),
            (Tag.FINDINGS, findings),
        ),
        "Return an _EnrichmentOutput with a concrete edit_plan for this slice.",
    )

    try:
        response: Any = await _ainvoke_structured_escalating(
            model,
            enrich_schema,
            [SystemMessage(content=_ENRICH_PROMPT), HumanMessage(content=human_content)],
            label="enrich_slice",
            base_cap=decompose_cap,
            window=enrich_window,
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
