"""Synthesis hierarchy nodes for the distributed onboarding engine.

This module implements Phase B of the onboarding graph (design Revision 2,
§2.2, §2.3) — the two-tier *documentation manager → section-worker* hierarchy
that lets synthesis fit a bounded local-model window **regardless of repo
size**, because **no LLM ever receives the whole manifest**:

- **Tier A — :func:`_doc_manager_node`** (documentation manager): builds the
  compact :func:`spine.work.onboarding.manifest_index.manifest_index`, starts
  from the deterministic
  :func:`spine.work.onboarding.synthesis_plan.deterministic_section_plan`
  skeleton, then makes ONE bare ``with_structured_output`` LLM call (no tools,
  no agentic loop) to *refine* grouping/ordering. On parse failure, empty, or
  incoherent output it falls back to the skeleton. Greenfield uses a fixed
  minimal plan.
- **Tier B — :func:`_section_worker_node`** (section worker): one bare
  ``with_structured_output`` LLM call per section, fanned out via ``Send``. Its
  input is ONLY the bounded fragment
  (:func:`spine.work.onboarding.manifest_index.resolve_fragment`, hard-capped
  to ``onboarding_section_token_cap``) + the section instruction + a short
  doc-level voice string. On failure it returns a ``SectionResult`` with
  ``status="error"`` carrying a GENERIC reason — never raw exception text
  (MEMORY rule: never leak tool-error text into generated docs).
- **Tier C — :func:`_assemble_docs_node`**: deterministic. Groups
  ``section_results`` by ``doc_id``, sorts by ``order``, concatenates markdown,
  and writes each ``<NAME>.md`` via :class:`WriteOnboardingDocTool`.
- :func:`_aggregate_synthesis_node`: verifies all four ``.md`` exist (→
  ``RuntimeError`` listing any missing, preserved from the legacy
  ``synthesis.py`` driver) and fails loudly if any section status is
  ``"error"``.

Both LLM tiers follow the bare-LLM-call primitive (the ``run_research_manager``
shape, NOT ``build_phase_agent``): :func:`spine.agents.helpers.resolve_chat_model`
→ ``model.with_structured_output(Schema)`` → a single ``ainvoke``, with the
instance/``.parsed``/dict coercion handled defensively by
:func:`spine.agents.helpers.coerce_structured_output`.

:func:`build_synthesis_graph` wires Tier A → B → C → aggregate into an
uncompiled ``StateGraph`` for the back-compat shim and the unit tests.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from spine.agents.helpers import (
    cap_completion_tokens,
    coerce_structured_output,
    resolve_chat_model,
    suppress_parsed_serializer_warning,
)
from spine.agents.prompt_format import Tag, hostage_layout, xml_blocks
from spine.work.onboarding.manifest import RepoManifest
from spine.work.onboarding.manifest_index import (
    manifest_index,
    resolve_fragment_from_dict,
    validate_fragment_keys,
)
from spine.work.onboarding.onboarding_state import OnboardingGraphState
from spine.work.onboarding.synthesis_plan import (
    SectionPlan,
    SectionPlanSet,
    SectionResult,
    deterministic_section_plan,
)
from spine.work.onboarding.synthesis_tools import (
    ONBOARDING_DOC_NAMES,
    WriteOnboardingDocTool,
    onboarding_docs_dir,
)

logger = logging.getLogger(__name__)

# Hard fallback for the per-fragment token ceiling when config carries no
# explicit ``onboarding_section_token_cap``.
_DEFAULT_SECTION_TOKEN_CAP = 6000
_DEFAULT_MAX_SECTIONS = 32
_DEFAULT_SECTION_MAX_COMPLETION_TOKENS = 2048


# ── Doc-level voice / role strings ───────────────────────────────────────────
#
# Shared across every worker authoring a given document so the fragmented
# sections read with one consistent voice (design Risk #5 mitigation).
_DOC_VOICE: dict[str, str] = {
    "PROJECT_DEFINITION": (
        "You author PROJECT_DEFINITION.md — what this project IS and DOES: its "
        "business purpose, core objectives, and domain boundaries. Write for a "
        "new contributor who needs the 'why' before the 'how'."
    ),
    "CODING_GUIDELINES": (
        "You author CODING_GUIDELINES.md — the typing, error-handling, testing, "
        "and naming rules contributors must follow. State each convention as a "
        "concrete, enforceable rule grounded in the supplied evidence."
    ),
    "ARCHITECTURE_MAP": (
        "You author ARCHITECTURE_MAP.md — the codemap: module responsibilities, "
        "primary execution paths, data flow, and dependencies. Be concrete; cite "
        "module names, paths, and key symbols from the supplied fragment."
    ),
    "SPINE_ASSISTANCE_REQUIREMENTS": (
        "You author SPINE_ASSISTANCE_REQUIREMENTS.md — instructions for Spine's "
        "Deep Agents: known Skill boundaries and context limits that prevent "
        "token-burn (which modules are large, where to use the codebase index "
        "rather than reading files)."
    ),
}

_DEFAULT_VOICE = (
    "You author one section of a Spine onboarding document. Write self-contained, "
    "well-structured markdown grounded ONLY in the supplied fragment."
)

_GENERIC_SECTION_ERROR = "section generation failed; section omitted"

# Bounded retries for a single section's bare-LLM structured call. A transient
# failure or one empty/unparseable response is retried up to this many extra
# times before the section is marked status="error" (finding #6) — so one
# flaky call no longer nukes the whole run, while we stay fail-loud after the
# budget is exhausted.
_SECTION_MAX_RETRIES = 2


# ── Manager (Tier A) prompts ─────────────────────────────────────────────────

_MANAGER_ROLE = (
    "You are the Spine onboarding documentation manager. You plan how four "
    "onboarding documents (PROJECT_DEFINITION, CODING_GUIDELINES, "
    "ARCHITECTURE_MAP, SPINE_ASSISTANCE_REQUIREMENTS) should be broken into "
    "small sections, each written by a separate worker from a bounded slice of "
    "the repository. You see only a COMPACT INDEX of the repo (names, roles, "
    "counts) — never the code itself."
)

_MANAGER_INSTRUCTIONS = (
    "You are given a deterministic DRAFT plan and a compact repository INDEX. "
    "Refine the draft: keep one section per natural unit, merge trivially small "
    "units, drop empty sections, and set a sensible reading ORDER per document. "
    "Preserve every section's `fragment_keys` exactly as drafted (they are "
    "stable manifest selectors — do NOT invent module/category/domain names not "
    "present in the index). Every document MUST keep at least one section. "
    "Return the full refined list of sections."
)


def _section_token_cap(config: RunnableConfig | None) -> int:
    """Per-fragment token ceiling, from config with a safe default."""
    cap = _DEFAULT_SECTION_TOKEN_CAP
    if config:
        configurable = config.get("configurable", {}) or {}
        spine_config = configurable.get("spine_config")
        if spine_config is not None:
            cap = int(getattr(spine_config, "onboarding_section_token_cap", cap) or cap)
    return cap if cap > 0 else _DEFAULT_SECTION_TOKEN_CAP


def _section_max_completion_tokens(config: RunnableConfig | None) -> int:
    """Per-section completion-token cap for the structured LLM call.

    Prevents runaway generation when the local model tries to fill the global
    max_completion_tokens window (16K) before the JSON schema closes, causing
    LengthFinishReasonError and a 290-450s wall-clock penalty per worker.
    """
    cap = _DEFAULT_SECTION_MAX_COMPLETION_TOKENS
    if config:
        configurable = config.get("configurable", {}) or {}
        spine_config = configurable.get("spine_config")
        if spine_config is not None:
            cap = int(
                getattr(
                    spine_config,
                    "onboarding_section_max_completion_tokens",
                    cap,
                )
                or cap
            )
    return cap if cap > 0 else _DEFAULT_SECTION_MAX_COMPLETION_TOKENS


def _max_sections(config: RunnableConfig | None) -> int:
    """Cap on the number of index modules / call volume, from config."""
    cap = _DEFAULT_MAX_SECTIONS
    if config:
        configurable = config.get("configurable", {}) or {}
        spine_config = configurable.get("spine_config")
        if spine_config is not None:
            cap = int(getattr(spine_config, "onboarding_max_sections", cap) or cap)
    return cap if cap > 0 else _DEFAULT_MAX_SECTIONS


def _manifest_from_state(state: OnboardingGraphState) -> RepoManifest:
    """Reconstruct the in-state manifest dict into a :class:`RepoManifest`."""
    data = state.get("manifest") or {}
    return RepoManifest.from_dict(data)


def _coerce_plan(response: Any) -> SectionPlanSet | None:
    """Coerce a ``with_structured_output`` response into a :class:`SectionPlanSet`.

    Delegates to :func:`spine.agents.helpers.coerce_structured_output`, which
    handles the three shapes seen across LangChain/provider versions (the
    Pydantic instance directly, ``AIMessage.parsed``, or a plain dict) and
    returns ``None`` when the shape is unrecognised so the caller falls back to
    the deterministic skeleton.
    """
    return coerce_structured_output(response, SectionPlanSet)


def _plan_is_coherent(
    sections: list[dict[str, Any]],
    index: dict[str, Any],
) -> bool:
    """A refined plan is usable only if it covers all four docs AND resolves.

    Two gates (design finding #5):

    1. **Coverage** — a weak local model may drop a document or emit a
       structurally-valid but empty plan; the plan must carry at least one
       section per document.
    2. **Resolvability** — every section's ``fragment_keys`` must reference only
       a known ``doc_id`` and only module/category/domain names present in the
       compact *index*. A single unresolvable selector means the LLM invented a
       name (→ hollow section), so the whole plan is rejected.

    On either failure the caller falls back to the deterministic skeleton, whose
    selectors are guaranteed to resolve.
    """
    if not sections:
        return False
    covered = {
        s.get("doc_id")
        for s in sections
        if isinstance(s, dict) and s.get("doc_id") in ONBOARDING_DOC_NAMES
    }
    if covered != set(ONBOARDING_DOC_NAMES):
        return False

    for s in sections:
        if not isinstance(s, dict):
            return False
        fragment_keys = dict(s.get("fragment_keys", {}) or {})
        fragment_keys.setdefault("doc_id", s.get("doc_id", ""))
        if validate_fragment_keys(index, fragment_keys):
            return False
    return True


def _section_to_dict(section: SectionPlan) -> dict[str, Any]:
    """Serialise a :class:`SectionPlan` to the plain dict the state carries."""
    return {
        "doc_id": section.doc_id,
        "order": int(section.order),
        "title": section.title,
        "fragment_keys": dict(section.fragment_keys or {}),
        "instruction": section.instruction,
    }


async def _doc_manager_node(
    state: OnboardingGraphState,
    config: RunnableConfig = None,  # noqa: RUF013 - LangGraph injects; None for direct calls
) -> dict[str, Any]:
    """Tier A: plan the onboarding sections from a compact index.

    Builds the bounded :func:`manifest_index`, starts from the deterministic
    skeleton, and makes ONE bare ``with_structured_output`` LLM call to refine
    grouping/ordering. Falls back to the skeleton on parse failure, empty, or
    incoherent output. Greenfield uses the fixed minimal skeleton (no LLM call
    needed, but the refine call is still attempted only for brownfield).

    Returns ``{"manifest_index": index, "sections": [...]}`` — ``sections`` is
    a seed list for the ``_slice_list_reducer`` channel.
    """
    work_id = state.get("work_id", "unknown")
    mode = state.get("mode", "brownfield")
    manifest = _manifest_from_state(state)

    max_modules = _max_sections(config)
    token_cap = _section_token_cap(config)
    index = manifest_index(manifest, max_modules=max_modules)

    skeleton = deterministic_section_plan(index, mode)

    # Greenfield: the fixed minimal skeleton is the plan — no LLM refinement.
    if mode == "greenfield":
        logger.info("[%s] doc_manager: greenfield minimal plan (%d sections)", work_id, len(skeleton))
        return {
            "manifest_index": index,
            "sections": skeleton,
            "section_token_cap": token_cap,
        }

    sections = await _refine_plan_with_llm(work_id, index, skeleton, config)
    return {
        "manifest_index": index,
        "sections": sections,
        "section_token_cap": token_cap,
    }


async def _refine_plan_with_llm(
    work_id: str,
    index: dict[str, Any],
    skeleton: list[dict[str, Any]],
    config: RunnableConfig | None,
) -> list[dict[str, Any]]:
    """Make the single bare manager call; return the skeleton on any failure."""
    try:
        model = resolve_chat_model(
            config, session_id=work_id, phase="onboarding/doc-manager"
        )

        index_json = json.dumps(index, ensure_ascii=False)
        draft_json = json.dumps(skeleton, ensure_ascii=False)
        blocks = xml_blocks(
            (Tag.ROLE, _MANAGER_ROLE),
            (Tag.WORKFLOW, _MANAGER_INSTRUCTIONS),
            (Tag.FINDINGS, f"Compact repository index:\n```json\n{index_json}\n```"),
            (Tag.SCRATCHPAD, f"Deterministic draft plan:\n```json\n{draft_json}\n```"),
        )
        prompt = hostage_layout(
            blocks,
            "Return the refined, ordered list of sections for all four documents.",
        )

        structured = model.with_structured_output(SectionPlanSet)
        with suppress_parsed_serializer_warning():
            response = await structured.ainvoke(
                [SystemMessage(content=_MANAGER_ROLE), HumanMessage(content=prompt)]
            )

        plan_set = _coerce_plan(response)
        if plan_set is None:
            logger.warning(
                "[%s] doc_manager: unrecognised LLM response shape → skeleton", work_id
            )
            return skeleton

        refined = [_section_to_dict(s) for s in plan_set.sections]
        if not _plan_is_coherent(refined, index):
            logger.warning(
                "[%s] doc_manager: refined plan rejected (incoherent or "
                "references unknown manifest keys; covers %d/4 docs) → skeleton",
                work_id,
                len({s.get("doc_id") for s in refined}),
            )
            return skeleton

        logger.info(
            "[%s] doc_manager: refined plan accepted (%d sections)", work_id, len(refined)
        )
        return refined
    except Exception as exc:  # noqa: BLE001 - any manager failure → deterministic floor
        logger.warning(
            "[%s] doc_manager: LLM refine failed (%s) → skeleton",
            work_id,
            type(exc).__name__,
        )
        return skeleton


# ── Section router (Tier A → Tier B fan-out) ─────────────────────────────────


def _section_router(state: OnboardingGraphState) -> list[Send]:
    """Fan out one ``Send("section_worker", ...)`` per planned section.

    The router resolves each section's bounded fragment HERE (via
    :func:`resolve_fragment_from_dict`, hard-capped) and ships it inside
    ``active_section`` so the worker node receives ONLY its fragment — no
    manifest, no read tool.

    The projection reads the in-state manifest DICT (``state["manifest"]``)
    directly, so the previous per-section ``RepoManifest.from_dict`` ->
    ``manifest.to_dict`` deep-copy round-trip (O(sections x full-manifest)) is
    gone — every section now slices the same shared dict (finding #8).
    """
    sections = list(state.get("sections", []) or [])
    if not sections:
        return []

    manifest_dict: dict[str, Any] = state.get("manifest") or {}
    # ``_section_router`` is a plain conditional-edge function and cannot read
    # config; the manager seeds ``section_token_cap`` into state for it.
    token_cap = int(
        state.get("section_token_cap", _DEFAULT_SECTION_TOKEN_CAP)
        or _DEFAULT_SECTION_TOKEN_CAP
    )

    sends: list[Send] = []
    for section in sections:
        fragment_keys = dict(section.get("fragment_keys", {}) or {})
        fragment_keys.setdefault("doc_id", section.get("doc_id", ""))
        fragment = resolve_fragment_from_dict(manifest_dict, fragment_keys, token_cap)
        active = {
            "doc_id": section.get("doc_id", ""),
            "order": int(section.get("order", 0) or 0),
            "title": section.get("title", ""),
            "instruction": section.get("instruction", ""),
            "fragment": fragment,
        }
        sends.append(Send("section_worker", {"active_section": active}))
    return sends


# ── Section worker (Tier B) ──────────────────────────────────────────────────


def _worker_prompt(active: dict[str, Any], voice: str) -> str:
    """Build the bounded per-section worker prompt (hostage layout)."""
    fragment_json = json.dumps(active.get("fragment", {}), ensure_ascii=False)
    blocks = xml_blocks(
        (Tag.ROLE, voice),
        (
            Tag.OBJECTIVE,
            f"Write the '{active.get('title', '')}' section. {active.get('instruction', '')}",
        ),
        (
            Tag.CONSTRAINTS,
            "Use ONLY the supplied fragment as your source of truth. Do not "
            "invent modules, symbols, or facts not present in it. Output "
            "self-contained markdown for THIS section only (no document-level "
            "title) — it will be concatenated with sibling sections.",
        ),
        (Tag.FINDINGS, f"Section fragment:\n```json\n{fragment_json}\n```"),
    )
    return hostage_layout(
        blocks,
        "Write the markdown for this one section now.",
    )


def _coerce_section_result(response: Any) -> SectionResult | None:
    """Coerce a worker ``with_structured_output`` response to a SectionResult.

    Delegates to :func:`spine.agents.helpers.coerce_structured_output` (the
    same instance / ``.parsed`` / dict-validate handling the doc manager uses).
    """
    return coerce_structured_output(response, SectionResult)


async def _section_worker_node(
    state: OnboardingGraphState,
    config: RunnableConfig = None,  # noqa: RUF013 - LangGraph injects; None for direct calls
) -> dict[str, Any]:
    """Tier B: author ONE section from its bounded fragment (bare LLM call).

    Input is ONLY ``state["active_section"]`` (fragment + instruction + title).
    Returns ``{"section_results": [one SectionResult dict]}`` for the
    ``operator.add`` channel. The bare structured call is retried up to
    :data:`_SECTION_MAX_RETRIES` extra times on exception or empty/unparseable
    output so a single transient failure does not nuke the run (finding #6).
    After the budget is exhausted it returns a result with ``status="error"``
    and a GENERIC reason (never raw exception text).
    """
    active = state.get("active_section", {}) or {}
    doc_id = active.get("doc_id", "")
    order = int(active.get("order", 0) or 0)
    work_id = state.get("work_id", "unknown")
    voice = _DOC_VOICE.get(doc_id, _DEFAULT_VOICE)

    def _error_result() -> dict[str, Any]:
        return {
            "doc_id": doc_id,
            "order": order,
            "markdown": "",
            "status": "error",
        }

    try:
        model = resolve_chat_model(
            config, session_id=work_id, phase="onboarding/section-worker"
        )
        comp_cap = _section_max_completion_tokens(config)
        model = cap_completion_tokens(model, comp_cap)

        prompt = _worker_prompt(active, voice)
        structured = model.with_structured_output(SectionResult)
        messages = [SystemMessage(content=voice), HumanMessage(content=prompt)]

        # Bounded retry: a single transient LLM failure or one empty/unparseable
        # response should not nuke the whole run (finding #6). Retry the bare
        # structured call up to _SECTION_MAX_RETRIES extra times, then fall back
        # to a generic status="error" (never leaking exception text).
        for attempt in range(_SECTION_MAX_RETRIES + 1):
            try:
                with suppress_parsed_serializer_warning():
                    response = await structured.ainvoke(messages)
                result = _coerce_section_result(response)
            except Exception as exc:  # noqa: BLE001 - never leak exception text into docs
                result = None
                if attempt >= _SECTION_MAX_RETRIES:
                    logger.warning(
                        "[%s] section_worker: section %s#%d failed after %d "
                        "attempt(s) (%s)",
                        work_id,
                        doc_id,
                        order,
                        attempt + 1,
                        type(exc).__name__,
                    )
                    return {"section_results": [_error_result()]}
                continue

            if result is not None and (result.markdown or "").strip():
                return {
                    "section_results": [
                        {
                            "doc_id": doc_id,
                            "order": order,
                            "markdown": result.markdown,
                            "status": "ok",
                        }
                    ]
                }

            if attempt >= _SECTION_MAX_RETRIES:
                logger.warning(
                    "[%s] section_worker: empty/unparseable section %s#%d after "
                    "%d attempt(s)",
                    work_id,
                    doc_id,
                    order,
                    attempt + 1,
                )
                return {"section_results": [_error_result()]}

        # Unreachable: the loop always returns. Defensive floor.
        return {"section_results": [_error_result()]}
    except Exception as exc:  # noqa: BLE001 - never leak exception text into docs
        logger.warning(
            "[%s] section_worker: section %s#%d failed (%s)",
            work_id,
            doc_id,
            order,
            type(exc).__name__,
        )
        return {"section_results": [_error_result()]}


# ── Document assembler (Tier C) ──────────────────────────────────────────────


def _doc_dir(workspace_root: str) -> Path:
    """Resolve the stable onboarding doc directory (matches the write tool)."""
    return onboarding_docs_dir(workspace_root)


def _assemble_docs_node(state: OnboardingGraphState) -> dict[str, Any]:
    """Tier C: group section results by doc, sort, concat, write each ``.md``.

    Deterministic — no LLM. Writes via :class:`WriteOnboardingDocTool` (idempotent
    ``overwrite_shorter=True``). Sections with ``status="error"`` are skipped in
    the concatenation (the aggregator fails loudly on any error), and a document
    with no OK sections falls back to a minimal generated placeholder so the
    all-four-exist contract is still observable before the aggregator runs.
    """
    workspace_root = state.get("workspace_root", "")
    work_id = state.get("work_id", "unknown")
    results = list(state.get("section_results", []) or [])

    doc_dir = _doc_dir(workspace_root)
    writer = WriteOnboardingDocTool(docs_dir=str(doc_dir))

    written: dict[str, str] = {}
    placeholder_only: list[str] = []

    for doc in ONBOARDING_DOC_NAMES:
        doc_sections = sorted(
            (r for r in results if r.get("doc_id") == doc and r.get("status") == "ok"),
            key=lambda r: int(r.get("order", 0) or 0),
        )
        body_parts = [
            (r.get("markdown") or "").strip()
            for r in doc_sections
            if (r.get("markdown") or "").strip()
        ]
        title = doc.replace("_", " ").title()
        if body_parts:
            content = f"# {title}\n\n" + "\n\n".join(body_parts) + "\n"
        else:
            # Idempotent placeholder so the all-four-exist invariant is still
            # observable, but record the doc so the aggregator FAILS the run:
            # a placeholder-only document carries no real content (finding #4).
            content = (
                f"# {title}\n\n"
                "_No content could be synthesised for this document._\n"
            )
            placeholder_only.append(doc)
        writer._run(doc=doc, content=content)
        written[doc] = str(doc_dir / f"{doc}.md")

    logger.info(
        "[%s] assemble_docs: wrote %d documents (%d placeholder-only)",
        work_id,
        len(written),
        len(placeholder_only),
    )
    return {"written": written, "placeholder_docs": placeholder_only}


# ── Aggregator ───────────────────────────────────────────────────────────────


def _aggregate_synthesis_node(state: OnboardingGraphState) -> dict[str, Any]:
    """Verify all four docs exist + no section errored, else ``RuntimeError``.

    Preserves the all-or-nothing contract of the legacy single-agent driver
    (``synthesis.py`` lines 205-223): a missing document is a hard failure. We
    additionally fail loudly when any section's status is ``"error"`` so a
    single bad section never produces a silently-truncated document.
    """
    workspace_root = state.get("workspace_root", "")
    work_id = state.get("work_id", "unknown")
    results = list(state.get("section_results", []) or [])

    errored = [
        f"{r.get('doc_id', '?')}#{r.get('order', '?')}"
        for r in results
        if r.get("status") == "error"
    ]
    if errored:
        raise RuntimeError(
            f"onboarding synthesis incomplete for work {work_id}: "
            f"{len(errored)} section(s) failed: {errored}."
        )

    doc_dir = _doc_dir(workspace_root)
    # Verify existence at the same stable directory WriteOnboardingDocTool
    # writes to, so the read/write path scheme can never drift.
    written: dict[str, str] = {}
    missing: list[str] = []
    for doc in ONBOARDING_DOC_NAMES:
        doc_path = doc_dir / f"{doc}.md"
        if doc_path.exists():
            written[doc] = str(doc_path)
        else:
            missing.append(doc)

    if missing:
        raise RuntimeError(
            f"onboarding synthesis incomplete for work {work_id}: "
            f"{len(missing)}/{len(ONBOARDING_DOC_NAMES)} document(s) not written: "
            f"{missing}. Wrote: {sorted(written)}."
        )

    # A document written as a placeholder-only stub (no OK sections produced any
    # real content) carries no substance. The "all four docs" contract means
    # four documents with REAL content, not four files (finding #4) — so a
    # placeholder-only doc must FAIL the run loudly even though its file exists.
    placeholder_docs = sorted(state.get("placeholder_docs", []) or [])
    if placeholder_docs:
        raise RuntimeError(
            f"onboarding synthesis incomplete for work {work_id}: "
            f"{len(placeholder_docs)}/{len(ONBOARDING_DOC_NAMES)} document(s) "
            f"have no synthesised content (placeholder-only): {placeholder_docs}."
        )

    logger.info(
        "[%s] aggregate_synthesis: verified all %d onboarding documents",
        work_id,
        len(ONBOARDING_DOC_NAMES),
    )
    return {"written": written}


# ── Graph builder ────────────────────────────────────────────────────────────


#: Destinations of the doc manager's ``_section_router`` (one ``Send`` per
#: section → ``section_worker``). Hoisted so the half-builder and the composed
#: onboarding graph share ONE route-map literal.
SECTION_ROUTE_MAP: list[str] = ["section_worker"]


def add_synthesis_nodes_and_edges(graph: StateGraph) -> None:
    """Add the synthesis-phase nodes + INTERIOR edges to ``graph`` in place.

    Adds ``doc_manager`` / ``section_worker`` / ``assemble_docs`` /
    ``aggregate_synthesis`` and the interior wiring (manager ``Send`` fan-out →
    section_worker → assemble_docs → aggregate_synthesis), but NOT the ``START``
    entry edge or any terminal edge — the caller owns those so the same wiring
    can serve both the standalone half-graph (``START`` → … → ``END``) and the
    composed onboarding graph (entered via ``aggregate_analysis`` →
    ``doc_manager``). This is the single source of truth for Phase B's topology,
    shared with
    :func:`spine.work.onboarding.onboarding_graph.build_onboarding_graph`.
    """
    graph.add_node("doc_manager", _doc_manager_node)
    graph.add_node("section_worker", _section_worker_node)
    graph.add_node("assemble_docs", _assemble_docs_node)
    graph.add_node("aggregate_synthesis", _aggregate_synthesis_node)

    graph.add_conditional_edges(
        "doc_manager",
        _section_router,
        SECTION_ROUTE_MAP,
    )
    graph.add_edge("section_worker", "assemble_docs")
    graph.add_edge("assemble_docs", "aggregate_synthesis")


def build_synthesis_graph() -> StateGraph:
    """Build the synthesis half of the onboarding graph (Tier A → B → C → agg).

    The returned graph is uncompiled (callers compile with whatever
    checkpointer they want). It expects an initial state carrying at least
    ``work_id``, ``workspace_root``, ``mode``, and ``manifest``
    (``RepoManifest.to_dict()``). It runs:

        doc_manager → (Send per section) → section_worker → assemble_docs
        → aggregate_synthesis → END

    Used by the :func:`spine.work.onboarding.synthesis.synthesize_artifacts`
    back-compat shim and the synthesis unit tests. The interior nodes/edges are
    shared with the composed graph via :func:`add_synthesis_nodes_and_edges`;
    this builder only adds the ``START`` and ``END`` boundary edges.
    """
    graph: StateGraph = StateGraph(OnboardingGraphState)
    add_synthesis_nodes_and_edges(graph)
    graph.add_edge(START, "doc_manager")
    graph.add_edge("aggregate_synthesis", END)
    return graph
