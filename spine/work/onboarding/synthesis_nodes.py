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
  doc-level voice string. The model fills the content-only
  :class:`spine.work.onboarding.synthesis_plan.SectionContent` schema (required
  ``overview``; no ``doc_id``/``order``/``status`` echo) and the worker renders
  the section markdown deterministically via
  :func:`spine.work.onboarding.synthesis_plan.render_section_markdown` — the
  model never authors document structure (same contract as
  ``specify_tools._WriteSpecificationInput``). On failure it returns a
  ``SectionResult``-shaped dict with ``status="error"`` carrying a GENERIC
  reason — never raw exception text (MEMORY rule: never leak tool-error text
  into generated docs).
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

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from spine.agents.helpers import (
    bind_structured_output,
    cap_completion_tokens,
    coerce_structured_output,
    resolve_chat_model,
    suppress_parsed_serializer_warning,
    suppress_reasoning,
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
    SectionContent,
    SectionPlan,
    SectionPlanSet,
    deterministic_section_plan,
    render_section_markdown,
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
    "You author one section of a Spine onboarding document, grounded ONLY in "
    "the supplied fragment."
)

_GENERIC_SECTION_ERROR = "section generation failed; section omitted"

# Bounded retries for a single section's bare-LLM structured call. A transient
# failure or one empty/unparseable response is retried up to this many extra
# times before the section is marked status="error" (finding #6) — so one
# flaky call no longer nukes the whole run, while we stay fail-loud after the
# budget is exhausted.
_SECTION_MAX_RETRIES = 2

# Base seconds for exponential backoff between a section worker's retries. The
# pre-fix loop fired all three attempts in ~30ms (trace 019ece15: 01:57:10.721
# → .753), giving a momentarily-unreachable model endpoint zero time to recover
# before the section was abandoned. Backoff is applied only after a *transient*
# (transport) failure — a content failure is deterministic and the corrective
# nudge, not a sleep, is what changes the outcome. attempt N waits
# ``base * 2**N`` seconds, capped at :data:`_SECTION_RETRY_BACKOFF_CAP_SECONDS`.
_SECTION_RETRY_BACKOFF_SECONDS = 1.5
_SECTION_RETRY_BACKOFF_CAP_SECONDS = 8.0

# Substrings / exception class-name markers identifying a TRANSIENT transport
# failure (endpoint unreachable, reset, or timed out) as opposed to a content
# failure. Provider-agnostic on purpose: the local GGUF endpoint surfaces these
# as ``APIError('CURL error: Could not connect to server')`` (trace 019ece15),
# while hosted providers raise ``APIConnectionError`` / ``APITimeoutError``.
_TRANSIENT_ERROR_MARKERS = (
    "could not connect",
    "connection error",
    "connection reset",
    "connection refused",
    "curl error",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "service unavailable",
    "bad gateway",
    "apiconnectionerror",
    "apitimeouterror",
)


class RetryableSynthesisError(RuntimeError):
    """Synthesis failed on TRANSIENT (infrastructure) section errors only.

    Subclasses :class:`RuntimeError` so existing ``except RuntimeError`` callers
    still treat the run as failed (the all-or-nothing contract is preserved for
    them), while newer callers can ``isinstance``-check to distinguish a
    recoverable infra blip — where the already-written sections should be kept
    and only the failed ones re-run once the endpoint is back — from a permanent
    failure. The completed sections are already persisted to disk by
    :func:`_assemble_docs_node` before this is raised.
    """

    def __init__(self, work_id: str, transient_sections: list[str]) -> None:
        self.work_id = work_id
        self.transient_sections = list(transient_sections)
        super().__init__(
            f"onboarding synthesis hit a transient failure for work {work_id}: "
            f"{len(transient_sections)} section(s) failed to reach the model "
            f"endpoint (retryable): {transient_sections}. Completed sections are "
            "already written; re-run to fill only the gaps."
        )


def _is_transient_error(exc: BaseException) -> bool:
    """Classify an exception as a transient transport failure vs a real error.

    Inspects the exception class name and message for connection/timeout markers
    (:data:`_TRANSIENT_ERROR_MARKERS`). Deliberately string-based so it survives
    across provider SDKs without importing every provider's exception hierarchy.
    """
    haystack = f"{type(exc).__name__} {exc}".lower()
    return any(marker in haystack for marker in _TRANSIENT_ERROR_MARKERS)


def _retry_backoff_seconds(attempt: int) -> float:
    """Exponential backoff (seconds) before the next transient retry."""
    delay = _SECTION_RETRY_BACKOFF_SECONDS * (2**attempt)
    return min(delay, _SECTION_RETRY_BACKOFF_CAP_SECONDS)


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

        monorepo_addendum = ""
        if index.get("is_monorepo"):
            pkg_names = [
                p.get("dotted_name", p.get("name", "?"))
                for p in (index.get("workspace_packages") or [])
            ]
            monorepo_addendum = (
                "\n\nThis is a MONOREPO containing multiple independent packages: "
                f"{', '.join(pkg_names)}. "
                "Treat each package as a standalone application or library — NOT as a "
                "module of a single application. Use the package's dotted_name as its "
                "canonical name in the documents you plan."
            )
        instructions = _MANAGER_INSTRUCTIONS + monorepo_addendum

        blocks = xml_blocks(
            (Tag.ROLE, _MANAGER_ROLE),
            (Tag.WORKFLOW, instructions),
            (Tag.FINDINGS, f"Compact repository index:\n```json\n{index_json}\n```"),
            (Tag.SCRATCHPAD, f"Deterministic draft plan:\n```json\n{draft_json}\n```"),
        )
        prompt = hostage_layout(
            blocks,
            "Return the refined, ordered list of sections for all four documents.",
        )

        structured = bind_structured_output(model, SectionPlanSet)
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


def _section_router(state: OnboardingGraphState) -> list[Send] | str:
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
        # No sections planned (degenerate/empty manifest, or a refine step that
        # produced an empty plan). Route straight to assembly: returning an
        # empty Send list would leave ``doc_manager`` with no outgoing edge and
        # hang the graph indefinitely. ``assemble_docs`` handles zero sections.
        return "assemble_docs"

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
    """Build the bounded per-section worker prompt (hostage layout).

    The contract matches the structured-output binding: the model returns a
    JSON object (``SectionContent``), NEVER raw markdown — document structure
    is rendered deterministically from its fields. The previous wording
    ("write the markdown now") contradicted the JSON output mode and let weak
    models resolve the tension by emitting a content-free envelope (trace
    019eaf55).
    """
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
            "invent modules, symbols, or facts not present in it. Return a "
            "JSON object matching the schema: put 1-3 paragraphs of prose in "
            "'overview', one 'entries' item (name, path, description) per "
            "concrete module/convention/component you cover, and any caveats "
            "in 'notes'. Inline markdown (backticks, bold) is allowed inside "
            "strings; do NOT include headings — document structure is "
            "rendered from your fields and concatenated with sibling sections.",
        ),
        (Tag.FINDINGS, f"Section fragment:\n```json\n{fragment_json}\n```"),
    )
    return hostage_layout(
        blocks,
        "Fill the section-content JSON for this one section now.",
    )


# Corrective nudge appended on retry attempts. Re-sending the identical
# messages just replays a deterministic failure (trace 019eaf55: all three
# attempts returned the same content-free envelope); naming the defect gives
# the retry a reason to land differently.
_RETRY_NUDGE = (
    "Your previous response did not contain usable section content. Return a "
    "JSON object matching the schema exactly: 'overview' MUST hold 1-3 "
    "non-empty paragraphs of prose for this section. Do not return an empty "
    "or metadata-only object. Do not include your reasoning, chain-of-thought, "
    "or any meta-commentary about how you're formatting the JSON — 'overview' "
    "is published verbatim as documentation prose, so it must contain ONLY "
    "the section content itself."
)


def _coerce_section_content(response: Any) -> SectionContent | None:
    """Coerce a worker ``with_structured_output`` response to SectionContent.

    Delegates to :func:`spine.agents.helpers.coerce_structured_output` (the
    same instance / ``.parsed`` / dict-validate handling the doc manager uses).
    """
    return coerce_structured_output(response, SectionContent)


# Chain-of-thought / self-narration phrases that occasionally leak into an
# otherwise schema-valid ``overview`` (trace 019f3a1d: ARCHITECTURE_MAP.md and
# PROJECT_DEFINITION.md sections were published containing the model's raw
# deliberation about how to format the JSON — hundreds of words of "Let's
# craft...", "wait, the user did not mention notes?", "the JSON begins now" —
# instead of section prose). Pydantic only checks ``overview`` is a non-empty
# string, which a leaked-reasoning blob satisfies just fine. Not
# prefix-anchored like :func:`spine.agents.helpers._looks_like_leaked_reasoning`
# — these leaks land anywhere in the text, often after a clean opening
# paragraph, so this scans the whole string. A false positive here just costs
# one retry attempt (unlike that helper, which discards a whole phase output),
# so the bar for inclusion is "would never appear in real documentation
# prose", not "impossible to coincidentally match".
_LEAKED_REASONING_MARKERS = (
    "</think>",
    "```",
    "<|tool_call",
    "let's craft",
    "let me craft",
    "i must not invent",
    "no extra text outside",
    "the json begins now",
    "the final json is",
    "json final",
    "wait, the user",
    "final answer is the json",
    "no additional commentary outside",
    "json_pseudofunction",
)


def _looks_like_leaked_content(overview: str) -> bool:
    """True if ``overview`` contains a chain-of-thought/self-narration leak."""
    lowered = overview.lower()
    return any(marker in lowered for marker in _LEAKED_REASONING_MARKERS)


async def _section_worker_node(
    state: OnboardingGraphState,
    config: RunnableConfig = None,  # noqa: RUF013 - LangGraph injects; None for direct calls
) -> dict[str, Any]:
    """Tier B: author ONE section from its bounded fragment (bare LLM call).

    Input is ONLY ``state["active_section"]`` (fragment + instruction + title).
    The model fills the content-only :class:`SectionContent` schema; the
    worker renders the markdown via :func:`render_section_markdown` using its
    own known ``doc_id``/``order``/``title`` (never the model's). Returns
    ``{"section_results": [one SectionResult dict]}`` for the ``operator.add``
    channel. The bare structured call is retried up to
    :data:`_SECTION_MAX_RETRIES` extra times — with :data:`_RETRY_NUDGE`
    appended so a retry isn't a verbatim replay — on exception or
    empty/unparseable output, so a single transient failure does not nuke the
    run (finding #6). After the budget is exhausted it returns a result with
    ``status="error"`` and a GENERIC reason (never raw exception text).
    """
    active = state.get("active_section", {}) or {}
    doc_id = active.get("doc_id", "")
    order = int(active.get("order", 0) or 0)
    work_id = state.get("work_id", "unknown")
    voice = _DOC_VOICE.get(doc_id, _DEFAULT_VOICE)

    def _error_result(reason_kind: str = "content") -> dict[str, Any]:
        return {
            "doc_id": doc_id,
            "order": order,
            "markdown": "",
            "status": "error",
            "reason_kind": reason_kind,
        }

    try:
        model = resolve_chat_model(
            config, session_id=work_id, phase="onboarding/section-worker"
        )
        comp_cap = _section_max_completion_tokens(config)
        model = suppress_reasoning(cap_completion_tokens(model, comp_cap))

        prompt = _worker_prompt(active, voice)
        structured = bind_structured_output(model, SectionContent)
        base_messages = [SystemMessage(content=voice), HumanMessage(content=prompt)]

        # Bounded retry: a single transient LLM failure or one empty/unparseable
        # response should not nuke the whole run (finding #6). Retry the bare
        # structured call up to _SECTION_MAX_RETRIES extra times — appending the
        # corrective nudge so the retry isn't a verbatim replay — then fall back
        # to a status="error" tagged transient vs content (never leaking
        # exception text). A transient (transport) failure additionally backs off
        # before retrying so a momentary endpoint outage has time to recover, and
        # propagates its kind so the aggregator can mark the run RETRYABLE rather
        # than discarding the sections that did complete (trace 019ece15).
        for attempt in range(_SECTION_MAX_RETRIES + 1):
            messages = (
                base_messages
                if attempt == 0
                else [*base_messages, HumanMessage(content=_RETRY_NUDGE)]
            )
            try:
                with suppress_parsed_serializer_warning():
                    response = await structured.ainvoke(messages)
                content = _coerce_section_content(response)
            except Exception as exc:  # noqa: BLE001 - never leak exception text into docs
                content = None
                transient = _is_transient_error(exc)
                if attempt >= _SECTION_MAX_RETRIES:
                    logger.warning(
                        "[%s] section_worker: section %s#%d failed after %d "
                        "attempt(s) (%s, %s)",
                        work_id,
                        doc_id,
                        order,
                        attempt + 1,
                        type(exc).__name__,
                        "transient" if transient else "content",
                    )
                    return {
                        "section_results": [
                            _error_result("transient" if transient else "content")
                        ]
                    }
                if transient:
                    await asyncio.sleep(_retry_backoff_seconds(attempt))
                continue

            if content is not None and content.overview.strip():
                if _looks_like_leaked_content(content.overview):
                    logger.warning(
                        "[%s] section_worker: section %s#%d attempt %d produced "
                        "leaked-reasoning content — discarding and retrying",
                        work_id,
                        doc_id,
                        order,
                        attempt + 1,
                    )
                else:
                    markdown = render_section_markdown(active.get("title", ""), content)
                    return {
                        "section_results": [
                            {
                                "doc_id": doc_id,
                                "order": order,
                                "markdown": markdown,
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
        transient = _is_transient_error(exc)
        logger.warning(
            "[%s] section_worker: section %s#%d failed (%s, %s)",
            work_id,
            doc_id,
            order,
            type(exc).__name__,
            "transient" if transient else "content",
        )
        return {
            "section_results": [
                _error_result("transient" if transient else "content")
            ]
        }


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
    """Verify all four docs exist with real content, else raise.

    Section failures are split by ``reason_kind`` (set by the section worker):

    - **transient** — the model endpoint was unreachable (connection/timeout).
      The completed sections are already written to disk by
      :func:`_assemble_docs_node`, so we raise :class:`RetryableSynthesisError`
      (a ``RuntimeError`` subclass) rather than discarding them: a re-run, once
      the endpoint is back, fills only the gaps. This is the trace-019ece15 fix
      for "9 transient section failures nuked ~25 good sections".
    - **content** — the model could not produce usable content for that section.
      These are *tolerated* as long as every document still carries real content:
      a content gap omits one section, it does not fail the run. The hard floors
      below still apply — a missing document (legacy all-or-nothing contract) or
      a placeholder-only document (finding #4) fails loudly, so a content failure
      that empties a whole document is still caught.
    """
    workspace_root = state.get("workspace_root", "")
    work_id = state.get("work_id", "unknown")
    results = list(state.get("section_results", []) or [])

    errored = [r for r in results if r.get("status") == "error"]

    def _label(r: dict[str, Any]) -> str:
        return f"{r.get('doc_id', '?')}#{r.get('order', '?')}"

    transient = [_label(r) for r in errored if r.get("reason_kind") == "transient"]
    if transient:
        # Infra blip, not a content defect — keep what completed; re-run the gaps.
        raise RetryableSynthesisError(work_id, transient)

    content_failed = [_label(r) for r in errored if r.get("reason_kind") != "transient"]
    if content_failed:
        # Tolerated: the section is omitted. The missing-doc and placeholder-only
        # floors below still fail the run if a gap left any document empty.
        logger.warning(
            "[%s] aggregate_synthesis: %d section(s) omitted (no usable content): "
            "%s",
            work_id,
            len(content_failed),
            content_failed,
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
        "[%s] aggregate_synthesis: verified all %d onboarding documents "
        "(%d section(s) omitted as degraded)",
        work_id,
        len(ONBOARDING_DOC_NAMES),
        len(content_failed),
    )
    return {"written": written, "degraded_sections": content_failed}


# ── Graph builder ────────────────────────────────────────────────────────────


#: Destinations of the doc manager's ``_section_router``: one ``Send`` per
#: section → ``section_worker``, OR a direct hop to ``assemble_docs`` when the
#: plan has zero sections (an empty Send list would dead-end the graph). Hoisted
#: so the half-builder and the composed onboarding graph share ONE route-map
#: literal.
SECTION_ROUTE_MAP: list[str] = ["section_worker", "assemble_docs"]


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
