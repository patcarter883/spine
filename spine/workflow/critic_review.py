"""SPINE critic review — two-tier critic with structural and agent checks.

The critic has two tiers:
1. **Structural** (fast, no LLM): checks artifacts exist, aren't empty, have
   basic structure. If structural fails, rework immediately — skip agent critic.
2. **Agent** (deep, LLM-based): quality review by the critic Deep Agent.
   Agent exceptions → rework, not crash.

Feedback is tagged by tier so the reworking phase knows what to address.

Context engineering: artifact content is on disk. The critic gets a short
preview inline with paths to full files, rather than inlining everything.
SpineContext is passed at invoke time.

Agent critic check is async to avoid event-loop binding errors when
subagents inherit the parent checkpointer — the sync ``invoke_with_retry``
runs in a thread pool which breaks ``asyncio.Lock`` objects bound to the
original event loop.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from spine.exceptions import CriticalContractFailure
from spine.models.enums import PhaseName, ReviewStatus
from spine.models.state import WorkflowState

# Phase → state field carrying the raw structured JSON produced by the
# phase agent. Critics MUST consume these instead of artifact previews;
# absence is a CriticalContractFailure.
_STRUCTURED_STATE_FIELD: dict[str, str] = {
    PhaseName.SPECIFY.value: "specification_json",
    PhaseName.PLAN.value: "plan_json",
}

logger = logging.getLogger(__name__)

# Hard cap on the length of a critic ``reason`` we persist/render. A degenerate
# critic once emitted a 126K-char repetition that the keyword fallback embedded
# verbatim and the work-detail UI rendered as a wall of repeated PASSED blocks
# (trace 019ef7fd).
_MAX_REASON_CHARS = 800


def _clip_reason(text: str) -> str:
    """Bound a critic reason: collapse consecutive duplicate lines (kills
    repetition-loop bloat) and hard-cap the length."""
    s = (text or "").strip()
    if not s:
        return s
    deduped: list[str] = []
    for ln in s.splitlines():
        if deduped and deduped[-1] == ln:
            continue
        deduped.append(ln)
    s = "\n".join(deduped)
    if len(s) > _MAX_REASON_CHARS:
        s = s[:_MAX_REASON_CHARS].rstrip() + f" … [reason truncated, {len(text)} chars total]"
    return s


def _format_prior_review(prior_review: dict[str, Any] | None) -> str:
    """Render the critic's own previous verdict for a rework pass.

    Returns "" when there is no prior verdict (first review). Otherwise
    renders the prior status + reason + suggestions so the critic can
    confirm whether each point was addressed instead of shifting the
    goalposts to brand-new issues every round.
    """
    if not prior_review:
        return ""
    reason = (prior_review.get("reason") or "").strip()
    suggestions = prior_review.get("suggestions") or []
    attempt = prior_review.get("attempt", 1)
    lines = [
        f"Your previous verdict on this phase (attempt {attempt}) was "
        f"NEEDS_REVISION for this reason:",
        f"  {reason}" if reason else "  (no reason recorded)",
    ]
    if suggestions:
        lines.append("You asked for these specific changes:")
        lines.extend(f"  - {s}" for s in suggestions)
    return "\n".join(lines)


def _build_review_prompt(
    *,
    reviewed_phase: str,
    structured_payload: str,
    description: str,
    spec_payload: str | None = None,
    prior_review: dict[str, Any] | None = None,
) -> str:
    """Build the user-message prompt for the critic agent.

    Wraps each dynamic block in a semantic XML tag and places the
    instruction in the hostage tail so small-model attention lands on
    "what to do" instead of trailing data. ``spec_payload`` is included
    only when reviewing the PLAN phase — that critic needs the spec's
    scope lists to check slice-level scope creep. ``prior_review`` is the
    critic's own previous verdict on this phase; when present (a rework
    pass) it is injected so the critic checks whether its prior asks were
    addressed rather than inventing new blocking issues each round.
    """
    from spine.agents.prompt_format import Tag, hostage_layout, xml_blocks

    findings_block = (
        f"```json\n{structured_payload}\n```" if structured_payload else ""
    )
    prior_block = _format_prior_review(prior_review)

    # When a specification is in scope (PLAN reviews), the critic may discover
    # that the requirement needs something the spec excludes/omits — a gap the
    # author cannot close by reworking this phase. Routing such a verdict
    # through NEEDS_REVISION burns the whole retry budget on plans that can
    # never pass (trace 019ed383: phase_max_retries demanded by the plan but
    # excluded by the spec's "no backend config changes"). Tell the critic to
    # escalate it as a spec contradiction instead.
    spec_contradiction_note = ""
    if spec_payload:
        spec_contradiction_note = (
            " IMPORTANT: if the ONLY thing blocking approval is that the "
            "<specification> excludes or omits something the requirement "
            "genuinely needs (so the author cannot fix it within this phase's "
            "scope), respond NEEDS_REVIEW and set blocker_category to "
            "'spec_contradiction' — do NOT respond NEEDS_REVISION, because "
            "reworking the plan cannot resolve a spec gap."
        )

    if prior_block:
        directive = (
            f"This is a REWORK review of the {reviewed_phase}-phase output. "
            "First, go through each point in the <critic_feedback> block above "
            "and decide whether the new payload ADDRESSES it. If every prior "
            "point is addressed, respond PASSED — do not block on issues you "
            "did not raise last round. Only emit NEEDS_REVISION if a prior, "
            "still-unaddressed point remains, or you find a genuinely NEW and "
            "BLOCKING semantic defect (not a schema/field-name nitpick). Keep "
            "your asks stable across rounds so the author can converge. "
            "Do NOT re-litigate slice ORGANISATION — splitting, merging, "
            "consolidating, renaming, or re-ordering slices that already form a "
            "valid dependency graph with full coverage is a structural taste "
            "preference, not a blocking defect. A deterministic validator "
            "already owns DAG validity, coverage, and granularity; never block a "
            "round on 'split this slice' / 'consolidate these slices' "
            "suggestions. Block only on a concrete SEMANTIC defect: a missing "
            "requirement, a dangling dependency, or a referenced symbol/method "
            "that no slice provides. "
            "Everything you need is in the tagged blocks — do not read files."
            + spec_contradiction_note
        )
    else:
        directive = (
            f"Review the {reviewed_phase}-phase structured output above. "
            "Do not attempt to read files or run commands; everything you "
            "need is in the tagged blocks. Respond with PASSED, "
            "NEEDS_REVISION, or NEEDS_REVIEW and include concrete reasons "
            "and suggestions."
            + spec_contradiction_note
        )

    return hostage_layout(
        xml_blocks(
            (Tag.OBJECTIVE, description or ""),
            (Tag.SPECIFICATION, spec_payload or ""),
            (Tag.CRITIC_FEEDBACK, prior_block),
            (Tag.FINDINGS, findings_block),
        ),
        directive,
    )


def _get_reviewed_phase(state: WorkflowState) -> str:
    """Determine which phase the critic is reviewing.

    First checks the explicit ``critic_reviewing`` field (set by the
    critic node function). Falls back to inspecting artifact keys.
    """
    # Explicit field takes priority
    critic_reviewing = state.get("critic_reviewing", "")
    if critic_reviewing:
        return critic_reviewing

    # Fallback: look at artifact keys
    artifacts = state.get("artifacts", {})
    non_critic_phases = [k for k in artifacts if k != PhaseName.CRITIC.value]
    if non_critic_phases:
        return non_critic_phases[-1]
    # Last resort
    feedback = state.get("feedback", [])
    if feedback:
        last = feedback[-1] if isinstance(feedback[-1], dict) else {}
        return last.get("phase", "unknown")
    return "unknown"


def structural_critic_check(state: WorkflowState, reviewed_phase: str) -> dict[str, Any]:
    """Fast, no-LLM structural check of a phase's output.

    Checks:
    - Artifacts exist for the reviewed phase
    - Artifact content is non-empty
    - Basic structure (has reasonable length for a document)

    Args:
        state: The current workflow state.
        reviewed_phase: The phase being reviewed.

    Returns:
        A dict with keys: ``status`` (ReviewStatus), ``tier``, ``reason``,
        ``suggestions``.
    """
    artifacts = state.get("artifacts", {})
    phase_artifacts = artifacts.get(reviewed_phase, {})

    if not phase_artifacts:
        return {
            "status": ReviewStatus.NEEDS_REVISION.value,
            "tier": "structural",
            "reason": f"No artifacts produced by {reviewed_phase} phase",
            "suggestions": [f"Ensure the {reviewed_phase} phase generates output documents"],
        }

    # Check each artifact has non-empty content
    for name, content in phase_artifacts.items():
        if not content or len(str(content).strip()) < 50:
            return {
                "status": ReviewStatus.NEEDS_REVISION.value,
                "tier": "structural",
                "reason": f"Artifact '{name}' from {reviewed_phase} is too short or empty",
                "suggestions": [
                    f"Expand the {name} artifact with more detail",
                    "Include specific technical decisions and rationale",
                ],
            }

    return {
        "status": ReviewStatus.PASSED.value,
        "tier": "structural",
        "reason": f"Structural check passed for {reviewed_phase}",
        "suggestions": [],
    }


async def agent_critic_check(
    state: WorkflowState,
    reviewed_phase: str,
    config: Any | None = None,
) -> dict[str, Any]:
    """Deep, LLM-based quality review by the critic agent.

    Delegates to the critic Deep Agent for a thorough review.
    If the agent raises an exception, returns NEEDS_REVISION rather than crashing.

    Context engineering: artifacts are materialized to disk and referenced
    by path. The critic gets a short preview inline (first 2000 chars) with
    the path to the full file, so it can read details on demand.

    Args:
        state: The current workflow state.
        reviewed_phase: The phase being reviewed.
        config: LangGraph runtime config (passed to the agent builder).

    Returns:
        A dict with keys: ``status``, ``tier``, ``reason``, ``suggestions``.
    """
    try:
        from spine.workflow.registry import get_registry

        registry = get_registry()
        critic_def = registry.get(PhaseName.CRITIC.value)
        if critic_def is None:
            logger.warning("Critic phase not registered, skipping agent review")
            return {
                "status": ReviewStatus.PASSED.value,
                "tier": "agent",
                "reason": "Critic agent not available, skipping",
                "suggestions": [],
            }

        # Build the critic agent and invoke it with retry
        from spine.agents.retry import ainvoke_with_retry
        from spine.agents.context import build_context
        from spine.agents.artifacts import materialize_artifacts

        critic_agent = critic_def.build_agent_fn(state, config)
        logger.info(
            "[%s] Critic agent built — invoking LLM review for phase '%s'",
            state.get("work_id", "?"), reviewed_phase,
        )

        # Materialize artifacts to disk so the critic can read them on demand.
        workspace_root = state.get("workspace_root", ".")
        work_id = state.get("work_id", "unknown")
        materialize_artifacts(state, workspace_root, work_id=work_id)

        # Critics review the structured JSON the phase agent produced — never
        # a truncated .md preview. Missing structured state is a contract
        # failure that must surface, not get papered over with a fallback.
        structured_field = _STRUCTURED_STATE_FIELD.get(reviewed_phase)
        if structured_field is None:
            raise CriticalContractFailure(
                phase=f"critic/{reviewed_phase}",
                reason=(
                    f"No structured-state field is registered for phase "
                    f"'{reviewed_phase}'. Critics require structured output; "
                    f"add an entry to _STRUCTURED_STATE_FIELD before enabling "
                    f"a critic for this phase."
                ),
            )
        structured_payload = state.get(structured_field)
        if not structured_payload or not isinstance(structured_payload, str):
            raise CriticalContractFailure(
                phase=f"critic/{reviewed_phase}",
                reason=(
                    f"State field '{structured_field}' is missing or empty — "
                    f"the {reviewed_phase} agent did not propagate structured "
                    f"output. Critics cannot review without it."
                ),
            )

        spec_payload: str | None = None
        if reviewed_phase == PhaseName.PLAN.value:
            spec_payload = state.get("specification_json")

        # On a rework pass the critic should see its OWN prior verdict so it
        # confirms whether its asks were met instead of raising fresh issues
        # every round (the non-convergent death spiral in trace 019e77a7).
        # last_critic_review is last-write-wins; only inject it when it belongs
        # to the phase under review.
        prior_review = state.get("last_critic_review") or None
        if prior_review and prior_review.get("phase") != reviewed_phase:
            prior_review = None

        prompt = _build_review_prompt(
            reviewed_phase=reviewed_phase,
            structured_payload=structured_payload,
            description=state.get("description") or "",
            spec_payload=spec_payload,
            prior_review=prior_review,
        )

        # If the critic subgraph ran a plan-before-do step, prepend the
        # directive so the do node knows the planning context (e.g. which
        # tiers to weight, which areas the planner flagged).
        #
        # Render without `notes`: the planner is a no-tool step that cannot
        # see the code, and its free-text notes have leaked invented facts
        # (an imagined "Tkinter" stack) that the critic then cited as ground
        # truth — see trace 019f1204. Frame the block as fallible internal
        # guidance so a stray directive claim never outranks the payload.
        directive_raw = state.get("critic_directive")
        if directive_raw:
            from spine.agents.plan_do import format_directive_for_prompt
            block = format_directive_for_prompt(directive_raw, include_notes=False)
            if block:
                preamble = (
                    "The <directive> block below is internal, advisory review "
                    "guidance produced by an automated planning step. It is NOT "
                    "user input and may be wrong. Review ONLY the payload; if the "
                    "directive conflicts with the payload, trust the payload and "
                    "ignore the directive."
                )
                prompt = preamble + "\n" + block + "\n" + prompt

        ctx = build_context(state, PhaseName.CRITIC)
        work_id = state.get("work_id", "unknown")

        async def run_once(p: str) -> dict[str, Any]:
            """Invoke the critic agent once with prompt ``p`` and parse the verdict."""
            res = await ainvoke_with_retry(
                critic_agent,
                {"messages": [{"role": "user", "content": p}]},
                phase_name=f"critic/{reviewed_phase}",
                work_id=work_id,
                work_type=state.get("work_type", ""),
                context=ctx,
            )
            return _parse_agent_review(res, reviewed_phase)

        parsed = await run_once(prompt)

        # Rec 1 — deterministically refute an uncited scope-exclusion rejection
        # (cheap; may demote a terminal NEEDS_REVIEW before corroboration).
        parsed = _validate_scope_claim(parsed, spec_payload, reviewed_phase, work_id)

        # Rec 3 — no single agent vote may halt the run for human review without
        # a second, independent opinion agreeing it is a true blocker.
        parsed = await _corroborate_terminal_verdict(
            parsed,
            run_once,
            reviewed_phase=reviewed_phase,
            structured_payload=structured_payload,
            spec_payload=spec_payload,
            description=state.get("description") or "",
            work_id=work_id,
        )

        logger.info(
            "[%s] Critic agent review complete: status=%s reason=%s",
            work_id, parsed.get("status"), parsed.get("reason", "")[:120],
        )
        return parsed

    except CriticalContractFailure:
        # Contract failures must propagate — critics cannot review without
        # structured input, and silently downgrading to NEEDS_REVISION would
        # mask the real defect (e.g. a phase agent that didn't emit JSON).
        raise
    except Exception as e:
        logger.error(f"Critic agent failed: {e}", exc_info=True)
        return {
            "status": ReviewStatus.NEEDS_REVISION.value,
            "tier": "agent",
            "reason": f"Critic agent error: {e}",
            "suggestions": ["Review the critic agent configuration and logs"],
        }


def _parse_agent_review(result: Any, reviewed_phase: str) -> dict[str, Any]:
    """Parse the critic agent's response into a structured review dict.

    First checks for structured output (CriticReview via response_format).
    Falls back to keyword parsing for backwards compatibility.
    """
    # Try structured output first (from response_format=CriticReview)
    structured = result.get("structured_response")
    if structured is not None:
        # Handle CriticReview Pydantic model
        if isinstance(structured, dict):
            data = structured
        elif hasattr(structured, "model_dump"):
            data = structured.model_dump()
        else:
            # Fallback for unexpected structured format
            return _parse_agent_review_fallback(result, reviewed_phase)

        status_raw = data.get("status", "NEEDS_REVISION")
        reason = _clip_reason(data.get("reason", "Structured review completed"))
        suggestions = data.get("suggestions", [])
        blocker_category = data.get("blocker_category")
        cited_exclusions = data.get("cited_exclusions") or []

        # Normalize status to lowercase (CriticReview uses uppercase)
        status = status_raw.lower() if isinstance(status_raw, str) else status_raw

        return {
            "status": status,
            "tier": "agent",
            "reason": reason,
            "suggestions": suggestions,
            "blocker_category": blocker_category,
            "cited_exclusions": cited_exclusions,
        }

    # Fallback to keyword parsing for backwards compatibility
    return _parse_agent_review_fallback(result, reviewed_phase)


def _parse_agent_review_fallback(result: Any, reviewed_phase: str) -> dict[str, Any]:
    """Fallback keyword parser for unstructured critic agent responses.

    Looks for PASSED, NEEDS_REVISION, or NEEDS_REVIEW keywords in the response.
    Falls back to NEEDS_REVISION if unclear.
    """
    messages = result.get("messages", [])
    last_message = messages[-1] if messages else None
    content = ""
    finish_reason = ""
    if last_message:
        content = getattr(last_message, "content", str(last_message))
        meta = getattr(last_message, "response_metadata", None) or {}
        finish_reason = str(meta.get("finish_reason", "")).lower()

    clipped = _clip_reason(content)

    # A structured-output critic that fell through to keyword parsing produced no
    # parseable CriticReview. If it ALSO hit the token ceiling
    # (finish_reason='length'), it degenerated/looped rather than rendering a
    # verdict — never salvage a PASS from that noise (trace 019ef7fd: a 126K-char
    # repetition keyword-matched to PASSED). Force a revision round instead.
    if finish_reason == "length":
        logger.warning(
            "critic fallback: response truncated at token limit without a "
            "structured verdict for phase '%s' — routing NEEDS_REVISION",
            reviewed_phase,
        )
        return {
            "status": ReviewStatus.NEEDS_REVISION.value,
            "tier": "agent",
            "reason": (
                "Critic response was truncated at the token limit "
                "(finish_reason=length) without a structured verdict — treating "
                "as NEEDS_REVISION rather than trusting a salvaged keyword. "
                f"Partial output: {clipped}"
            ),
            "suggestions": [],
            "cited_exclusions": [],
        }

    content_upper = content.upper()

    # Negation guard: prose like "did not pass" / "cannot be passed" must never
    # salvage a PASS from a bare "PASSED" substring. Fail closed on any negated
    # pass token before trusting an affirmative one.
    negated_pass = re.search(r"\b(?:NOT|NO|CANNOT|CAN'T|NEVER)\b[^.]*\bPASS", content_upper)

    if "NEEDS_REVIEW" in content_upper:
        status = ReviewStatus.NEEDS_REVIEW.value
    elif "NEEDS_REVISION" in content_upper:
        status = ReviewStatus.NEEDS_REVISION.value
    elif not negated_pass and re.search(r"\bPASSED\b", content_upper):
        status = ReviewStatus.PASSED.value
    else:
        # Unclear / negated / ambiguous response → needs revision (conservative).
        # A free-text verdict with no clear PASSED token must not fail open.
        status = ReviewStatus.NEEDS_REVISION.value

    return {
        "status": status,
        "tier": "agent",
        "reason": f"Agent review of {reviewed_phase}: {clipped}",
        "suggestions": [],
        "cited_exclusions": [],
    }


# ── Scope-claim refutation + terminal-verdict corroboration ──────────────────
# A weak critic model can hallucinate that in-scope work is "scope creep" and
# fire a terminal NEEDS_REVIEW that halts the whole run (trace 019ed849: it
# claimed embedding/reranker config — both in scope_inclusions — were
# excluded). Two guards run on the parsed verdict before it leaves
# agent_critic_check:
#   1. _validate_scope_claim — deterministically overturn a scope-exclusion
#      rejection that cites no real scope_exclusions bullet.
#   2. _corroborate_terminal_verdict — never let a single agent vote escalate
#      to human review without a second, independent opinion agreeing.

# Phrasing a critic uses when it claims the plan does TOO MUCH and reaches into
# an EXCLUDED area. Deliberately omits the spec_contradiction direction (plan
# NEEDS something the spec omits) — that opposite claim is handled by
# corroboration, not by this refutation.
_SCOPE_CREEP_MARKERS = (
    "scope creep",
    "scope_creep",
    "out of scope",
    "out-of-scope",
    "outside scope",
    "outside the scope",
    "outside of scope",
    "exceeds scope",
    "exceeds the scope",
    "scope violation",
    "violates scope",
    "violates the scope",
    "scope_exclusions",
)

_SCOPE_REMOVAL_VERBS = ("remove", "delete", "drop", "strip", "exclude")


def _norm_tokens(text: str) -> set[str]:
    """Lowercase alphanumeric tokens (length > 2) for fuzzy text matching."""
    return {t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(t) > 2}


def _citation_matches_exclusion(citation: str, exclusions: list[str]) -> bool:
    """True when ``citation`` plausibly refers to a real scope_exclusions bullet.

    Accepts a substring match either direction, or ≥0.5 Jaccard overlap on
    content tokens — tolerant of light paraphrase while still rejecting a
    citation that has nothing to do with any declared exclusion.
    """
    cl = (citation or "").strip().lower()
    if not cl:
        return False
    ctoks = _norm_tokens(citation)
    for ex in exclusions:
        el = (ex or "").strip().lower()
        if not el:
            continue
        if cl in el or el in cl:
            return True
        etoks = _norm_tokens(ex)
        if ctoks and etoks:
            overlap = len(ctoks & etoks) / len(ctoks | etoks)
            if overlap >= 0.5:
                return True
    return False


def _load_scope_lists(spec_payload: str | None) -> tuple[list[str], list[str]] | None:
    """Parse (scope_inclusions, scope_exclusions) from a Specification JSON.

    Returns None when the payload is absent or unparseable — callers then skip
    scope validation rather than guessing.
    """
    if not spec_payload or not isinstance(spec_payload, str):
        return None
    try:
        spec = json.loads(spec_payload)
    except (ValueError, TypeError):
        return None
    if not isinstance(spec, dict):
        return None
    incl = [str(x) for x in (spec.get("scope_inclusions") or []) if str(x).strip()]
    excl = [str(x) for x in (spec.get("scope_exclusions") or []) if str(x).strip()]
    return incl, excl


def _validate_scope_claim(
    parsed: dict[str, Any],
    spec_payload: str | None,
    reviewed_phase: str,
    work_id: str,
) -> dict[str, Any]:
    """Refute an UNSUPPORTED scope-exclusion rejection (rec 1).

    Fires only for PLAN reviews carrying a spec, and never on a
    spec_contradiction verdict (the opposite, legitimate claim). When the
    verdict alleges scope creep / an exclusion violation but cites no
    scope_exclusions bullet that actually matches the spec:

    - a terminal NEEDS_REVIEW is demoted to NEEDS_REVISION, its reason
      rewritten to RETAIN the in-scope work, and removal-style suggestions
      dropped — so the run reworks with corrected feedback instead of halting
      on a fabricated blocker;
    - a non-terminal NEEDS_REVISION is left to run (it may carry legitimate
      defects) but gets a short caution appended so the author does not delete
      in-scope work.

    Genuinely-cited scope objections pass through untouched.
    """
    if reviewed_phase != PhaseName.PLAN.value:
        return parsed
    if parsed.get("blocker_category") == "spec_contradiction":
        return parsed
    status = parsed.get("status")
    if status not in (
        ReviewStatus.NEEDS_REVISION.value,
        ReviewStatus.NEEDS_REVIEW.value,
    ):
        return parsed

    scope_lists = _load_scope_lists(spec_payload)
    if scope_lists is None:
        return parsed
    _inclusions, exclusions = scope_lists

    citations = [c for c in (parsed.get("cited_exclusions") or []) if str(c).strip()]
    reason = str(parsed.get("reason") or "")
    suggestions = parsed.get("suggestions") or []
    haystack = (reason + " " + " ".join(str(s) for s in suggestions)).lower()
    is_scope_claim = bool(citations) or any(m in haystack for m in _SCOPE_CREEP_MARKERS)
    if not is_scope_claim:
        return parsed

    # Supported if at least one citation matches a real exclusion bullet.
    if any(_citation_matches_exclusion(c, exclusions) for c in citations):
        return parsed

    out = dict(parsed)
    if status == ReviewStatus.NEEDS_REVIEW.value:
        notice = (
            "[scope objection overturned] A scope-creep / out-of-scope objection "
            "was raised but cited no matching scope_exclusions bullet"
            + (f" (offered: {citations})" if citations else "")
            + ". The flagged work is within the declared scope and MUST be "
            "retained — items in scope_inclusions are IN scope by definition. "
            "Do not remove them; address only concrete, in-scope defects."
        )
        if exclusions:
            notice += f" Actual scope_exclusions: {exclusions}."
        kept = [
            s
            for s in suggestions
            if not (
                any(v in str(s).lower() for v in _SCOPE_REMOVAL_VERBS)
                and any(m in str(s).lower() for m in ("scope", "exclud"))
            )
        ]
        out["status"] = ReviewStatus.NEEDS_REVISION.value
        out["blocker_category"] = None
        out["suggestions"] = kept
        out["reason"] = notice + " | original: " + reason
        logger.warning(
            "[%s] critic scope verdict OVERTURNED (terminal→revision): no cited "
            "exclusion matched spec for phase '%s'. citations=%s",
            work_id,
            reviewed_phase,
            citations,
        )
    else:
        # Non-terminal: keep status/suggestions (may hold real defects); only
        # caution the author against acting on the unsupported scope objection.
        out["reason"] = (
            reason
            + " | NOTE: any scope-exclusion objection must cite a real "
            "scope_exclusions bullet; items in scope_inclusions are in scope "
            "and must not be removed."
        )
        logger.info(
            "[%s] critic scope objection uncited (non-terminal) for phase '%s'; "
            "annotated. citations=%s",
            work_id,
            reviewed_phase,
            citations,
        )
    return out


def _build_reconsideration_prompt(
    *,
    reviewed_phase: str,
    structured_payload: str,
    spec_payload: str | None,
    description: str,
    first_verdict: dict[str, Any],
) -> str:
    """Prompt for an independent second opinion on a terminal verdict (rec 3)."""
    from spine.agents.prompt_format import Tag, hostage_layout, xml_blocks

    findings_block = f"```json\n{structured_payload}\n```" if structured_payload else ""
    prior = (
        f"A prior review returned {first_verdict.get('status')} and wants to HALT "
        "this run for human review. Its stated reason:\n"
        f"  {first_verdict.get('reason', '')}\n"
    )
    directive = (
        f"Give a SECOND, INDEPENDENT opinion on the {reviewed_phase}-phase output "
        "above. Terminal human review is expensive and must be reserved for "
        "issues the author genuinely CANNOT resolve by reworking this phase. "
        "Decide for yourself from the payload and specification:\n"
        "- If the blocking issue is real AND unresolvable by rework (e.g. the "
        "spec truly omits/excludes something the requirement needs), respond "
        "NEEDS_REVIEW (set blocker_category='spec_contradiction' if it is a spec "
        "gap).\n"
        "- If the author could plausibly fix it by reworking, respond "
        "NEEDS_REVISION.\n"
        "- If the prior objection does not hold up against the specification "
        "(e.g. it calls in-scope work 'scope creep'), respond PASSED.\n"
        "Judge only the tagged blocks; do not read files."
    )
    return hostage_layout(
        xml_blocks(
            (Tag.OBJECTIVE, description or ""),
            (Tag.SPECIFICATION, spec_payload or ""),
            (Tag.CRITIC_FEEDBACK, prior),
            (Tag.FINDINGS, findings_block),
        ),
        directive,
    )


def _is_terminal_verdict(parsed: dict[str, Any]) -> bool:
    """A verdict that would escalate to human review on a single agent vote."""
    return (
        parsed.get("status") == ReviewStatus.NEEDS_REVIEW.value
        or parsed.get("blocker_category") == "spec_contradiction"
    )


async def _corroborate_terminal_verdict(
    parsed: dict[str, Any],
    run_once: Any,
    *,
    reviewed_phase: str,
    structured_payload: str,
    spec_payload: str | None,
    description: str,
    work_id: str,
) -> dict[str, Any]:
    """Require a second opinion before a single vote halts the run (rec 3).

    ``run_once`` is an async callable ``(prompt) -> parsed_verdict`` that
    re-invokes the critic agent. If the verdict would terminally escalate, run
    one independent reconsideration pass; downgrade to NEEDS_REVISION unless the
    second review also calls it a blocker. Non-terminal verdicts are returned
    unchanged (no extra LLM call).
    """
    if not _is_terminal_verdict(parsed):
        return parsed

    recon_prompt = _build_reconsideration_prompt(
        reviewed_phase=reviewed_phase,
        structured_payload=structured_payload,
        spec_payload=spec_payload,
        description=description,
        first_verdict=parsed,
    )
    try:
        second = await run_once(recon_prompt)
    except Exception as e:  # corroboration is best-effort — never crash the critic
        logger.warning(
            "[%s] terminal-verdict corroboration failed (%s); keeping first verdict",
            work_id,
            e,
        )
        return parsed

    if _is_terminal_verdict(second):
        logger.info(
            "[%s] terminal critic verdict corroborated by second review (2nd=%s)",
            work_id,
            second.get("status"),
        )
        return parsed

    out = dict(parsed)
    out["status"] = ReviewStatus.NEEDS_REVISION.value
    out["blocker_category"] = None
    out["reason"] = (
        "[terminal escalation not corroborated] A second independent review did "
        f"not agree this requires human review (it returned {second.get('status')}). "
        "Downgraded to revision so the author can attempt a fix. Original basis: "
        + str(parsed.get("reason") or "")
    )
    merged = list(parsed.get("suggestions") or [])
    for s in second.get("suggestions") or []:
        if s not in merged:
            merged.append(s)
    out["suggestions"] = merged
    logger.warning(
        "[%s] terminal critic verdict NOT corroborated (2nd=%s) → downgraded to "
        "revision",
        work_id,
        second.get("status"),
    )
    return out


def critic_router(state: WorkflowState) -> str:
    """Conditional edge function for critic nodes.

    Reads the feedback that ``call_critic`` already wrote into state and
    returns the routing key.  Does NOT re-run the review — that would
    duplicate every LLM call.

    Routing:
    - ``"passed"`` → proceed to next phase
    - ``"needs_revision"`` → rework the previous phase (if retries remain)
    - ``"needs_review"`` → flag for human review (stop workflow)
    - ``"failed"`` → stop workflow as failed

    Args:
        state: The current workflow state (already updated by call_critic).

    Returns:
        A routing key string for the conditional edge.
    """
    if state.get("status") == "failed":
        return "failed"

    # Single source of truth: the critic mapper writes last_critic_review with
    # the canonical phase_status from the subgraph. Reading feedback[-1] was
    # fragile because feedback is operator.add and any prior "passed" entry
    # from another tier or phase could shadow the current critic's verdict.
    lcr = state.get("last_critic_review") or {}
    if not lcr:
        logger.warning(
            "critic_router: last_critic_review missing — routing needs_revision"
        )
        return "needs_revision"

    reviewed_phase = lcr.get("phase") or _get_reviewed_phase(state)
    review_status = lcr.get("status", ReviewStatus.NEEDS_REVISION.value)

    review = {
        "status": review_status,
        "tier": lcr.get("tier", "unknown"),
        "reason": lcr.get("reason", ""),
        "suggestions": lcr.get("suggestions", []),
        # Early-escalation signals computed by _critic_result_mapper. When set,
        # the loop stops short of the full retry budget (stagnation, or a spec
        # contradiction the author cannot resolve by reworking this phase).
        "escalate": lcr.get("escalate", False),
        "escalation_kind": lcr.get("escalation_kind"),
        "stagnation_streak": lcr.get("stagnation_streak", 0),
        "churn_streak": lcr.get("churn_streak", 0),
    }
    decision = _handle_review_outcome(state, reviewed_phase, review)
    retry_count = state.get("retry_count", {})
    logger.info(
        "[%s] critic_router: phase=%s status=%s retries=%d/%d streak=%d churn=%d kind=%s → %s",
        state.get("work_id", "?"),
        reviewed_phase,
        review_status,
        retry_count.get(reviewed_phase, 0),
        state.get("max_retries", 3),
        review.get("stagnation_streak", 0),
        review.get("churn_streak", 0),
        review.get("escalation_kind"),
        decision,
    )
    return decision


def _handle_review_outcome(
    state: WorkflowState,
    reviewed_phase: str,
    review: dict[str, Any],
) -> str:
    """Process a review result, managing retry counts and routing.

    If the review passed, return "passed".
    If needs revision and retries remain, return "needs_revision".
    If needs revision but retries exceeded, return "needs_review".
    If needs human review directly, return "needs_review".
    """
    status = review["status"]

    if status == ReviewStatus.PASSED.value:
        return "passed"

    if status == ReviewStatus.NEEDS_REVIEW.value:
        return "needs_review"

    # Early escalation: the mapper flagged this verdict as unresolvable by
    # another rework round (stagnation — the same asks recurring — or a spec
    # contradiction). Stop here instead of spending the rest of the budget on
    # plans that cannot converge.
    if review.get("escalate"):
        logger.warning(
            f"Phase '{reviewed_phase}' escalated early "
            f"(kind={review.get('escalation_kind')}, "
            f"streak={review.get('stagnation_streak', 0)}) → human review"
        )
        return "needs_review"

    # NEEDS_REVISION — check retry count
    retry_count = state.get("retry_count", {})
    phase_retries = retry_count.get(reviewed_phase, 0)
    max_retries = state.get("max_retries", 3)

    if phase_retries >= max_retries:
        logger.warning(
            f"Phase '{reviewed_phase}' exceeded max retries "
            f"({phase_retries}/{max_retries}), flagging for human review"
        )
        return "needs_review"

    # Increment retry count (the _merge_dicts reducer will handle the update)
    # Note: the actual state update happens in the phase node; here we just route
    return "needs_revision"
