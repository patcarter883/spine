"""SPINE cross-run experience — distil critic feedback into reusable lessons.

The write path (:func:`capture_run_experience`) runs at the end of a workflow
run: it reads the run's accumulated critic / adversarial feedback and turns the
defects that were flagged for revision into compact :class:`ExperienceLesson`
records, persisted via :class:`spine.persistence.experience_store.ExperienceStore`.

The read path (:func:`resolve_experience_block`) runs when a phase agent is
built: it pulls the lessons recorded for that phase and renders a small
``<learned_experience>`` block injected into the agent's system prompt, so a
defect the critic caught on a previous run is guarded against up front.

**Why both paths anchor to the main repo root, never the run's worktree:**
code-producing runs execute in a throwaway git worktree
(:class:`spine.git.sandbox.WorktreeSandbox`) whose ``.spine/`` is a fresh
checkout — writes there are rolled back, and ``.spine/experience`` isn't even
present. Capture therefore uses the *base* config (main root); injection loads
a fresh :class:`SpineConfig` (whose ``workspace_root`` auto-resolves to the main
repo) rather than the worktree path carried in ``WorkflowState``.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from spine.models.types import ExperienceLesson
from spine.persistence.experience_store import ExperienceStore

logger = logging.getLogger(__name__)

# Verdicts worth learning from — the critic asked for a change or escalated.
_REVISION_STATUSES = {"needs_revision", "needs_review"}
# Structural-tier feedback ("artifact too short / empty") is generic boilerplate,
# not a reusable design lesson. Only the LLM/human review tiers carry one.
_LESSON_TIERS = {"agent", "adversarial", "human"}

_MAX_TRIGGER_CHARS = 240
_MAX_LESSON_CHARS = 360
# Don't let a single pathological run flood the store; dedup + per-phase cap in
# the store bound the total, this bounds one run's contribution.
_MAX_LESSONS_PER_RUN = 6
# How many lessons to inject into a phase prompt — small on purpose.
_INJECT_LIMIT = 5

# Terminal statuses whose feedback is noise (a crash / abort), not a lesson.
_SKIP_CAPTURE_STATUSES = {"failed", "cancelled", "stalled"}


# ── Store location ───────────────────────────────────────────────────────────
def experience_store_for(config: Any) -> ExperienceStore:
    """Build an :class:`ExperienceStore` rooted at the project's main repo.

    ``config.experience_path`` is relative by default (``.spine/experience``);
    it is anchored under ``config.workspace_root`` so the store lives at the
    project root regardless of the process CWD.
    """
    base = getattr(config, "experience_path", None) or ".spine/experience"
    path = Path(base)
    root = getattr(config, "workspace_root", "") or ""
    if not path.is_absolute() and root:
        path = Path(root) / base
    return ExperienceStore(str(path))


# ── Text helpers ─────────────────────────────────────────────────────────────
def _clip(text: str, limit: int) -> str:
    s = " ".join((text or "").split())
    if len(s) > limit:
        s = s[:limit].rstrip() + "…"
    return s


def _lesson_text(reason: str, suggestions: list[str]) -> str:
    """Derive the reusable guidance from a review's reason + suggestions.

    The critic's ``suggestions`` are imperative and reusable, so they make the
    best lesson body; fall back to the ``reason`` when there are none.
    """
    clean = [str(s).strip() for s in (suggestions or []) if str(s).strip()]
    body = "; ".join(clean) if clean else (reason or "")
    return _clip(body, _MAX_LESSON_CHARS)


# ── Write path ───────────────────────────────────────────────────────────────
def _attribute_phase(entry: dict[str, Any], reworked: list[str]) -> str | None:
    """Best-effort: map a flat feedback entry to the phase it concerns.

    Feedback list entries are not reliably phase-tagged (the critic writes
    ``{status, tier, reason, suggestions}``). We attribute via, in order:
    an explicit ``phase`` field; a ``[phase]`` / "<phase> phase" token in the
    reason (subgraph wrappers embed these); or — when exactly one phase was
    reworked this run — that phase. Otherwise we decline to guess (return None).
    """
    explicit = entry.get("phase")
    if explicit:
        return str(explicit)
    reason = (entry.get("reason") or "").lower()
    for cand in reworked:
        if f"[{cand}]" in reason or f"{cand} phase" in reason:
            return cand
    if len(reworked) == 1:
        return reworked[0]
    return None


def distill_run_experience(result: dict[str, Any], config: Any) -> list[ExperienceLesson]:
    """Distil a run's flagged defects into reusable lessons (no I/O).

    Combines two sources:

    1. **Terminal escalation verdicts** — ``last_critic_review`` /
       ``last_adversarial_review`` when they ended in revision/review. These are
       fully phase-attributed and carry the blocking defect.
    2. **Converged-after-rework rounds** — needs_revision entries in the
       ``feedback`` list, attributed to the phase(s) named in ``retry_count``.
       This captures lessons even when the run ultimately passed (the terminal
       verdict for that phase is then PASSED and carries no defect).

    De-duplicated within the run and capped at :data:`_MAX_LESSONS_PER_RUN`.
    """
    feedback = result.get("feedback") or []
    retry_count = result.get("retry_count") or {}
    category = result.get("task_category")
    work_id = result.get("work_id", "unknown")
    created = datetime.now().isoformat()

    lessons: list[ExperienceLesson] = []
    seen: set[tuple[str, str]] = set()

    def add(phase: Any, reason: str, suggestions: list[str], tier: str, salience: int) -> None:
        if not phase:
            return
        text = _lesson_text(reason, suggestions)
        if not text:
            return
        key = (str(phase), " ".join(text.lower().split()))
        if key in seen:
            return
        seen.add(key)
        lessons.append(
            ExperienceLesson(
                id=uuid.uuid4().hex[:12],
                work_id=work_id,
                phase=str(phase),
                category=category,
                trigger=_clip(reason, _MAX_TRIGGER_CHARS) or text,
                lesson=text,
                # Freeze the dedup identity to this pre-generalization text so the
                # downstream LLM rewrite can't paraphrase a recurring defect into a
                # fresh key (see ExperienceLesson.dedup_key).
                dedup_basis=text,
                source_tier=tier or "agent",
                salience=max(1, int(salience or 1)),
                created_at=created,
            )
        )

    # Source 1 — terminal escalation verdicts (fully phase-attributed).
    for review_key, default_tier in (
        ("last_critic_review", "agent"),
        ("last_adversarial_review", "adversarial"),
    ):
        rv = result.get(review_key)
        if isinstance(rv, dict) and rv.get("status") in _REVISION_STATUSES:
            add(
                rv.get("phase"),
                rv.get("reason", ""),
                rv.get("suggestions") or [],
                rv.get("tier") or default_tier,
                int(rv.get("attempt") or 1),
            )

    # Source 2 — converged-after-rework rounds from the feedback list.
    reworked = [p for p, c in retry_count.items() if isinstance(c, int) and c >= 1]
    if reworked:
        for entry in feedback:
            if not isinstance(entry, dict):
                continue
            if entry.get("status") not in _REVISION_STATUSES:
                continue
            if entry.get("tier") not in _LESSON_TIERS:
                continue
            phase = _attribute_phase(entry, reworked)
            if phase is None:
                continue
            add(
                phase,
                entry.get("reason", ""),
                entry.get("suggestions") or [],
                entry.get("tier") or "agent",
                int(retry_count.get(phase, 1)),
            )

    return lessons[:_MAX_LESSONS_PER_RUN]


# ── LLM generalisation pass ──────────────────────────────────────────────────
class _GeneralizedLesson(BaseModel):
    """One rewritten lesson, mapped back to its input by index."""

    index: int = Field(description="0-based index of the input lesson this maps to")
    lesson: str = Field(
        default="",
        description="The generalised, reusable rule — no run-specific identifiers",
    )
    drop: bool = Field(
        default=False,
        description="True when the defect is too run-specific to yield a reusable rule",
    )


class _GeneralizationResult(BaseModel):
    """Structured output for the generalisation pass."""

    lessons: list[_GeneralizedLesson] = Field(default_factory=list)


def _generalize_system_prompt() -> str:
    from spine.agents.prompt_format import Tag, xml_block

    return (
        xml_block(
            Tag.ROLE,
            "You distil one-off code-review defects into general, reusable "
            "engineering rules for an AI coding agent. The agent will read your "
            "rules before it works, to avoid repeating a class of mistake.",
        )
        + "\n\n"
        + xml_block(
            Tag.CONSTRAINTS,
            "- Rewrite each flagged defect as ONE short, imperative rule that "
            "prevents the same CLASS of mistake on a DIFFERENT task.\n"
            "- Strip every run-specific identifier: slice ids, file paths, "
            "symbol/method names, requirement ids, and concrete numbers.\n"
            "- Preserve the original phase's concern; do not invent new scope.\n"
            "- Keep each rule to a single sentence.\n"
            "- If a defect is purely run-specific and yields no reusable rule, "
            "set drop=true for that index instead of forcing a vague platitude.",
        )
        + "\n\n"
        + xml_block(
            Tag.OUTPUT_SCHEMA,
            "Return JSON {\"lessons\": [{\"index\": int, \"lesson\": str, "
            "\"drop\": bool}, ...]} with exactly one entry per input index.",
        )
    )


async def generalize_lessons(
    lessons: list[ExperienceLesson], config: Any
) -> list[ExperienceLesson]:
    """Rewrite run-specific lessons into general rules via one LLM call.

    Best-effort: returns the input unchanged on any failure (no model, parse
    error, timeout). Lessons the model marks ``drop`` are removed; the rest keep
    all their metadata with only the ``lesson`` text rewritten.
    """
    if not lessons:
        return lessons
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        from spine.agents.helpers import bind_structured_output, resolve_chat_model
        from spine.agents.prompt_format import Tag, hostage_layout, xml_blocks

        # config=None → resolve_chat_model loads SpineConfig and resolves the
        # optional ``experience`` phase override, else the default provider.
        model = resolve_chat_model(None, phase="experience")
        bound = bind_structured_output(model, _GeneralizationResult)

        numbered = "\n".join(
            f"[{i}] phase={le.phase} | flagged: {le.trigger} | lesson: {le.lesson}"
            for i, le in enumerate(lessons)
        )
        prompt = hostage_layout(
            xml_blocks((Tag.FINDINGS, numbered)),
            "Generalise each numbered defect above into a reusable rule. Return "
            "one entry per index in the JSON schema.",
        )
        res = await bound.ainvoke(
            [SystemMessage(content=_generalize_system_prompt()), HumanMessage(content=prompt)]
        )
        if not isinstance(res, _GeneralizationResult):
            # Some providers return a dict; coerce best-effort.
            res = _GeneralizationResult.model_validate(res)

        # Trust the model's `index` only when it is a valid, unique, in-range
        # position. A 1-based renumbering or a duplicate index would otherwise
        # graft a generalized rule onto the WRONG input lesson (keeping that
        # lesson's phase/trigger); reject those and fall back to the original.
        by_index: dict[int, _GeneralizedLesson] = {}
        for g in res.lessons:
            if 0 <= g.index < len(lessons) and g.index not in by_index:
                by_index[g.index] = g
        out: list[ExperienceLesson] = []
        for i, le in enumerate(lessons):
            g = by_index.get(i)
            if g is None:
                out.append(le)
                continue
            if g.drop:
                continue
            text = (g.lesson or "").strip()
            out.append(le.model_copy(update={"lesson": _clip(text, _MAX_LESSON_CHARS)}) if text else le)
        logger.info(
            "experience generalisation: %d in → %d out", len(lessons), len(out)
        )
        return out
    except Exception:  # noqa: BLE001 — generalisation is best-effort
        logger.debug("experience generalisation failed (non-fatal)", exc_info=True)
        return lessons


async def capture_run_experience(
    result: dict[str, Any],
    config: Any,
    final_status: str,
) -> int:
    """Distil, optionally generalise, and persist a run's lessons.

    Best-effort — never raises. Returns the number of new lessons written.
    Capture is skipped when disabled by config or when the run ended in a
    crash/abort status whose feedback is noise rather than a learnable defect.
    """
    try:
        if not getattr(config, "experience_capture", True):
            return 0
        if final_status in _SKIP_CAPTURE_STATUSES:
            return 0
        lessons = distill_run_experience(result, config)
        if not lessons:
            return 0
        if getattr(config, "experience_generalize", True):
            lessons = await generalize_lessons(lessons, config)
            if not lessons:
                return 0
        added = experience_store_for(config).add_many(lessons)
        if added:
            logger.info(
                "[%s] captured %d cross-run experience lesson(s)",
                result.get("work_id", "?"),
                added,
            )
        return added
    except Exception:  # noqa: BLE001 — capture must never break run finalisation
        logger.debug("experience capture failed (non-fatal)", exc_info=True)
        return 0


# ── Read path ────────────────────────────────────────────────────────────────
def format_experience_block(lessons: list[ExperienceLesson]) -> str:
    """Render lessons as a ``<learned_experience>`` system-prompt block."""
    if not lessons:
        return ""
    from spine.agents.prompt_format import Tag, xml_block

    lines = [
        "Past reviews of this phase flagged the issues below. Check your output "
        "against each before finishing — do not repeat them:"
    ]
    lines.extend(f"- {le.lesson}" for le in lessons)
    return xml_block(Tag.LEARNED_EXPERIENCE, "\n".join(lines))


def resolve_experience_block(
    phase: str,
    *,
    category: str | None = None,
    config: Any | None = None,
) -> str:
    """Return the injectable experience block for ``phase`` (best-effort).

    Loads a fresh :class:`SpineConfig` when none is given so the store resolves
    against the main repo root rather than a run's worktree. Returns ``""`` when
    injection is disabled, no lessons exist, or anything goes wrong.
    """
    try:
        if config is None:
            from spine.config import SpineConfig

            config = SpineConfig.load()
        if not getattr(config, "experience_injection", True):
            return ""
        lessons = experience_store_for(config).for_phase(
            phase, category=category, limit=_INJECT_LIMIT
        )
        return format_experience_block(lessons)
    except Exception:  # noqa: BLE001 — injection is best-effort
        logger.debug("experience injection failed (non-fatal)", exc_info=True)
        return ""
