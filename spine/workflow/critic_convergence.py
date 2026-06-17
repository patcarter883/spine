"""Critic convergence detection — measure whether rework rounds are progressing.

The critic-retry loop re-runs a phase (specify/plan) up to ``max_retries``
times whenever the critic returns NEEDS_REVISION.  That budget is wasted when
the author keeps producing a plan the critic rejects for the *same* reasons
round after round (trace 019ed383: three plan attempts, the critic's three
core asks — split the slice, add the getters, resolve the schema mismatch —
recurring verbatim, never addressed, then a forced human escalation).

This module provides deterministic (no-LLM) helpers to:

* decide whether a new critic verdict substantially *repeats* the prior one
  (``is_repeat_verdict``), so the loop can short-circuit a stagnating retry
  budget instead of burning every attempt; and
* compute exactly which prior asks are *still unaddressed*
  (``unaddressed_points``), so the rework prompt can tell the author "you did
  NOT address X last time" rather than re-listing the whole verdict.

A "point" is one feedback ask — a critic suggestion, or the verdict reason
when no suggestions were given.  Two points are considered the same ask when
their normalized content-word sets overlap (Jaccard) at or above
``POINT_MATCH_THRESHOLD``.  A verdict repeats the prior one when at least
``REPEAT_VERDICT_RATIO`` of the prior points recur in it.
"""

from __future__ import annotations

import re
from typing import Any

# A new NEEDS_REVISION verdict whose content-word overlap with a prior point
# meets this Jaccard threshold is treated as the *same* ask recurring.
POINT_MATCH_THRESHOLD = 0.6

# A verdict is a "repeat" of the prior verdict when at least this fraction of
# the prior verdict's points recur in it.
REPEAT_VERDICT_RATIO = 0.5

# Number of *consecutive* repeat verdicts that trips early escalation. With a
# limit of 2: the first repeat is a warning round (the author still has the
# explicit "still unaddressed" delta to act on); a second consecutive repeat
# means the loop is not converging and is escalated regardless of remaining
# retry budget.
STAGNATION_LIMIT = 2

# Short, high-frequency words carry no signal for "is this the same ask"; they
# would inflate Jaccard overlap between unrelated points. Kept deliberately
# small — domain words (config, slice, schema, provider…) must survive.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "if", "then", "this", "that",
        "these", "those", "is", "are", "was", "were", "be", "been", "being",
        "to", "of", "in", "on", "for", "with", "as", "at", "by", "from",
        "it", "its", "into", "should", "must", "needs", "need", "add",
        "adding", "ensure", "make", "use", "using", "which", "not", "no",
        "do", "does", "via", "per", "each", "all", "any", "you", "your",
    }
)

_WORD_RE = re.compile(r"[a-z0-9_]+")


def _tokens(text: str) -> set[str]:
    """Lower-case content-word set for a feedback point.

    Snake_case / dotted identifiers are kept whole (``get_embedding_providers``,
    ``phase_max_retries``) because those are the load-bearing tokens that
    distinguish one critic ask from another.
    """
    if not text:
        return set()
    words = _WORD_RE.findall(text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _strong_tokens(tokens: set[str]) -> set[str]:
    """Domain identifiers (snake_case symbols like ``get_embedding_providers``).

    A shared one is a near-certain "same ask" signal — two critic points that
    both name ``phase_max_retries`` are about the same thing regardless of how
    much qualifier prose surrounds them.
    """
    return {t for t in tokens if "_" in t}


def _point_similarity(a: str, b: str) -> float:
    """Similarity of two feedback points in 0.0–1.0.

    Base measure is Jaccard overlap of content-word sets. When the two points
    share a domain identifier (snake_case symbol), the score is floored at the
    match threshold — a recurring symbol reference shouldn't be missed just
    because one round wrapped it in extra qualifier words.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    jaccard = inter / union if union else 0.0
    if _strong_tokens(ta) & _strong_tokens(tb):
        return max(jaccard, POINT_MATCH_THRESHOLD)
    return jaccard


def review_points(review: dict[str, Any] | None) -> list[str]:
    """Extract the comparable asks from a critic verdict.

    Prefers the explicit ``suggestions`` list; falls back to a single-element
    list holding the ``reason`` when no suggestions were emitted. Empty/blank
    entries are dropped.
    """
    if not review:
        return []
    suggestions = [
        s.strip()
        for s in (review.get("suggestions") or [])
        if isinstance(s, str) and s.strip()
    ]
    if suggestions:
        return suggestions
    reason = (review.get("reason") or "").strip()
    return [reason] if reason else []


def _point_recurs(point: str, candidates: list[str], threshold: float) -> bool:
    """True when ``point`` matches any candidate at/above ``threshold``."""
    return any(_point_similarity(point, c) >= threshold for c in candidates)


def unaddressed_points(
    prior_review: dict[str, Any] | None,
    current_review: dict[str, Any] | None,
    *,
    threshold: float = POINT_MATCH_THRESHOLD,
) -> list[str]:
    """Prior asks that still appear in the current verdict.

    Returns the *prior* point text (the author's own earlier ask) for every
    prior point that recurs in the current verdict's points — i.e. the asks
    the rework round failed to resolve. The current reason is also folded into
    the candidate set so an ask that moved from a suggestion to the headline
    reason still counts as recurring.
    """
    prior = review_points(prior_review)
    if not prior:
        return []
    current = review_points(current_review)
    cur_reason = (current_review or {}).get("reason", "")
    if cur_reason and cur_reason not in current:
        current = [*current, cur_reason]
    return [p for p in prior if _point_recurs(p, current, threshold)]


def is_repeat_verdict(
    prior_review: dict[str, Any] | None,
    current_review: dict[str, Any] | None,
    *,
    threshold: float = POINT_MATCH_THRESHOLD,
    ratio: float = REPEAT_VERDICT_RATIO,
) -> bool:
    """True when the current verdict substantially repeats the prior one.

    Defined as: at least ``ratio`` of the prior verdict's points recur in the
    current verdict. With no prior points there is nothing to repeat, so the
    result is ``False`` (the first revision round is never "stagnant").
    """
    prior = review_points(prior_review)
    if not prior:
        return False
    recurring = unaddressed_points(prior_review, current_review, threshold=threshold)
    return (len(recurring) / len(prior)) >= ratio


def next_stagnation_streak(
    prior_review: dict[str, Any] | None,
    current_review: dict[str, Any] | None,
) -> int:
    """Updated consecutive-repeat streak after the current verdict.

    Reads the prior streak off ``prior_review['stagnation_streak']`` (0 when
    absent). Increments it when the current verdict repeats the prior one;
    resets to 0 otherwise.
    """
    if is_repeat_verdict(prior_review, current_review):
        return int((prior_review or {}).get("stagnation_streak", 0) or 0) + 1
    return 0
