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

# Number of *consecutive* goalpost-shift verdicts that trips early escalation.
# A goalpost shift is the OPPOSITE failure to stagnation: each NEEDS_REVISION
# round raises a wholly *new* set of asks instead of re-raising the prior ones
# (trace 019f01c2: the plan critic rejected a structurally-valid plan five
# times, every round with fresh re-slice/consolidate nitpicks, so the
# repeat-based stagnation streak never moved off 0 and the loop burned the
# entire retry budget). Detecting churn caps the loop at ~3 attempts: round 1
# sets the baseline, two consecutive shifts mean the critic is moving the
# goalposts rather than driving convergence.
CHURN_LIMIT = 2

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


def is_goalpost_shift(
    prior_review: dict[str, Any] | None,
    current_review: dict[str, Any] | None,
    *,
    threshold: float = POINT_MATCH_THRESHOLD,
    ratio: float = REPEAT_VERDICT_RATIO,
) -> bool:
    """True when the current verdict moves the goalposts off the prior one.

    A goalpost shift is a NEEDS_REVISION round that, against a real prior
    verdict, raises a substantially *new* set of asks rather than repeating
    the prior ones. It is the mirror image of :func:`is_repeat_verdict`:

    * no prior points → nothing to shift from → ``False`` (the first revision
      round is never a shift);
    * a repeat verdict is stagnation, not churn → ``False``;
    * otherwise (a real prior verdict whose asks did NOT recur) → ``True``.
    """
    if not review_points(prior_review):
        return False
    return not is_repeat_verdict(
        prior_review, current_review, threshold=threshold, ratio=ratio
    )


def next_churn_streak(
    prior_review: dict[str, Any] | None,
    current_review: dict[str, Any] | None,
) -> int:
    """Updated consecutive-goalpost-shift streak after the current verdict.

    Reads the prior streak off ``prior_review['churn_streak']`` (0 when
    absent). Increments it when the current verdict shifts the goalposts off
    the prior one; resets to 0 otherwise (no prior, or a repeat — repeats are
    owned by :func:`next_stagnation_streak`).
    """
    if is_goalpost_shift(prior_review, current_review):
        return int((prior_review or {}).get("churn_streak", 0) or 0) + 1
    return 0


def compute_streaks(
    prior_review: dict[str, Any] | None,
    current_review: dict[str, Any] | None,
    current_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Source-aware convergence accounting for one NEEDS_REVISION round.

    Verdicts reach the critic loop from three sources — the LLM agent critic
    (``verdict_source`` absent or ``"agent"``), harness guards like the
    truncation fallback (``"guard"``), and the deterministic plan validators
    (``"gate"``). Comparing verdicts ACROSS sources manufactures goalpost
    shifts: a guard notice shares no asks with the agent verdict before it,
    nor with the gate findings after it, so both transitions score as churn
    (trace 019f260c: agent → truncation guard → reference gate hit
    CHURN_LIMIT and parked a converging plan while the agent critic voted
    PASSED). Streaks therefore only advance within a single source chain:

    * ``guard`` verdicts freeze both streaks and carry the last real agent
      baseline through untouched — harness noise is not evidence either way;
    * ``gate`` verdicts compare against the prior round's deterministic
      outcome (``prior_review['reference_gate']``, falling back to the prior
      verdict itself when that round was gate-sourced). A first-time gate
      finding freezes the streaks; recurring violations stagnate; shifting
      violations churn;
    * ``agent`` verdicts compare against the last agent-sourced revision ask
      (``streak_baseline``), carried through intervening guard/gate rounds.

    Returns the new ``stagnation_streak``, ``churn_streak``,
    ``unaddressed_points``, and the ``streak_baseline`` dict to store on the
    round's ``last_critic_review``.
    """
    cur = current_review or {}
    prior = prior_review or {}
    cur_source = cur.get("verdict_source") or "agent"
    prior_source = prior.get("verdict_source") or "agent"
    prior_stag = int(prior.get("stagnation_streak", 0) or 0)
    prior_churn = int(prior.get("churn_streak", 0) or 0)

    # The last real agent ask-set as of the prior round: the prior verdict
    # itself when it was an agent revision ask, else whatever baseline it
    # carried forward. Status literals match ReviewStatus values; kept as
    # strings so this module stays import-free.
    if prior_source == "agent":
        if prior.get("status") in ("needs_revision", "needs_review") and review_points(prior):
            agent_baseline: dict[str, Any] = {
                "reason": prior.get("reason", ""),
                "suggestions": list(prior.get("suggestions") or []),
            }
        else:
            agent_baseline = {}
    else:
        agent_baseline = prior.get("streak_baseline") or {}

    frozen = {
        "stagnation_streak": prior_stag,
        "churn_streak": prior_churn,
        "unaddressed_points": [],
    }

    if cur_source == "guard":
        return {**frozen, "streak_baseline": agent_baseline}

    if cur_source == "gate":
        # An agent PASS overridden by the gate means the agent ask-chain
        # converged — drop the stale baseline so a later agent revision is a
        # fresh round, not a phantom goalpost shift off long-addressed asks.
        baseline = {} if cur.get("agent_status") == "passed" else agent_baseline
        prior_gate = prior.get("reference_gate") or {}
        if review_points(prior_gate):
            cmp_prior = prior_gate
        elif prior_source == "gate":
            # Deterministic verdict without a gate record (e.g. the
            # structural validator): its findings ARE the prior lcr points.
            cmp_prior = {
                "reason": prior.get("reason", ""),
                "suggestions": list(prior.get("suggestions") or []),
            }
        else:
            # First-time deterministic finding: not evidence the loop is
            # stagnating or churning.
            return {**frozen, "streak_baseline": baseline}
        cmp_prior = {
            **cmp_prior,
            "stagnation_streak": prior_stag,
            "churn_streak": prior_churn,
        }
        cmp_cur = current_gate if review_points(current_gate) else cur
        return {
            "stagnation_streak": next_stagnation_streak(cmp_prior, cmp_cur),
            "churn_streak": next_churn_streak(cmp_prior, cmp_cur),
            "unaddressed_points": unaddressed_points(cmp_prior, cmp_cur),
            "streak_baseline": baseline,
        }

    # Agent verdict: compare against the carried agent baseline, resuming the
    # streak values from wherever the chain left off.
    cmp_prior = (
        {
            **agent_baseline,
            "stagnation_streak": prior_stag,
            "churn_streak": prior_churn,
        }
        if review_points(agent_baseline)
        else {}
    )
    return {
        "stagnation_streak": next_stagnation_streak(cmp_prior, cur),
        "churn_streak": next_churn_streak(cmp_prior, cur),
        "unaddressed_points": unaddressed_points(cmp_prior, cur),
        "streak_baseline": {
            "reason": cur.get("reason", ""),
            "suggestions": list(cur.get("suggestions") or []),
        },
    }
