"""Neutralise slice-verifier verdicts that are internally incoherent.

agripath probe 27 (run cc9c2611): the ``database-migrations-farm`` slice came
back NOT_VERIFIED on 15 criteria, and 14 of the 15 ``detail`` strings were
BYTE-IDENTICAL::

    "The file is named ContactController but the class is actually named
     ContactController in the file."

That sentence asserts X is not X. It was applied to criteria about migrations
("php artisan migrate succeeds", "JSONB column type used for metadata"), and
seven of the criteria being judged belonged to a different slice. The judge had
the wrong file in context and then repeated one non-reason across the whole
checklist — this model's documented repetition behaviour surfacing in the judge
rather than in an editor.

A slice is never re-judged within a cycle, so the fabrication became the rework
loop's instruction for three more cycles.

WHAT THIS GATE MUST NOT DO
--------------------------

*Change how many criteria are failing.* An earlier version collapsed the
checklist to a single entry, and adversarial review proved that strictly worse
than no gate at all: ``compose._total_gap_count`` counts failing entries, and
``_verify_result_mapper`` feeds that count to the best-state ratchet. Reporting
1 instead of 15 made the FIRST, unfixed cycle look like the best state, so every
honest later cycle scored as a regression and ``restore_best`` deleted the real
fixes from disk. Measured against the production functions: totals went
``[1, 6, 4]`` with the workspace reverted to cycle-1 code and the run parked,
versus ``[15, 6, 4, 2]`` converging with the gate off.

*Promote anything.* Marking entries ``passed=True`` would trip
``_reconcile_verdict``'s all-passed shortcut and flip the slice to VERIFIED on a
verdict just declared untrustworthy.

*Discard concurrent real findings.* Only entries whose detail IS the repeated
boilerplate are rewritten; a genuine failure sitting alongside it is untouched.

So the gate rewrites text and nothing else: same entries, same pass/fail flags,
same count — the fabricated reason is replaced with an honest one telling the
rework loop the verdict is unreliable and not to synthesise fixes from it.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

# A detail repeated across at least this many FAILING entries, and across at
# least this share of them, is boilerplate rather than per-criterion reasoning.
# Probe 27 was 14/15. The denominator is failing entries only, so interleaved
# passing entries cannot dilute the share below the threshold.
_MIN_REPEATS = 3
_MIN_REPEAT_SHARE = 0.6

# "named X ... but ... named X" — the SAME identifier on both sides of a
# contrast conjunction.
_CONTRAST = re.compile(
    r"\b(?:named|called)\s+[`'\"]?(\w{3,})[`'\"]?\b"
    r"(?:(?!\bnamed\b|\bcalled\b).){0,120}?"
    r"\b(?:but|however|whereas|although)\b"
    r"(?:(?!\bnamed\b|\bcalled\b).){0,120}?"
    r"\b(?:named|called)\s+[`'\"]?(\w{3,})[`'\"]?\b",
    re.IGNORECASE | re.DOTALL,
)

# A contrast that NEGATES is coherent, not tautological: "a column named farm_id
# exists but the model has NO property named farm_id" says something true about
# two different things. Probe 27's sentence asserts both halves positively and
# is therefore contentless. Without this, seven realistic Laravel findings were
# wrongly flagged in review.
_NEGATION = re.compile(
    r"\b(?:no|not|never|none|without|missing|absent|lacks?|lacking|"
    r"does\s?n[o']t|is\s?n[o']t|are\s?n[o']t|was\s?n[o']t|were\s?n[o']t|"
    r"cannot|can\s?not|fails?|failed|undefined|unknown)\b",
    re.IGNORECASE,
)


def _norm(detail: str) -> str:
    """Whitespace- and case-normalised key, so trivial jitter still groups."""
    return re.sub(r"\s+", " ", (detail or "").strip().lower())


def is_self_contradictory(detail: str) -> bool:
    """True when *detail* contrasts an identifier against itself, positively.

    ``"The file is named ContactController but the class is actually named
    ContactController"`` states X is not X. A negated contrast — ``"a column
    named farm_id exists but there is no property named farm_id"`` — is a real
    finding and must not match.
    """
    if not detail:
        return False
    for m in _CONTRAST.finditer(detail):
        if m.group(1).lower() != m.group(2).lower():
            continue
        # Negation anywhere in the sentence means the two halves are asserting
        # different things about the identifier, which is coherent.
        if _NEGATION.search(detail):
            return False
        return True
    return False


def _boilerplate(checklist: list[dict]) -> tuple[str, int] | None:
    """The dominant failing detail and its count, when one dominates."""
    failing = [
        e for e in checklist
        if isinstance(e, dict) and not e.get("passed") and (e.get("detail") or "").strip()
    ]
    if len(failing) < _MIN_REPEATS:
        return None
    counts = Counter(_norm(e.get("detail", "")) for e in failing)
    key, count = counts.most_common(1)[0]
    if count < _MIN_REPEATS or count / len(failing) < _MIN_REPEAT_SHARE:
        return None
    return key, count


def reject_unreliable_verdict(
    verification_result: dict[str, Any],
    work_id: str = "unknown",
    slice_id: str = "unknown",
) -> int:
    """Rewrite fabricated reasons in place, preserving every count and flag.

    Returns the number of details rewritten, 0 when the verdict looks sane.
    Never adds, removes or re-flags a checklist entry — see the module
    docstring for why changing the failing count corrupts the best-state
    ratchet.
    """
    checklist = verification_result.get("checklist")
    if not isinstance(checklist, list):
        return 0

    found = _boilerplate(checklist)
    if found is None:
        return 0
    key, count = found

    # Repetition alone is not evidence — several criteria can genuinely share
    # one cause ("the pest suite failed to run"). Require incoherence too.
    sample = next(
        (e.get("detail", "") for e in checklist
         if isinstance(e, dict) and _norm(e.get("detail", "")) == key),
        "",
    )
    if not is_self_contradictory(sample):
        return 0

    note = (
        f"UNRELIABLE VERDICT (verdict gate): the judge gave this same "
        f"self-contradictory reason for {count} criteria — {sample.strip()!r} — "
        f"which asserts a thing is not itself. Treat this slice's verdict as "
        f"unjudged: re-verify it, and do NOT synthesise fixes from the reason "
        f"above."
    )

    rewritten = 0
    for entry in checklist:
        if not isinstance(entry, dict) or entry.get("passed"):
            continue
        if _norm(entry.get("detail", "")) != key:
            continue  # a genuine concurrent finding — leave it exactly as-is
        entry["detail"] = note
        rewritten += 1

    # Replace the same fabricated text wherever it was echoed, but only if it
    # is the whole of that entry — a real gap must survive.
    for field in ("gaps", "recommendations"):
        items = verification_result.get(field)
        if isinstance(items, list) and items:
            verification_result[field] = [
                note if _norm(str(it)) == key else it for it in items
            ]

    logger.warning(
        "[%s] Slice-verifier %r: verdict gate neutralised an incoherent verdict "
        "— %d failing criteria shared one self-contradictory reason; details "
        "replaced, counts and pass/fail flags preserved",
        work_id, slice_id, rewritten,
    )
    return rewritten
