"""Reject slice-verifier verdicts that are internally incoherent.

agripath probe 27 (run cc9c2611): the ``database-migrations-farm`` slice came
back NOT_VERIFIED on 15 criteria, and 14 of the 15 ``detail`` strings were
BYTE-IDENTICAL::

    "The file is named ContactController but the class is actually named
     ContactController in the file."

That sentence asserts X is not X. It was applied to criteria about migrations
("php artisan migrate succeeds", "JSONB column type used for metadata"), and
seven of the criteria being judged belonged to a different slice entirely. The
judge had the wrong file in context and then repeated one non-reason across the
whole checklist — the documented repetition behaviour of this model surfacing
in the judge rather than in an editor.

The cost is not one bad verdict. A slice is never re-judged within a cycle
(``_run_slice_verifier_node`` makes exactly one call per slice), so a bogus
NOT_VERIFIED costs a full gap_plan + re-implement + re-verify cycle, consumes
one of the bounded verify attempts, and — because ``_total_gap_count`` counts
every ``passed=False`` entry — inflates the convergence arithmetic the
patience/ratchet logic depends on. In probe 27 the fabricated reason
propagated into the next three cycles' gap plans as the rework instruction.

This gate does NOT promote anything. Marking the entries ``passed=True`` would
hit ``_reconcile_verdict``'s all-passed shortcut and silently flip the slice to
VERIFIED on the strength of a verdict we just declared untrustworthy. Instead
it collapses the fabricated entries into a single honest failure, so the slice
still fails but the rework loop is told the verdict was unreliable rather than
being handed 15 phantom fixes to chase.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

# A detail repeated across at least this many entries, AND across at least this
# share of the checklist, is boilerplate rather than per-criterion reasoning.
# Probe 27 was 14/15 (93%). A healthy verdict repeats a detail at most
# incidentally — two criteria can legitimately share one cause.
_MIN_REPEATS = 3
_MIN_REPEAT_SHARE = 0.6

# "named X but ... named X" — the contrast conjunction with the SAME identifier
# on both sides. Deliberately narrow: it must be the same token, so a genuine
# "named Foo but the class is named Bar" is untouched.
_CONTRAST = re.compile(
    r"\b(?:named|called)\s+[`'\"]?(\w{3,})[`'\"]?\b"
    r"(?P<mid>(?:(?!\bnamed\b|\bcalled\b).){0,120}?)"
    r"\b(?:but|however|whereas|although)\b"
    r"(?:(?!\bnamed\b|\bcalled\b).){0,120}?"
    r"\b(?:named|called)\s+[`'\"]?(\w{3,})[`'\"]?\b",
    re.IGNORECASE | re.DOTALL,
)


def is_self_contradictory(detail: str) -> bool:
    """True when *detail* contrasts an identifier against itself.

    ``"The file is named ContactController but the class is actually named
    ContactController"`` states X is not X, which is never a real finding.
    """
    if not detail:
        return False
    for m in _CONTRAST.finditer(detail):
        if m.group(1) == m.group(3):
            return True
    return False


def _repeated_detail(checklist: list[dict]) -> tuple[str, int] | None:
    """The boilerplate detail and its count, when one dominates the checklist."""
    details = [
        (e.get("detail") or "").strip()
        for e in checklist
        if isinstance(e, dict) and not e.get("passed") and (e.get("detail") or "").strip()
    ]
    if len(details) < _MIN_REPEATS:
        return None
    top, count = Counter(details).most_common(1)[0]
    if count < _MIN_REPEATS or count / len(checklist) < _MIN_REPEAT_SHARE:
        return None
    return top, count


def reject_unreliable_verdict(
    verification_result: dict[str, Any],
    work_id: str = "unknown",
    slice_id: str = "unknown",
) -> int:
    """Collapse a self-contradictory or copy-pasted verdict, in place.

    Returns the number of checklist entries collapsed, 0 when the verdict looks
    sane. Never promotes a slice and never re-calls the model — a slice cannot
    be re-judged within a cycle, so the only safe move is to stop the fabricated
    detail from becoming the rework loop's instruction.
    """
    checklist = verification_result.get("checklist")
    if not isinstance(checklist, list) or len(checklist) < _MIN_REPEATS:
        return 0

    repeated = _repeated_detail(checklist)
    if repeated is None:
        return 0
    detail, count = repeated

    # Repetition alone is not proof — two criteria can share one real cause.
    # Require the repeated text to also be incoherent on its face.
    if not is_self_contradictory(detail):
        return 0

    failed = [e for e in checklist if isinstance(e, dict) and not e.get("passed")]
    criteria = [str(e.get("criterion", "?")) for e in failed]

    note = (
        f"REJECTED by verdict gate: the judge returned the same self-contradictory "
        f"reason for {count} of {len(checklist)} criteria — {detail!r} — which "
        f"asserts a thing is not itself. The verdict for this slice is unreliable "
        f"and its per-criterion reasons must not be treated as findings. "
        f"Re-verify this slice; do not synthesise fixes from the text above. "
        f"Affected criteria: {'; '.join(criteria[:8])}"
        + (f" (+{len(criteria) - 8} more)" if len(criteria) > 8 else "")
    )

    # Keep exactly one failing entry so the slice stays NOT_VERIFIED (a
    # zero-failure checklist would trip the all-passed promotion), and drop the
    # rest so _total_gap_count is not inflated by fabrications.
    kept = [e for e in checklist if isinstance(e, dict) and e.get("passed")]
    kept.append({"criterion": "verdict reliability", "passed": False, "detail": note})
    verification_result["checklist"] = kept
    verification_result["verdict"] = "NOT_VERIFIED"
    verification_result["gaps"] = [note]
    verification_result["recommendations"] = [
        "Re-run verification for this slice; the previous verdict was incoherent."
    ]

    logger.warning(
        "[%s] Slice-verifier %r: verdict gate rejected an incoherent verdict — "
        "%d of %d criteria shared one self-contradictory reason; collapsed to a "
        "single re-verify finding",
        work_id, slice_id, count, len(checklist),
    )
    return len(failed)
