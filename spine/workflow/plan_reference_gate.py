"""Deterministic reference-symbol gate for PLAN critic review.

Trace 019f2077: the plan critic burned four rework rounds on plans whose
``reference_symbols`` named UIApi methods that do not exist
(``UIApi.get_llm_providers`` for the real ``UIApi.get_providers``), and the
underlying blocker — the spec demands embedding/reranker provider UI while its
``scope_exclusions`` forbid the UIApi changes that requires — was never
surfaced as a spec_contradiction, so the loop stagnated into a human review
with a still-broken plan.

This gate closes both holes at critic_plan time, with no LLM call:

* Every slice ``reference_symbols`` entry must resolve somewhere: in the
  codebase index, in some slice's ``provides`` (the cross-slice contract), or
  as an obvious external-library name. Anything else is DANGLING → a
  deterministic NEEDS_REVISION naming the exact symbol, with a "did you mean"
  suggestion fuzzy-matched from the real symbols of the same owner class and
  the slice's target files.
* A dangling symbol whose DIRECT owner is protected by a spec
  ``scope_exclusions`` bullet (e.g. ``UIApi.add_embedding_provider`` under "No
  changes to SpineConfig or UIApi schemas") cannot be fixed by reworking the
  plan — creating it is excluded. When such a symbol stays dangling on a
  SECOND consecutive round despite the gate naming it exactly, the gate
  escalates NEEDS_REVIEW with ``blocker_category='spec_contradiction'``, which
  the critic result mapper already routes to a SPECIFY amendment instead of
  another futile plan rework. Persistence is required and only the final
  owner segment is exclusion-matched — a single round's fuzzy match on a
  generic package prefix parked run 019f2104 on a fabricated contradiction.
* ``provides`` entries must be NEW symbols (the field's documented contract).
  A slice that "provides" a symbol already in the codebase is respecifying
  live code — run 019f20e0 provided the existing ``UIApi.get_config`` and its
  workers then invented exception-raising acceptance criteria the existing
  fail-safe convention contradicts, deadlocking implement/verify. Flagged as
  NEEDS_REVISION with the real definition's location.

Symbol names are normalized before matching: planners have emitted
``file.py:symbol`` / ``slice-id:symbol`` forms instead of dotted names.

The gate fails open on every lookup problem (no index, no db, query error):
a missing index must never manufacture a violation.
"""

from __future__ import annotations

import difflib
import json
import logging
from typing import Any

from spine.agents.plan_synthesis import (
    _is_external_reference,
    _leaf,
    _symbol_exists_in_index,
)
from spine.models.enums import ReviewStatus
from spine.workflow.critic_review import _norm_tokens

logger = logging.getLogger(__name__)

# difflib cutoff for a "did you mean" candidate. 0.6 accepts
# get_llm_providers→get_providers while rejecting unrelated names.
_NEAR_MISS_CUTOFF = 0.6

# At most this many indexed files are scanned for near-miss candidates per
# dangling symbol (the owner class's file(s) plus the slice's target files).
_MAX_CANDIDATE_FILES = 4


def _normalize_symbol(sym: str) -> str:
    """Strip planner format drift from a symbol name.

    Plans have emitted ``spine/ui_api/api.py:get_config`` and
    ``slice-id:render_form`` instead of dotted qualified names (run 019f20e0).
    The part after the last ``:`` is the actual symbol; call parens are
    dropped. Plain dotted names pass through unchanged.
    """
    s = (sym or "").strip().split("(", 1)[0].strip()
    if ":" in s:
        s = s.rsplit(":", 1)[-1].strip()
    return s


def _owner_segments(sym: str) -> list[str]:
    """Qualifier segments of a dotted symbol, leaf excluded.

    ``'spine.ui_api.UIApi.get_providers'`` → ``['spine', 'ui_api', 'UIApi']``;
    an unqualified name has no owner and returns ``[]``.
    """
    s = (sym or "").strip().split("(", 1)[0].strip()
    parts = [p for p in s.split(".") if p]
    return parts[:-1]


def _find_symbol_files(db_path: str | None, name: str) -> list[str]:
    """File paths the index lists for *name*, ``[]`` on any failure."""
    if not db_path or not name:
        return []
    try:
        from spine.agents.tools.codebase_query import find_symbol

        raw = find_symbol(db_path, name)
    except Exception:  # noqa: BLE001 — index unavailable ⇒ no candidates
        return []
    if not raw:
        return []
    try:
        matches = json.loads(raw).get("matches") or []
    except (ValueError, AttributeError):
        return []
    out: list[str] = []
    for m in matches:
        fp = m.get("file_path")
        if fp and fp not in out:
            out.append(fp)
    return out


def _list_file_symbols(db_path: str | None, file_path: str) -> list[str]:
    """Indexed symbol names for *file_path*, ``[]`` on any failure."""
    if not db_path or not file_path:
        return []
    try:
        from spine.agents.tools.codebase_query import list_file_symbols

        return list_file_symbols(db_path, file_path) or []
    except Exception:  # noqa: BLE001
        return []


def _near_miss(
    db_path: str | None, ref: str, target_files: list[str]
) -> str | None:
    """Closest REAL symbol to the dangling *ref*, or None.

    Candidate pool: every indexed symbol in (a) the file(s) where the ref's
    owner class actually lives — found via the index, since the owner usually
    lives outside the slice's own target files — and (b) the slice's target
    files. Matching compares leaf names so ``UIApi.get_llm_providers`` lands on
    ``UIApi.get_providers`` regardless of how the index qualifies it.
    """
    owner = _owner_segments(ref)
    files: list[str] = []
    if owner:
        files.extend(_find_symbol_files(db_path, owner[-1]))
    for f in target_files or []:
        if f and f not in files:
            files.append(f)

    by_leaf: dict[str, str] = {}
    for fp in files[:_MAX_CANDIDATE_FILES]:
        for cand in _list_file_symbols(db_path, fp):
            leaf = _leaf(cand)
            if leaf:
                by_leaf.setdefault(leaf, cand)
    if not by_leaf:
        return None

    close = difflib.get_close_matches(
        _leaf(ref), list(by_leaf), n=1, cutoff=_NEAR_MISS_CUTOFF
    )
    return by_leaf[close[0]] if close else None


def _matching_exclusion(ref: str, exclusions: list[str]) -> str | None:
    """The scope_exclusions bullet protecting *ref*'s DIRECT owner, or None.

    A dangling ``UIApi.add_embedding_provider`` matches "No changes to
    SpineConfig or UIApi schemas" because the direct owner ``UIApi`` appears
    (as a token) in the bullet. Only the FINAL owner segment is matched:
    intermediate package segments are far too generic — run 019f2104 falsely
    matched the ``spine`` package prefix of
    ``spine.ui._pages.config_view.st`` against an exclusion that merely
    contained the word "spine", parking the whole run on a fabricated
    contradiction. Unqualified refs never match — a bare name carries too
    little signal to pin a contradiction on the spec.
    """
    owner = _owner_segments(ref)
    if not owner or len(owner[-1]) <= 2:
        return None
    direct = owner[-1].lower()
    for ex in exclusions:
        if direct in _norm_tokens(ex):
            return ex
    return None


def _scope_exclusions(specification_json: str | None) -> list[str]:
    if not specification_json or not isinstance(specification_json, str):
        return []
    try:
        spec = json.loads(specification_json)
    except (ValueError, TypeError):
        return []
    if not isinstance(spec, dict):
        return []
    return [str(x) for x in (spec.get("scope_exclusions") or []) if str(x).strip()]


def check_reference_symbols(
    plan_data: dict[str, Any],
    specification_json: str | None,
    prior_gate: dict[str, Any] | None,
    *,
    db_path: str | None,
) -> dict[str, Any] | None:
    """Validate every slice's ``reference_symbols`` against ground truth.

    Returns None when everything resolves (or the index is unavailable —
    ``_symbol_exists_in_index`` is permissive on failure, so a missing index
    disables the gate rather than flunking every plan). Otherwise returns a
    review-shaped dict (status / tier / reason / suggestions /
    blocker_category / cited_exclusions) plus ``dangling_leafs``, which the
    caller persists so the NEXT round can detect symbols that stayed dangling
    despite this round's exact feedback.

    ``prior_gate`` is the previous round's gate result for this phase (or
    None/{} on the first round).
    """
    slices = plan_data.get("feature_slices") or []
    if not isinstance(slices, list):
        return None

    provided_leafs: set[str] = set()
    for s in slices:
        if isinstance(s, dict):
            for p in s.get("provides") or []:
                leaf = _leaf(_normalize_symbol(str(p)))
                if leaf:
                    provided_leafs.add(leaf)

    exclusions = _scope_exclusions(specification_json)
    prior_leafs = set((prior_gate or {}).get("dangling_leafs") or [])

    dangling: list[dict[str, Any]] = []
    redefined: list[dict[str, Any]] = []
    for s in slices:
        if not isinstance(s, dict):
            continue
        sid = s.get("id", "?")
        target_files = [f for f in (s.get("target_files") or []) if f]
        for ref in s.get("reference_symbols") or []:
            ref = _normalize_symbol(str(ref))
            if not ref:
                continue
            leaf = _leaf(ref)
            # External-library names and Python builtins are not contracts —
            # bare aliases ('st.form'), module-qualified imports
            # ('spine.ui._pages.config_view.st', run 019f2104), and research
            # "Calls:" artifacts like 'open' / 'logger.exception' (run
            # 019f34b7: two rework rounds burned on those two names).
            if _is_external_reference(ref):
                continue
            if _symbol_exists_in_index(db_path, ref):
                continue
            if leaf in provided_leafs:
                continue  # cross-slice contract — synthesis validation owns it
            dangling.append(
                {
                    "slice": sid,
                    "symbol": ref,
                    "leaf": leaf,
                    "near_miss": _near_miss(db_path, ref, target_files),
                    "exclusion": _matching_exclusion(ref, exclusions),
                }
            )
        # `provides` must name NEW symbols — that is the field's documented
        # contract. A slice that "provides" something already in the codebase
        # is respecifying live code: run 019f20e0 provided UIApi.get_config /
        # get_providers (which exist) and its workers then wrote acceptance
        # criteria inventing exception semantics the existing fail-safe
        # convention contradicts — an unimplementable deadlock the editor can
        # never satisfy. Flag it so the planner either names the new symbol
        # distinctly or declares a modification of the existing one via
        # reference_symbols. Uses an exact/qualified index lookup (no leaf
        # fallback) so a new method that merely shares a short name with some
        # unrelated existing symbol is not misflagged.
        for p in s.get("provides") or []:
            sym = _normalize_symbol(str(p))
            if not sym:
                continue
            files = _find_symbol_files(db_path, sym)
            if files:
                redefined.append({"slice": sid, "symbol": sym, "file": files[0]})

    if not dangling and not redefined:
        return None

    # A contradiction is a dangling symbol the plan cannot legally acquire:
    # its owner is exclusion-protected AND it already survived a full rework
    # round in which the gate named it exactly. Persistence is REQUIRED — a
    # single round's fuzzy exclusion match is not enough evidence to park the
    # whole run (run 019f2104: a first-round false positive escalated a
    # fabricated contradiction straight to human review). A true
    # contradiction costs one extra ~2-minute round; a false park costs the
    # run.
    contradictions = [
        d for d in dangling if d["exclusion"] and d["leaf"] in prior_leafs
    ]

    lines: list[str] = []
    suggestions: list[str] = []
    for d in dangling:
        msg = (
            f"slice '{d['slice']}' references '{d['symbol']}', which does not "
            f"exist in the codebase index and is not provided by any slice"
        )
        if d["near_miss"]:
            msg += f" — did you mean '{d['near_miss']}'?"
            suggestions.append(
                f"Replace '{d['symbol']}' with the existing symbol "
                f"'{d['near_miss']}' (or add the new name to a producer "
                f"slice's `provides` and depend on that slice)."
            )
        else:
            suggestions.append(
                f"Reference only existing symbols in slice '{d['slice']}', or "
                f"add '{d['symbol']}' to a producer slice's `provides` and "
                f"depend on that slice."
            )
        if d["exclusion"] and d["leaf"] not in prior_leafs:
            msg += (
                f" NOTE: creating '{d['symbol']}' appears to be excluded by "
                f"scope_exclusions ('{d['exclusion']}') — if this reference "
                "persists unresolved next round, the gate will escalate a "
                "spec contradiction."
            )
        lines.append(msg)
    for r in redefined:
        lines.append(
            f"slice '{r['slice']}' declares '{r['symbol']}' in `provides`, but "
            f"that symbol ALREADY EXISTS in the codebase ({r['file']}) — "
            f"`provides` is only for NEW symbols"
        )
        suggestions.append(
            f"In slice '{r['slice']}': if the slice MODIFIES the existing "
            f"'{r['symbol']}', move it to `reference_symbols` and state the "
            f"modification (matching the existing signature and error-handling "
            f"convention) in execution_requirements; if it creates something "
            f"new, give it a name that does not collide with {r['file']}."
        )

    if contradictions:
        cited = sorted({d["exclusion"] for d in contradictions})
        symbols = sorted({d["symbol"] for d in contradictions})
        reason = (
            "Deterministic reference-symbol gate: the plan requires "
            f"symbol(s) {symbols} that do not exist in the codebase, and the "
            f"specification's scope_exclusions {cited} forbid creating them. "
            "No plan rework can satisfy both — the specification must be "
            "amended (allow the API change, or drop/relax the requirement "
            "that needs it). Full findings:\n"
            + "\n".join(f"- {ln}" for ln in lines)
        )
        result = {
            "status": ReviewStatus.NEEDS_REVIEW.value,
            "tier": "structural",
            "reason": reason,
            "suggestions": suggestions,
            "blocker_category": "spec_contradiction",
            "cited_exclusions": cited,
        }
    else:
        n = len(dangling) + len(redefined)
        reason = (
            "Deterministic reference-symbol gate: "
            f"{n} symbol contract violation{'' if n == 1 else 's'}:\n"
            + "\n".join(f"- {ln}" for ln in lines)
        )
        result = {
            "status": ReviewStatus.NEEDS_REVISION.value,
            "tier": "structural",
            "reason": reason,
            "suggestions": suggestions,
            "blocker_category": None,
            "cited_exclusions": [],
        }

    result["dangling_leafs"] = sorted({d["leaf"] for d in dangling})
    logger.warning(
        "plan reference-symbol gate: %d dangling symbol(s), %d redefined "
        "provides, %d spec contradiction(s) → %s",
        len(dangling),
        len(redefined),
        len(contradictions),
        result["status"],
    )
    return result
