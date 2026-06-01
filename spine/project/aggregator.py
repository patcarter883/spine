"""SPINE project coverage aggregator — read-only, deterministic, no LLM.

Given a :class:`ProjectSpec`, this walks the project's member work_ids, reads each
one's latest checkpointed ``WorkflowState`` (verification result + specification),
and computes per-requirement coverage by pure SET LOGIC over normalized-exact
string matches. There is intentionally NO fuzzy or semantic matching: the whole
point of this layer is to replace LLM judgment with a computed invariant.

Coverage semantics (deliberate, not a bug):
- A requirement is matched to a member only when the member's specification lists
  that exact requirement text (after ``strip().casefold()``). ``"Log user activity"``
  and ``"Log user actions"`` are DIFFERENT requirements and never match.
- Coverage reflects members that have RUN AND PASSED verification. A member whose
  checkpoint state is absent — purged when its plan was approved, or never run —
  is treated as unverified (it contributes no spec and is not counted as passed),
  not as a failure.

Async note: ``CheckpointStore.get_state`` is async-only, so this function is async.
Sync callers (CLI, UIApi) wrap it with ``asyncio.run``.
"""

from __future__ import annotations

import json
from typing import Any

from spine.config import SpineConfig
from spine.models.types import ProjectSpec


def _normalize(text: str) -> str:
    """Canonical form for exact requirement matching."""
    return text.strip().casefold()


def _member_requirements(state: dict[str, Any] | None) -> set[str]:
    """Normalized requirement strings declared in a member's specification.

    Returns an empty set when the state is absent or carries no parseable
    specification — such a member simply covers no requirements.
    """
    if not state:
        return set()
    raw = state.get("specification_json")
    spec: Any = None
    if isinstance(raw, dict):
        spec = raw
    elif isinstance(raw, str) and raw.strip():
        try:
            spec = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return set()
    if not isinstance(spec, dict):
        return set()
    reqs = spec.get("requirements") or []
    return {_normalize(r) for r in reqs if isinstance(r, str) and r.strip()}


def _member_passed(state: dict[str, Any] | None) -> bool:
    """True only when the member ran verification and it passed."""
    return bool(state and state.get("verification_passed") is True)


async def aggregate_project_coverage(
    spec: ProjectSpec, config: SpineConfig
) -> dict[str, Any]:
    """Compute deterministic coverage + phase rollup for a project.

    Returns a dict::

        {
          "project_id": str,
          "total_members": int,
          "members_with_state": int,
          "verified_members": int,
          "requirements": [
            {"id", "text", "status", "covering": [...], "verified": [...]}
          ],
          "summary": {"satisfied": int, "partial": int, "unsatisfied": int},
          "phases": [
            {"id", "title", "status", "requirement_ids": [...]}
          ],
        }
    """
    from spine.persistence.checkpoint import CheckpointStore

    store = CheckpointStore(db_path=config.checkpoint_path)
    states: dict[str, dict[str, Any] | None] = {}
    try:
        for wid in spec.member_work_ids:
            states[wid] = await store.get_state(wid)
    finally:
        await store.close()

    # Pre-compute per-member facts once.
    member_reqs = {wid: _member_requirements(st) for wid, st in states.items()}
    passed = {wid for wid, st in states.items() if _member_passed(st)}

    requirements: list[dict[str, Any]] = []
    status_by_req: dict[str, str] = {}
    counts = {"satisfied": 0, "partial": 0, "unsatisfied": 0}

    for req in spec.requirements:
        target = _normalize(req.text)
        covering = [wid for wid, reqs in member_reqs.items() if target in reqs]
        verified = [wid for wid in covering if wid in passed]

        if covering and set(covering) <= set(verified):
            status = "satisfied"
        elif verified:
            status = "partial"
        else:
            status = "unsatisfied"

        counts[status] += 1
        status_by_req[req.id] = status
        requirements.append(
            {
                "id": req.id,
                "text": req.text,
                "status": status,
                "covering": sorted(covering),
                "verified": sorted(verified),
            }
        )

    phases: list[dict[str, Any]] = []
    for phase in spec.roadmap.phases:
        statuses = [status_by_req.get(rid, "unsatisfied") for rid in phase.requirement_ids]
        if statuses and all(s == "satisfied" for s in statuses):
            phase_status = "complete"
        elif any(s in ("satisfied", "partial") for s in statuses):
            phase_status = "in_progress"
        else:
            phase_status = "pending"
        phases.append(
            {
                "id": phase.id,
                "title": phase.title,
                "status": phase_status,
                "requirement_ids": list(phase.requirement_ids),
            }
        )

    return {
        "project_id": spec.id,
        "total_members": len(spec.member_work_ids),
        "members_with_state": sum(1 for st in states.values() if st),
        "verified_members": len(passed),
        "requirements": requirements,
        "summary": counts,
        "phases": phases,
    }
