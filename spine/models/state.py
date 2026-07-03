"""SPINE workflow state — LangGraph state schema and reducers."""

from __future__ import annotations

import operator
from typing import Annotated

from typing_extensions import TypedDict


def _merge_dicts(left: dict, right: dict) -> dict:
    """Merge two dicts, with right overwriting left for overlapping keys.

    Used as a LangGraph reducer for dict-typed state fields that should
    accumulate across phases (retry_count, etc.).
    """
    merged = {**left, **right}
    return merged


# Upper bound on retained ``feedback`` entries. The list accumulates on every
# critic round, adversarial round, verify failure, gate failure, and subgraph
# error; routing reads ``last_critic_review`` (not this list) and consumers only
# read the tail or iterate, so keeping the most recent N entries bounds
# unbounded growth in deep-rework runs without losing actionable recent
# feedback (finding #9). Generous enough that normal runs are unaffected.
_MAX_FEEDBACK_ENTRIES = 40


def _append_capped_feedback(left: list, right: list) -> list:
    """Append feedback entries, bounding total growth to the most recent N."""
    combined = (left or []) + (right or [])
    if len(combined) > _MAX_FEEDBACK_ENTRIES:
        return combined[-_MAX_FEEDBACK_ENTRIES:]
    return combined


def _merge_read_cache(left: dict, right: dict) -> dict:
    """Reducer for the cross-invocation read_cache.

    Entries are keyed by file path or by ``call::<tool>::<args-fingerprint>``;
    values are small metadata dicts (n_lines, symbols, turn, summary). Right
    wins on key collisions so the newest probe — which has the freshest turn
    counter — survives. Concurrent Send() researchers each contribute their
    own keys; non-overlapping keys merge cleanly.
    """
    if not left:
        return dict(right or {})
    if not right:
        return dict(left)
    return {**left, **right}


def _merge_artifacts(left: dict, right: dict) -> dict:
    """Deep-merge artifacts dicts so per-file entries aren't lost.

    Artifacts have a two-level structure: ``{phase: {filename: content}}``.
    A shallow merge would replace the entire inner dict for a phase key,
    destroying any files that weren't re-produced (e.g. individual slice
    files from the tasks phase).  This reducer merges at the file level
    instead, so returning ``{"tasks": {"tasks.md": summary}}`` only
    overwrites ``tasks.md`` — any ``slice-1.md``, ``slice-2.md``, etc.
    from a prior run are preserved.
    """
    merged = {**left}
    for phase_key, phase_artifacts in right.items():
        if not phase_artifacts:
            # Empty update — e.g. a phase exception handler returning
            # ``{phase: {}}``. Do NOT clobber previously-accumulated artifacts
            # for this phase: a transient error after a successful pass must not
            # blank the phase in state (finding #7). Skip the key entirely.
            continue
        if not isinstance(phase_artifacts, dict):
            # Non-dict value — overwrite the key
            merged[phase_key] = phase_artifacts
        elif phase_key in merged and isinstance(merged.get(phase_key), dict):
            # Both sides are dicts — merge at the file level
            merged[phase_key] = {**merged[phase_key], **phase_artifacts}
        else:
            # Left side missing or not a dict — use right's value
            merged[phase_key] = phase_artifacts
    return merged


class PhaseResult(TypedDict, total=False):
    """Lightweight summary of a phase subgraph's output.

    Stored in ``WorkflowState.phase_results`` so the parent graph can
    track progress without carrying full artifact content.
    """

    phase: str
    status: str  # "success" | "needs_review" | "error"
    artifact_count: int
    artifact_names: list[str]
    error: str | None


class WorkflowState(TypedDict, total=False):
    """State schema for the SPINE workflow StateGraph.

    Fields with reducers accumulate across node executions:
    - artifacts: phase output documents merge by key
    - feedback: review feedback appends to a list
    - retry_count: per-phase retry counts merge by phase name
    - phase_results: per-phase summary dicts merge by key
    """

    work_id: str
    work_type: str
    description: str  # Original work description — only used by SPECIFY (first spec
    # phase) and TASKS (first quick-workflow phase). Downstream
    # phases work from artifacts on disk, not the raw description.
    current_phase: str
    phase_index: int
    retry_count: Annotated[dict, _merge_dicts]
    max_retries: int
    artifacts: Annotated[dict, _merge_artifacts]
    feedback: Annotated[list, _append_capped_feedback]
    status: str
    prompt_request: dict | None
    critic_reviewing: str  # Phase the current critic node is reviewing
    last_critic_review: dict | None  # {"phase": str, "status": str, "tier": str,
    # "reason": str, "suggestions": list[str], "attempt": int,
    # "stagnation_streak": int, "unaddressed_points": list[str],
    # "blocker_category": str | None, "escalate": bool, "escalation_kind": str | None}
    # Written by _critic_result_mapper; consumed by critic_router and by the
    # specify/plan synthesizers when retry_count > 0. Last-write-wins — this
    # is the single source of truth for routing, decoupled from the
    # (capped-append) `feedback` list. The convergence fields (stagnation_streak,
    # unaddressed_points) are computed by spine.workflow.critic_convergence and
    # drive early escalation + the "still unaddressed" rework delta.
    # ── Adversarial review (critical work types) ──
    # The adversarial stage runs after critic_plan for critical_task /
    # critical_reviewed_task. It red-teams the approved plan; autonomously-
    # fixable findings loop back to PLAN, human-judgement findings escalate.
    # Its rework budget is tracked SEPARATELY from the critic's retry_count so
    # the two loops never consume each other's attempts.
    adversarial_retry_count: int  # Adversarial→plan rework rounds taken. Plain
    # int, last-write-wins (like verify_attempts) — NOT routed through the
    # retry_count _merge_dicts reducer, so critic accounting is untouched.
    max_adversarial_retries: int  # Budget for adversarial reworks; seeded from
    # SpineConfig.max_adversarial_retries alongside max_retries.
    adversarial_plan_completed: bool  # True when adversarial_plan passed.
    last_adversarial_review: dict | None  # Mirror of last_critic_review for the
    # adversarial stage. Written by _adversarial_result_mapper, read by
    # adversarial_router and (on loopback) the PLAN synthesizer's rework block.
    needs_review_kind: str | None  # Why the workflow paused for human review:
    # "spec_amendment" (spec contradiction the plan can't resolve),
    # "stagnation" (rework rounds stopped converging), "retries_exhausted"
    # (max_retries hit), "critic_flagged" (critic returned NEEDS_REVIEW),
    # "adversarial_flagged" (adversarial review needs human judgement), or
    # "adversarial_exhausted" (adversarial rework budget consumed).
    # Surfaced in the human_review interrupt payload so the reviewer knows
    # whether to amend upstream artifacts vs. nudge a rework.
    workspace_root: str  # Project root directory for deep agent backends
    phase_results: Annotated[dict, _merge_dicts]  # phase → PhaseResult
    needs_review_phase: str | None  # Which phase triggered human review
    human_feedback: dict | None  # Human-review decision from the interrupt:
    # {"action": "rework"|"approve"|"abort", "feedback": str, "_review_target": str}.
    # Written by _human_review_interrupt, read by the human_review router. MUST be
    # a declared channel — LangGraph commits only declared channels, so without
    # this the update is silently dropped, the router reads {} and defaults the
    # action to "abort", and EVERY resume (rework and approve alike) collapses to
    # abort/re-park regardless of what the human chose. See trace 019f1628.
    plan_id: str | None  # Optional reference to an approved planning work item.
    # For execution work items spawned from a plan: references the planning work
    # item that spawned this item. None for standalone quick/critical_quick items.
    spawned_work_ids: Annotated[list[str], operator.add]  # IDs of execution work
    # items spawned from this planning item. Empty for standalone execution items
    # and for planning items that haven't been approved yet.
    execution_waves: list[list[dict]]  # Pre-sorted waves of slice dicts from the
    # scheduler, consumed by IMPLEMENT to dispatch slices in dependency order.
    # Each inner list is one wave of independent slices that can run concurrently.
    verify_attempts: int  # How many gap-fix cycles attempted (starts at 0).
    # Incremented by the verify result mapper when verification fails.
    # The first _VERIFY_MIN_CYCLES cycles are unconditional; beyond that a
    # cycle is granted only while the total gap count strictly decreases
    # (progress-based budget, ceiling _VERIFY_MAX_CYCLES — run 019f2194 was
    # cut off at 3 verify passes while converging 18→12→8).
    verify_gap_totals: list[int]  # Total open-gap count per verify pass, in
    # order. Written whole by the verify result mapper (last-write-wins);
    # consecutive entries drive the strictly-decreasing progress check.
    verify_best: dict | None  # Best-state ratchet marker: {"total": int} for
    # the lowest gap total achieved; the snapshot itself lives on disk under
    # the verify artifact dir (run 019f2579: a late regressed cycle must not
    # leave worse code on disk than the best cycle produced).
    verify_regression_retries: int  # LEGACY (kept so resumed checkpoints
    # deserialize): the one-shot regression retry was subsumed by the
    # patience-based budget — a restored regression is just a stall cycle,
    # and patience grants the next cycle from the restored best state.

    # ── Phase Completion Invariants (prevent rework misinterpretation) ──
    # These boolean flags track whether critical phase operations completed
    # successfully, preventing the system from re-interpreting empty/failed
    # artifacts as intentionally empty work.
    gap_plan_produced: bool  # True when gap_plan.md was successfully created
    exploration_executed: bool  # True when SPECIFY/PLAN research rounds ran

    # Phase completion flags
    spec_completed: bool  # True when SPECIFY phase completed successfully
    plan_completed: bool  # True when PLAN phase completed successfully
    execution_waves_present: bool  # True when execution_waves is non-empty (fail-closed
    # invariant for IMPLEMENT prerequisite gate)
    tasks_completed: bool  # True when TASKS phase completed successfully
    implement_completed: bool  # True when IMPLEMENT phase completed successfully
    verify_completed: bool  # True when VERIFY phase completed successfully
    critic_specify_completed: bool  # True when CRITIC_SPECIFY phase passed
    critic_plan_completed: bool  # True when CRITIC_PLAN phase passed
    # adversarial_plan_completed declared above with the adversarial fields.
    gap_plan_completed: bool  # True when GAP_PLAN phase completed successfully

    # Additional tracking fields
    verification_findings: list[dict]  # Structured VerificationResult objects from subagents
    verification_attempted: bool  # True when VERIFY phase ran
    verification_passed: bool  # True when VERIFY phase passed
    implementation_files_written: bool  # True when IMPLEMENT wrote files
    files_written: list[str]  # Sorted, de-duplicated paths the implementer
    # reported creating/modifying. Forwarded from the IMPLEMENT subgraph by
    # _implement_result_mapper and read by the scope-boundary check to enforce
    # Specification.hard_boundaries deterministically. Last-write-wins (no
    # reducer): each IMPLEMENT run, including gap-fix re-runs, reports its own
    # file set rather than accumulating across cycles.
    slices_dispatched: bool  # True when IMPLEMENT dispatched slice subagents
    gaps_identified: int  # Number of gaps found by GAP_PLAN
    work_units_count: int  # Number of work units from TASKS phase
    feature_slices_count: int  # Number of feature slices from PLAN phase

    # Structured phase outputs (raw JSON strings) — sourced from the phase
    # agent's structured tool call (write_specification / write_structured_plan)
    # and consumed by the matching critic. Critics MUST work from these
    # structured fields; missing values are a CriticalContractFailure.
    specification_json: str | None
    plan_json: str | None

    # RAG retrieved context (for SPECIFY phase early commitment)
    task_category: str | None  # Classified task category for vector filtering
    retrieved_context: list[dict]  # Retrieved code chunks from vector store
    classification_confidence: float  # 0.0-1.0 confidence from classify_task —
    # used by the pre_research_gate to decide whether to skip the exploration
    # loop and synthesize directly from recalled chunks.

    # Shared deduper cache for the run — survives across phases and rework
    # cycles via the LangGraph checkpointer. Populated by ReadCacheMiddleware
    # inside every agent.ainvoke() and merged back into state by each node.
    # Subgraph state mappers seed their own copy from this field and bubble
    # the post-invocation cache back via the same key.
    read_cache: Annotated[dict, _merge_read_cache]
