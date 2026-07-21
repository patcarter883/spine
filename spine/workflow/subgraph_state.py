"""Per-phase subgraph state schemas for the SPINE orchestrator.

Each subgraph has its own TypedDict so DA agent message history
and phase-internal state don't leak into the parent graph's state.
"""

from __future__ import annotations

from operator import add as _op_add
from operator import or_ as _op_or
from typing import Annotated, Any

from typing_extensions import TypedDict

from spine.models.state import _merge_read_cache


class BaseSubgraphState(TypedDict, total=False):
    """Fields shared by all phase subgraphs."""

    phase: str
    work_id: str
    work_type: str
    description: str  # Only used by SPECIFY (always) and TASKS (quick workflows).
    # Other phases work from prior artifacts on disk.
    workspace_root: str
    retry_count: int
    feedback: list
    messages: list[Any]
    artifacts_output: dict  # {filename: content} — what this phase produced
    phase_status: str  # "success" | "needs_review" | "error"
    last_critic_review: dict | None  # Forwarded from parent WorkflowState so
    # synthesizers can render the critic's most recent verdict in rework prompts
    # without scanning the accumulating `feedback` list.
    # Shared dedupe cache for ReadCacheMiddleware — seeded by the state mapper
    # from WorkflowState.read_cache and written back by nodes after every
    # agent.ainvoke(). Merged with _merge_read_cache so parallel Send()
    # researchers do not clobber each other's contributions.
    read_cache: Annotated[dict, _merge_read_cache]


class SpecifySubgraphState(BaseSubgraphState, total=False):
    """SPECIFY phase — produces specification.md + specification.json."""

    specification_json: str  # Raw specification.json content (for downstream phases)
    agent_response: str  # Agent's text response (captured for artifact fallback)
    task_category: str | None  # Classified task category from early commitment
    retrieved_context: list[dict]  # Retrieved code chunks from vector store
    classification_confidence: float  # 0.0-1.0 from classify_task
    # SubagentDirective from the plan-before-do split (no-tool LLM call that
    # the do node consumes). Declared so LangGraph doesn't drop it between
    # nodes. See spine.agents.plan_do.
    specify_directive: dict


class PlanSubgraphState(BaseSubgraphState, total=False):
    """PLAN phase — reads spec (if available), produces plan.md + plan.json."""

    spec_path: str  # None for quick workflows (no SPECIFY phase)
    has_spec: bool  # True when a specification artifact exists
    plan_json: str  # Raw plan.json content (set by run_agent, read by save_artifacts)
    execution_waves: list  # Computed after agent completes (for IMPLEMENT)
    plan_directive: dict  # SubagentDirective from plan-before-do split


class TasksSubgraphState(BaseSubgraphState, total=False):
    """TASKS phase — reads plan, produces tasks.md + slice-*.md."""

    plan_path: str
    spec_path: str  # Only for spec/critical_spec workflows
    tasks_directive: dict  # SubagentDirective from plan-before-do split


def _slice_list_reducer(
    existing: list[dict] | None,
    update: list[dict] | dict | None,
) -> list[dict]:
    """Reducer for pending_slices / completed_slices / failed_slices.

    Update shapes:
      - ``list[dict]``  -> append (initial seed from state mapper).
      - ``dict``        -> ``{"add": [...], "remove": ["<slice_id>", ...]}``
                            directive; ids are matched against ``slice["id"]``.
      - ``None``        -> no-op.

    ``remove`` is idempotent: an id absent from ``existing`` is a no-op.
    Within a super-step LangGraph applies parallel-Send updates through
    the reducer sequentially, so two sibling Sends each emitting
    ``{"remove": [...]}`` compose correctly.
    """
    base = list(existing or [])
    if update is None:
        return base
    if isinstance(update, dict):
        remove_ids = set(update.get("remove") or [])
        if remove_ids:
            base = [s for s in base if s.get("id") not in remove_ids]
        adds = update.get("add") or []
        if adds:
            base.extend(adds)
        return base
    return base + list(update)


def _bool_or(existing: bool | None, update: bool | None) -> bool:
    """OR-reducer for completion-invariant bools.

    Parallel Send branches (one per slice) each return ``True`` for
    ``slices_dispatched`` / ``implementation_files_written`` in the same
    super-step. A plain ``bool`` channel rejects the concurrent writes with
    ``InvalidUpdateError("Can receive only one value per step")`` (trace
    019e784c crashed IMPLEMENT on any multi-slice plan). OR composes them:
    once any branch sets it True it stays True.
    """
    return bool(existing) or bool(update)


class ImplementSubgraphState(BaseSubgraphState, total=False):
    """IMPLEMENT phase — reads plan artifacts, dispatches slice-implementers.

    Slice lists use ``_slice_list_reducer`` so a node can atomically
    remove a slice (by id) and add new ones in a single update — this
    is what lets the dispatch loop terminate even though pending and
    failed slices are repeatedly re-routed via conditional edges.

    Single-file decomposition (``split_slices`` node) replaces each multi-file
    slice with the head of a single-file chain. The extra sequencing metadata
    rides *inside* the slice dicts already flowing through the lists above — no
    new channels — using these private (underscore-prefixed) keys:

    - ``_parent_slice_id`` — id of the originating multi-file slice.
    - ``_all_files`` — every target file of the parent (for sibling context).
    - ``_sibling_queue`` — ordered remaining sub-slices; ``slice_implementer``
      promotes ``[0]`` to ``pending_slices`` on success so a parent's files
      land sequentially while other parents proceed in parallel.
    - ``_file_index`` / ``_file_total`` — 1-based position within the chain.
    - ``_validate_slice_criteria`` — True only on the last file; that
      implementer runs the slice-level acceptance checks.
    - ``_decompose_depth`` — inherited from the parent so a failed sub-slice is
      still eligible for fallback micro-slicing.
    """

    plan_path: str
    gap_plan_path: str | None  # Set when re-running for a gap fix
    # Last cycle's verification findings (forwarded on gap reworks) — the
    # editors render each slice's PASSING checklist criteria as a
    # do-not-break block, countering wholesale-regeneration regressions
    # (run 019f2579: cycle 6 regressed 9→23 open gaps).
    verification_findings: list[dict]

    # Transient — populated per-Send by ``_route_slices``.
    active_slice: dict
    # Transient — written by the per-branch plan node, read by the do node.
    # Each parallel Send branch has its own state copy, so last-write-wins
    # semantics are exactly what we want here (one writer per branch).
    # See spine.agents.plan_do.SubagentDirective.
    active_slice_directive: dict

    # Slice lists, all using the same custom reducer.
    pending_slices: Annotated[list[dict], _slice_list_reducer]
    completed_slices: Annotated[list[dict], _slice_list_reducer]
    failed_slices: Annotated[list[dict], _slice_list_reducer]

    # Monotonic count of implementer/decomposer node executions. Summed across
    # parallel Send branches (operator.add) so ``_route_slices`` can enforce a
    # hard dispatch ceiling and abort a decompose/same-file runaway instead of
    # fanning out hundreds of Sends (trace 019efd92: 687 executions / 1.33M tok).
    slice_dispatch_count: Annotated[int, _op_add]

    # ── Phase Completion Invariants ──
    # OR-reduced: parallel slice-implementer Send branches each write True in
    # the same super-step, which a plain bool channel rejects (trace 019e784c).
    slices_dispatched: Annotated[bool, _bool_or]  # True when slice-implementers dispatched
    implementation_files_written: Annotated[bool, _bool_or]  # True when code files created

    # Sorted, de-duplicated list of every file the implementer reported
    # touching. Written once by ``synthesize_implementation`` (single node,
    # so plain last-write semantics), consumed by the scope-boundary gate.
    files_written: list[str]


class VerifySubgraphState(BaseSubgraphState, total=False):
    """VERIFY phase — confirms implementation."""

    tasks_path: str
    spec_path: str | None  # Only for spec/critical_spec workflows
    plan_path: str | None
    execution_waves: list  # Execution waves from PLAN phase (for Send dispatch)

    # Transient — populated per-Send by ``_verify_router``.
    slice: dict
    # Per-branch SubagentDirective written by the plan node, read by the do
    # node. See ImplementSubgraphState for the same pattern.
    active_slice_directive: dict

    # Accumulated per-slice verdicts from parallel Send dispatch (operator.add)
    verification_results: Annotated[list[dict], _op_add]

    # Per-slice convergence inputs (gap-fix cycles only, forwarded by
    # _verify_state_mapper): last cycle's verdicts + the files the rework
    # rewrote. seed_prior_results carries forward VERIFIED verdicts for
    # untouched slices and records their ids so _verify_router skips them.
    prior_verification_findings: list[dict]
    files_written: list[str]
    reverify_skipped_ids: list[str]

    # ── Phase Completion Invariants ──
    verification_attempted: bool  # True when verify agent ran (vs. skipped)
    verification_passed: bool  # True when verification confirmed passing

    verification_findings: list[dict]  # Structured VerificationResult objects from subagents


class CriticSubgraphState(BaseSubgraphState, total=False):
    """CRITIC phase — reviews a preceding phase's output."""

    reviewed_phase: str
    reviewed_phase_path: str
    artifacts: dict  # Phase artifacts from parent WorkflowState — needed
    # by structural_critic_check to verify artifacts exist.
    # Structured outputs the critic reviews (forwarded from parent state).
    # The agent_critic_check fails closed if the relevant field is empty.
    specification_json: str | None
    plan_json: str | None
    # Review results written by the subgraph nodes. These MUST be declared
    # here — LangGraph drops state updates for keys not in the schema, which
    # silently strips the agent/structural verdicts before the result mapper
    # can read them (manifests as "No review performed" feedback).
    structural_result: dict | None
    agent_result: dict | None
    validation_result: dict | None
    # NOTE: no critic_directive channel — the critic's plan→do directive step
    # was removed after twice poisoning reviews with invented requirements
    # (traces 019f1204, 019f2131).
    # The critic's own prior verdict, forwarded from the parent's
    # last_critic_review — feeds the REWORK prompt (goalpost pinning) and the
    # reference-symbol gate's cross-round persistence check.
    last_critic_review: dict | None
    # Reference-symbol gate outcome for THIS round ({} when the gate passed).
    # The result mapper stashes it under last_critic_review["reference_gate"]
    # so the next round can see which symbols were already flagged.
    reference_gate_result: dict | None
    # Records of prior-round literal_fixes that structural_check applied
    # MECHANICALLY this round (the rework left the flagged text in place).
    # Declared so the update survives LangGraph channel filtering and the
    # result mapper can propagate the patched plan to the parent.
    literal_fixes_applied: list[dict] | None


class AdversarialSubgraphState(BaseSubgraphState, total=False):
    """ADVERSARIAL phase — red-teams the approved plan (always reviews PLAN)."""

    reviewed_phase: str
    reviewed_phase_path: str
    artifacts: dict  # Phase artifacts from parent WorkflowState.
    # Structured outputs the reviewer reads (forwarded from parent state).
    specification_json: str | None
    plan_json: str | None
    # Verdict written by the agent_check node — MUST be declared here or
    # LangGraph drops the update before the result mapper can read it.
    agent_result: dict | None
    phase_status: str | None
    adversarial_directive: dict  # SubagentDirective for the agent plan→do split.


class GapPlanSubgraphState(BaseSubgraphState, total=False):
    """GAP_PLAN phase — reads verify feedback, produces gap_plan.md.

    Does NOT re-explore the codebase — uses the existing codebase map
    and plan artifacts from the original planning phase.
    """

    verify_path: str
    plan_path: str
    gap_plan_json: str  # Raw gap_plan.json content (for downstream phases)
    gap_plan_directive: dict  # SubagentDirective for plan-before-do split


class ExplorationSubgraphState(BaseSubgraphState, total=False):
    """Multi-node exploration → synthesis subgraph (SPECIFY, PLAN).

    Accumulates findings across parallel explore rounds via
    ``operator.add``, then routes to synthesis when research
    is sufficient.
    """

    # Exploration loop control
    research_round: int  # Current round number (0-based)
    max_rounds: int  # Safety valve — max exploration rounds (default 3)
    manager_decision: str  # "explore" | "done" — set by research_manager

    # Set True (OR-merged across parallel branches) when any explore_do branch
    # exhausted its supervisor cycle budget without completing. The sufficiency
    # router treats this as a hard "proceed to synthesis" signal: a researcher
    # that already blew its full per-topic budget will not converge by being
    # handed the same budget on overlapping topics next round, so looping
    # further only burns time/tokens (and risks tripping the dispatcher stall
    # timeout). Write-once-true; no reset needed since it forces the loop to end.
    recursion_capped_seen: Annotated[bool, _op_or]

    # Accumulated research (operator.add reducer merges per-round findings)
    topics: Annotated[list[str], _op_add]  # Areas being explored this round
    findings: Annotated[list[dict], _op_add]  # ResearchFindings dicts from explore nodes
    scratchpad: Annotated[str, _op_add]  # Working memory accumulator for GC

    # Findings inherited from an earlier phase's research_log.json
    # (e.g. SPECIFY's findings injected into PLAN). Seeded once by the
    # state mapper; read-only inside the subgraph. Kept separate from
    # ``findings`` so it does not pollute the topic-dedup, manager-summary,
    # or persisted-research-log paths for the current phase.
    prior_phase_findings: list[dict]

    # ── Structural-retry carryover (trace 019eb940) ──
    # Seeded into the fresh-thread retry input by subgraph_wrapper from
    # CriticalContractFailure.carryover when synthesis failed AFTER
    # exploration succeeded. ``findings_carried_over`` routes the retry
    # straight to synthesize (skipping classify/recall and the research
    # loop); ``synthesis_cap_escalated`` starts the synthesizer at the
    # raised completion cap — the prior attempt truncated at the base cap,
    # so re-rolling at it would truncate identically.
    findings_carried_over: bool
    synthesis_cap_escalated: bool

    # Per-Send transient: the evidence dossier produced by the explore_do
    # node and consumed by the summarise node in the same branch. One
    # writer per branch so no reducer is needed.
    exploration_evidence: dict

    # Per-Send transient: compact digest of prior-round findings rendered
    # by the dispatching router (render_covered_ground) and consumed by
    # explore_do, so round-2+ researchers build on already-mapped ground
    # instead of re-fetching it. Rides the Send payload only — never
    # written back as a channel update.
    covered_ground: str

    # Recall hits per topic from the topic_lookup node. Populated for the
    # latest round's NEW topics only — older rounds' entries are not
    # carried forward (already-explored topics won't be re-sent by the
    # router). Each hit is {symbol_name, file_path, symbol_type, lang,
    # enriched_summary, similarity}. Last-write-wins (no reducer).
    topic_recall_hits: dict[str, list[dict]]

    # Classification from early commitment (for SPECIFY scope constraint)
    task_category: str | None
    classification_confidence: float  # 0.0-1.0 from classify_task —
    # used by the pre_research_gate to decide whether to skip the exploration
    # loop and synthesize directly from recalled chunks.
    retrieved_context: list[dict]  # Chunks pulled by the pre_research_gate;
    # injected into the SPECIFY synthesizer prompt when present.

    # Synthesis output
    agent_response: str  # Final spec/plan text from synthesizer

    # ── Phase Completion Invariants (prevent rework misinterpretation) ──
    # Track whether the exploration loop actually executed vs. was skipped,
    # and whether the synthesis produced valid output.
    exploration_happened: bool  # True when research rounds executed (vs. skipped)
    synthesis_completed: bool  # True when synthesizer produced valid output

    # SPECIFY-specific fields
    specification_json: str  # Raw specification.json content (only used in SPECIFY phase)

    # PLAN-specific fields
    spec_path: str  # Path to specification.md (for PLAN explore agents)
    has_spec: bool  # True when a specification artifact exists
    specification_json: str  # Raw specification.json content (only used in SPECIFY phase)
    plan_json: str  # Raw plan.json content (only used in PLAN phase)
    execution_waves: list  # Computed execution waves (only used in PLAN phase)
