"""SPINE workflow composer — builds a LangGraph StateGraph from a WorkType.

The composer reads the WorkType, determines the phase sequence, and wires
the graph with conditional edges for critic review rework loops.

Each critic instance gets a unique node name (e.g. ``critic_specify``,
``critic_plan``) so the same critic function can appear multiple times in
a workflow graph — each reviewing a different preceding phase.

Artifact gates are wired as **nodes** (not just conditional edge functions)
so they can write ``status = "needs_review"`` and feedback entries to state
when they fail. This ensures the dispatcher detects the human-review condition
instead of silently marking the work as completed.

Currently, only the plan→implement transition is gated.  Verify always runs
after implement — if implement produced nothing, verify detects and reports
that; there is no reason for a human review gate between those two phases.

Phase sequences by WorkType:
    task:              SPECIFY → PLAN → CRITIC_PLAN → IMPLEMENT → VERIFY
    critical_task:     SPECIFY → PLAN → CRITIC_PLAN → ADVERSARIAL_PLAN → IMPLEMENT → VERIFY
    reviewed_task:     SPECIFY → PLAN → CRITIC_PLAN  (graph ENDs; awaits human approval)
    critical_reviewed: SPECIFY → PLAN → CRITIC_PLAN → ADVERSARIAL_PLAN  (graph ENDs; awaits human approval)

The ADVERSARIAL_PLAN stage red-teams the approved plan: autonomously-fixable
findings loop back to PLAN on a separate retry budget, human-judgement
findings escalate to review.

Reviewed work types intentionally terminate after ``critic_plan``. The
graph never runs IMPLEMENT/VERIFY directly — when the human approves
the plan via ``approve_and_spawn``, fresh ``task`` work items are
spawned for execution. Letting the graph fall through to IMPLEMENT
would defeat the entire purpose of the human-review gate.
"""

import logging
import re
from typing import Any, Callable, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import interrupt

from spine.models.enums import PhaseName, ReviewStatus, WorkType
from spine.models.state import WorkflowState
from spine.workflow.phase_progress import mark_phase_started
from spine.workflow.registry import get_registry
from spine.workflow import critic_convergence
from spine.workflow.critic_review import critic_router
from spine.workflow.artifact_gate import (
    make_artifact_gate_node,
    artifact_gate_router,
    check_scope_boundaries,
)
from spine.agents.artifacts import artifact_path
from spine.workflow.subgraph_wrapper import (
    make_subgraph_node,
    make_success_result_mapper,
)
from spine.workflow.subgraphs.verify_subgraph import build_verify_subgraph
from spine.workflow.subgraphs.implement_subgraph import build_implement_subgraph
from spine.workflow.subgraphs.specify_subgraph import build_specify_subgraph
from spine.workflow.subgraphs.plan_subgraph import build_plan_subgraph
from spine.workflow.subgraphs.critic_subgraph import build_critic_subgraph
from spine.workflow.subgraphs.exploration_subgraph import build_exploration_subgraph
from spine.workflow.subgraphs.gap_plan_subgraph import build_gap_plan_subgraph
from spine.workflow.subgraphs.adversarial_subgraph import build_adversarial_subgraph
from spine.workflow.artifact_gate import (
    make_prerequisite_gate_node,
    _check_spec_prerequisite,
    _check_plan_prerequisite,
    _check_implement_prerequisite,
    _check_verify_prerequisite,
)

# ── Subgraph builder registry ──────────────────────────────────────────
# Used by the per-phase checkpointer to recompile subgraphs at runtime
# with phase-specific SQLite databases instead of sharing the parent's.

_SUBGRAPH_BUILDER_REGISTRY: dict[str, Callable] = {}


def register_subgraph_builder(phase: str, builder: Callable) -> None:
    """Register a subgraph builder so per-phase checkpointers can recompile."""
    _SUBGRAPH_BUILDER_REGISTRY[phase] = builder


def get_subgraph_builder(phase: str) -> Callable | None:
    """Get the registered builder for a phase, or None."""
    return _SUBGRAPH_BUILDER_REGISTRY.get(phase)


# Register all phase builders at import time.
register_subgraph_builder(PhaseName.VERIFY.value, build_verify_subgraph)
register_subgraph_builder(PhaseName.IMPLEMENT.value, build_implement_subgraph)
register_subgraph_builder(PhaseName.SPECIFY.value, build_specify_subgraph)
register_subgraph_builder(PhaseName.PLAN.value, build_plan_subgraph)
# Critic is parameterized by reviewed_phase — register keyed variants.
register_subgraph_builder(f"{PhaseName.CRITIC.value}_tasks", build_critic_subgraph)
register_subgraph_builder(f"{PhaseName.CRITIC.value}_plan", build_critic_subgraph)
register_subgraph_builder(f"{PhaseName.CRITIC.value}_specify", build_critic_subgraph)
# Adversarial reviews the PLAN; the node is named "adversarial_plan".
register_subgraph_builder(
    f"{PhaseName.ADVERSARIAL.value}_plan", build_adversarial_subgraph
)
register_subgraph_builder(PhaseName.GAP_PLAN.value, build_gap_plan_subgraph)

logger = logging.getLogger(__name__)


# Feature flags for per-phase subgraph migration.
# During rollout, phases can be enabled independently.
_SUBGRAPH_ENABLED: dict[str, bool] = {
    PhaseName.VERIFY.value: True,
    PhaseName.IMPLEMENT.value: True,
    PhaseName.TASKS.value: True,
    PhaseName.SPECIFY.value: True,
    PhaseName.PLAN.value: True,
    PhaseName.CRITIC.value: True,
    PhaseName.GAP_PLAN.value: True,
}

# Feature flags for exploration subgraph rollout.
# When True, SPECIFY/PLAN use the multi-node research_manager → explore →
# synthesize subgraph instead of the linear run_agent → save_artifacts
# subgraph.  Default: SPECIFY enabled, PLAN pending implementation.
_USE_EXPLORATION_SUBGRAPH: dict[str, bool] = {
    PhaseName.SPECIFY.value: True,
    PhaseName.PLAN.value: True,
}


def _base_state_mapper(parent_state: WorkflowState, config) -> dict:
    """Base fields shared by all phase state mappers."""
    return {
        "work_id": parent_state.get("work_id", "unknown"),
        "work_type": parent_state.get("work_type", ""),
        "description": parent_state.get("description", ""),
        "workspace_root": parent_state.get("workspace_root", "."),
        "feedback": parent_state.get("feedback", []),
        "last_critic_review": parent_state.get("last_critic_review"),
        "messages": [],
        "artifacts_output": {},
        "phase_status": "",
        # Seed the subgraph with the run-wide dedupe cache. The subgraph
        # accumulates further entries during the phase and bubbles the
        # merged dict back via its phase result mapper.
        "read_cache": dict(parent_state.get("read_cache") or {}),
    }


def _load_prior_research(
    workspace_root: str,
    work_id: str,
    phase: str,
) -> tuple[list[str], list[dict]]:
    """Read research_log.json from the phase's artifact dir.

    Returns ``(topics, findings)`` so the rework pass can seed the
    exploration subgraph with prior research instead of starting fresh
    and repeating the same exploration the critic just rejected.
    Returns empty lists if the log is missing or unreadable.
    """
    import json as _json

    from pathlib import Path as _Path

    log_path = (
        _Path(workspace_root)
        / ".spine"
        / "artifacts"
        / work_id
        / phase
        / "research_log.json"
    )
    if not log_path.exists():
        return [], []
    try:
        data = _json.loads(log_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [], []
    topics = data.get("topics") or []
    findings = data.get("findings") or []
    return (
        [t for t in topics if isinstance(t, str)],
        [f for f in findings if isinstance(f, dict)],
    )


def _specify_state_mapper(parent_state: WorkflowState, config) -> dict:
    workspace_root = parent_state.get("workspace_root", ".")
    work_id = parent_state.get("work_id", "")
    retry_count = parent_state.get("retry_count", {}).get(PhaseName.SPECIFY.value, 0)
    base = _base_state_mapper(parent_state, config)
    # Always attempt to load prior research, regardless of retry_count.
    # CriticalContractFailure on a critic gate does not increment
    # retry_count but does re-enter the phase, so gating prior-research
    # seeding on retry_count > 0 caused the research_manager to re-issue
    # the same architectural topics on every re-entry. _load_prior_research
    # returns ([], []) when no research_log.json exists, so the fresh-run
    # path is unaffected.
    prior_topics, prior_findings = _load_prior_research(
        workspace_root, work_id, PhaseName.SPECIFY.value
    )
    if prior_topics or prior_findings:
        base["topics"] = prior_topics
        base["findings"] = prior_findings
    return {
        **base,
        "phase": PhaseName.SPECIFY.value,
        "retry_count": retry_count,
        "scratchpad": parent_state.get("scratchpad", ""),
        "task_category": parent_state.get("task_category"),
    }


def _plan_state_mapper(parent_state: WorkflowState, config) -> dict:
    work_id = parent_state.get("work_id", "")
    workspace_root = parent_state.get("workspace_root", ".")
    crit_retry = parent_state.get("retry_count", {}).get(PhaseName.PLAN.value, 0)
    # The adversarial stage loops the plan back here on its OWN budget. Use
    # current_phase (last node to route in — set by the result mappers) to tell
    # an adversarial-driven rework from a critic-driven one. On an adversarial
    # loopback, drive rework mode off the adversarial round and surface the
    # adversarial findings instead of the critic's stale PASS verdict.
    came_from_adversarial = (
        parent_state.get("current_phase") == PhaseName.ADVERSARIAL.value
    )
    retry_count = (
        parent_state.get("adversarial_retry_count", 0)
        if came_from_adversarial
        else crit_retry
    )
    base = _base_state_mapper(parent_state, config)
    if came_from_adversarial:
        # _render_rework_feedback renders the subgraph's last_critic_review
        # slot; point it at the adversarial verdict so the rework prompt shows
        # what the red-team flagged. Parent state is untouched — the critic's
        # own last_critic_review and convergence accounting stay intact.
        base["last_critic_review"] = parent_state.get("last_adversarial_review")
    # Always attempt to load prior research (see _specify_state_mapper).
    prior_topics, prior_findings = _load_prior_research(
        workspace_root, work_id, PhaseName.PLAN.value
    )
    if prior_topics or prior_findings:
        base["topics"] = prior_topics
        base["findings"] = prior_findings
    # Inject SPECIFY's research findings into a separate channel so PLAN
    # researchers and the PLAN manager start from the architectural map
    # instead of re-mapping it. Topics are NOT seeded — they would pollute
    # PLAN's topic-dedup (different angles on the same module are valid
    # PLAN topics even when SPECIFY already touched the area).
    _, specify_findings = _load_prior_research(
        workspace_root, work_id, PhaseName.SPECIFY.value
    )
    if specify_findings:
        base["prior_phase_findings"] = specify_findings
    # All work types now run specify, so has_spec is always True
    return {
        **base,
        "phase": PhaseName.PLAN.value,
        "retry_count": retry_count,
        "spec_path": artifact_path(work_id, PhaseName.SPECIFY.value),
        "has_spec": True,
        "scratchpad": parent_state.get("scratchpad", ""),
    }


def _gap_plan_implicated(workspace_root: str, work_id: str) -> tuple[set, set]:
    """(slice_ids, file_paths) the current gap plan's remediations implicate.

    Read from gap_plan.json on disk (the same artifact ``_gap_fixes_body``
    renders for the editors). Empty sets on any load/shape problem — the
    per-slice convergence filter then behaves exactly as before.
    """
    import json as _json
    from pathlib import Path as _Path

    from spine.agents.artifacts import artifact_path as _artifact_path

    try:
        gap_dir = _artifact_path(work_id, PhaseName.GAP_PLAN.value)
        data = _json.loads(
            (_Path(workspace_root) / gap_dir / "gap_plan.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, ValueError):
        return set(), set()
    ids: set = set()
    files: set = set()
    items = data.get("remediation_items") if isinstance(data, dict) else None
    for item in items or []:
        if not isinstance(item, dict):
            continue
        if item.get("slice_id"):
            ids.add(str(item["slice_id"]))
        for fix in item.get("fixes") or []:
            if isinstance(fix, dict) and fix.get("file_path"):
                files.add(str(fix["file_path"]))
    return ids, files


def _implement_state_mapper(parent_state: WorkflowState, config) -> dict:
    work_id = parent_state.get("work_id", "")
    verify_attempts = parent_state.get("verify_attempts", 0)
    execution_waves = parent_state.get("execution_waves", []) or []
    pending: list[dict] = []
    for wave in execution_waves:
        if isinstance(wave, list):
            for sl in wave:
                if isinstance(sl, dict) and sl.get("id"):
                    pending.append(sl)
    # Per-slice convergence: on a gap-fix rework, re-dispatch ONLY the slices
    # verification failed. Re-implementing already-VERIFIED slices wastes the
    # cycle budget and risks regressing them (trace 019f2040 regressed a
    # passing slice; run 019f20e0 re-edited both slices every cycle when one
    # was a single gap from converging). Slices with no verdict at all stay in
    # (unknown ≠ passed); if filtering would empty the list, keep it
    # unfiltered — an empty IMPLEMENT dispatch is a contract violation.
    if verify_attempts > 0:
        findings = parent_state.get("verification_findings") or []
        verified_ids = {
            f.get("slice_name")
            for f in findings
            if isinstance(f, dict) and f.get("verdict") == "VERIFIED"
        }
        # Cross-slice reattribution: a VERIFIED slice is REOPENED when the
        # gap plan implicates it — by slice_id or by naming one of its files
        # in a fix. Probe 21 (run ad237d70): the test slice crashed on a
        # table-name mismatch whose fix belonged in the VERIFIED model
        # slice; the filter below locked that slice out and the loop
        # re-edited the one slice that could not fix it until the cap.
        reopened_ids, implicated_files = _gap_plan_implicated(
            parent_state.get("workspace_root", "."), work_id
        )
        verified_ids -= reopened_ids
        if implicated_files:
            verified_ids = {
                sid for sid in verified_ids
                if not any(
                    str(t) in implicated_files
                    for s in pending if s.get("id") == sid
                    for t in (s.get("target_files") or [])
                )
            }
        kept = [s for s in pending if s.get("id") not in verified_ids]
        if verified_ids and kept and len(kept) < len(pending):
            logger.info(
                "[%s] implement rework: re-dispatching %d/%d slice(s); "
                "already VERIFIED: %s",
                work_id, len(kept), len(pending),
                sorted(verified_ids & {s.get("id") for s in pending}),
            )
            pending = kept
    return {
        **_base_state_mapper(parent_state, config),
        "phase": PhaseName.IMPLEMENT.value,
        "retry_count": parent_state.get("retry_count", {}).get(PhaseName.IMPLEMENT.value, 0),
        "plan_path": artifact_path(work_id, PhaseName.PLAN.value),
        "gap_plan_path": artifact_path(work_id, PhaseName.GAP_PLAN.value) if verify_attempts > 0 else None,
        # On gap reworks the editors render each slice's PASSING checklist
        # criteria as a do-not-break block (run 019f2579: a wholesale
        # re-synthesis regressed 9→23 open gaps late in the run).
        "verification_findings": (
            parent_state.get("verification_findings") or []
            if verify_attempts > 0
            else []
        ),
        "pending_slices": pending,
        "completed_slices": [],
        "failed_slices": [],
    }


def _verify_state_mapper(parent_state: WorkflowState, config) -> dict:
    """Map parent WorkflowState to VerifySubgraphState."""
    work_id = parent_state.get("work_id", "")
    verify_attempts = parent_state.get("verify_attempts", 0)
    # All work types now run specify, so has_spec is always True
    return {
        **_base_state_mapper(parent_state, config),
        "phase": PhaseName.VERIFY.value,
        "retry_count": parent_state.get("retry_count", {}).get(PhaseName.VERIFY.value, 0),
        "plan_path": artifact_path(work_id, PhaseName.PLAN.value),
        "spec_path": artifact_path(work_id, PhaseName.SPECIFY.value),
        "execution_waves": parent_state.get("execution_waves", []),
        # Per-slice convergence inputs (gap-fix cycles only): the previous
        # cycle's verdicts plus the files the rework actually rewrote. A slice
        # that was VERIFIED last cycle and whose targets were not touched
        # since keeps its verdict without a re-verification round trip.
        "prior_verification_findings": (
            parent_state.get("verification_findings") or []
            if verify_attempts > 0
            else []
        ),
        "files_written": parent_state.get("files_written", []),
    }


def _critic_state_mapper(reviewed_phase: str):
    """Create a state mapper for a critic subgraph reviewing a specific phase."""

    def mapper(parent_state: WorkflowState, config) -> dict:
        work_id = parent_state.get("work_id", "")
        return {
            **_base_state_mapper(parent_state, config),
            "phase": PhaseName.CRITIC.value,
            "retry_count": parent_state.get("retry_count", {}).get(reviewed_phase, 0),
            "reviewed_phase": reviewed_phase,
            "reviewed_phase_path": artifact_path(work_id, reviewed_phase),
            "artifacts": parent_state.get("artifacts", {}),
            # Structured outputs the critic reviews — must round-trip into
            # the critic subgraph state, not get re-read from disk.
            "specification_json": parent_state.get("specification_json"),
            "plan_json": parent_state.get("plan_json"),
            # The critic's own prior verdict. agent_critic_check needs it to
            # render the REWORK prompt (confirm prior asks, don't shift
            # goalposts) and the reference-symbol gate needs its
            # "reference_gate" entry to spot symbols that stayed dangling
            # across rounds. Without this key both features silently no-op.
            "last_critic_review": parent_state.get("last_critic_review"),
        }

    return mapper


def _verify_failure_reason(subgraph_result: dict) -> str:
    """Compose a concrete needs_review reason from per-slice verdicts.

    Names the failing slices and their gaps so the reviewer sees *why*
    verification did not pass. Falls back to the synthesized
    ``verification.md`` summary, then to a generic message — so the reason is
    never a bare pointer to an artifact that may have been cleared.
    """
    findings = subgraph_result.get("verification_findings") or []
    failed = [
        f
        for f in findings
        if isinstance(f, dict) and f.get("verdict") not in ("VERIFIED", None)
    ]
    if failed:
        lines = []
        for f in failed:
            name = f.get("slice_name", "unknown slice")
            gaps = [str(g) for g in (f.get("gaps") or []) if g]
            if gaps:
                lines.append(f"{name}: " + "; ".join(gaps[:3]))
            else:
                lines.append(f"{name}: did not pass verification")
        reason = f"{len(failed)} slice(s) did not pass verification:\n" + "\n".join(
            f"- {ln}" for ln in lines
        )
        return reason[:2000]
    # Fall back to the synthesized summary the verify subgraph carried back.
    artifacts = subgraph_result.get("artifacts_output") or {}
    if isinstance(artifacts, dict):
        summary = str(artifacts.get("verification.md", "")).strip()
        if summary:
            return summary[:2000]
    return (
        "Verification did not pass and no detailed report was produced. "
        "Manual review required."
    )


def _verify_failure_suggestions(subgraph_result: dict) -> list[str]:
    """Aggregate per-slice recommendations into actionable suggestions."""
    findings = subgraph_result.get("verification_findings") or []
    suggestions: list[str] = []
    for f in findings:
        if not isinstance(f, dict):
            continue
        for rec in f.get("recommendations") or []:
            text = str(rec).strip()
            if text and text not in suggestions:
                suggestions.append(text)
    return suggestions[:10]


# Gap-fix cycle budget. The first _VERIFY_MIN_CYCLES gap-fix cycles are
# unconditional (the pre-existing behavior); beyond that, further cycles are
# granted while the run keeps setting new BEST (lowest) failed-criteria
# totals, with a patience of one non-improving cycle (the verifier is an LLM
# judge with per-cycle noise — run 019f25b8's checklist fails went 50→28→30→17
# and a strict-decrease rule would have stopped at the 30). Two consecutive
# cycles without a new best stop the run; _VERIFY_MAX_CYCLES bounds the total.
# The best-state ratchet makes patience safe: a regressed cycle is restored
# before the next one runs.
_VERIFY_MIN_CYCLES = 2
# 8 gives the final-mile mode runway: cycles are ~2 min, the ratchet makes
# each one safe, and patience-2 stops genuinely stuck runs long before this.
_VERIFY_MAX_CYCLES = 8


def _ratchet_files(parent_state: WorkflowState) -> list[str]:
    """Workspace-relative files the best-state ratchet snapshots/restores.

    The implementation files (every slice's target_files plus whatever the
    implementer reported writing) and the verify artifacts describing them —
    restoring code without its matching verification.json would hand gap_plan
    findings for the wrong state.
    """
    work_id = parent_state.get("work_id", "")
    files: set[str] = {str(f) for f in (parent_state.get("files_written") or []) if f}
    for wave in parent_state.get("execution_waves") or []:
        if isinstance(wave, list):
            for s in wave:
                if isinstance(s, dict):
                    files |= {str(f) for f in (s.get("target_files") or []) if f}
    vdir = artifact_path(work_id, PhaseName.VERIFY.value)
    files.add(f"{vdir}/verification.json")
    files.add(f"{vdir}/verification.md")
    return sorted(files)


def _total_gap_count(findings: Any) -> int | None:
    """Total FAILED acceptance criteria across slice findings, or None.

    Counts checklist entries with ``passed=False`` when a checklist is
    present — the ``gaps`` list is free-text itemization whose granularity
    varies per verifier call (run 019f25b8: checklist fails were flat at 17
    for four cycles while gap-entry totals bounced 17→24→13→19, burning the
    regression retry on phantom noise). Falls back to gap entries for
    findings without a checklist; a failing finding with neither still
    counts as one so a degenerate finding can't fake convergence to zero.
    """
    if not isinstance(findings, list) or not findings:
        return None
    total = 0
    for f in findings:
        if not isinstance(f, dict):
            return None
        checklist = f.get("checklist")
        if isinstance(checklist, list) and checklist:
            total += sum(
                1
                for it in checklist
                if isinstance(it, dict) and not it.get("passed")
            )
            continue
        if f.get("verdict") == "VERIFIED":
            continue
        gaps = f.get("gaps")
        total += len(gaps) if isinstance(gaps, list) and gaps else 1
    return total


def _cycles_since_best(totals: list[int]) -> int:
    """Consecutive trailing cycles without a new best (lowest) total."""
    best = None
    since = 0
    for t in totals:
        if best is None or t < best:
            best = t
            since = 0
        else:
            since += 1
    return since


def _verify_result_mapper(subgraph_result: dict, parent_state: WorkflowState) -> dict[str, Any]:
    """Map VerifySubgraphState output back to parent WorkflowState.

    When verification fails (phase_status="needs_review"), decides between a
    gap-fix cycle and human review: the first ``_VERIFY_MIN_CYCLES`` cycles are
    unconditional, then further cycles are granted while the run keeps setting
    new best failed-criteria totals, tolerating one non-improving cycle
    (verifier noise; the ratchet restores regressed cycles first), ceiling
    ``_VERIFY_MAX_CYCLES``.
    """
    base = make_success_result_mapper(PhaseName.VERIFY.value)(subgraph_result, parent_state)
    phase_status = subgraph_result.get("phase_status", "")
    findings_override: list[dict] | None = None
    ratchet_note = ""
    if phase_status == "needs_review":
        verify_attempts = parent_state.get("verify_attempts", 0)
        totals = list(parent_state.get("verify_gap_totals") or [])
        current_total = _total_gap_count(subgraph_result.get("verification_findings"))
        if current_total is not None:
            totals.append(current_total)
            base["verify_gap_totals"] = totals
        # Progress = a new BEST total, with patience: the verifier is an LLM
        # judge with per-cycle noise, so a single non-improving cycle must not
        # kill a converging run (run 019f25b8: checklist fails went 50→28→30→
        # 17… — a strict-decrease rule would have stopped at the 30 and lost
        # the drop to 17). Judged only on a total from THIS round — an
        # uncountable round must not ride on stale history.
        stall = _cycles_since_best(totals) if current_total is not None else 99
        improving = current_total is not None and stall < 2

        # ── Best-state ratchet ──
        # The editor re-synthesizes from scratch each cycle, so a rework is a
        # variance draw; run 019f2579 converged 43→22→11→9 then a bad draw
        # regressed to 23 and the loop stopped with the WORSE code on disk.
        # Snapshot each new best (lowest total); on a regression, restore it —
        # code, verification artifacts, and state findings together — so the
        # loop's floor is monotone and the final state is always the best one.
        regression_restored = False
        if current_total is not None:
            from spine.workflow import verify_snapshot as _snap

            work_id = parent_state.get("work_id", "")
            ws_root = parent_state.get("workspace_root", ".")
            best_total = (parent_state.get("verify_best") or {}).get("total")
            if best_total is None or current_total < best_total:
                if _snap.snapshot_best(
                    ws_root,
                    work_id,
                    _ratchet_files(parent_state),
                    subgraph_result.get("verification_findings") or [],
                    current_total,
                ):
                    base["verify_best"] = {"total": current_total}
            elif current_total > best_total and _snap.restore_best(ws_root, work_id):
                regression_restored = True
                findings_override = _snap.load_best_findings(ws_root, work_id)
                ratchet_note = (
                    f" | NOTE: this cycle REGRESSED the implementation "
                    f"({best_total}→{current_total} open gaps); the workspace "
                    f"has been RESTORED to the best state ({best_total} gaps) "
                    f"and the findings below describe the restored state."
                )

        # A regressed cycle was already restored to the best state above, so
        # granting the patience cycle retries from the BEST code, not the
        # regression — patience subsumes the old one-shot regression retry.
        if verify_attempts < _VERIFY_MIN_CYCLES or (
            improving and verify_attempts < _VERIFY_MAX_CYCLES
        ):
            if verify_attempts >= _VERIFY_MIN_CYCLES:
                logger.info(
                    "[%s] verify: granting extra gap-fix cycle %d/%d "
                    "(totals=%s stall=%d%s)",
                    parent_state.get("work_id", "?"),
                    verify_attempts + 1,
                    _VERIFY_MAX_CYCLES,
                    totals[-4:],
                    stall,
                    ", retrying from restored best" if regression_restored else "",
                )
            base["status"] = "needs_gap_fix"
            base["verify_attempts"] = verify_attempts + 1
        else:
            base["status"] = "needs_review"
            base["needs_review_phase"] = PhaseName.VERIFY.value
        base["feedback"] = base.get("feedback", []) + [
            {
                "status": "needs_review",
                "tier": "verify",
                "reason": _verify_failure_reason(subgraph_result) + ratchet_note,
                "suggestions": _verify_failure_suggestions(subgraph_result),
            }
        ]
    elif phase_status == "error":
        base["status"] = "failed"
    # Set completion invariants
    base["verify_completed"] = phase_status == "success"
    base["verification_attempted"] = True
    base["verification_passed"] = phase_status == "success"
    vf = subgraph_result.get("verification_findings", [])
    if vf:
        base["verification_findings"] = vf
    if findings_override:
        # A restored regression must hand downstream consumers (gap_plan, the
        # implement rework, per-slice convergence) the findings that match the
        # RESTORED code, not the regressed cycle's.
        base["verification_findings"] = findings_override
    return base


def _implement_result_mapper(subgraph_result: dict, parent_state: WorkflowState) -> dict[str, Any]:
    """Map ImplementSubgraphState output back to parent WorkflowState."""
    base = make_success_result_mapper(PhaseName.IMPLEMENT.value)(subgraph_result, parent_state)
    phase_status = subgraph_result.get("phase_status", "")
    if phase_status == "needs_review":
        base["status"] = "needs_review"
        base["needs_review_phase"] = PhaseName.IMPLEMENT.value
    elif phase_status == "error":
        base["status"] = "failed"
    # Set completion invariants
    base["implement_completed"] = phase_status == "success"
    base["slices_dispatched"] = subgraph_result.get("slices_dispatched", False)
    base["implementation_files_written"] = subgraph_result.get("implementation_files_written", False)
    # Forward the implementer's reported file set so the scope-boundary check
    # (and any downstream consumer) can read it from parent state.
    files_written = subgraph_result.get("files_written", []) or []
    base["files_written"] = files_written

    # Deterministic anti-drift gate: if IMPLEMENT wrote inside a declared
    # hard_boundary, downgrade an otherwise-successful phase to human review.
    # Run only when the phase didn't already fail/need review — a real failure
    # takes precedence and its own routing should not be masked.
    if base.get("status") not in ("needs_review", "failed"):
        scope_ok, scope_reason = check_scope_boundaries(
            {**parent_state, "files_written": files_written}
        )
        if not scope_ok:
            base["status"] = "needs_review"
            base["needs_review_phase"] = PhaseName.IMPLEMENT.value
            base["feedback"] = base.get("feedback", []) + [
                {
                    "status": "needs_review",
                    "tier": "structural",
                    "reason": f"Scope-boundary gate: {scope_reason}",
                    "suggestions": [],
                }
            ]
    return base


def _specify_result_mapper(subgraph_result: dict, parent_state: WorkflowState) -> dict[str, Any]:
    """Map SpecifySubgraphState output back to parent WorkflowState."""
    base = make_success_result_mapper(PhaseName.SPECIFY.value)(subgraph_result, parent_state)
    phase_status = subgraph_result.get("phase_status", "")
    if phase_status == "needs_review":
        base["status"] = "needs_review"
        base["needs_review_phase"] = PhaseName.SPECIFY.value
    elif phase_status == "error":
        base["status"] = "failed"
    # Forward early commitment results to parent state
    if subgraph_result.get("task_category"):
        base["task_category"] = subgraph_result["task_category"]
    if subgraph_result.get("retrieved_context"):
        base["retrieved_context"] = subgraph_result["retrieved_context"]
    if "classification_confidence" in subgraph_result:
        base["classification_confidence"] = subgraph_result["classification_confidence"]
    # Forward structured spec JSON so CRITIC_SPECIFY can read it from state.
    if subgraph_result.get("specification_json"):
        base["specification_json"] = subgraph_result["specification_json"]
    # Set completion invariants
    base["spec_completed"] = phase_status == "success"
    return base


def _plan_result_mapper(subgraph_result: dict, parent_state: WorkflowState) -> dict[str, Any]:
    """Map PlanSubgraphState output back to parent WorkflowState."""
    base = make_success_result_mapper(PhaseName.PLAN.value)(subgraph_result, parent_state)
    phase_status = subgraph_result.get("phase_status", "")
    if phase_status == "needs_review":
        base["status"] = "needs_review"
        base["needs_review_phase"] = PhaseName.PLAN.value
    elif phase_status == "error":
        base["status"] = "failed"
    # Forward execution waves so IMPLEMENT can use wave-based dispatch
    execution_waves = subgraph_result.get("execution_waves", [])
    if execution_waves:
        base["execution_waves"] = execution_waves
    # Forward structured plan JSON so CRITIC_PLAN can read it from state.
    if subgraph_result.get("plan_json"):
        base["plan_json"] = subgraph_result["plan_json"]
    # When this PLAN run was an adversarial-driven rework (the adversarial stage
    # loops the plan back here on its OWN budget), the reworked plan is
    # effectively a new plan and deserves a fresh critic budget. Reset
    # retry_count[plan] to 0 so the subsequent critic_plan does not inherit the
    # prior plan's exhausted count and escalate prematurely — the outer loop is
    # still bounded by max_adversarial_retries (finding: adversarial/critic
    # retry-budget coupling). retry_count uses a right-wins merge reducer.
    if parent_state.get("current_phase") == PhaseName.ADVERSARIAL.value:
        base["retry_count"] = {PhaseName.PLAN.value: 0}
    # Set completion invariants
    base["plan_completed"] = phase_status == "success"
    base["feature_slices_count"] = len(subgraph_result.get("feature_slices", []))
    return base


def _critic_result_mapper(reviewed_phase: str):
    """Create a result mapper for a critic subgraph reviewing a specific phase.
    
    Two-tier review logic:
    1. Structural check: If it fails, the agent check still runs (for feedback).
    2. Agent check: Determines the final status (PASSED, NEEDS_REVISION, NEEDS_REVIEW).
    
    The effective feedback comes from the agent if available, otherwise from structural.
    """

    def mapper(subgraph_result: dict, parent_state: WorkflowState) -> dict[str, Any]:
        base: dict[str, Any] = {
            "current_phase": PhaseName.CRITIC.value,
            "status": "running",
            "prompt_request": None,
        }

        structural_result = subgraph_result.get("structural_result", {})
        agent_result = subgraph_result.get("agent_result", {})

        if agent_result:
            effective_result = agent_result
        elif structural_result:
            # Deterministic tier standing in for the whole review — mark it
            # gate-sourced so streak accounting compares like with like.
            effective_result = {"verdict_source": "gate", **structural_result}
        else:
            effective_result = {
                "status": "passed",
                "tier": "structural",
                "verdict_source": "guard",
                "reason": "No review performed",
                "suggestions": [],
            }

        # Stamp the reviewed phase so downstream consumers (notably cross-run
        # experience capture) can attribute this feedback entry without parsing
        # it out of the free-text reason. Copy so we don't alias the subgraph's
        # dict into both ``feedback`` and ``last_critic_review``.
        effective_result = {**effective_result, "phase": reviewed_phase}
        base["feedback"] = [effective_result]

        phase_status = subgraph_result.get("phase_status", "")
        prior_attempts = parent_state.get("retry_count", {}).get(reviewed_phase, 0)
        max_retries = parent_state.get("max_retries", 3)

        # The prior verdict for THIS phase (last-write-wins). Only meaningful
        # when it belongs to the same phase — critic_specify and critic_plan
        # don't interleave, but guard anyway so a cross-phase record can't be
        # mistaken for a prior round.
        prior_lcr = parent_state.get("last_critic_review") or {}
        prior_for_phase = (
            prior_lcr if prior_lcr.get("phase") == reviewed_phase else {}
        )

        # Attempt accounting is verdict-source-aware, like the streaks below:
        # a guard verdict (the critic's own response was truncated — already
        # retried once inside the critic node) is harness noise, and charging
        # it a retry attempt pushes a healthy rework loop toward escalation
        # (trace 019f404a: a guard round consumed attempt 4-of-5). The FIRST
        # guard round for a phase is free; consecutive guard rounds DO count,
        # so a critic that always truncates still exhausts the budget instead
        # of looping forever.
        is_guard_verdict = effective_result.get("verdict_source") == "guard"
        prior_was_guard = prior_for_phase.get("verdict_source") == "guard"
        if is_guard_verdict and not prior_was_guard:
            new_attempt = prior_attempts
        else:
            new_attempt = prior_attempts + 1

        # Convergence detection: is this rework round repeating the previous
        # verdict (stagnation), or shifting the goalposts to wholly new asks
        # (churn)? Both are non-convergence; together they cap the loop well
        # short of the full retry budget. ``unaddressed`` lists the prior asks
        # that still recur, for the rework prompt's "STILL NOT ADDRESSED"
        # block. Accounting is verdict-source-aware: guard verdicts freeze the
        # streaks, gate verdicts compare gate-vs-gate, agent verdicts compare
        # against the carried agent baseline (trace 019f260c: cross-source
        # comparison scored a truncation round as two goalpost shifts and
        # parked a converging plan).
        unaddressed: list[str] = []
        stagnation_streak = 0
        churn_streak = 0
        streak_baseline: dict[str, Any] = {}
        if phase_status == ReviewStatus.NEEDS_REVISION.value:
            streaks = critic_convergence.compute_streaks(
                prior_for_phase,
                effective_result,
                current_gate=subgraph_result.get("reference_gate_result"),
            )
            unaddressed = streaks["unaddressed_points"]
            stagnation_streak = streaks["stagnation_streak"]
            churn_streak = streaks["churn_streak"]
            streak_baseline = streaks["streak_baseline"]

        # Escalation decision — collapse every "stop reworking" condition into a
        # single flag so the mapper (state) and critic_router (edge) agree.
        blocker_category = effective_result.get("blocker_category")
        spec_contradiction = blocker_category == "spec_contradiction"

        # Symbol existence is OWNED by the deterministic reference gate
        # (attribute-aware since e1d7f71). An agent-tier spec_contradiction
        # whose evidence is "symbol X does not exist / is not defined" while
        # the gate PASSED this round is the critic re-litigating stale
        # prior-round feedback with no ground truth — run a24d4fca parked on
        # 'self._base is not defined' one round after the gate started
        # resolving that exact attribute. Demote to a normal revision round;
        # if the objection is real the gate will flag it itself next round.
        gate_clean = not subgraph_result.get("reference_gate_result")
        if (
            spec_contradiction
            and gate_clean
            and effective_result.get("verdict_source", "agent") == "agent"
            and re.search(
                r"(?:does not exist|not defined|not present|not provided|"
                r"missing (?:from|in) the codebase|not in the codebase)",
                effective_result.get("reason", ""),
                re.IGNORECASE,
            )
        ):
            logger.warning(
                "[%s] critic spec_contradiction demoted: evidence is symbol "
                "non-existence but the reference gate passed this round",
                parent_state.get("work_id", "?"),
            )
            spec_contradiction = False
            blocker_category = None
            effective_result = {
                **effective_result,
                "blocker_category": None,
                "reason": effective_result.get("reason", "")
                + " [DEMOTED: the deterministic reference gate validated every "
                "referenced symbol this round — including instance attributes "
                "— so the non-existence claim above is refuted. Address any "
                "other points and drop this objection.]",
            }
            if phase_status == ReviewStatus.NEEDS_REVIEW.value:
                phase_status = ReviewStatus.NEEDS_REVISION.value
            # The un-demoted dict was already stamped into feedback above —
            # refresh it so state carries the demotion note, not the claim.
            base["feedback"] = [effective_result]
        stagnated = stagnation_streak >= critic_convergence.STAGNATION_LIMIT
        churning = churn_streak >= critic_convergence.CHURN_LIMIT
        retries_exhausted = (
            phase_status == ReviewStatus.NEEDS_REVISION.value
            and new_attempt >= max_retries
        )
        escalate = (
            phase_status == ReviewStatus.NEEDS_REVIEW.value
            or spec_contradiction
            or stagnated
            or churning
            or retries_exhausted
        )
        if spec_contradiction:
            escalation_kind = "spec_amendment"
        elif stagnated:
            escalation_kind = "stagnation"
        elif churning:
            escalation_kind = "non_convergence"
        elif retries_exhausted:
            escalation_kind = "retries_exhausted"
        elif phase_status == ReviewStatus.NEEDS_REVIEW.value:
            escalation_kind = "critic_flagged"
        else:
            escalation_kind = None

        # Single source of truth for the next routing decision. Derived from
        # the subgraph's phase_status (canonical) plus the effective review
        # dict — never from the operator.add `feedback` list, which can carry
        # stale entries from earlier phases or tiers.
        base["last_critic_review"] = {
            "phase": reviewed_phase,
            "status": phase_status,
            "tier": effective_result.get("tier", "unknown"),
            # Which chain produced this verdict (agent | guard | gate) and the
            # last real agent ask-set carried through non-agent rounds — both
            # consumed by critic_convergence.compute_streaks next round.
            "verdict_source": effective_result.get("verdict_source", "agent"),
            "streak_baseline": streak_baseline,
            "reason": effective_result.get("reason", ""),
            "suggestions": effective_result.get("suggestions", []),
            "attempt": new_attempt,
            "stagnation_streak": stagnation_streak,
            "churn_streak": churn_streak,
            "unaddressed_points": unaddressed,
            "blocker_category": blocker_category,
            "escalate": escalate,
            "escalation_kind": escalation_kind,
            # This round's reference-symbol gate outcome ({} when it passed,
            # absent for non-PLAN critics). Round-trips through the critic
            # state mapper so the next round's gate can escalate symbols that
            # stayed dangling despite this round's exact feedback.
            "reference_gate": subgraph_result.get("reference_gate_result") or {},
            # Mechanically-applicable corrections from THIS round's verdict —
            # next round's structural_check applies any the rework leaves in
            # place (critic_subgraph.apply_literal_fixes).
            "literal_fixes": effective_result.get("literal_fixes", []),
        }

        # Mechanical-fix propagation: when structural_check patched the plan
        # (prior-round literal fixes the rework failed to apply), the parent
        # must carry the PATCHED document — downstream implement dispatch and
        # the next rework prompt read it from parent state.
        if subgraph_result.get("literal_fixes_applied"):
            if subgraph_result.get("plan_json"):
                base["plan_json"] = subgraph_result["plan_json"]
            if subgraph_result.get("artifacts"):
                base["artifacts"] = subgraph_result["artifacts"]

        # Rework of a workspace-mutating phase invalidates the per-work_id
        # symbol cache: the failed attempt (or the human, during a
        # needs_review pause) may have edited files, and serving the
        # pre-edit lookup results to the retry would mask those changes.
        # Spec/plan rework leaves the workspace untouched — keep the cache.
        _mutating_phases = (PhaseName.IMPLEMENT.value, PhaseName.VERIFY.value)
        if (
            phase_status in (ReviewStatus.NEEDS_REVIEW.value, ReviewStatus.NEEDS_REVISION.value)
            and reviewed_phase in _mutating_phases
        ):
            from spine.agents import symbol_cache

            symbol_cache.clear(parent_state.get("work_id", ""))

        if phase_status in (
            ReviewStatus.NEEDS_REVIEW.value,
            ReviewStatus.NEEDS_REVISION.value,
        ):
            base["retry_count"] = {reviewed_phase: new_attempt}

        if phase_status == "error":
            base["status"] = "failed"
        elif escalate:
            # Pause for human review. A spec contradiction targets SPECIFY so a
            # "rework" action amends the spec rather than retrying a plan that
            # can't satisfy it; every other escalation reworks the phase the
            # critic just reviewed.
            base["status"] = "needs_review"
            base["needs_review_phase"] = (
                PhaseName.SPECIFY.value if spec_contradiction else reviewed_phase
            )
            base["needs_review_kind"] = escalation_kind
        else:
            # PASSED or a NEEDS_REVISION round that still has budget to converge.
            base["status"] = "running"

        if reviewed_phase == PhaseName.SPECIFY.value:
            base["critic_specify_completed"] = True
        elif reviewed_phase == PhaseName.PLAN.value:
            base["critic_plan_completed"] = True

        return base

    return mapper


# ── Adversarial review (critical work types) ──
# Runs after critic_plan. Structurally wired like a critic node, but reviews
# the PLAN with a red-team agent and tracks its own retry budget
# (adversarial_retry_count / max_adversarial_retries) so the critic's
# retry_count is never touched.


def _adversarial_state_mapper(parent_state: WorkflowState, config) -> dict:
    """Map parent WorkflowState to AdversarialSubgraphState (reviews PLAN)."""
    work_id = parent_state.get("work_id", "")
    return {
        **_base_state_mapper(parent_state, config),
        "phase": PhaseName.ADVERSARIAL.value,
        # The adversarial agent's own rework round (separate budget). Passed so
        # the agent/logging see the right round; never the critic's retry_count.
        "retry_count": parent_state.get("adversarial_retry_count", 0),
        "reviewed_phase": PhaseName.PLAN.value,
        "reviewed_phase_path": artifact_path(work_id, PhaseName.PLAN.value),
        "artifacts": parent_state.get("artifacts", {}),
        "specification_json": parent_state.get("specification_json"),
        "plan_json": parent_state.get("plan_json"),
    }


def _adversarial_result_mapper(
    subgraph_result: dict, parent_state: WorkflowState
) -> dict[str, Any]:
    """Map AdversarialSubgraphState output back to parent WorkflowState.

    Mirrors :func:`_critic_result_mapper`'s escalation shape but on the
    SEPARATE adversarial budget. Writes ``last_adversarial_review`` (the
    router's source of truth) and appends the verdict to ``feedback`` so the
    PLAN synthesizer renders it on a loopback. Never writes ``retry_count`` or
    ``last_critic_review`` — the critic's accounting stays pristine.
    """
    base: dict[str, Any] = {
        "current_phase": PhaseName.ADVERSARIAL.value,
        "status": "running",
        "prompt_request": None,
    }

    agent_result = subgraph_result.get("agent_result", {})
    if agent_result:
        effective_result = agent_result
    else:
        effective_result = {
            "status": ReviewStatus.PASSED.value,
            "tier": "adversarial",
            "reason": "No adversarial review performed",
            "suggestions": [],
        }

    # Adversarial always reviews the PLAN (see registration above); stamp it so
    # experience capture attributes the feedback entry without reason-parsing.
    effective_result = {**effective_result, "phase": PhaseName.PLAN.value}
    base["feedback"] = [effective_result]

    phase_status = subgraph_result.get("phase_status", "") or effective_result.get(
        "status", ""
    )
    prior = parent_state.get("adversarial_retry_count", 0)
    max_adv = parent_state.get("max_adversarial_retries", 2)
    new_attempt = prior + 1

    blocker_category = effective_result.get("blocker_category")
    spec_contradiction = blocker_category == "spec_contradiction"

    # Escalation decision on the adversarial budget. A NEEDS_REVISION verdict
    # loops back to PLAN while budget remains; once the budget is spent it
    # escalates instead of looping forever. A NEEDS_REVIEW verdict (or a spec
    # contradiction) escalates immediately — those need human judgement.
    is_revision = phase_status == ReviewStatus.NEEDS_REVISION.value
    is_review = phase_status == ReviewStatus.NEEDS_REVIEW.value
    exhausted = is_revision and prior >= max_adv
    escalate = is_review or spec_contradiction or exhausted

    if spec_contradiction:
        escalation_kind = "spec_amendment"
    elif exhausted:
        escalation_kind = "adversarial_exhausted"
    elif is_review:
        escalation_kind = "adversarial_flagged"
    else:
        escalation_kind = None

    base["last_adversarial_review"] = {
        "phase": PhaseName.PLAN.value,
        "status": phase_status,
        "tier": effective_result.get("tier", "adversarial"),
        "reason": effective_result.get("reason", ""),
        "suggestions": effective_result.get("suggestions", []),
        "attempt": new_attempt if is_revision else prior,
        "blocker_category": blocker_category,
        "escalate": escalate,
        "escalation_kind": escalation_kind,
    }

    # Count the round only when we actually loop back to PLAN.
    if is_revision and not escalate:
        base["adversarial_retry_count"] = new_attempt

    if phase_status == "error":
        base["status"] = "failed"
    elif escalate:
        base["status"] = "needs_review"
        # A spec contradiction targets SPECIFY so a "rework" amends the spec;
        # every other escalation points the reviewer at the PLAN.
        base["needs_review_phase"] = (
            PhaseName.SPECIFY.value if spec_contradiction else PhaseName.PLAN.value
        )
        base["needs_review_kind"] = escalation_kind
    elif is_revision:
        # Loop back to PLAN with budget remaining.
        base["status"] = "running"
    else:
        # PASSED — the plan survived the red-team.
        base["status"] = "running"
        base["adversarial_plan_completed"] = True

    return base


def adversarial_router(state: WorkflowState) -> str:
    """Conditional edge function for the adversarial node.

    Reads ``last_adversarial_review`` (written by
    :func:`_adversarial_result_mapper`) and returns the routing key:
    - ``"passed"`` → proceed (implement gate, or END → awaiting_approval)
    - ``"needs_revision"`` → loop the plan back to PLAN (budget remains)
    - ``"needs_review"`` → escalate (human review / flag)
    - ``"failed"`` → stop the workflow as failed
    """
    if state.get("status") == "failed":
        return "failed"

    lar = state.get("last_adversarial_review") or {}
    if not lar:
        logger.warning(
            "adversarial_router: last_adversarial_review missing — routing needs_revision"
        )
        return "needs_revision"

    status = lar.get("status", ReviewStatus.NEEDS_REVISION.value)
    decision: str
    if status == ReviewStatus.PASSED.value:
        decision = "passed"
    elif status == ReviewStatus.NEEDS_REVIEW.value or lar.get("escalate"):
        # Direct human-judgement verdict, or a revision whose budget is spent.
        decision = "needs_review"
    else:
        decision = "needs_revision"

    logger.info(
        "[%s] adversarial_router: status=%s rounds=%d/%d kind=%s → %s",
        state.get("work_id", "?"),
        status,
        state.get("adversarial_retry_count", 0),
        state.get("max_adversarial_retries", 2),
        lar.get("escalation_kind"),
        decision,
    )
    return decision


def _gap_plan_state_mapper(parent_state: WorkflowState, config) -> dict:
    """Map parent WorkflowState to GapPlanSubgraphState."""
    work_id = parent_state.get("work_id", "")
    return {
        **_base_state_mapper(parent_state, config),
        "phase": PhaseName.GAP_PLAN.value,
        "retry_count": 0,
        "verify_path": artifact_path(work_id, PhaseName.VERIFY.value),
        "plan_path": artifact_path(work_id, PhaseName.PLAN.value),
    }


def _gap_plan_result_mapper(subgraph_result: dict, parent_state: WorkflowState) -> dict[str, Any]:
    """Map GapPlanSubgraphState output back to parent WorkflowState."""
    base = make_success_result_mapper(PhaseName.GAP_PLAN.value)(subgraph_result, parent_state)
    phase_status = subgraph_result.get("phase_status", "")
    if phase_status == "needs_review":
        base["status"] = "needs_review"
        base["needs_review_phase"] = PhaseName.GAP_PLAN.value
    elif phase_status == "error":
        base["status"] = "failed"
    # Set completion invariants
    base["gap_plan_completed"] = phase_status == "success"
    base["gaps_identified"] = len(subgraph_result.get("gaps", []))
    return base


# ── Phase sequences per work type ──
# Each tuple is (node_name, reviewed_phase_or_None).
# For critic nodes, reviewed_phase tells the critic which phase to review.
#
# Reviewed work types (REVIEWED_TASK, CRITICAL_REVIEWED_TASK) terminate
# after critic_plan. The graph reaching END is the human-review gate:
# the dispatcher relabels the resulting "completed" status as
# "awaiting_approval", and the user reviews and either approves
# (spawning fresh execution tasks via approve_and_spawn) or requests
# revision (which re-runs the planning graph with feedback).
#
# Letting reviewed types fall through to IMPLEMENT/VERIFY would skip
# the human gate entirely — the entire point of the workflow.

WORKFLOW_SEQUENCES: dict[str, list[tuple[str, str | None]]] = {
    WorkType.TASK.value: [
        (PhaseName.SPECIFY.value, None),
        (PhaseName.PLAN.value, None),
        (f"{PhaseName.CRITIC.value}_plan", PhaseName.PLAN.value),
        (PhaseName.IMPLEMENT.value, None),
        (PhaseName.VERIFY.value, None),
    ],
    WorkType.CRITICAL_TASK.value: [
        (PhaseName.SPECIFY.value, None),
        (PhaseName.PLAN.value, None),
        (f"{PhaseName.CRITIC.value}_plan", PhaseName.PLAN.value),
        (f"{PhaseName.ADVERSARIAL.value}_plan", PhaseName.PLAN.value),
        (PhaseName.IMPLEMENT.value, None),
        (PhaseName.VERIFY.value, None),
    ],
    WorkType.REVIEWED_TASK.value: [
        (PhaseName.SPECIFY.value, None),
        (PhaseName.PLAN.value, None),
        (f"{PhaseName.CRITIC.value}_plan", PhaseName.PLAN.value),
    ],
    WorkType.CRITICAL_REVIEWED_TASK.value: [
        (PhaseName.SPECIFY.value, None),
        (PhaseName.PLAN.value, None),
        (f"{PhaseName.CRITIC.value}_plan", PhaseName.PLAN.value),
        (f"{PhaseName.ADVERSARIAL.value}_plan", PhaseName.PLAN.value),
    ],
}


def _human_review_interrupt(state: WorkflowState) -> dict:
    """Interrupt the graph for human review between phases.

    The workflow pauses here. A human (via UI or CLI) reviews the
    current state and calls ``Command(resume={...})`` to continue.

    Returns:
        A dict with the human decision and feedback.
    """
    needs_review_phase = state.get("needs_review_phase", "")
    feedback = state.get("feedback", [])
    phase_results = state.get("phase_results", {})

    # Feedback lists can contain non-dict entries (e.g. tuples from state
    # merges) — use the last dict entry, matching how the dispatcher filters.
    last_fb = next((f for f in reversed(feedback) if isinstance(f, dict)), {})
    lcr = state.get("last_critic_review") or {}
    review_info = {
        "phase": needs_review_phase or state.get("current_phase", ""),
        "reason": last_fb.get("reason", "No reason provided"),
        "suggestions": last_fb.get("suggestions", []),
        "phase_results": phase_results,
        # Why the workflow paused — lets the reviewer (UI/CLI) distinguish a
        # spec amendment from a non-converging rework loop or an exhausted
        # budget, and shows which asks remained unaddressed.
        "kind": state.get("needs_review_kind"),
        "unaddressed_points": lcr.get("unaddressed_points", []),
    }

    # interrupt() pauses the graph. Human response comes back via Command(resume=...)
    human_decision = interrupt(review_info)

    # Stash the review target on the decision BEFORE we clear needs_review_phase
    # below. LangGraph applies this node's state update before evaluating the
    # outgoing conditional edge, so the router would otherwise read a nulled
    # needs_review_phase and collapse "rework" → "abort" (and "approve" → the
    # wrong phase via its current_phase fallback).
    decision = dict(human_decision) if isinstance(human_decision, dict) else {"action": "abort"}
    decision.setdefault("_review_target", needs_review_phase or state.get("current_phase", ""))

    return {
        "human_feedback": decision,
        "needs_review_phase": None,
        "needs_review_kind": None,
    }


def _flag_needs_review_terminal(state: WorkflowState) -> dict:
    """Terminal sink for needs_review on autonomous work types.

    Reviewed work types pause at the ``human_review`` interrupt for a human
    to resume. Autonomous types (task / critical_task) have no human in the
    loop, so blocking on the interrupt strands the run. This node instead
    records the flagged status and lets the graph reach END cleanly — the
    dispatcher then surfaces ``needs_review`` to the caller (e.g. ``spine
    status``) rather than an indefinitely-paused interrupt.
    """
    return {
        "status": "needs_review",
        "needs_review_phase": state.get("needs_review_phase")
        or state.get("current_phase", ""),
    }


def _make_human_review_router(phase_seq: list[tuple[str, str | None]]):
    """Create a human review router that knows the phase sequence.

    Returns a router function for add_conditional_edges that routes:
    - "rework" → back to the needs_review_phase node
    - "approve" → to the phase after needs_review_phase in the sequence
    - "abort" → END
    """

    # Build a lookup: phase_name → index in phase_seq
    phase_index = {}
    for idx, (name, _) in enumerate(phase_seq):
        phase_index[name] = idx

    def router(state: WorkflowState) -> str:
        human_feedback = state.get("human_feedback", {})
        action = (
            human_feedback.get("action", "abort") if isinstance(human_feedback, dict) else "abort"
        )

        # The interrupt node nulls needs_review_phase in the same update the
        # router reads, so prefer the target it stashed on the decision; fall
        # back to live state only if an older resume payload lacks it.
        target = None
        if isinstance(human_feedback, dict):
            target = human_feedback.get("_review_target")
        target = target or state.get("needs_review_phase")

        if action == "rework":
            if target and target in phase_index:
                return target
            # Fallback: if we can't find the phase, still try rework
            return target or "abort"

        if action == "approve":
            if target and target in phase_index:
                idx = phase_index[target]
                if idx + 1 < len(phase_seq):
                    return phase_seq[idx + 1][0]
                return END
            # Fallback: advance from current_phase
            current = state.get("current_phase", "")
            if current in phase_index:
                idx = phase_index[current]
                if idx + 1 < len(phase_seq):
                    return phase_seq[idx + 1][0]
            return END

        return "abort"

    return router


def _phase_status_router(state: WorkflowState) -> str:
    """Generic post-phase router.

    Reads ``state["status"]`` after a phase subgraph completes and decides
    whether to proceed to the next node or halt for human review.  This is
    the missing guard that previously let needs_review propagate forward
    silently (Bug B): the result mapper set status=needs_review, but the
    bare ``graph.add_edge(phase, next)`` ignored it.

    Returns:
        ``"proceed"`` when status is ``"running"`` (the phase succeeded),
        ``"needs_review"`` when status is ``"needs_review"`` or anything
        unexpected.  Routing targets are bound at edge-construction time
        in :func:`build_workflow_graph`.
    """
    status = state.get("status", "running")
    if status == "needs_review":
        return "needs_review"
    if status == "failed":
        return "failed"
    return "proceed"


def _verify_router(state: WorkflowState) -> str:
    """Post-verify conditional edge router.

    Unlike the generic ``_phase_status_router``, this router can send a
    failed verification into a ``gap_plan → implement → verify`` loop
    instead of immediately flagging for human review.

    Returns:
        ``"passed"`` when verification succeeded,
        ``"needs_gap_fix"`` when verification failed with gap attempts remaining,
        ``"needs_review"`` when verification failed and gap attempts exhausted,
        ``"failed"`` for hard errors.
    """
    status = state.get("status", "running")
    if status == "needs_gap_fix":
        return "needs_gap_fix"
    if status == "needs_review":
        return "needs_review"
    if status == "failed":
        return "failed"
    return "passed"


def _gate_node_name(source_node: str, next_node: str) -> str:
    """Derive a unique node name for a gate between two phases."""
    return f"gate_{source_node}_to_{next_node}"


def build_workflow_graph(
    work_type: str,
    checkpointer: BaseCheckpointSaver | None = None,
    start_from_phase: str | None = None,
) -> Any:
    """Build a compiled LangGraph StateGraph for the given work type.

    The graph wires phase nodes with:
    - Sequential edges for non-critic phases
    - Conditional edges after critic nodes that route to rework or next phase
    - Artifact gate nodes that check prerequisites before verify/implement
    - Retry counting and needs_review escalation

    Critic nodes are named ``critic_{reviewed_phase}`` to allow multiple
    critic instances in a single workflow (e.g. critical_task has 2).

    Args:
        work_type: One of "task", "critical_task", "reviewed_task",
            "critical_reviewed_task".
        checkpointer: Optional BaseCheckpointSaver for persistence.
        start_from_phase: Optional phase name to start execution from.
            When set, the START edge routes directly to this phase instead
            of the first phase in the sequence. Used by restart_from_phase
            to resume a stalled job partway through the workflow.

    Returns:
        A compiled StateGraph ready for ``.invoke()`` or ``.stream()``.

    Raises:
        ValueError: If the work_type is not recognised.
    """
    if work_type not in WORKFLOW_SEQUENCES:
        raise ValueError(
            f"Unknown work type '{work_type}'. Must be one of: {list(WORKFLOW_SEQUENCES.keys())}"
        )

    phase_seq = WORKFLOW_SEQUENCES[work_type]
    registry = get_registry()

    # Reviewed work types (reviewed_task / critical_reviewed_task) END at
    # critic_plan and are designed to pause for a human. Autonomous types
    # (task / critical_task) have no human watching — routing their
    # needs_review verdicts into the human_review interrupt strands the run
    # waiting on input that never comes (trace 019e77a7: a plain `task` hit
    # the interrupt after the rework cap). For those, terminate cleanly with
    # status=needs_review instead.
    is_reviewed = work_type in (
        WorkType.REVIEWED_TASK.value,
        WorkType.CRITICAL_REVIEWED_TASK.value,
    )
    needs_review_target = "human_review" if is_reviewed else "flag_needs_review"

    # Validate start_from_phase if provided
    if start_from_phase is not None:
        valid_nodes = {name for name, _ in phase_seq}
        if start_from_phase not in valid_nodes:
            raise ValueError(
                f"Phase '{start_from_phase}' is not a valid node for work type "
                f"'{work_type}'. Valid nodes: {sorted(valid_nodes)}"
            )

    # ── Build the graph ──
    graph = StateGraph(WorkflowState)

    # Collect which edges need artifact gates.
    # We gate based on the *target* node: implement requires plan artifacts.
    # Verify always runs after implement — it's the phase that confirms
    # implementation meets requirements.  If implement produced nothing,
    # verify can detect and report that; there is no reason for a human
    # review gate between implement and verify.
    gate_edges: dict[tuple[str, str], str] = {}  # (src, dst) → required_phase
    _human_review_targets: dict[str, dict[str, str]] = {}  # node → {rework, approve}
    for i, (node_name, _reviewed_phase) in enumerate(phase_seq):
        if i < len(phase_seq) - 1:
            next_node_name = phase_seq[i + 1][0]
            if next_node_name == PhaseName.IMPLEMENT.value:
                gate_edges[(node_name, next_node_name)] = PhaseName.PLAN.value

    # ── Exploration subgraph override ──
    # When exploration mode is enabled for a phase, replace the standard
    # linear subgraph builder with the multi-node research loop.
    # This must happen before the phase node loop below so the builder
    # lookup finds the exploration subgraph instead of the standard one.
    if _USE_EXPLORATION_SUBGRAPH.get(PhaseName.SPECIFY.value, False):
        register_subgraph_builder(
            PhaseName.SPECIFY.value,
            lambda: build_exploration_subgraph(phase=PhaseName.SPECIFY.value),
        )
    if _USE_EXPLORATION_SUBGRAPH.get(PhaseName.PLAN.value, False):
        register_subgraph_builder(
            PhaseName.PLAN.value,
            lambda: build_exploration_subgraph(phase=PhaseName.PLAN.value),
        )

    # Add all phase/critic nodes
    for node_name, reviewed_phase in phase_seq:
        if node_name.startswith(PhaseName.ADVERSARIAL.value):
            # Adversarial node — single-tier red-team subgraph reviewing PLAN.
            # Checked before the critic branch and the generic _SUBGRAPH_ENABLED
            # path so "adversarial_plan" routes here (it does not start with
            # "critic", so there is no collision).
            adv_subgraph = build_adversarial_subgraph().compile()
            graph.add_node(
                node_name,
                make_subgraph_node(
                    adv_subgraph,
                    node_name,
                    _adversarial_state_mapper,
                    _adversarial_result_mapper,
                    use_per_phase_checkpointer=True,
                ),
            )
        elif node_name.startswith(PhaseName.CRITIC.value):
            # Critic node — use subgraph if enabled, else legacy
            _reviewed = reviewed_phase or "unknown"
            if _SUBGRAPH_ENABLED.get(PhaseName.CRITIC.value, False):
                critic_subgraph = build_critic_subgraph(_reviewed).compile()
                graph.add_node(
                    node_name,
                    make_subgraph_node(
                        critic_subgraph,
                        node_name,
                        _critic_state_mapper(_reviewed),
                        _critic_result_mapper(_reviewed),
                        use_per_phase_checkpointer=True,
                    ),
                )
            else:
                critic_def = registry.require(PhaseName.CRITIC.value)
                critic_fn = critic_def.call_fn or critic_def.subgraph_node_fn
                if critic_fn is None:
                    raise ValueError(
                        f"Critic phase '{node_name}' has no call_fn or subgraph_node_fn"
                    )
                graph.add_node(
                    node_name,
                    _make_critic_node(critic_fn, _reviewed),
                )
        elif _SUBGRAPH_ENABLED.get(node_name, False):
            # Subgraph node (new style) — look up builder, mapper, and result mapper
            # from the module-level registry and lookup tables.
            _STATE_MAPPERS = {
                PhaseName.VERIFY.value: _verify_state_mapper,
                PhaseName.IMPLEMENT.value: _implement_state_mapper,
                PhaseName.SPECIFY.value: _specify_state_mapper,
                PhaseName.PLAN.value: _plan_state_mapper,
                PhaseName.GAP_PLAN.value: _gap_plan_state_mapper,
            }
            _RESULT_MAPPERS = {
                PhaseName.VERIFY.value: _verify_result_mapper,
                PhaseName.IMPLEMENT.value: _implement_result_mapper,
                PhaseName.SPECIFY.value: _specify_result_mapper,
                PhaseName.PLAN.value: _plan_result_mapper,
                PhaseName.GAP_PLAN.value: _gap_plan_result_mapper,
            }

            builder_fn = _SUBGRAPH_BUILDER_REGISTRY.get(node_name)
            if builder_fn is not None:
                subgraph = builder_fn().compile()
                state_mapper = _STATE_MAPPERS[node_name]
                result_mapper = _RESULT_MAPPERS[node_name]

                def _build_node(
                    phase: str,
                    sub: Any,
                    sm: Any,
                    rm: Any,
                ) -> Any:
                    return make_subgraph_node(
                        sub,
                        phase,
                        sm,
                        rm,
                        use_per_phase_checkpointer=True,
                    )

                graph.add_node(
                    node_name,
                    _build_node(
                        node_name,
                        subgraph,
                        state_mapper,
                        result_mapper,
                    ),
                )
            else:
                # Fallback to legacy for phases not yet migrated
                phase_def = registry.require(node_name)
                if phase_def.subgraph_node_fn:
                    graph.add_node(
                        node_name, _make_legacy_node(node_name, phase_def.subgraph_node_fn)
                    )
                elif phase_def.call_fn:
                    graph.add_node(node_name, _make_legacy_node(node_name, phase_def.call_fn))
                else:
                    raise ValueError(f"Phase '{node_name}' has no call_fn or subgraph_node_fn")
        else:
            # Legacy phase node — wrap with phase-start tracking
            phase_def = registry.require(node_name)
            if phase_def.call_fn is None:
                raise ValueError(f"Phase '{node_name}' has no call_fn (legacy mode)")
            graph.add_node(node_name, _make_legacy_node(node_name, phase_def.call_fn))

    # Add human review interrupt node (once, not per gate)
    # Build the human_review conditional-edge map.  The router can return:
    #   - a phase node name (rework or approve → that node),
    #   - "abort" → END,
    #   - END (approve past last phase).
    # Every possible return value must appear in the map or LangGraph raises
    # KeyError at runtime.
    _hr_ends: dict[str, str] = {"abort": END, END: END}
    for name, _ in phase_seq:
        _hr_ends[name] = name
    # gap_plan only exists when verify is in the sequence (recovery loop).
    if PhaseName.VERIFY.value in {n for n, _ in phase_seq}:
        _hr_ends[PhaseName.GAP_PLAN.value] = PhaseName.GAP_PLAN.value

    graph.add_node("human_review", _human_review_interrupt)
    graph.add_conditional_edges(
        "human_review",
        _make_human_review_router(phase_seq),
        _hr_ends,
    )

    # Terminal needs_review sink for autonomous work types. Sets the flagged
    # status and ends the graph instead of blocking on the human interrupt.
    if not is_reviewed:
        graph.add_node("flag_needs_review", _flag_needs_review_terminal)
        graph.add_edge("flag_needs_review", END)

    # Reviewed work types terminate after critic_plan and never reach
    # IMPLEMENT/VERIFY/GAP_PLAN. Compiling the prereq gates and the
    # gap_plan recovery node anyway would leave conditional edges pointing
    # at non-existent targets (LangGraph rejects this at compile time).
    seq_phase_names = {name for name, _ in phase_seq}
    has_implement = PhaseName.IMPLEMENT.value in seq_phase_names
    has_verify = PhaseName.VERIFY.value in seq_phase_names

    # Add gap_plan node — not in WORKFLOW_SEQUENCES because it's a
    # conditional node reached via the verify_router, not a linear step.
    # Only relevant when verify is present (gap_plan is the recovery loop
    # after a failed verification).
    if has_verify:
        if _SUBGRAPH_ENABLED.get(PhaseName.GAP_PLAN.value, False):
            gap_subgraph = build_gap_plan_subgraph().compile()
            graph.add_node(
                PhaseName.GAP_PLAN.value,
                make_subgraph_node(
                    gap_subgraph,
                    PhaseName.GAP_PLAN.value,
                    _gap_plan_state_mapper,
                    _gap_plan_result_mapper,
                    use_per_phase_checkpointer=True,
                ),
            )
        else:
            # Legacy gap_plan node — wrap with phase-start tracking
            gap_plan_def = registry.require(PhaseName.GAP_PLAN.value)
            if gap_plan_def.call_fn is None:
                raise ValueError(f"Phase '{PhaseName.GAP_PLAN.value}' has no call_fn (legacy mode)")
            graph.add_node(PhaseName.GAP_PLAN.value, _make_legacy_node(PhaseName.GAP_PLAN.value, gap_plan_def.call_fn))

    # ── Prerequisite Gate Nodes ─────────────────────────────────────────
    # These gates check phase completion invariants before allowing phases to run.
    # They are wired inline with the graph edges to block empty progression.
    # Only add a gate when its target phase is part of this work type's
    # sequence — wiring a gate that targets a missing node breaks compile.

    # Gate: PLAN requires SPECIFY completed (PLAN is in every work type)
    prereq_gate_plan = make_prerequisite_gate_node(_check_spec_prerequisite, PhaseName.PLAN.value)
    graph.add_node("prereq_gate_plan", prereq_gate_plan)

    if has_implement:
        prereq_gate_implement = make_prerequisite_gate_node(
            _check_plan_prerequisite, PhaseName.IMPLEMENT.value
        )
        graph.add_node("prereq_gate_implement", prereq_gate_implement)

    if has_verify:
        prereq_gate_verify = make_prerequisite_gate_node(
            _check_implement_prerequisite, PhaseName.VERIFY.value
        )
        graph.add_node("prereq_gate_verify", prereq_gate_verify)

        prereq_gate_gap_plan = make_prerequisite_gate_node(
            _check_verify_prerequisite, PhaseName.GAP_PLAN.value
        )
        graph.add_node("prereq_gate_gap_plan", prereq_gate_gap_plan)

    # Add artifact gate nodes and their outgoing conditional edges.
    # Gate nodes route to ``next_node`` on proceed or ``human_review`` on
    # needs_review.  Adding the conditional edges here (rather than in the
    # per-phase loop below) ensures the gate's outgoing edges exist
    # regardless of whether its source is a phase node or a critic node —
    # the prior placement skipped the gate-outgoing edges when the gate
    # source was a critic, leaving the gate as a dead-end node.
    for (src, dst), required_phase in gate_edges.items():
        gate_name = _gate_node_name(src, dst)
        graph.add_node(
            gate_name,
            make_artifact_gate_node(required_phase, dst),
        )
        # Route through prerequisite gate when target has one.
        # Without this, critic_plan → gate_* → IMPLEMENT bypasses
        # prereq_gate_implement, skipping the plan_completed invariant check.
        prereq_gate_map = {
            PhaseName.PLAN.value: "prereq_gate_plan",
            PhaseName.IMPLEMENT.value: "prereq_gate_implement",
            PhaseName.VERIFY.value: "prereq_gate_verify",
        }
        actual_dst = prereq_gate_map.get(dst, dst)
        graph.add_conditional_edges(
            gate_name,
            artifact_gate_router,
            {
                "proceed": actual_dst,
                "needs_review": needs_review_target,
            },
        )

    # Wire the graph
    if start_from_phase:
        graph.add_edge(START, start_from_phase)
    else:
        graph.add_edge(START, phase_seq[0][0])

    for i, (node_name, reviewed_phase) in enumerate(phase_seq):
        is_last = i == len(phase_seq) - 1
        next_node = phase_seq[i + 1][0] if not is_last else None

        # Check if there's a gate between this node and the next
        edge_key = (node_name, next_node) if next_node else None
        has_gate = edge_key in gate_edges if edge_key else False

        # Determine the target for "proceed" (after phase succeeds)
        # This includes both artifact gates AND prerequisite gates
        def get_proceed_target(next_node_name: str | None) -> str:
            """Get the target node, potentially routing through prerequisite gate."""
            if next_node_name is None:
                return END
            # Map target phases to their prerequisite gates
            prereq_gate = {
                PhaseName.PLAN.value: "prereq_gate_plan",
                PhaseName.IMPLEMENT.value: "prereq_gate_implement",
                PhaseName.VERIFY.value: "prereq_gate_verify",
            }
            return prereq_gate.get(next_node_name, next_node_name)

        if node_name.startswith(PhaseName.ADVERSARIAL.value):
            # Adversarial node → conditional edge. "passed" proceeds (implement
            # gate for critical_task, END for critical_reviewed_task);
            # "needs_revision" loops the plan back to PLAN (NOT the previous
            # node, which is critic_plan); "needs_review" escalates.
            if has_gate and next_node:
                adv_proceed_target: str = _gate_node_name(node_name, next_node)
            elif not is_last and next_node:
                adv_proceed_target = get_proceed_target(next_node)
            else:
                adv_proceed_target = END

            graph.add_conditional_edges(
                node_name,
                adversarial_router,
                {
                    "passed": adv_proceed_target,
                    "needs_revision": PhaseName.PLAN.value,  # rework loop → plan
                    "needs_review": needs_review_target,  # human gate or terminal flag
                    "failed": END,
                },
            )
        elif node_name.startswith(PhaseName.CRITIC.value):
            # Critic node → conditional edge
            pre_critic = phase_seq[i - 1][0] if i > 0 else phase_seq[0][0]

            # Determine where "passed" routes to
            if has_gate and next_node:
                gate_name = _gate_node_name(node_name, next_node)
                critic_proceed_target: str = gate_name
            elif not is_last and next_node:
                critic_proceed_target = get_proceed_target(next_node)
            else:
                critic_proceed_target = END

            graph.add_conditional_edges(
                node_name,
                critic_router,
                {
                    "passed": critic_proceed_target,
                    "needs_revision": pre_critic,  # rework loop
                    "needs_review": needs_review_target,  # human gate or terminal flag
                    "failed": END,
                },
            )
        elif has_gate and next_node:
            # Route to the artifact gate node; its outgoing conditional edges
            # were registered in the gate-node loop above.
            gate_name = _gate_node_name(node_name, next_node)
            graph.add_edge(node_name, gate_name)
        elif node_name == PhaseName.VERIFY.value:
            # Verify uses its own router for gap-fix loop support.
            # On passed → END. On needs_gap_fix → prereq_gate_gap_plan → gap_plan → implement → verify.
            # On needs_review (gap attempts exhausted) → human_review.
            graph.add_conditional_edges(
                node_name,
                _verify_router,
                {
                    "passed": END,
                    "needs_gap_fix": "prereq_gate_gap_plan",
                    "needs_review": needs_review_target,
                    "failed": END,
                },
            )
        elif next_node is not None:
            # Status guard: if the phase produced needs_review, route
            # to human_review instead of charging into the next phase.
            # Without this, a failing specify silently feeds an empty
            # plan, which feeds an empty critic, and so on — burning
            # tokens and exceeding budgets.
            # Also route through prerequisite gate for the next phase.
            target = get_proceed_target(next_node)
            graph.add_conditional_edges(
                node_name,
                _phase_status_router,
                {
                    "proceed": target,
                    "needs_review": needs_review_target,
                    "failed": END,
                },
            )
        else:
            # Terminal phase: still guard so a needs_review on the
            # final phase routes to human_review for resume support.
            graph.add_conditional_edges(
                node_name,
                _phase_status_router,
                {
                    "proceed": END,
                    "needs_review": needs_review_target,
                    "failed": END,
                },
            )

    # Wire prerequisite gates to their target phases via conditional edges
    # (failure routes to human_review, success routes to target phase).
    # Gates that were skipped above must also be skipped here.
    graph.add_conditional_edges(
        "prereq_gate_plan",
        _phase_status_router,
        {
            "proceed": PhaseName.PLAN.value,
            "needs_review": needs_review_target,
            "failed": END,
        },
    )
    if has_implement:
        graph.add_conditional_edges(
            "prereq_gate_implement",
            _phase_status_router,
            {
                "proceed": PhaseName.IMPLEMENT.value,
                "needs_review": needs_review_target,
                "failed": END,
            },
        )
    if has_verify:
        graph.add_conditional_edges(
            "prereq_gate_verify",
            _phase_status_router,
            {
                "proceed": PhaseName.VERIFY.value,
                "needs_review": needs_review_target,
                "failed": END,
            },
        )
        graph.add_conditional_edges(
            "prereq_gate_gap_plan",
            _phase_status_router,
            {
                "proceed": PhaseName.GAP_PLAN.value,
                "needs_review": needs_review_target,
                "failed": END,
            },
        )
        # Wire gap_plan → implement (gap-fix loop: verify → gap_plan → implement → verify)
        graph.add_edge(PhaseName.GAP_PLAN.value, PhaseName.IMPLEMENT.value)

    # Compile with optional checkpointer
    compile_kwargs: dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    return graph.compile(**compile_kwargs)


def _make_critic_node(
    critic_fn: Any,
    reviewed_phase: str,
) -> Any:
    """Create a critic node function that knows which phase it reviews.

    Wraps the generic ``call_critic`` so the reviewed phase is determined
    by the graph position, not by inspecting state artifacts.

    The wrapper is async because ``call_critic`` is async — LangGraph
    handles async node functions natively.

    Args:
        critic_fn: The base critic call function (async).
        reviewed_phase: The phase this critic instance reviews.

    Returns:
        An async node function with the correct reviewed_phase.
    """

    async def critic_node(state: WorkflowState, config: Optional[RunnableConfig] = None) -> dict:
        """Critic node that reviews a specific phase."""
        mark_phase_started(state, f"critic_{reviewed_phase}")
        # Inject which phase this critic reviews into state
        # so _get_reviewed_phase and critic_router can use it
        augmented_state = {**state, "critic_reviewing": reviewed_phase}
        result = await critic_fn(augmented_state, config)
        return result

    return critic_node


def _make_legacy_node(
    phase_name: str,
    call_fn: Any,
) -> Any:
    """Create a legacy node function with phase-start tracking.

    Wraps the generic phase call function so it marks the phase as
    started before executing. This ensures the UI shows the correct
    phase immediately.

    Args:
        phase_name: The phase identifier (e.g. "specify", "plan").
        call_fn: The async call function for this phase.

    Returns:
        An async node function that marks the phase started, then calls
        the original phase function.
    """

    async def legacy_node(state: WorkflowState, config: Optional[RunnableConfig] = None) -> dict:
        """Legacy node wrapper with phase-start tracking."""
        mark_phase_started(state, phase_name)
        result = await call_fn(state, config)
        return result

    return legacy_node


def get_restart_phases(work_type: str) -> list[str]:
    """Return the list of valid phase names for restart_from_phase.

    Filters out critic and adversarial review nodes since restarting into a
    review doesn't make sense — they are always called after the phase they
    review (PLAN).

    Args:
        work_type: One of the valid WorkType values.

    Returns:
        Sorted list of non-review phase names from the workflow sequence.

    Raises:
        ValueError: If the work_type is not recognised.
    """
    if work_type not in WORKFLOW_SEQUENCES:
        raise ValueError(
            f"Unknown work type '{work_type}'. Must be one of: {list(WORKFLOW_SEQUENCES.keys())}"
        )
    return sorted(
        name
        for name, _ in WORKFLOW_SEQUENCES[work_type]
        if not name.startswith(PhaseName.CRITIC.value)
        and not name.startswith(PhaseName.ADVERSARIAL.value)
    )
