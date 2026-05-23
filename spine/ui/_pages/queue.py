"""SPINE Queue page — pending jobs, active job with phase/timing, recent history."""

from __future__ import annotations

import json

import streamlit as st

from spine.ui.utils import format_duration, status_icon
from spine.ui_api import UIApi

# ── Fragment refresh interval (seconds) ──
# Each data section auto-refreshes via @st.fragment(run_every=...) so
# only that fragment re-renders, preserving widget state elsewhere on
# the page (e.g. form inputs on the Submit Work page).
_POLL_INTERVAL = 10


# ── Helpers ──

_PHASE_SEQUENCE: dict[str, list[str]] = {
    "task": ["specify", "plan", "critic_plan", "implement", "verify"],
    "critical_task": [
        "specify",
        "critic_specify",
        "plan",
        "critic_plan",
        "implement",
        "verify",
    ],
    "reviewed_task": ["specify", "plan", "critic_plan", "implement", "verify"],
    "critical_reviewed_task": [
        "specify",
        "critic_specify",
        "plan",
        "critic_plan",
        "implement",
        "verify",
    ],
}

_PHASE_EMOJI = {
    "specify": "📐",
    "plan": "📋",
    "critic": "🔍",
    "critic_specify": "🔍",
    "critic_plan": "🔍",
    "critic_tasks": "🔍",
    "tasks": "📦",
    "implement": "🛠️",
    "verify": "✅",
}


def _render_phase_bar(phases: list[str], current: str) -> None:
    """Horizontal progress bar: completed / current / upcoming."""
    if not phases:
        return
    cols = st.columns(len(phases))
    # Resolve the current phase to an index.  Prefer exact match, then
    # match against the critic-of-phase form, then fall back to bare
    # phase name.  This handles both pre-merge node names ("critic_plan")
    # and post-merge canonical phase ("plan").
    try:
        current_idx = phases.index(current)
    except ValueError:
        # Try variants: e.g. current="critic" matched against "critic_plan"
        # by finding the first phase that starts with current_ or ends with
        # _current.
        current_idx = -1
        for i, p in enumerate(phases):
            if p == current or p.startswith(f"{current}_") or p.endswith(f"_{current}"):
                current_idx = i
                break

    for i, (col, phase) in enumerate(zip(cols, phases)):
        icon = _PHASE_EMOJI.get(phase, "⚙️")
        # Pretty label: "critic_plan" → "𝘊 Plan", "specify" → "Specify"
        if phase.startswith("critic_"):
            label = f"𝘊 {phase[len('critic_') :].title()}"
        else:
            label = phase.title()
        if current_idx < 0:
            # Unknown phase — show all as upcoming
            col.caption(f"○ {icon} {label}")
        elif i < current_idx:
            col.caption(f"✓ {icon} {label}")
        elif i == current_idx:
            col.markdown(f"**● {icon} {label}**")
        else:
            col.caption(f"○ {icon} {label}")


# ── Fragment sections ──


@st.fragment(run_every=_POLL_INTERVAL)
def _render_summary(api: UIApi) -> None:
    """Summary metrics — auto-refreshing fragment."""
    overview = api.get_queue_overview()
    summary = overview.get("status_summary", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pending", summary.get("pending", 0))
    c2.metric("Running", summary.get("running", 0))
    c3.metric("Failed", summary.get("failed", 0) + summary.get("stalled", 0))
    c4.metric("Completed", summary.get("completed", 0))


@st.fragment(run_every=_POLL_INTERVAL)
def _render_active_job(api: UIApi) -> None:
    """Active job detail — auto-refreshing fragment."""
    overview = api.get_queue_overview()
    active = overview.get("active")
    worker_status = api.get_worker_status()
    worker_running = worker_status.get("running", False)

    # Worker not running — show the start button and return.
    if not worker_running:
        st.info("⏸️ RalphLoopWorker is not running. Jobs will not be processed until started.")
        if st.button("▶️ Start Worker", use_container_width=True):
            from spine.work.ralph_worker import get_worker

            worker = get_worker(api._config)
            worker.start()
            st.rerun()
        return

    if not active:
        st.markdown(":gray[No active job — queue is idle or all jobs completed]")
        # Reset stuck items button (useful even when no active job)
        if st.button("🔄 Reset stuck running items", key="reset_stuck"):
            reset_count = api.reset_stuck_items()
            if reset_count:
                st.success(
                    f"Reset {reset_count} stuck item(s) back to pending. They will be reprocessed."
                )
            else:
                st.info("No stuck running items found.")
            st.rerun()
        return

    # ── Active job present — render once, cleanly ──
    # Prefer the work_id (UUID prefix from dispatcher) over the queue
    # sequence number for the displayed ID.  The queue PK is purely a
    # queue-internal sequence; work_id is the canonical identifier.
    work_id = active.get("work_id") or ""
    queue_id = active.get("id")
    display_id = work_id or f"queue-{queue_id}"

    status = active.get("status", "unknown")
    current_phase = active.get("current_phase") or "starting"
    work_type = active.get("work_type", "spec")
    created_at = active.get("created_at") or ""
    updated_at = active.get("updated_at") or ""
    description = active.get("description", "")

    st.subheader("🔄 Active Job")
    st.caption(
        f"Work ID: `{display_id}`" + (f"  ·  Queue #{queue_id}" if work_id and queue_id else "")
    )

    # Description
    if description:
        st.markdown(f"**{description[:200]}**")

    # Metrics row
    meta = st.columns(4)
    meta[0].metric("Work ID", f"`{display_id}`")
    meta[1].metric("Type", work_type)
    meta[2].metric("Status", status.title())
    phase_label = current_phase.replace("critic_", "𝘊 ").title() if current_phase else "Starting"
    meta[3].metric("Current Phase", phase_label)

    # Stop button for active running job
    if work_id:
        st.divider()
        if st.button("⏹ Stop Work", key=f"stop_work_{work_id}"):
            result = api.stop_work(work_id)
            st.success(f"Stop requested for work `{work_id}`. The job will be cancelled.")
            st.rerun()

    # Phase progress bar
    phases = _PHASE_SEQUENCE.get(work_type, [])
    if phases:
        st.markdown("**Progress:**")
        _render_phase_bar(phases, current_phase)
    else:
        st.caption(f"Current phase: {current_phase}")

    # Timing
    if created_at:
        try:
            from datetime import datetime as _dt

            start = _dt.fromisoformat(created_at)
            duration = format_duration(start)
            st.caption(
                f"Started {created_at[:19]}  ·  "
                f"Last updated {updated_at[:19] or '—'}  ·  "
                f"Elapsed: {duration}"
            )
        except Exception:
            st.caption(f"Started {created_at[:19]}")

    # If failed, show result/reason
    if status == "failed":
        result = active.get("result", "")
        if result:
            st.error("**Error:**")
            st.code(result)

    # Timing-based stalled heuristic (fallback when enrichment fails)
    if not active.get("work_status") and created_at:
        try:
            from datetime import datetime as _dt, timezone

            start = _dt.fromisoformat(created_at)
            elapsed = (
                _dt.now(timezone.utc) - start.replace(tzinfo=timezone.utc)
            ).total_seconds()
            if elapsed > 300:
                st.warning(
                    f"⚠️ This job has been running for {int(elapsed)}s and may be stalled."
                )
        except Exception:
            pass

    # Stalled detection (RC4)
    work_status = active.get("work_status")
    if work_status == "stalled":
        st.warning(
            "⚠️ This job appears to be **stalled**. "
            "The workflow hasn't produced output since the stall timeout. "
            "Use the Reset button below to retry."
        )

    # Reset stuck items button
    st.divider()
    if st.button("🔄 Reset stuck running items", key="reset_stuck"):
        reset_count = api.reset_stuck_items()
        if reset_count:
            st.success(
                f"Reset {reset_count} stuck item(s) back to pending. They will be reprocessed."
            )
        else:
            st.info("No stuck running items found.")
        st.rerun()


@st.fragment(run_every=_POLL_INTERVAL)
def _render_pending(api: UIApi) -> None:
    """Pending jobs list — auto-refreshing fragment."""
    overview = api.get_queue_overview()
    pending = overview.get("pending", [])

    if not pending:
        st.caption("No jobs waiting in the queue.")
    else:
        for item in pending:
            with st.container(border=True):
                pc1, pc2, pc3, pc4 = st.columns([4, 1, 2, 1])
                pc1.markdown(f"**{item.get('description', '')[:120]}**")
                pc2.write(item.get("work_type", ""))
                pc3.caption(f"Enqueued {format_duration(item.get('enqueued_at'))} ago")
                # Stop button for pending items
                work_id = item.get("work_id")
                if work_id and pc4.button("⏹ Stop", key=f"stop_pending_{item.get('id')}"):
                    api.stop_work(work_id)
                    st.success(f"Stop requested for work `{work_id}`.")
                    st.rerun()


@st.fragment(run_every=_POLL_INTERVAL)
def _render_recent(api: UIApi) -> None:
    """Recent results list — auto-refreshing fragment."""
    overview = api.get_queue_overview()
    recent = overview.get("recent", [])

    if not recent:
        st.caption("No completed jobs yet.")
    else:
        for item in recent:
            queue_status = item.get("status", "completed")
            icon = status_icon(queue_status)

            result_raw = item.get("result", "")
            inner_status = None
            if result_raw:
                try:
                    inner = json.loads(result_raw) if isinstance(result_raw, str) else result_raw
                    inner_status = inner.get("status") if isinstance(inner, dict) else None
                except (json.JSONDecodeError, TypeError):
                    pass
            display_status = inner_status or queue_status

            with st.container(border=True):
                rc1, rc2, rc3 = st.columns([4, 1, 2])
                rc1.markdown(f"{icon} **{item.get('description', '')[:100]}**")
                rc2.write(display_status)
                elapsed = format_duration(item.get("started_at"), item.get("completed_at"))
                rc3.caption(f"Finished {item.get('completed_at', '')[:10]} · Took {elapsed}")


# ── Page ──


def render(api: UIApi) -> None:
    """Render the Queue page."""
    st.title("🚦 Queue")

    # ── Summary metrics ──
    _render_summary(api)

    # ── Active job ──
    st.divider()
    st.subheader("▸ Active Job")
    _render_active_job(api)

    # ── Pending jobs ──
    st.divider()
    st.subheader("▸ Pending Jobs")
    _render_pending(api)

    # ── Recent history ──
    st.divider()
    st.subheader("▸ Recent Results")
    _render_recent(api)
