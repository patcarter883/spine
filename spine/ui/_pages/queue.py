"""SPINE Queue page — pending jobs, active job with phase/timing, recent history."""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from spine.ui_api import UIApi


# ── Helpers ──


def _elapsed(iso_str: str | None) -> str:
    """Human-readable elapsed time from an ISO timestamp."""
    if not iso_str:
        return "—"
    try:
        start = datetime.fromisoformat(iso_str)
        delta = datetime.now() - start
        total_secs = int(delta.total_seconds())
        if total_secs < 60:
            return f"{total_secs}s"
        mins = total_secs // 60
        hours = mins // 60
        if hours > 0:
            return f"{hours}h {mins % 60}m"
        return f"{mins}m {total_secs % 60}s"
    except (ValueError, TypeError):
        return "—"


_PHASE_SEQUENCE: dict[str, list[str]] = {
    "quick": ["tasks", "implement", "verify"],
    "critical_quick": ["tasks", "critic", "implement", "verify"],
    "spec": ["specify", "plan", "critic", "tasks", "implement", "verify"],
    "critical_spec": [
        "specify",
        "critic-specify",
        "plan",
        "critic-plan",
        "tasks",
        "critic-tasks",
        "implement",
        "verify",
    ],
}

_PHASE_EMOJI = {
    "specify": "📐",
    "plan": "📋",
    "critic": "🔍",
    "critic-specify": "🔍",
    "critic-plan": "🔍",
    "critic-tasks": "🔍",
    "tasks": "📦",
    "implement": "🛠️",
    "verify": "✅",
}


def _render_phase_bar(phases: list[str], current: str) -> None:
    """Horizontal progress bar: completed / current / upcoming."""
    if not phases:
        return
    cols = st.columns(len(phases))
    current_base = current.rsplit("_", 1)[-1] if "_" in current else current
    try:
        current_idx = next(
            i for i, p in enumerate(phases)
            if p == current or p == current_base or p.startswith(current_base)
        )
    except StopIteration:
        current_idx = -1

    for i, (col, phase) in enumerate(zip(cols, phases)):
        icon = _PHASE_EMOJI.get(phase, "⚙️")
        label = phase.replace("critic-", "𝘊 ").replace("-", " ").title().replace("Critic ", "𝘊 ")
        label = phase.capitalize()
        if current_idx < 0:
            # Unknown phase — show all as upcoming
            col.caption(f"○ {icon} {label}")
        elif i < current_idx:
            col.caption(f"✓ {icon} {label}")
        elif i == current_idx:
            col.markdown(f"**● {icon} {label}**")
        else:
            col.caption(f"○ {icon} {label}")


# ── Page ──


def render(api: UIApi) -> None:
    """Render the Queue page."""
    st.title("🚦 Queue")

    overview = api.get_queue_overview()
    summary = overview.get("status_summary", {})
    active = overview.get("active")
    pending = overview.get("pending", [])
    recent = overview.get("recent", [])

    # ── Summary metrics ──
    c1, c2, c3 = st.columns(3)
    c1.metric("Pending", summary.get("pending", 0))
    c2.metric("Running", summary.get("running", 0))
    c3.metric("Completed", summary.get("completed", 0))

    # ── Active job ──
    st.divider()
    st.subheader("▸ Active Job")

    worker_status = api.get_worker_status()
    if not worker_status.get("running"):
        st.info("⏸️ RalphLoopWorker is not running. Jobs will not be processed until started.")
        if st.button("▶️ Start Worker", use_container_width=True):
            from spine.work.ralph_worker import get_worker

            worker = get_worker(api._config)
            worker.start()
            st.rerun()
    elif not active:
        st.info("No job is currently running. Submit work from the **Submit Work** page.")
    else:
        # Description line
        st.markdown(f"**{active.get('description', '')[:150]}**")

        meta = st.columns(4)
        meta[0].metric("Work ID", f"`{active.get('id', '')}`")
        meta[1].metric("Type", active.get("work_type", ""))
        meta[2].metric("Total Time", _elapsed(active.get("created_at")))

        current_phase = active.get("current_phase", "starting")
        meta[3].metric("Current Phase", current_phase.title() if current_phase else "Starting")

        # Phase progress bar
        phases = _PHASE_SEQUENCE.get(active.get("work_type", ""), [])
        if phases:
            _render_phase_bar(phases, current_phase)

        # Timing detail
        st.caption(
            f"Started {active.get('created_at', '')[:19]}  ·  "
            f"Last updated {active.get('updated_at', '')[:19]}  ·  "
            f"Status: `{active.get('status', '')}`"
        )

    # ── Pending jobs ──
    st.divider()
    st.subheader("▸ Pending Jobs")
    if not pending:
        st.caption("No jobs waiting in the queue.")
    else:
        for item in pending:
            with st.container(border=True):
                pc1, pc2, pc3 = st.columns([4, 1, 2])
                pc1.markdown(f"**{item.get('description', '')[:120]}**")
                pc2.write(item.get("work_type", ""))
                pc3.caption(f"Enqueued {_elapsed(item.get('enqueued_at'))} ago")

    # ── Recent history ──
    st.divider()
    st.subheader("▸ Recent Results")
    if not recent:
        st.caption("No completed jobs yet.")
    else:
        for item in recent:
            emoji = "✅" if item.get("status") == "completed" else "❌"
            with st.container(border=True):
                rc1, rc2, rc3 = st.columns([4, 1, 2])
                rc1.markdown(f"{emoji} **{item.get('description', '')[:100]}**")
                rc2.write(item.get("status", ""))
                elapsed = _elapsed(item.get("started_at"))
                rc3.caption(f"Finished {item.get('completed_at', '')[:10]} · Took {elapsed}")
