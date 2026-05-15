"""SPINE Queue page — pending jobs, active job with phase/timing, recent history."""

from __future__ import annotations

import json

import streamlit as st

from spine.ui.utils import format_duration, status_icon
from spine.ui_api import UIApi


# ── Polling fallback interval (seconds) ──
# When the WebSocket bridge is disconnected or unavailable, this meta
# refresh tag ensures the page eventually updates without manual reload.
_POLL_INTERVAL = 30


# ── Helpers ──


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
            i
            for i, p in enumerate(phases)
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

    # ── Polling fallback ──
    # Ensure the page auto-refreshes even when the WebSocket bridge is
    # unavailable (port conflict, browser security, start-up race, etc.).
    # The meta refresh tag triggers a full Streamlit rerun at a fixed
    # interval, providing a graceful degradation path when live push
    # events are not arriving.
    st.markdown(
        f'<meta http-equiv="refresh" content="{_POLL_INTERVAL}">',
        unsafe_allow_html=True,
    )

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
        meta[2].metric("Total Time", format_duration(active.get("created_at")))

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
                pc3.caption(f"Enqueued {format_duration(item.get('enqueued_at'))} ago")

    # ── Recent history ──
    st.divider()
    st.subheader("▸ Recent Results")
    if not recent:
        st.caption("No completed jobs yet.")
    else:
        for item in recent:
            # Queue item status is now derived from the work result
            # (completed / failed / needs_review).
            queue_status = item.get("status", "completed")
            icon = status_icon(queue_status)

            # Also try to extract the inner work status for the tooltip
            # when the queue status was back-compat set to "completed".
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
