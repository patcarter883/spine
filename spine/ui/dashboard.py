"""Dashboard page — list of all work items with status and progress."""

import time
from datetime import datetime, timezone

import streamlit as st

from spine.config import dashboard_config
from spine.ui.utils import (
    get_active_work_items,
    format_phase_icon,
    format_phase_color,
)


def render_dashboard() -> None:
    """Render the main dashboard page showing all work items."""
    st.title("📊 SPINE Dashboard")

    # ── Polling / auto-refresh controls ──
    col_poll, col_status, _ = st.columns([2, 2, 4])

    with col_poll:
        auto_refresh = st.toggle("Auto-refresh", value=True)
        raw_interval_s = st.select_slider(
            "Poll interval",
            options=[1, 2, 5, 10, 30],
            value=2,
            disabled=not auto_refresh,
            label_visibility="collapsed",
            help="How often to check for status updates (seconds).",
        )
        cfg = dashboard_config(raw_interval_s)
        poll_interval_s = cfg["poll_interval_s"]

    with col_status:
        if "dashboard_last_refresh" not in st.session_state:
            st.session_state.dashboard_last_refresh = datetime.now(timezone.utc)
        if auto_refresh:
            st.caption(
                f"🔄 Polling every {poll_interval_s}s "
                f"(last: {st.session_state.dashboard_last_refresh.strftime('%H:%M:%S')})"
            )

    # Fetch work items
    work_items = get_active_work_items()
    st.session_state.dashboard_last_refresh = datetime.now(timezone.utc)

    # Summary metrics
    active = sum(1 for w in work_items if w["phase"] not in ("COMPLETE", "ERROR", "BLOCKED"))
    complete = sum(1 for w in work_items if w["phase"] == "COMPLETE")
    blocked = sum(1 for w in work_items if w["phase"] == "BLOCKED")
    errors = sum(1 for w in work_items if w["phase"] == "ERROR")

    # Pending review = HUMAN_REVIEW or critic gate pending
    pending_review = sum(
        1 for w in work_items
        if w["phase"] == "HUMAN_REVIEW" or w.get("critic_gate_result") == "NEEDS_REVISION"
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Active", active)
    col2.metric("Complete", complete)
    col3.metric("Blocked", blocked)
    col4.metric("Needs Review", pending_review)
    if errors > 0:
        col5.metric("Errors", errors, delta=None, help=f"{errors} work items in error state")
    else:
        col5.metric("Errors", 0)

    # ── Phase Distribution Chart ──
    if work_items:
        phase_counts: dict[str, int] = {}
        for w in work_items:
            p = w.get("phase", "INIT")
            phase_counts[p] = phase_counts.get(p, 0) + 1

        if len(phase_counts) > 1:
            st.subheader("Phase Distribution")
            chart_data = {
                "Phase": list(phase_counts.keys()),
                "Count": list(phase_counts.values()),
            }
            st.bar_chart(chart_data, x="Phase", y="Count", use_container_width=True)

    # ── Quick Actions ──
    st.markdown("---")
    qa1, qa2, qa3 = st.columns(3)
    if qa1.button("➕ Start New Work", key="qa_new_work", use_container_width=True):
        st.session_state.page = "New Work"
        st.rerun()
    if qa2.button("⏳ View Queue", key="qa_queue", use_container_width=True):
        st.session_state.page = "Task Queue"
        st.rerun()
    if pending_review > 0 and qa3.button(
        f"👤 Review Pending ({pending_review})", key="qa_review", use_container_width=True
    ):
        # Navigate to first item needing review
        for w in work_items:
            if w["phase"] == "HUMAN_REVIEW" or w.get("critic_gate_result") == "NEEDS_REVISION":
                st.session_state.selected_work_id = w["thread_id"]
                st.session_state.page = "Work Detail"
                st.rerun()
                break

    if not work_items:
        st.info("No work items yet. Click **New Work** to get started.")
        st.divider()
        with st.expander("💡 What is SPINE?"):
            st.markdown(
                "SPINE is a deterministic AI agent harness that orchestrates "
                "multi-agent workflows through a state machine. It supports "
                "parallel execution, critic gates, and file reservations."
            )
        if auto_refresh:
            time.sleep(poll_interval_s)
            st.rerun()
        return

    # Work item cards
    st.subheader("Active Work Items")

    for item in work_items:
        progress = min(1.0, item["completed_tasks"] / max(1, item.get("total_tasks", 1)))
        icon = format_phase_icon(item["phase"])
        color = format_phase_color(item["phase"])

        with st.container():
            col_icon, col_title, col_phase, col_progress, col_actions = st.columns(
                [1, 3, 2, 3, 2]
            )

            col_icon.write(icon)
            col_title.write(f"**{item['requirement']}**")
            col_phase.write(f"[{color}]**{item['phase']}**[/]")

            col_progress.progress(
                progress,
                text=f"{item['completed_tasks']}/{item.get('total_tasks', 1)} tasks",
            )

            if col_actions.button("View", key=f"view_{item['thread_id']}"):
                st.session_state.selected_work_id = item["thread_id"]
                st.session_state.page = "Work Detail"
                st.rerun()

    # Footer
    st.divider()
    st.caption(
        f"{len(work_items)} work item(s) total • "
        f"Click 'View' to see details • "
        f"Auto-refresh is {'on' if auto_refresh else 'off'}"
    )

    # Auto-refresh polling loop
    if auto_refresh and any(
        w["phase"] not in ("COMPLETE", "ERROR") for w in work_items
    ):
        time.sleep(poll_interval_s)
        st.rerun()
