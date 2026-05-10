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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active", active)
    col2.metric("Complete", complete)
    col3.metric("Blocked", blocked)
    if errors > 0:
        col4.metric("Errors", errors, delta=None, help=f"{errors} work items in error state")
    else:
        col4.metric("Errors", 0)

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
        progress = item["completed_tasks"] / max(1, item.get("total_tasks", 1))
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
