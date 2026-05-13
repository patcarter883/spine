"""SPINE Dashboard page — overview of all work."""

from __future__ import annotations

import streamlit as st

from spine.ui_api import UIApi
from spine.ui.utils import status_icon, truncate


def render(api: UIApi) -> None:
    """Render the dashboard page."""
    st.title("🦴 SPINE Dashboard")

    # ── Quick stats ──
    col1, col2, col3, col4 = st.columns(4)

    running = api.list_work(status="running", limit=100)
    completed = api.list_work(status="completed", limit=100)
    needs_review = api.list_work(status="needs_review", limit=100)
    failed = api.list_work(status="failed", limit=100)

    col1.metric("Running", len(running))
    col2.metric("Completed", len(completed))
    col3.metric("Needs Review", len(needs_review))
    col4.metric("Failed", len(failed))

    st.divider()

    # ── Recent work ──
    st.subheader("Recent Work")
    all_work = api.list_work(limit=20)

    if not all_work:
        st.info("No work items yet. Submit work to get started!")
        return

    for item in all_work:
        status = item.get("status", "unknown")
        icon = status_icon(status)
        phase = item.get("current_phase", "")
        desc = truncate(item.get("description", ""), 80)

        with st.container():
            col1, col2, col3 = st.columns([1, 4, 2])
            col1.write(f"{icon}")
            col2.write(f"**{item.get('id', '')}** — {desc}")
            col3.write(f"{phase or status}")

    # ── Worker status ──
    st.divider()
    st.subheader("Worker Status")
    try:
        worker_status = api.get_worker_status()
        if worker_status.get("running"):
            st.success("RalphLoopWorker is running")
        else:
            st.warning("RalphLoopWorker is not running")
        queue = worker_status.get("queue", {})
        if queue:
            st.json(queue)
    except Exception:
        st.info("Worker status unavailable")
