"""SPINE Dashboard page — overview of all work."""

from __future__ import annotations

import streamlit as st

from spine.ui.pages import get as get_page
from spine.ui.utils import status_icon, truncate
from spine.ui_api import UIApi

# ── Fragment refresh interval (seconds) ──
_POLL_INTERVAL = 10


@st.fragment(run_every=_POLL_INTERVAL)
def _render_stats(api: UIApi) -> None:
    """Quick stats — auto-refreshing fragment."""
    running = api.list_work(status="running", limit=100)
    completed = api.list_work(status="completed", limit=100)
    needs_review = api.list_work(status="needs_review", limit=100)
    failed = api.list_work(status="failed", limit=100)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Running", len(running))
    col2.metric("Completed", len(completed))
    col3.metric("Needs Review", len(needs_review))
    col4.metric("Failed", len(failed))


@st.fragment(run_every=_POLL_INTERVAL)
def _render_recent_work(api: UIApi) -> None:
    """Recent work list — auto-refreshing fragment."""
    all_work = api.list_work(limit=20)

    if not all_work:
        st.info("No work items yet. Submit work to get started!")
        return

    for item in all_work:
        work_id = item.get("id", "")
        status = item.get("status", "unknown")
        icon = status_icon(status)
        phase = item.get("current_phase", "")
        desc = truncate(item.get("description", ""), 80)

        with st.container():
            col1, col2, col3 = st.columns([1, 4, 2])
            col1.write(f"{icon}")

            # Clickable button to navigate to work detail page
            if col2.button(
                f"**{work_id}** — {desc}",
                key=f"dash_{work_id}",
                use_container_width=True,
            ):
                st.switch_page(
                    get_page("work-detail"),
                    query_params={"work_id": work_id},
                )
            col3.write(f"{phase or status}")


@st.fragment(run_every=_POLL_INTERVAL)
def _render_worker_status(api: UIApi) -> None:
    """Worker status — auto-refreshing fragment."""
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


def render(api: UIApi) -> None:
    """Render the dashboard page."""
    st.title("🦴 SPINE Dashboard")

    # ── Quick stats ──
    _render_stats(api)

    st.divider()

    # ── Recent work ──
    st.subheader("Recent Work")
    _render_recent_work(api)

    # ── Worker status ──
    st.divider()
    st.subheader("Worker Status")
    _render_worker_status(api)
