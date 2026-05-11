"""Task queue page — view and manage the SPINE task queue."""

import time

import streamlit as st

from spine.ui.utils import (
    clear_completed_queue_tasks,
    enqueue_task,
    get_queue_items,
    get_queue_status,
    navigate_to_work,
    retry_queue_task,
)


def render_task_queue() -> None:
    """Render the task queue page."""
    st.title("📋 Task Queue")

    # ── Auto-refresh controls ──
    col_refresh, col_interval, _ = st.columns([2, 2, 4])

    with col_refresh:
        auto_refresh = st.toggle("Auto-refresh", value=True)

    with col_interval:
        refresh_interval = st.select_slider(
            "Refresh interval",
            options=[1, 2, 5, 10],
            value=5,
            disabled=not auto_refresh,
            label_visibility="collapsed",
        )

    # ── Worker Controls ──
    st.subheader("Worker Controls")
    col_start, col_pause, col_clear = st.columns(3)

    with col_start:
        if st.button("▶ Start Worker", type="primary"):
            try:
                from spine.work.ralph_worker import get_worker

                worker = get_worker()
                if not worker.status.get("running"):
                    worker.start()
                    st.success("Worker started")
                else:
                    st.info("Worker already running")
            except Exception as e:
                st.error(f"Failed to start worker: {e}")

    with col_pause:
        if st.button("⏸ Pause Worker"):
            try:
                from spine.work.ralph_worker import get_worker

                worker = get_worker()
                if worker.status.get("running"):
                    worker.pause()
                    st.success("Worker paused")
                else:
                    st.info("Worker not running")
            except Exception as e:
                st.error(f"Failed to pause worker: {e}")

    with col_clear:
        if st.button("🗑 Clear Completed"):
            removed = clear_completed_queue_tasks()
            st.success(f"Cleared {removed} completed task(s)")

    st.divider()

    # ── Summary Cards ──
    status = get_queue_status()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pending", status.get("pending", 0))
    col2.metric("Running", status.get("running", 0))
    col3.metric("Completed", status.get("success", 0))
    col4.metric("Failed", status.get("failed", 0))

    st.divider()

    # ── Enqueue Task Form ──
    st.subheader("Enqueue New Task")

    with st.form("enqueue_task_form"):
        requirement = st.text_area(
            "Requirement",
            placeholder="Enter the task requirement...",
            height=80,
            key="enqueue_requirement",
        )

        col_method, col_priority = st.columns([2, 1])
        with col_method:
            method = st.selectbox(
                "Method",
                ["Quick Work", "Full Spec Work", "Full Spec Project"],
                index=0,
            )
        with col_priority:
            priority = st.slider("Priority", min_value=0, max_value=10, value=0)

        submitted = st.form_submit_button("Enqueue Task", type="primary")

        if submitted:
            if requirement.strip():
                task_id = enqueue_task(
                    requirement=requirement.strip(),
                    method=method,
                    priority=priority,
                )
                if task_id:
                    st.success(f"Task enqueued with ID: {task_id}")
                else:
                    st.error("Failed to enqueue task")
            else:
                st.error("Please enter a requirement")

    st.divider()

    # ── Queue Items Sections ──
    # Pending items
    st.subheader("Pending Tasks")
    pending_items = get_queue_items(status="pending")
    if pending_items:
        for item in pending_items:
            _render_queue_item(item)
    else:
        st.info("No pending tasks")

    st.divider()

    # Running items
    st.subheader("Running Tasks")
    running_items = get_queue_items(status="running")
    if running_items:
        for item in running_items:
            _render_queue_item(item)
    else:
        st.info("No running tasks")

    st.divider()

    # Completed items
    st.subheader("Completed Tasks")
    completed_items = get_queue_items(status="success")
    if completed_items:
        for item in completed_items:
            _render_queue_item(item, show_view=True)
    else:
        st.info("No completed tasks")

    st.divider()

    # Failed items
    st.subheader("Failed Tasks")
    failed_items = get_queue_items(status="failed")
    if failed_items:
        for item in failed_items:
            _render_queue_item(item, show_retry=True)
    else:
        st.info("No failed tasks")

    # ── Auto-refresh polling ──
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


def _render_queue_item(item: dict, show_view: bool = False, show_retry: bool = False) -> None:
    """Render a single queue item."""
    with st.container():
        col_id, col_type, col_created, col_actions = st.columns([2, 2, 2, 2])

        col_id.write(f"**{item['id'][:8]}...**")
        col_type.write(item.get("task_type", "unknown"))
        col_created.write(item.get("created_at", "")[:19] if item.get("created_at") else "")

        # View button for completed items
        if show_view:
            result = item.get("result")
            if result and isinstance(result, dict):
                thread_id = result.get("thread_id")
                if thread_id:
                    if col_actions.button("View", key=f"view_{item['id']}"):
                        navigate_to_work(thread_id)
                        st.rerun()
                else:
                    col_actions.write("-")
            else:
                col_actions.write("-")
        # Retry button for failed items
        elif show_retry:
            if col_actions.button("Retry", key=f"retry_{item['id']}"):
                if retry_queue_task(item["id"]):
                    st.success("Task retried successfully")
                    st.rerun()
                else:
                    st.error("Failed to retry task")
        else:
            col_actions.write("-")

        # Show error for failed items
        if item.get("error") and item["status"] == "failed":
            with st.expander("Error details", expanded=False):
                st.error(item["error"])