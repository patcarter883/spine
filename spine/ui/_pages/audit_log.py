"""SPINE Audit Log page — view workflow audit events."""

from __future__ import annotations

import streamlit as st

from spine.ui_api import UIApi
from spine.ui.utils import format_timestamp


def render(api: UIApi) -> None:
    """Render the audit log page."""
    st.title("📋 Audit Log")

    col1, col2 = st.columns(2)
    with col1:
        work_id_filter = st.text_input("Filter by Work ID", "")
    with col2:
        event_type_filter = st.selectbox(
            "Filter by Event Type",
            options=[
                None,
                "work_submitted",
                "work_completed",
                "work_failed",
                "phase_start",
                "phase_complete",
                "critic_review",
            ],
            format_func=lambda x: "All" if x is None else x,
        )

    limit = st.slider("Events per page", 10, 500, 100)

    events = api.get_audit_log(
        work_id=work_id_filter or None,
        event_type=event_type_filter,
        limit=limit,
    )

    if not events:
        st.info("No audit events found.")
        return

    st.subheader(f"{len(events)} Events")

    for event in events:
        event_type = event.get("event_type", "unknown")
        work_id = event.get("work_id", "")
        phase = event.get("phase", "")
        timestamp = format_timestamp(event.get("timestamp"))
        details = event.get("details", {})

        with st.expander(f"[{timestamp}] {event_type} — {work_id} ({phase})"):
            if details:
                st.json(details)
            else:
                st.write("No details available.")
