"""SPINE Work Status page — view details of a specific work item."""

from __future__ import annotations

import streamlit as st

from spine.ui_api import UIApi
from spine.ui.utils import format_timestamp, status_icon


def render(api: UIApi) -> None:
    """Render the work status page."""
    st.title("🔍 Work Status")

    work_id = st.text_input("Work ID", placeholder="Enter work item ID")

    if not work_id:
        st.info("Enter a work item ID to view its status.")
        return

    entry = api.get_work(work_id)
    if entry is None:
        st.error(f"Work item '{work_id}' not found.")
        return

    # ── Status display ──
    status = entry.get("status", "unknown")
    icon = status_icon(status)

    st.header(f"{icon} {work_id}")

    col1, col2 = st.columns(2)
    col1.write(f"**Status:** {status}")
    col1.write(f"**Type:** {entry.get('work_type', 'N/A')}")
    col1.write(f"**Phase:** {entry.get('current_phase', 'N/A')}")
    col2.write(f"**Created:** {format_timestamp(entry.get('created_at'))}")
    col2.write(f"**Updated:** {format_timestamp(entry.get('updated_at'))}")

    # ── Description ──
    st.divider()
    st.subheader("Description")
    st.write(entry.get("description", "N/A"))

    # ── Result ──
    result = entry.get("result", {})
    if isinstance(result, dict):
        if result.get("artifacts"):
            st.subheader("Artifacts")
            for phase, names in result["artifacts"].items():
                st.write(f"**{phase}**: {', '.join(names) if isinstance(names, list) else names}")

        if result.get("error"):
            st.error(f"Error: {result['error']}")

    # ── Action buttons ──
    st.divider()
    if status == "needs_review":
        st.warning("This work item needs human review.")
        _human_input = st.text_input("Your input / decision")
        if st.button("Resume with input"):
            st.info("Resume functionality coming soon.")
    elif status == "running":
        st.info("Work is currently in progress.")
