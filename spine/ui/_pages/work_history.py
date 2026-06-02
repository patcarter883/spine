"""SPINE Work History page — list and filter all work items."""

from __future__ import annotations

import streamlit as st

from spine.ui_api import UIApi
from spine.ui.utils import format_timestamp, normalize_artifacts, status_icon, truncate


def render(api: UIApi) -> None:
    """Render the work history page."""
    st.title("📜 Work History")

    # ── Filters ──
    col1, col2 = st.columns([1, 3])
    with col1:
        status_filter = st.selectbox(
            "Filter by status",
            options=[None, "running", "completed", "needs_review", "failed"],
            format_func=lambda x: "All" if x is None else x.replace("_", " ").title(),
        )
        limit = st.slider("Items per page", 5, 100, 25)

    items = api.list_work(status=status_filter, limit=limit)

    if not items:
        st.info("No work items match the filter.")
        return

    # ── Table ──
    st.subheader(f"{len(items)} Work Items")

    for item in items:
        work_id = item.get("id", "N/A")
        status = item.get("status", "unknown")
        icon = status_icon(status)

        with st.expander(f"{icon} {work_id} — {truncate(item.get('description', ''), 60)}"):
            if st.button("View Details", key=f"view_{work_id}"):
                from spine.ui.pages import get as get_page

                st.switch_page(get_page("work-detail"), query_params={"work_id": work_id})
            col1, col2 = st.columns(2)
            col1.write(f"**Status:** {status}")
            col1.write(f"**Type:** {item.get('work_type', 'N/A')}")
            col1.write(f"**Phase:** {item.get('current_phase', 'N/A')}")
            col2.write(f"**Created:** {format_timestamp(item.get('created_at'))}")
            col2.write(f"**Updated:** {format_timestamp(item.get('updated_at'))}")

            description = item.get("description", "")
            if description:
                st.write("**Description:**")
                st.write(description)

            result = item.get("result", {})
            if isinstance(result, dict) and result.get("artifacts"):
                st.write("**Artifacts:**")
                for label, text in normalize_artifacts(result["artifacts"]):
                    st.write(f"- {label}: {text}" if label else f"- {text}")
