"""SPINE Human Review page — manage work items needing attention."""

from __future__ import annotations

import streamlit as st

from spine.ui_api import UIApi
from spine.ui.utils import format_timestamp, status_icon, truncate


def render(api: UIApi) -> None:
    """Render the human review page."""
    st.title("👤 Human Review")

    items = api.list_work(status="needs_review", limit=50)

    if not items:
        st.success("No work items need human review!")
        return

    st.warning(f"{len(items)} work item(s) need your attention.")

    for item in items:
        work_id = item.get("id", "unknown")
        icon = status_icon(item.get("status", "unknown"))
        desc = truncate(item.get("description", ""), 80)

        with st.expander(f"{icon} {work_id} — {desc}"):
            st.write(f"**Type:** {item.get('work_type', 'N/A')}")
            st.write(f"**Phase:** {item.get('current_phase', 'N/A')}")
            st.write(f"**Created:** {format_timestamp(item.get('created_at'))}")
            st.write(f"**Updated:** {format_timestamp(item.get('updated_at'))}")

            st.write("**Description:**")
            st.write(item.get("description", ""))

            result = item.get("result", {})
            if isinstance(result, dict):
                if result.get("prompt_request"):
                    st.write("**Prompt Request:**")
                    st.json(result["prompt_request"])

            st.divider()
            action = st.selectbox(
                "Action",
                options=["approve", "revise", "cancel"],
                key=f"action_{work_id}",
            )

            _notes = st.text_area(
                "Notes / Feedback",
                placeholder="Provide feedback or instructions...",
                key=f"notes_{work_id}",
            )

            if st.button("Submit Decision", key=f"submit_{work_id}"):
                st.info(f"Decision: {action} for {work_id}. Resume with feedback coming soon.")
