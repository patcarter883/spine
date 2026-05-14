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

            # Inline resume form
            st.divider()
            action_choice = st.radio(
                "Action",
                ["Rework from flagged phase", "Approve and proceed"],
                horizontal=True,
                key=f"hr_action_{work_id}",
            )
            resume_action = "rework" if action_choice.startswith("Rework") else "approve"
            feedback = st.text_area(
                "Feedback",
                placeholder="What needs to change, or why you approve...",
                key=f"hr_feedback_{work_id}",
            )
            col1, col2 = st.columns([1, 3])
            if col1.button("▶ Resume", type="primary", key=f"hr_resume_{work_id}"):
                if not feedback.strip():
                    st.error("Please provide feedback before resuming.")
                else:
                    result = api.resume_work(work_id, feedback.strip(), resume_action)
                    st.success(
                        f"Resumed! Status: {result['status']} | Action: {result['action']}"
                    )
                    st.rerun()
            if col2.button("View Details", key=f"hr_view_{work_id}"):
                from spine.ui.pages import get as get_page

                st.switch_page(
                    get_page("work-detail"),
                    query_params={"work_id": work_id},
                )
