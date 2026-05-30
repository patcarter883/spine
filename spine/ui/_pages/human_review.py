"""SPINE Human Review page — index of work items needing human attention.

This page is only an index. The actual review, feedback, and approval
actions live on the Work Detail page; each item here links there.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from spine.ui_api import UIApi
from spine.ui.pages import get as get_page
from spine.ui.utils import format_timestamp, status_icon, truncate


_REVIEW_STATUSES = ("needs_review", "awaiting_approval")


def render(api: UIApi) -> None:
    """Render the human review index."""
    st.title("👤 Human Review")
    st.caption(
        "Work items awaiting your review or approval. "
        "Open a task to see the critic's feedback and respond."
    )

    items = _fetch_review_items(api, limit=50)

    if not items:
        st.success("No work items need human review!")
        return

    st.warning(f"{len(items)} work item(s) need your attention.")

    for item in items:
        _render_review_item(item)


def _fetch_review_items(api: UIApi, limit: int) -> list[dict[str, Any]]:
    """Collect work items across every status that requires human review."""
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for status in _REVIEW_STATUSES:
        for item in api.list_work(status=status, limit=limit):
            work_id = item.get("id")
            if not work_id or work_id in seen:
                continue
            seen.add(work_id)
            merged.append(item)

    merged.sort(
        key=lambda r: (r.get("updated_at") or r.get("created_at") or ""),
        reverse=True,
    )
    return merged[:limit]


def _render_review_item(item: dict[str, Any]) -> None:
    """Render summary details for one item plus a link to its detail page."""
    work_id = item.get("id", "unknown")
    status = item.get("status", "unknown")
    icon = status_icon(status)
    desc = truncate(item.get("description", ""), 80)

    with st.expander(f"{icon} {work_id} — {desc}"):
        st.write(f"**Status:** `{status}`")
        st.write(f"**Type:** {item.get('work_type', 'N/A')}")
        st.write(f"**Phase:** {item.get('current_phase', 'N/A')}")
        st.write(f"**Created:** {format_timestamp(item.get('created_at'))}")
        st.write(f"**Updated:** {format_timestamp(item.get('updated_at'))}")

        st.write("**Description:**")
        st.write(item.get("description", ""))

        label = (
            "📋 Review & Approve"
            if status == "awaiting_approval"
            else "📝 Review & Respond"
        )
        if st.button(label, key=f"hr_view_{work_id}", type="primary"):
            st.switch_page(
                get_page("work-detail"),
                query_params={"work_id": work_id},
            )
