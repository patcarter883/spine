"""SPINE UI utility functions."""

from __future__ import annotations


# ── Status display helpers ──

STATUS_COLORS = {
    "running": "🔵",
    "completed": "🟢",
    "needs_review": "🟡",
    "failed": "🔴",
    "pending": "⚪",
}

STATUS_BADGE = {
    "running": "🔄",
    "completed": "✅",
    "needs_review": "⚠️",
    "failed": "❌",
    "pending": "⏳",
}


def status_icon(status: str) -> str:
    """Return an emoji icon for a work status."""
    return STATUS_BADGE.get(status, "❓")


def status_color(status: str) -> str:
    """Return a color emoji for a work status."""
    return STATUS_COLORS.get(status, "⚪")


def format_timestamp(ts: str | None) -> str:
    """Format an ISO timestamp for display."""
    if not ts:
        return "N/A"
    try:
        return ts[:19].replace("T", " ")
    except (ValueError, IndexError):
        return ts


def truncate(text: str, max_len: int = 100) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


# ── Navigation helpers ──
# With st.navigation / st.Page, pages have URL paths like /work-detail.
# Deep links use standard query params: /work-detail?work_id=abc


def create_work_link(work_id: str, text: str | None = None) -> str:
    """Create a clickable Markdown link to the work detail page.

    Args:
        work_id: The work item ID.
        text: Optional display text. If None, uses the work_id.

    Returns:
        Markdown formatted link string for use in st.markdown().
    """
    display_text = text or work_id
    return f"[{display_text}](/work-detail?work_id={work_id})"


def navigate_to_work(work_id: str) -> None:
    """Navigate to the work detail page for a specific work ID.

    Uses the page registry to switch to the work detail page with
    the work_id query parameter set.

    Args:
        work_id: The work item ID to navigate to.
    """
    import streamlit as st

    from spine.ui.pages import get as get_page

    work_detail_page = get_page("work-detail")
    if work_detail_page:
        st.switch_page(work_detail_page, query_params={"work_id": work_id})
    else:
        # Fallback: set query param and switch by URL path
        st.query_params["work_id"] = work_id
        st.switch_page(get_page("dashboard"))
