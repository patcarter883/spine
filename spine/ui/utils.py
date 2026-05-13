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

def create_work_link(work_id: str, text: str | None = None) -> str:
    """Create a clickable link to the work detail page for a specific work ID.
    
    Args:
        work_id: The work item ID.
        text: Optional display text. If None, uses the work_id.
        
    Returns:
        Markdown formatted link string.
    """
    display_text = text or work_id
    return f"[{display_text}](?work_id={work_id})"


def get_work_id_from_params() -> str | None:
    """Extract work ID from URL query parameters.
    
    Returns:
        Work ID if found in query params, otherwise None.
    """
    import streamlit as st
    return st.query_params.get("work_id")


def set_work_id_param(work_id: str) -> None:
    """Set work ID in URL query parameters.
    
    Args:
        work_id: The work item ID to set.
    """
    import streamlit as st
    st.query_params["work_id"] = work_id


def clear_work_id_param() -> None:
    """Clear work ID from URL query parameters."""
    import streamlit as st
    if "work_id" in st.query_params:
        del st.query_params["work_id"]
