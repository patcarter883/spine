"""SPINE UI utility functions."""

from __future__ import annotations

from datetime import datetime


# ── Status display helpers ──

STATUS_COLORS = {
    "running": "🔵",
    "completed": "🟢",
    "needs_review": "🟡",
    "awaiting_approval": "🟣",
    "failed": "🔴",
    "pending": "⚪",
    "stalled": "🟠",
}

STATUS_BADGE = {
    "running": "🔄",
    "completed": "✅",
    "needs_review": "⚠️",
    "awaiting_approval": "⏳",
    "failed": "❌",
    "pending": "⏳",
    "stalled": "🕐",
}


def status_icon(status: str) -> str:
    """Return an emoji icon for a work status."""
    return STATUS_BADGE.get(status, "❓")


def status_color(status: str) -> str:
    """Return a color emoji for a work status."""
    return STATUS_COLORS.get(status, "⚪")


def status_color_css(status: str) -> str:
    """Return a CSS color string for a work status."""
    return {
        "running": "blue",
        "stalled": "orange",
        "completed": "green",
        "needs_review": "yellow",
        "failed": "red",
        "pending": "gray",
    }.get(status, "white")


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


# ── Duration formatting ──


def format_duration(start_iso: str | None, end_iso: str | None = None) -> str:
    """Human-readable elapsed time between two timestamps.

    When *end_iso* is provided, computes ``end - start`` (actual duration).
    When *end_iso* is ``None``, computes ``now - start`` (live elapsed time).

    Args:
        start_iso: ISO 8601 start timestamp string, or ``None``.
        end_iso: ISO 8601 end timestamp string. When ``None``, uses the
            current time instead.

    Returns:
        A human-readable duration string like ``"5m 30s"``, ``"2h 15m"``,
        or ``"7s"``. Returns ``"—`` when inputs are invalid or ``None``.
    """
    if not start_iso:
        return "—"
    try:
        start = datetime.fromisoformat(start_iso)
        if end_iso:
            end = datetime.fromisoformat(end_iso)
            delta = end - start
        else:
            delta = datetime.now() - start
        total_secs = max(0, int(delta.total_seconds()))
        if total_secs < 60:
            return f"{total_secs}s"
        mins = total_secs // 60
        hours = mins // 60
        if hours > 0:
            return f"{hours}h {mins % 60}m"
        return f"{mins}m {total_secs % 60}s"
    except (ValueError, TypeError):
        return "—"


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
