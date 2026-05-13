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
