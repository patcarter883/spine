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
    "cancelled": "⚫",
}

STATUS_BADGE = {
    "running": "🔄",
    "completed": "✅",
    "needs_review": "⚠️",
    "awaiting_approval": "⏳",
    "failed": "❌",
    "pending": "⏳",
    "stalled": "🕐",
    "cancelled": "⏹️",
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


__all__ = [
    'create_work_link',
    'format_duration',
    'format_timestamp',
    'navigate_to_work',
    'normalize_artifacts',
    'slugify',
    'status_color',
    'status_color_css',
    'status_icon',
    'truncate',
]


def slugify(text: str) -> str:
    """Convert text to a URL-friendly slug.

    The input is converted to lowercase, every contiguous run of non-alphanumeric
    characters is replaced by a single hyphen, and any leading or trailing
    hyphens are stripped.

    Args:
        text: The string to slugify.

    Returns:
        A hyphen-separated slug derived from the input text.
    """
    import re
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')

def normalize_artifacts(artifacts: object) -> list[tuple[str, str]]:
    """Normalize a persisted ``result["artifacts"]`` value into display rows.

    Two shapes are persisted in the wild: the workflow dispatcher stores a
    ``{phase: [names]}`` mapping (see ``dispatcher.py``), while onboarding
    stores a flat ``[names]`` list (see ``onboarding/engine.py``). Returns a
    list of ``(label, text)`` rows that render identically for either shape —
    ``label`` is the phase name for the mapping shape and empty for the
    flat-list shape, so callers can render ``"{label}: {text}"`` or just
    ``"{text}"`` when the label is empty.
    """
    if isinstance(artifacts, dict):
        rows: list[tuple[str, str]] = []
        for phase, names in artifacts.items():
            text = ", ".join(str(n) for n in names) if isinstance(names, list) else str(names)
            rows.append((str(phase), text))
        return rows
    if isinstance(artifacts, list):
        return [("", ", ".join(str(n) for n in artifacts))]
    # Unexpected scalar — render it rather than crash.
    return [("", str(artifacts))]


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


def format_bytes(num_bytes: int) -> str:
    """Format a byte count as a human-readable string.

    Args:
        num_bytes: Number of bytes to format.

    Returns:
        A string representation with appropriate unit (B, KB, MB, GB).
        For values >= 1024, uses one decimal place and round-half-up.

    Examples:
        >>> format_bytes(0)
        '0 B'
        >>> format_bytes(512)
        '512 B'
        >>> format_bytes(1024)
        '1.0 KB'
        >>> format_bytes(1536)
        '1.5 KB'
        >>> format_bytes(1048576)
        '1.0 MB'
    """
    from decimal import Decimal, ROUND_HALF_UP

    if num_bytes < 1024:
        return f"{num_bytes} B"

    # Scale to appropriate unit
    for unit in ["KB", "MB", "GB"]:
        if num_bytes < 1024 ** (list(["B", "KB", "MB", "GB"]).index(unit) + 1):
            value = num_bytes / (1024 ** (list(["B", "KB", "MB", "GB"]).index(unit)))
            # Round half-up to one decimal place
            rounded = float(Decimal(str(value)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))
            return f"{rounded:.1f} {unit}"

    # Fallback for very large numbers (>= 1TB)
    value = num_bytes / (1024 ** 3)
    rounded = float(Decimal(str(value)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))
    return f"{rounded:.1f} GB"

def truncate_middle(text: str, max_len: int) -> str:
    """
    Truncate text by replacing the middle with an ellipsis while maintaining exact length.

    Args:
        text: The string to truncate.
        max_len: The maximum length of the returned string.

    Returns:
        The truncated string of exactly max_len characters (or less if text is shorter).

    Examples:
        >>> truncate_middle('abcde', 5)
        'abcde'
        >>> truncate_middle('abcdefghijk', 7)
        'ab...jk'  # Example result; actual distribution may vary slightly
        >>> truncate_middle('', 5)
        ''
        >>> truncate_middle('abcde', 2)
        'ab'
    """
    if not text:
        return ''
    if len(text) <= max_len:
        return text
    if max_len < 3:
        return text[:max_len]
    
    # max_len >= 3 and len(text) > max_len
    if max_len == 3:
        return f'{text[0]}…{text[-1]}'
    
    # Calculate split: need max_len characters total, with one '…' in middle
    # So we need (max_len - 1) visible characters, distributed as evenly as possible
    available = max_len - 1
    start_count = available // 2
    end_count = available - start_count
    
    start = text[:start_count]
    end = text[-end_count:]
    return f'{start}…{end}'

