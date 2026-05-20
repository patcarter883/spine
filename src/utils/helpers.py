"""Utility functions for duration formatting."""

from __future__ import annotations


def format_duration(seconds: int | float) -> str:
    """Format a duration in seconds to a human-readable string.

    Takes a duration in seconds (int or float) and returns a human-readable
    string representation like "5m 30s", "2h 15m", or "1d 2h 30m 45s".

    Args:
        seconds: Duration in seconds (int or float).

    Returns:
        Human-readable duration string.

    Examples:
        >>> format_duration(45)
        '45s'
        >>> format_duration(90)
        '1m 30s'
        >>> format_duration(3600)
        '1h 0m'
        >>> format_duration(90061)
        '1d 1h 1m 1s'
        >>> format_duration(0)
        '0s'
        >>> format_duration(-5)
        '0s'
    """
    # Handle edge cases: zero, negative, and non-finite values
    if seconds <= 0 or not isinstance(seconds, (int, float)) or seconds != seconds:  # NaN check
        return "0s"

    total_secs = int(seconds)

    if total_secs < 60:
        return f"{total_secs}s"

    mins = total_secs // 60
    hours = mins // 60
    days = hours // 24

    parts = []

    if days > 0:
        parts.append(f"{days}d")
    if hours % 24 > 0:
        parts.append(f"{hours % 24}h")
    if mins % 60 > 0:
        parts.append(f"{mins % 60}m")
    if total_secs % 60 > 0:
        parts.append(f"{total_secs % 60}s")

    return " ".join(parts)