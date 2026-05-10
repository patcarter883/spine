"""Time conversion utilities for transforming configuration values.

Provides deterministic conversion from seconds-based config values
to milliseconds consumed by runtime logic, with input validation.
"""

from __future__ import annotations

from typing import Union


def seconds_to_ms(seconds: Union[int, float]) -> int:
    """Convert a seconds value to milliseconds.

    Args:
        seconds: Duration in seconds. Must be non-negative.

    Returns:
        Equivalent duration in milliseconds.

    Raises:
        ValueError: If seconds is negative.
        TypeError: If seconds is not a number.

    Examples:
        >>> seconds_to_ms(2)
        2000
        >>> seconds_to_ms(0)
        0
        >>> seconds_to_ms(1.5)
        1500
    """
    if not isinstance(seconds, (int, float)):
        raise TypeError(f"Expected int or float, got {type(seconds).__name__}")
    if isinstance(seconds, bool):
        raise TypeError(f"Expected int or float, got bool")
    if seconds < 0:
        raise ValueError(f"Seconds must be non-negative, got {seconds}")
    return int(round(seconds * 1000))


def normalize_interval(seconds: Union[int, float, None], default: int = 2) -> int:
    """Normalize a poll interval value to milliseconds, with fallback.

    Accepts None or a number. Returns ``default * 1000`` when *seconds*
    is None, negative, or zero.

    Args:
        seconds: Raw interval in seconds (from config/UI).
        default: Fallback value in seconds when input is invalid (default 2).

    Returns:
        Normalized interval in milliseconds.

    Examples:
        >>> normalize_interval(2)
        2000
        >>> normalize_interval(None)
        2000
        >>> normalize_interval(-1)
        2000
    """
    if seconds is None:
        return default * 1000
    try:
        ms = seconds_to_ms(seconds)
    except (ValueError, TypeError):
        return default * 1000
    if ms <= 0:
        return default * 1000
    return ms


__all__ = [
    "seconds_to_ms",
    "normalize_interval",
]
