"""SPINE configuration package.

Provides helpers to load, validate, and transform configuration values
before they reach runtime logic.
"""

from __future__ import annotations

from typing import Any, Union

from spine.utils.time_conversion import normalize_interval, seconds_to_ms


def dashboard_config(raw_poll_seconds: Union[int, float, None] = None) -> dict[str, Any]:
    """Build the dashboard auto-refresh config from a raw seconds value.

    Converts the user-facing interval (seconds) to the internal representation
    (milliseconds) with validation/normalization.

    Args:
        raw_poll_seconds: Poll interval in seconds as provided by the UI slider.
                          Passed through ``normalize_interval`` for safety.

    Returns:
        Dict with ``poll_interval_ms`` (int) and ``poll_interval_s`` (int) keys.
    """
    poll_ms = normalize_interval(raw_poll_seconds)
    return {
        "poll_interval_ms": poll_ms,
        "poll_interval_s": int(round(poll_ms / 1000)),
    }


__all__ = [
    "dashboard_config",
    "normalize_interval",
    "seconds_to_ms",
]
