"""Tests for duration formatting utilities."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.ui.utils import format_duration


# ── Single-timestamp (live elapsed) tests ──


class TestFormatDurationSingleTimestamp:
    """Tests for format_duration with a single start timestamp (no end)."""

    def test_format_duration_none_returns_dash(self):
        assert format_duration(None) == "—"

    def test_format_duration_under_60_seconds(self):
        start = (datetime.now() - timedelta(seconds=35)).isoformat()
        result = format_duration(start)
        # Should be between 35-36s depending on when it runs
        assert result.endswith("s")
        secs = int(result.rstrip("s"))
        assert 34 <= secs <= 37

    def test_format_duration_minutes_and_seconds(self):
        start = (datetime.now() - timedelta(minutes=5, seconds=30)).isoformat()
        result = format_duration(start)
        # Allow a 1-second tolerance
        assert "5m 29s" in result or "5m 30s" in result

    def test_format_duration_hours_and_minutes(self):
        start = (datetime.now() - timedelta(hours=2, minutes=15)).isoformat()
        result = format_duration(start)
        assert "2h 15m" in result or "2h 14m" in result

    def test_format_duration_invalid_iso_returns_dash(self):
        assert format_duration("not-a-date") == "—"

    def test_format_duration_empty_string_returns_dash(self):
        assert format_duration("") == "—"

    def test_format_duration_end_none_uses_now(self):
        """Explicit end_iso=None should behave same as omitting it."""
        start = (datetime.now() - timedelta(minutes=3)).isoformat()
        result_with_none = format_duration(start, None)
        result_without = format_duration(start)
        # Both should show ~3m
        assert result_with_none == result_without


# ── Dual-timestamp (actual duration) tests ──


class TestFormatDurationDualTimestamp:
    """Tests for format_duration with both start and end timestamps."""

    def test_exact_5_minutes(self):
        start = "2024-01-01T10:00:00"
        end = "2024-01-01T10:05:00"
        assert format_duration(start, end) == "5m 0s"

    def test_exact_90_seconds_as_1m_30s(self):
        start = "2024-01-01T10:00:00"
        end = "2024-01-01T10:01:30"
        assert format_duration(start, end) == "1m 30s"

    def test_exact_7_seconds(self):
        start = "2024-01-01T10:00:00"
        end = "2024-01-01T10:00:07"
        assert format_duration(start, end) == "7s"

    def test_two_hours_30_minutes(self):
        start = "2024-01-01T10:00:00"
        end = "2024-01-01T12:30:00"
        assert format_duration(start, end) == "2h 30m"

    def test_three_hours_exact(self):
        start = "2024-01-01T10:00:00"
        end = "2024-01-01T13:00:00"
        assert format_duration(start, end) == "3h 0m"

    def test_end_before_start_clamped_to_zero(self):
        """When end < start, return 0s not a negative value."""
        start = "2024-01-01T10:05:00"
        end = "2024-01-01T10:00:00"  # end is before start
        assert format_duration(start, end) == "0s"

    def test_start_none_with_end_returns_dash(self):
        """If start is None, return dash regardless of end."""
        assert format_duration(None, "2024-01-01T10:30:00") == "—"

    def test_both_invalid_returns_dash(self):
        assert format_duration("bad", "bad") == "—"

    def test_empty_start_with_valid_end_returns_dash(self):
        assert format_duration("", "2024-01-01T10:30:00") == "—"

    def test_exact_zero_duration(self):
        """When start equals end, return 0s."""
        ts = "2024-01-01T10:00:00"
        assert format_duration(ts, ts) == "0s"

    def test_large_duration(self):
        """Multi-day duration formats as hours."""
        start = "2024-01-01T00:00:00"
        end = "2024-01-03T14:25:00"  # 2 days + 14h 25m = 62h 25m
        assert format_duration(start, end) == "62h 25m"
