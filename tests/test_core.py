"""Tests for core formatting utilities."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.helpers import format_duration


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_seconds_only(self):
        """Test formatting seconds only (under 1 minute)."""
        assert format_duration(45) == "45s"
        assert format_duration(1) == "1s"
        assert format_duration(59) == "59s"

    def test_minutes_and_seconds(self):
        """Test formatting minutes and seconds."""
        assert format_duration(90) == "1m 30s"
        assert format_duration(125) == "2m 5s"
        assert format_duration(3599) == "59m 59s"

    def test_hours_only(self):
        """Test formatting hours only."""
        assert format_duration(3600) == "1h 0m"
        assert format_duration(7200) == "2h 0m"

    def test_hours_and_minutes(self):
        """Test formatting hours and minutes."""
        assert format_duration(3660) == "1h 1m"
        assert format_duration(5430) == "1h 30m"

    def test_days_only_minutes_seconds(self):
        """Test formatting days, hours, minutes, and seconds."""
        assert format_duration(90061) == "1d 1h 1m 1s"
        assert format_duration(86400) == "1d 0h 0m 0s"

    def test_zero(self):
        """Test zero duration."""
        assert format_duration(0) == "0s"

    def test_negative(self):
        """Test negative duration (should return 0s)."""
        assert format_duration(-5) == "0s"
        assert format_duration(-100) == "0s"

    def test_float_values(self):
        """Test float duration values (should truncate to int)."""
        assert format_duration(45.9) == "45s"
        assert format_duration(90.5) == "1m 30s"

    def test_very_large_duration(self):
        """Test very large duration values."""
        # 30 days
        assert format_duration(2592000) == "30d 0h 0m 0s"

    def test_exact_minute(self):
        """Test exact minute boundary."""
        assert format_duration(60) == "1m 0s"

    def test_exact_hour(self):
        """Test exact hour boundary."""
        assert format_duration(3600) == "1h 0m"

    def test_exact_day(self):
        """Test exact day boundary."""
        assert format_duration(86400) == "1d 0h 0m 0s"
