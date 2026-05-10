"""Tests for time conversion utilities and dashboard config."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from spine.utils.time_conversion import normalize_interval, seconds_to_ms
from spine.config import dashboard_config


class TestSecondsToMs:
    """Tests for seconds_to_ms conversion."""

    def test_converts_integer_seconds(self):
        assert seconds_to_ms(2) == 2000

    def test_converts_float_seconds(self):
        assert seconds_to_ms(1.5) == 1500

    def test_zero_seconds(self):
        assert seconds_to_ms(0) == 0

    def test_large_value(self):
        assert seconds_to_ms(3600) == 3_600_000

    def test_raises_on_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            seconds_to_ms(-1)

    def test_raises_on_string(self):
        with pytest.raises(TypeError):
            seconds_to_ms("2")

    def test_raises_on_none(self):
        with pytest.raises(TypeError):
            seconds_to_ms(None)

    def test_raises_on_bool(self):
        with pytest.raises(TypeError):
            seconds_to_ms(True)

    def test_rounding(self):
        assert seconds_to_ms(0.001) == 1
        assert seconds_to_ms(0.0004) == 0


class TestNormalizeInterval:
    """Tests for normalize_interval with fallback."""

    def test_normal_positive(self):
        assert normalize_interval(2) == 2000

    def test_none_falls_back(self):
        assert normalize_interval(None) == 2000

    def test_negative_falls_back(self):
        assert normalize_interval(-5) == 2000

    def test_zero_falls_back(self):
        assert normalize_interval(0) == 2000

    def test_string_falls_back(self):
        assert normalize_interval("bad") == 2000

    def test_custom_default(self):
        assert normalize_interval(None, default=5) == 5000

    def test_bool_falls_back(self):
        assert normalize_interval(True) == 2000


class TestDashboardConfig:
    """Tests for the dashboard_config builder."""

    def test_basic_conversion(self):
        cfg = dashboard_config(2)
        assert cfg["poll_interval_ms"] == 2000
        assert cfg["poll_interval_s"] == 2

    def test_none_input(self):
        cfg = dashboard_config(None)
        assert cfg["poll_interval_ms"] == 2000
        assert cfg["poll_interval_s"] == 2

    def test_negative_input(self):
        cfg = dashboard_config(-1)
        assert cfg["poll_interval_ms"] == 2000

    def test_float_input(self):
        cfg = dashboard_config(1.5)
        assert cfg["poll_interval_ms"] == 1500
