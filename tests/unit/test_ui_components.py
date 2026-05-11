"""Unit tests for reusable Streamlit component formatting functions.

Verifies:
- phase_badge produces an HTML span with the correct color and phase icon
- status_badge maps status strings to correct colors
- Edge cases: empty strings, unknown values, mixed casing
"""

import pytest

from spine.ui.components import phase_badge, status_badge


class TestPhaseBadge:
    """phase_badge must return HTML with colored phase text."""

    def test_complete_phase(self):
        result = phase_badge("COMPLETE")
        assert "🏁" in result
        assert "COMPLETE" in result
        assert 'style="color:green"' in result
        assert result.startswith("<span")
        assert result.endswith("</span>")
        assert "<strong>" in result

    def test_planning_phase(self):
        result = phase_badge("PLANNING")
        assert "📋" in result
        assert "PLANNING" in result
        assert 'style="color:blue"' in result

    def test_error_phase(self):
        result = phase_badge("ERROR")
        assert "❌" in result
        assert 'style="color:red"' in result

    def test_unknown_phase_defaults_to_white(self):
        result = phase_badge("MYSTERY_PHASE")
        assert 'style="color:white"' in result
        assert "MYSTERY_PHASE" in result

    def test_empty_phase_defaults_to_bullet(self):
        result = phase_badge("")
        assert 'style="color:white"' in result
        # Empty phase gets the default icon (bullet) but no phase text
        assert "•" in result

    def test_never_contains_bbcode(self):
        for phase in ("COMPLETE", "PLANNING", "EXECUTION", "VERIFICATION", "INIT"):
            result = phase_badge(phase)
            assert "[" not in result
            assert "[/]" not in result
            assert "[/" not in result


class TestStatusBadge:
    """status_badge maps status strings to colored HTML."""

    def test_success_status(self):
        result = status_badge("success")
        assert 'style="color:green"' in result
        assert "<strong>success</strong>" in result

    def test_running_status(self):
        result = status_badge("running")
        assert 'style="color:blue"' in result

    def test_failed_status(self):
        result = status_badge("failed")
        assert 'style="color:red"' in result

    def test_blocked_status(self):
        result = status_badge("blocked")
        assert 'style="color:red"' in result

    def test_pending_status(self):
        result = status_badge("pending")
        assert 'style="color:gray"' in result

    def test_unknown_status_defaults_to_gray(self):
        result = status_badge("unknown_thing")
        assert 'style="color:gray"' in result

    def test_empty_status(self):
        result = status_badge("")
        assert 'style="color:gray"' in result

    def test_color_lookup_case_insensitive(self):
        """Color mapping is case-insensitive, but displayed text preserves input case."""
        result = status_badge("SUCCESS")
        assert 'style="color:green"' in result
        assert "<strong>SUCCESS</strong>" in result

    def test_never_contains_bbcode(self):
        for status in ("success", "failed", "running", "pending", "blocked", "error"):
            result = status_badge(status)
            assert "[" not in result
            assert "[/]" not in result
            assert "[/" not in result


__all__ = []
