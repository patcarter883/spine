"""Unit tests for UI utility formatting functions.

Verifies:
- colored_html produces valid HTML with correct style attributes
- format_phase_icon returns correct emoji for each phase
- format_phase_color returns correct CSS color names
- Edge cases: empty text, unknown phases, special characters
"""

import pytest

from spine.ui.utils import (
    colored_html,
    format_phase_icon,
    format_phase_color,
    PHASE_ICONS,
    PHASE_COLORS,
)


class TestColoredHtml:
    """colored_html must produce an HTML span with inline color+style."""

    def test_basic_colored_text(self):
        result = colored_html("COMPLETE", "green")
        assert result == '<span style="color:green"><strong>COMPLETE</strong></span>'

    def test_with_icon_prefix(self):
        result = colored_html("🏁 COMPLETE", "green")
        assert "🏁 COMPLETE" in result
        assert 'style="color:green"' in result
        assert "<strong>" in result
        assert "</strong>" in result

    def test_different_colors(self):
        for color in ("red", "blue", "yellow", "cyan", "magenta", "white"):
            result = colored_html("test", color)
            assert f'style="color:{color}"' in result
            assert "<strong>test</strong>" in result

    def test_hex_color(self):
        result = colored_html("custom", "#ff5722")
        assert 'style="color:#ff5722"' in result

    def test_empty_text(self):
        result = colored_html("", "green")
        assert result == '<span style="color:green"><strong></strong></span>'

    def test_text_with_special_chars(self):
        result = colored_html("a < b & c > d", "red")
        assert "a < b & c > d" in result


class TestFormatPhaseIcon:
    """format_phase_icon returns correct emoji for known phases."""

    def test_all_known_phases_match_constants(self):
        for phase, expected_icon in PHASE_ICONS.items():
            assert format_phase_icon(phase) == expected_icon

    def test_unknown_phase_returns_bullet(self):
        assert format_phase_icon("NONEXISTENT_PHASE") == "•"

    def test_empty_string_returns_bullet(self):
        assert format_phase_icon("") == "•"

    def test_case_sensitive(self):
        assert format_phase_icon("complete") == "•"
        assert format_phase_icon("COMPLETE") == "🏁"

    def test_none_returns_bullet(self):
        assert format_phase_icon(None) == "•"


class TestFormatPhaseColor:
    """format_phase_color returns correct color names for known phases."""

    def test_all_known_phases_match_constants(self):
        for phase, expected_color in PHASE_COLORS.items():
            assert format_phase_color(phase) == expected_color

    def test_complete_is_green(self):
        assert format_phase_color("COMPLETE") == "green"

    def test_error_is_red(self):
        assert format_phase_color("ERROR") == "red"

    def test_unknown_phase_returns_white(self):
        assert format_phase_color("UNKNOWN_PHASE") == "white"

    def test_empty_string_returns_white(self):
        assert format_phase_color("") == "white"

    def test_case_sensitive(self):
        assert format_phase_color("complete") == "white"
        assert format_phase_color("COMPLETE") == "green"

    def test_none_returns_white(self):
        assert format_phase_color(None) == "white"


__all__ = []
