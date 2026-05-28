"""Tests for the shared token estimator in spine.agents._tokens."""

from __future__ import annotations

from spine.agents._tokens import count_tokens


def test_empty_string_returns_zero():
    assert count_tokens("") == 0


def test_simple_string_positive():
    assert count_tokens("hello world") > 0


def test_count_roughly_scales_with_length():
    short = count_tokens("hello")
    long = count_tokens("hello " * 200)
    assert long > short * 10


def test_unicode_does_not_raise():
    # Mix of emoji and CJK should not blow up.
    out = count_tokens("hello 🌍 世界 — done")
    assert out > 0
