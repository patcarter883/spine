"""Tests for task-aware research breadth limits (trace 019ec965).

High-confidence, well-understood categories (Frontend/UI) should explore with
the leaner ceilings; everything else keeps the full default breadth.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spine.config import SpineConfig
from spine.workflow.subgraphs.exploration_subgraph import _effective_research_limits


class TestEffectiveResearchLimits:
    def test_high_confidence_frontend_is_lean(self):
        cfg = SpineConfig.load()
        rounds, parallel = _effective_research_limits("Frontend/UI", 0.95)
        assert rounds == cfg.research_lean_max_rounds
        assert parallel == cfg.research_lean_max_parallel_explores
        # Lean must be no broader than the default.
        assert rounds <= cfg.research_max_rounds
        assert parallel <= cfg.research_max_parallel_explores

    def test_low_confidence_frontend_is_full(self):
        cfg = SpineConfig.load()
        assert _effective_research_limits("Frontend/UI", 0.70) == (
            cfg.research_max_rounds,
            cfg.research_max_parallel_explores,
        )

    def test_high_confidence_non_lean_category_is_full(self):
        cfg = SpineConfig.load()
        assert _effective_research_limits("Backend/API", 0.99) == (
            cfg.research_max_rounds,
            cfg.research_max_parallel_explores,
        )

    def test_none_category_is_full(self):
        cfg = SpineConfig.load()
        assert _effective_research_limits(None, 0.0) == (
            cfg.research_max_rounds,
            cfg.research_max_parallel_explores,
        )

    def test_threshold_is_inclusive(self):
        cfg = SpineConfig.load()
        # confidence exactly at the lean threshold qualifies for lean limits.
        rounds, parallel = _effective_research_limits(
            "Frontend/UI", cfg.research_lean_confidence
        )
        assert (rounds, parallel) == (
            cfg.research_lean_max_rounds,
            cfg.research_lean_max_parallel_explores,
        )
