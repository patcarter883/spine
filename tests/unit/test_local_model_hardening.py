"""Regression tests for local-model hardening fixes (trace audit 2026-06-17).

Covers the five gaps the trace audit surfaced that the prior fixes did not:

1. Connection-failure circuit breaker (traces 019ece87, 019ed360).
2. write_structured_plan degenerate-output sanitization (trace 019ed44a).
3. Fallback-decomposer prompt bounding + length salvage (trace 019ed3dc).
4. Hard recursion ceiling on phase subgraphs (trace 019ece87).
5. Window-relative hard pre-send prompt guard (traces 019ed3dc/019ed413).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ── Fix 1: connection-failure circuit breaker ──────────────────────────────


class _FakeConnError(Exception):
    """Stands in for openai.APIError('CURL error: Could not connect to server')."""


class _Boom:
    """Agent stub whose ainvoke always raises a connection-unreachable error."""

    def __init__(self, exc):
        self._exc = exc
        self.calls = 0

    async def ainvoke(self, *_a, **_k):
        self.calls += 1
        raise self._exc


class _OkThenNothing:
    async def ainvoke(self, *_a, **_k):
        return {"messages": []}


def test_is_connection_unreachable_matches_curl_and_refused():
    from spine.agents import retry

    assert retry._is_connection_unreachable(
        _FakeConnError("CURL error: Could not connect to server")
    )
    assert retry._is_connection_unreachable(Exception("Connection refused"))
    # Must NOT trip on transient server / rate-limit errors.
    assert not retry._is_connection_unreachable(Exception("503 Service Unavailable"))
    assert not retry._is_connection_unreachable(Exception("429 rate limit"))


@pytest.mark.asyncio
async def test_breaker_trips_after_threshold_consecutive_failures(monkeypatch):
    from spine.agents import retry

    monkeypatch.setenv("SPINE_CONN_FAILURE_THRESHOLD", "3")
    retry.reset_conn_breaker()
    exc = _FakeConnError("CURL error: Could not connect to server")

    # Each call exhausts its own per-call retries (max_retries=0 → one try),
    # so 3 calls = 3 consecutive connection failures = breaker trips.
    seen_unreachable = False
    for _ in range(3):
        try:
            await retry.ainvoke_with_retry(
                _Boom(exc), {"messages": []}, max_retries=0, base_delay=0
            )
        except retry.ServerUnreachable:
            seen_unreachable = True
            break
        except _FakeConnError:
            pass
    assert seen_unreachable, "breaker should raise ServerUnreachable at threshold"
    retry.reset_conn_breaker()


@pytest.mark.asyncio
async def test_breaker_resets_on_success(monkeypatch):
    from spine.agents import retry

    monkeypatch.setenv("SPINE_CONN_FAILURE_THRESHOLD", "3")
    retry.reset_conn_breaker()
    retry._note_conn_failure()
    retry._note_conn_failure()
    # A genuine success must clear the counter so a flaky-then-healthy server
    # never trips on stale failures.
    await retry.ainvoke_with_retry(_OkThenNothing(), {"messages": []}, max_retries=0)
    assert retry._consecutive_conn_failures == 0
    retry.reset_conn_breaker()


# ── Fix 2: write_structured_plan degenerate-output sanitization ─────────────


def _valid_slice():
    return {
        "id": "slice-1",
        "title": "Do the thing",
        "target_files": ["a.py"],
        "execution_requirements": "implement it",
        "dependencies": [],
        "acceptance_criteria": ["it works"],
        "complexity": "small",
    }


def test_feature_slices_drops_non_mapping_repetition():
    from spine.agents.plan_tools import _StructuredWritePlanInput

    # Repetition collapse: 1 real slice + 3561 `False` scalars (trace 019ed44a).
    args = {
        "architecture_overview": "x",
        "feature_slices": [_valid_slice()] + [False] * 3561,
        "testing_strategy": "y",
    }
    model = _StructuredWritePlanInput.model_validate(args)
    assert len(model.feature_slices) == 1
    assert model.feature_slices[0].id == "slice-1"


def test_feature_slices_truncated_to_max_length():
    from spine.agents.plan_tools import _MAX_FEATURE_SLICES, _StructuredWritePlanInput

    args = {
        "architecture_overview": "x",
        "feature_slices": [_valid_slice() for _ in range(_MAX_FEATURE_SLICES + 50)],
        "testing_strategy": "y",
    }
    model = _StructuredWritePlanInput.model_validate(args)
    assert len(model.feature_slices) == _MAX_FEATURE_SLICES


def test_feature_slices_normal_input_unchanged():
    from spine.agents.plan_tools import _StructuredWritePlanInput

    args = {
        "architecture_overview": "x",
        "feature_slices": [_valid_slice(), {**_valid_slice(), "id": "slice-2"}],
        "testing_strategy": "y",
    }
    model = _StructuredWritePlanInput.model_validate(args)
    assert [s.id for s in model.feature_slices] == ["slice-1", "slice-2"]


# ── Fix 4: subgraph recursion limit config ─────────────────────────────────


def test_subgraph_recursion_limit_default_and_env(monkeypatch):
    from spine.config import SpineConfig

    assert SpineConfig.load().subgraph_recursion_limit == 80
    monkeypatch.setenv("SPINE_SUBGRAPH_RECURSION_LIMIT", "40")
    assert SpineConfig.load().subgraph_recursion_limit == 40


# ── Fix 3: decomposer traceback bound config ───────────────────────────────


def test_decompose_max_traceback_chars_config(monkeypatch):
    from spine.config import SpineConfig

    assert SpineConfig.load().decompose_max_traceback_chars == 4000
    monkeypatch.setenv("SPINE_DECOMPOSE_MAX_TRACEBACK_CHARS", "1000")
    assert SpineConfig.load().decompose_max_traceback_chars == 1000


# ── Fix 5: window-relative hard prompt guard ───────────────────────────────


def test_window_hard_ceiling():
    from spine.agents.synthesis_budget import window_hard_ceiling

    assert window_hard_ceiling(0, 4000) == 0  # cloud/legacy → no guard
    assert window_hard_ceiling(60000, 4000) == 60000 - 4000 - 512
    assert window_hard_ceiling(60000, 4000, completion_floor=1000) == 60000 - 4000 - 1000
