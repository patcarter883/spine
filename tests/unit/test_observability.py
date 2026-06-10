"""Regression tests: the test suite must never trace to LangSmith.

The autouse ``_no_langsmith_tracing`` fixture in ``tests/conftest.py``
strips the LangSmith credentials that ``spine.config`` loads from the
repo's ``.env``, which makes ``work_run_tracing`` a no-op. These tests
pin that contract so dispatcher/onboarding tests can't silently start
emitting real traces again.
"""

from __future__ import annotations

import os
from unittest import mock

from spine.observability import work_run_tracing


def test_langsmith_credentials_absent_under_pytest() -> None:
    assert "LANGSMITH_API_KEY" not in os.environ
    assert "LANGCHAIN_API_KEY" not in os.environ
    assert "SPINE_TRACE_ALL" not in os.environ


def test_work_run_tracing_is_noop_without_api_key() -> None:
    with mock.patch("langchain_core.tracers.context.tracing_v2_enabled") as enabled:
        with work_run_tracing("w-1", "task"):
            pass
    enabled.assert_not_called()
