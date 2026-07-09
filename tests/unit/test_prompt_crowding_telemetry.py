"""Prompt-crowding telemetry: inlined blocks are individually capped but no
assembly point asserts the SUM — the warn line makes stacking creep visible
in logs before it degrades into squeezed completions and truncation guards.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from spine.agents.helpers import warn_if_prompt_crowds_window


def test_small_prompt_is_silent(caplog):
    with caplog.at_level(logging.WARNING, logger="spine.agents.helpers"):
        est = warn_if_prompt_crowds_window(
            [SystemMessage(content="hi"), HumanMessage(content="there")],
            label="unit-test",
        )
    assert est > 0
    assert "prompt-crowding" not in caplog.text


def test_crowding_prompt_warns(caplog, monkeypatch):
    monkeypatch.setattr("spine.agents.helpers._PROMPT_WARN_TOKENS", 10)
    with caplog.at_level(logging.WARNING, logger="spine.agents.helpers"):
        warn_if_prompt_crowds_window(
            [HumanMessage(content="word " * 100)], label="unit-test"
        )
    assert "prompt-crowding: unit-test" in caplog.text


def test_estimation_failure_is_fail_open(monkeypatch):
    class Weird:
        @property
        def content(self):
            raise RuntimeError("boom")

    # Must never raise, returns 0.
    assert warn_if_prompt_crowds_window([Weird()], label="x") == 0
