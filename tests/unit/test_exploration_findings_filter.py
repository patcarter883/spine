"""Sentinel error-findings must not reach LLM-facing summaries."""
from __future__ import annotations

from spine.agents.exploration_agents import _summarize_findings
from spine.workflow.subgraphs.exploration_subgraph import _format_findings


def _good_finding() -> dict:
    return {
        "topic": "config loading",
        "summary": "Config is loaded from .spine/config.yaml via spine.config.load_config.",
        "patterns": ["yaml-config"],
        "file_map": {"spine/config.py": "load_config entrypoint"},
        "dependencies": ["pyyaml"],
    }


def _error_finding() -> dict:
    return {
        "summary": "Research failed for topic 'auth': RuntimeError: boom",
        "patterns": [],
        "file_map": {},
        "dependencies": [],
        "error": True,
    }


def test_summarize_findings_drops_error_sentinels():
    out = _summarize_findings([_good_finding(), _error_finding()])
    assert "Research failed" not in out
    assert "config loading" in out


def test_format_findings_drops_error_sentinels():
    out = _format_findings([_good_finding(), _error_finding()])
    assert "Research failed" not in out
    assert "config loading" in out


def test_format_findings_all_errors_returns_placeholder():
    out = _format_findings([_error_finding(), _error_finding()])
    assert out == "(no codebase research was performed)"


def test_summarize_findings_all_errors_returns_empty_join():
    # No exception, no sentinel content leaks through.
    out = _summarize_findings([_error_finding(), _error_finding()])
    assert "Research failed" not in out
