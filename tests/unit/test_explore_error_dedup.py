"""Errored explore branches must still register their topic for dedup.

When a researcher subagent hits the recursion cap and salvage cannot
produce structured findings, the explore node falls through to an error
sentinel. That sentinel is excluded from the LLM-facing findings summary
but it MUST still carry the ``topic`` field so the round-stable
``_new_topics`` filter sees the topic as attempted — otherwise the
research_manager will re-issue the same topic on every subsequent round
(and on every rework attempt), burning tokens on doomed re-exploration.
"""
from __future__ import annotations

from spine.workflow.subgraphs.exploration_subgraph import _new_topics
from spine.agents.exploration_agents import _summarize_findings


def _error_sentinel(topic: str, files: dict[str, str] | None = None) -> dict:
    """Mirror the sentinel built in run_explore_node's except block."""
    return {
        "topic": topic,
        "summary": f"Research failed for topic '{topic}': GraphRecursionError: ...",
        "patterns": [],
        "file_map": files or {},
        "dependencies": [],
        "error": True,
        "partial": True,
    }


def test_new_topics_skips_errored_topic():
    state = {
        "topics": ["how is CLI logging configured?", "what does X do?"],
        "findings": [_error_sentinel("how is CLI logging configured?")],
    }
    remaining = _new_topics(state)  # type: ignore[arg-type]
    assert "how is CLI logging configured?" not in remaining
    assert "what does X do?" in remaining


def test_new_topics_skips_errored_topic_with_recall_suffix():
    # _new_topics normalises the recall suffix, so an enriched topic with
    # "— recall symbols: …" still matches the bare proposed topic.
    state = {
        "topics": ["how is CLI logging configured?"],
        "findings": [
            _error_sentinel(
                "how is CLI logging configured? — recall symbols: spine/cli.py, spine/logging.py"
            )
        ],
    }
    assert _new_topics(state) == []  # type: ignore[arg-type]


def test_error_sentinel_still_excluded_from_findings_summary():
    # Dedup-visibility must not also leak the failure noise into the
    # LLM-facing summary the manager reads.
    out = _summarize_findings([_error_sentinel("foo")])
    assert "Research failed" not in out
    assert "foo" not in out


def test_error_sentinel_carries_attempted_files_when_cache_present():
    sentinel = _error_sentinel(
        "what does X do?",
        files={"spine/x.py": "attempted during failed exploration"},
    )
    # Schema check: the dedup-visible fields the eviction filter and
    # manager prompt look at must be present and well-typed.
    assert isinstance(sentinel["file_map"], dict)
    assert "spine/x.py" in sentinel["file_map"]
    assert sentinel["error"] is True
    assert sentinel["partial"] is True
