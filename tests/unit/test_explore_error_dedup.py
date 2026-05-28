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


def test_error_sentinel_surfaces_topic_with_neutral_marker():
    """The summary now includes attempted-but-failed topics with a
    neutral "attempted; no usable findings" marker. The exception/error
    text MUST still be filtered (per
    [[feedback_no_error_text_in_research_results]]) — only the bare
    topic name and a generic outcome may surface.

    This is the change that lets the research_manager see prior round
    coverage and avoid re-proposing the same topic under different
    wording (trace 019e6e53).
    """
    out = _summarize_findings([_error_sentinel("foo")])
    # No error text — the memory-pinned rule.
    assert "Research failed" not in out
    assert "RuntimeError" not in out
    # But the topic IS surfaced so the manager knows it was attempted.
    assert "foo" in out
    assert "attempted" in out.lower()


# ── Near-duplicate (paraphrase) dedup ──────────────────────────────────────


def test_new_topics_drops_paraphrased_topic():
    """Regression for trace 019e6e53.

    The local model paraphrases prior topics across rounds — same
    underlying question, different wording. ``_normalise_topic`` does
    exact-string match and lets this through; ``_topics_near_duplicate``
    must catch it via content-word overlap.
    """
    prior = "How does the CLI entrypoint currently parse and handle command-line arguments?"
    paraphrase = "How does the command-line interface parse and handle flags?"
    state = {
        "topics": [paraphrase],
        "findings": [_error_sentinel(prior)],
    }
    assert _new_topics(state) == []  # type: ignore[arg-type]


def test_new_topics_keeps_unrelated_topic():
    """Dedup must not over-fire on topics that share a few common words."""
    prior = "How does the CLI entrypoint parse command-line arguments?"
    unrelated = "How is the LangSmith trace project configured for runs?"
    state = {
        "topics": [unrelated],
        "findings": [_error_sentinel(prior)],
    }
    assert _new_topics(state) == [unrelated]  # type: ignore[arg-type]


def test_new_topics_handles_recall_suffix_in_dedup():
    """The enriched recall suffix on the prior topic must not defeat the
    paraphrase check."""
    prior_enriched = (
        "How does the CLI entrypoint parse command-line arguments? "
        "— recall symbols: parse_args (spine/cli/__init__.py)"
    )
    paraphrase = "How does the command-line interface handle flags and arguments?"
    state = {
        "topics": [paraphrase],
        "findings": [_error_sentinel(prior_enriched)],
    }
    # The exact-match check strips the suffix; the paraphrase check sees
    # the suffix-stripped prior topic too.
    assert _new_topics(state) == []  # type: ignore[arg-type]


def test_topics_near_duplicate_threshold_calibration():
    """Smoke-check the calibration: the exact 019e6e53 example must trip."""
    from spine.agents.exploration_agents import _topics_near_duplicate

    a = "How does the CLI entrypoint currently parse and handle command-line arguments?"
    b = "How does the command-line interface parse and handle flags?"
    assert _topics_near_duplicate(a, b) is True

    # Different topics that happen to mention CLI should not collide.
    c = "How is the CLI entrypoint exposed as a console_scripts entry?"
    d = "How are agent retries configured at the LangGraph layer?"
    assert _topics_near_duplicate(c, d) is False


# ── Explored-topic roll-up (per-topic outcome surfaced to manager) ─────────


def _good_finding(topic: str, files: dict[str, str]) -> dict:
    return {
        "topic": topic,
        "summary": f"Real findings for {topic}",
        "patterns": [],
        "file_map": files,
        "dependencies": [],
    }


def test_explored_roll_up_pairs_topic_with_files_examined():
    from spine.agents.exploration_agents import _render_explored_topic_roll_up

    topics = ["How is logging configured?"]
    findings = [
        _good_finding(
            "How is logging configured?",
            {"spine/logging.py": "configure_logging entry"},
        )
    ]
    out = _render_explored_topic_roll_up(topics, findings)
    assert "How is logging configured?" in out
    assert "1 file(s) examined" in out


def test_explored_roll_up_marks_sentinel_topics_attempted():
    from spine.agents.exploration_agents import _render_explored_topic_roll_up

    topics = ["How is verbose handled?"]
    findings = [_error_sentinel("How is verbose handled?")]
    out = _render_explored_topic_roll_up(topics, findings)
    assert "How is verbose handled?" in out
    assert "attempted" in out.lower()
    # Memory rule — no error text in any rendering path.
    assert "RuntimeError" not in out
    assert "Research failed" not in out


def test_explored_roll_up_flags_topics_without_findings():
    """A topic that was proposed but never produced a finding (e.g. router
    dropped it as a near-dupe) should surface as 'no result recorded' so
    the manager doesn't re-propose it as if it were new ground."""
    from spine.agents.exploration_agents import _render_explored_topic_roll_up

    topics = ["How is foo?", "How is bar?"]
    findings = [_good_finding("How is foo?", {"spine/foo.py": "foo"})]
    out = _render_explored_topic_roll_up(topics, findings)
    assert "How is foo?" in out
    assert "How is bar?" in out
    assert "no result recorded" in out


def test_explored_roll_up_returns_empty_on_round_one():
    from spine.agents.exploration_agents import _render_explored_topic_roll_up

    assert _render_explored_topic_roll_up([], []) == ""


def test_explored_roll_up_matches_enriched_finding_topic_to_bare_topic():
    """The router stamps a recall suffix on dispatched topics, so the
    finding's topic ends up enriched. The roll-up must still pair it
    with the bare topic the manager originally emitted.
    """
    from spine.agents.exploration_agents import _render_explored_topic_roll_up

    bare = "How does retry work?"
    enriched = bare + " — recall symbols: retry_with_backoff (spine/agents/retry.py)"
    out = _render_explored_topic_roll_up(
        [bare],
        [_good_finding(enriched, {"spine/agents/retry.py": "retry entrypoint"})],
    )
    # Topic line uses the bare form — the manager wrote that wording.
    assert f"- {bare} —" in out
    # Outcome reflects the actual finding.
    assert "1 file(s) examined" in out
    # Enriched form must not be duplicated on a separate line.
    assert "recall symbols" not in out


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
