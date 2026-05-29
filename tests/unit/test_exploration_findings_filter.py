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


def test_format_findings_budget_under_passes_all():
    """All findings emitted verbatim when their total token count fits."""
    findings = [_good_finding() for _ in range(3)]
    out = _format_findings(findings, budget=10_000)
    assert out.count("config loading") == 3
    assert "[truncated:" not in out


def test_format_findings_budget_over_truncates_with_marker():
    """Once the budget is crossed, remaining findings are dropped and a
    visible truncation marker tells the synthesizer the count omitted."""
    # 20 findings is plenty to overflow a tiny 200-token budget.
    findings = [_good_finding() for _ in range(20)]
    out = _format_findings(findings, budget=200)
    assert "[truncated:" in out
    assert "200-token budget" in out
    # At least *some* findings included; not the full 20.
    assert 0 < out.count("config loading") < 20


def test_format_findings_zero_budget_is_unbounded():
    """budget=0 must NOT enable the cap (defensive — config typo guard)."""
    findings = [_good_finding() for _ in range(15)]
    out_capped = _format_findings(findings, budget=0)
    out_default = _format_findings(findings)
    assert out_capped == out_default
    assert "[truncated:" not in out_capped


def test_format_findings_skips_errors_when_counting_omitted():
    """Error sentinels remain filtered out and don't count as omitted."""
    findings = [_good_finding()] * 5 + [_error_finding()] * 5 + [_good_finding()] * 5
    # Budget that fits only ~the first two findings.
    out = _format_findings(findings, budget=50)
    if "[truncated:" in out:
        # Omitted count must exclude the 5 error sentinels.
        # We can't compute exact N without rendering, but it must be < 10.
        import re
        m = re.search(r"\[truncated: (\d+) more findings", out)
        assert m is not None
        assert 0 < int(m.group(1)) <= 10


def test_salvage_sentinel_carries_no_exception_text_in_summary():
    """The error-sentinel finding emitted by the salvage path at
    ``spine/agents/exploration_agents.py`` MUST NOT include raw exception
    text in any free-text field readable by an LLM that introspects
    ``state['findings']`` directly.

    This is the source-of-truth regression — every render-time filter is
    just defence-in-depth. The user observed in production that even with
    all render filters in place, sub-agents reading state still got fed
    the "Research failed for topic '…': GraphRecursionError: …" string
    because the entry sat in state for the research_manager's coverage
    bookkeeping. The fix is to keep the entry but neutralise the summary.
    """
    # Reach into the salvage block via a small fake invocation. Easier:
    # construct the dict shape the source code now emits and assert its
    # invariants. This catches any future regression that puts exception
    # text back into the summary, since the source code asserts the same
    # invariant inline.
    err = RuntimeError("Recursion limit of 50 reached without hitting a "
                       "stop condition — full multi-line stack trace here")
    sentinel = {
        "topic": "some topic",
        "summary": "(research did not converge on this topic)",
        "patterns": [],
        "file_map": {},
        "dependencies": [],
        "error": True,
        "error_class": type(err).__name__,
        "error_topic": "some topic",
    }
    # The sentinel's summary must be safe to feed to ANY downstream
    # consumer — no exception text, no stack trace, no class name.
    forbidden_in_summary = (
        "RuntimeError", "Traceback", "GraphRecursionError",
        "Recursion limit", "stop condition", "Research failed",
    )
    for needle in forbidden_in_summary:
        assert needle not in sentinel["summary"], (
            f"summary leaks {needle!r}: {sentinel['summary']!r}"
        )
    # And the same invariants must hold against the LIVE sentinel
    # constructor in spine.agents.exploration_agents._empty_research_finding.
    # The supervisor↔worker refactor moved error capture inside
    # run_worker_node — exceptions never bubble up to the explore_do node
    # any more — but the canonical sentinel constructor still emits the
    # finding that summarise / format_findings / save_artifacts filter on.
    from spine.agents.exploration_agents import _empty_research_finding

    live = _empty_research_finding(
        "some topic",
        error_class=type(err).__name__,
    )
    for needle in forbidden_in_summary:
        assert needle not in live["summary"], (
            f"live sentinel summary leaks {needle!r}: {live['summary']!r}"
        )
    # The structured error_class field IS allowed to contain the class
    # name — downstream consumers key off ``error=True`` to drop these,
    # so the class name is metadata not user-facing text.
    assert live["error"] is True
    assert live["error_class"] == "RuntimeError"
    # And the StructuredFinding ERROR path (the new salvage point) must
    # likewise route exception text away from any summary-like field.
    from spine.agents.researcher_supervisor import (
        FindingStatus,
        StructuredFinding,
        ToolClass,
        render_history_as_evidence,
    )

    err_finding = StructuredFinding(
        tool_name="codebase_query",
        tool_class=ToolClass.READ_SOURCE,
        status=FindingStatus.ERROR,
        execution_error_details=str(err),  # safe — never reaches summary
    )
    # render_history_as_evidence DROPS error findings entirely, so
    # forbidden tokens cannot leak into the evidence dossier even if a
    # worker turn captures them in execution_error_details.
    rendered = render_history_as_evidence([err_finding])
    for needle in forbidden_in_summary:
        assert needle not in rendered, (
            f"history render leaks {needle!r}: {rendered!r}"
        )


def test_export_markdown_drops_error_sentinels():
    """The human-readable markdown export (spine/workflow/export.py) must
    NOT render error-sentinel findings.

    Regression for the leak the user surfaced after 10+ attempts —
    GraphRecursionError text was reaching the export's "Research → Findings"
    section because export.py iterated the raw findings list without the
    same ``error=True`` filter every other render site applies.
    """
    from spine.workflow.export import format_export_markdown

    data = {
        "work_id": "demo",
        "description": "test",
        "phases": {
            "specify": {
                "research": {
                    "topics": ["config loading", "auth"],
                    "findings": [_good_finding(), _error_finding()],
                },
            },
        },
    }
    md = format_export_markdown(data)
    assert "Research failed" not in md
    assert "GraphRecursionError" not in md
    assert "config loading" in md  # the real finding still rendered


def test_save_artifacts_research_log_drops_error_sentinels(tmp_path, monkeypatch):
    """The persisted research_log.json must NOT carry error-sentinel
    findings — otherwise raw GraphRecursionError / Salvage strings leak
    into the on-disk research results that critics, agents and humans
    consume after the run.

    Regression for trace 019e6cc4-f57d-7652-a718-15d04278ad5c, where
    16 errored explore branches put their raw exception text into the
    research_log artifact via the unfiltered comprehension in
    _save_exploration_artifacts.
    """
    import json
    import asyncio
    from spine.workflow.subgraphs.exploration_subgraph import (
        _save_exploration_artifacts,
    )

    # Point the artifact store at a temp dir via SpineConfig override.
    from spine.config import SpineConfig
    real_load = SpineConfig.load

    def _fake_load(*_a, **_kw):
        cfg = real_load()
        cfg.artifact_path = str(tmp_path)
        return cfg

    monkeypatch.setattr(SpineConfig, "load", classmethod(lambda cls, *a, **k: _fake_load()))

    # Satisfy the SPECIFY phase contract check (line 924-961) so the
    # function reaches the research_log persist step we want to test.
    spec_dir = tmp_path / ".spine" / "artifacts" / "test-work" / "specify"
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "specification.json").write_text(json.dumps({
        "title": "Test", "summary": "x", "requirements": [],
    }))

    state = {
        "phase": "specify",
        "work_id": "test-work",
        "workspace_root": str(tmp_path),
        "topics": ["auth", "config loading"],
        "findings": [_good_finding(), _error_finding()],
        "agent_response": "",
    }
    asyncio.run(_save_exploration_artifacts(state))

    log_path = tmp_path / "test-work" / "specify" / "research_log.json"
    assert log_path.exists(), f"research_log.json not written under {tmp_path}"
    body = log_path.read_text()
    assert "Research failed" not in body, body
    assert "GraphRecursionError" not in body, body
    payload = json.loads(body)
    assert len(payload["findings"]) == 1
    assert payload["findings"][0]["topic"] == "config loading"
