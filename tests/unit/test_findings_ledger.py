"""The durable codebase ledger (A2): a compact file→role map that survives
the findings-budget compaction the synthesizers apply.

``build_findings_ledger`` is a pure reduction over each finding's ``file_map``;
it must dedupe by file, skip error findings, let the current phase win over an
inherited prior-phase map, and bound its own size. The synthesizers inject it
as a ``<codebase_ledger>`` block that is NOT subject to ``compress_findings`` /
the ``_research_text`` cap, so it carries the full structural map even when the
verbose ``<findings>`` are trimmed.
"""
from __future__ import annotations

from spine.agents.exploration_agents import build_findings_ledger
from spine.agents.prompt_format import Tag


def _f(topic: str, file_map: dict[str, str]) -> dict:
    return {"topic": topic, "summary": f"summary for {topic}", "file_map": file_map}


def test_empty_findings_yield_empty_block():
    # Empty → "" so the caller's xml_blocks elides the section entirely.
    assert build_findings_ledger([]) == ""
    assert build_findings_ledger([{"topic": "x", "summary": "s"}]) == ""


def test_lists_every_file_with_role():
    out = build_findings_ledger(
        [
            _f("ui", {"spine/ui/api.py": "UIApi REST surface"}),
            _f("cfg", {"spine/config.py": "config loader"}),
        ]
    )
    assert "- spine/ui/api.py: UIApi REST surface" in out
    assert "- spine/config.py: config loader" in out
    # Header marks it as durable so the model knows not to expect it trimmed.
    assert "NEVER trimmed" in out


def test_error_findings_are_skipped():
    out = build_findings_ledger(
        [
            {"error": True, "file_map": {"should/skip.py": "leaked"}},
            _f("ok", {"keep/me.py": "kept"}),
        ]
    )
    assert "should/skip.py" not in out
    assert "keep/me.py" in out


def test_dedupes_by_file_current_phase_role_wins():
    prior = [_f("prior", {"shared.py": "(stale prior role)"})]
    current = [_f("now", {"shared.py": "live current role"})]
    out = build_findings_ledger(current, prior_findings=prior)
    assert out.count("shared.py") == 1
    assert "live current role" in out
    assert "stale prior role" not in out


def test_prior_only_files_are_retained():
    out = build_findings_ledger(
        [_f("now", {"a.py": "a"})],
        prior_findings=[_f("before", {"b.py": "b"})],
    )
    assert "a.py" in out and "b.py" in out


def test_role_and_total_size_are_bounded():
    huge_role = "x" * 5000
    many = [_f(f"t{i}", {f"file{i}.py": huge_role}) for i in range(200)]
    out = build_findings_ledger(many, max_files=10, max_role_chars=80, max_chars=2000)
    assert len(out) <= 2000
    # Per-line role truncation kicks in (no 5000-char line survives).
    assert huge_role not in out
    # Over the file cap → an explicit "and N more" marker, never silent.
    assert "more file" in out


def test_non_dict_and_missing_file_map_tolerated():
    out = build_findings_ledger(
        [None, "junk", {"file_map": "not-a-dict"}, _f("ok", {"real.py": "r"})]  # type: ignore[list-item]
    )
    assert "real.py" in out


def test_ledger_tag_exists():
    # The synthesizers render the ledger under this tag; guard the contract.
    assert Tag.CODEBASE_LEDGER.value == "codebase_ledger"
