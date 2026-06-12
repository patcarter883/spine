"""Window-aware synthesis budgeting + evidence compression (trace 019eb3dd).

Covers spine.agents.synthesis_budget and spine.agents.evidence_compression:
the budget ledger arithmetic, legacy fallback for providers without a
declared context_window, evidence allocation, the structural recall degrade,
and the map-reduce findings digest (including per-batch failure fallback and
the kill switch).
"""
from __future__ import annotations

import asyncio


from spine.agents._tokens import count_tokens
from spine.agents.evidence_compression import (
    compress_findings,
    compress_recall_chunks,
)
from spine.agents.synthesis_budget import (
    MIN_INPUT_BUDGET,
    SynthesisBudget,
    allocate_evidence,
    escalated_completion_cap,
    estimate_tool_payload_reserve,
    resolve_synthesis_budget,
    synthesis_completion_cap,
)
from spine.config import SpineConfig


def _config_with_provider(monkeypatch, provider: dict, **spine_overrides):
    cfg = SpineConfig(
        providers={"llm": [provider], "phases": {"specify": {"provider": provider["name"]}}},
        **spine_overrides,
    )
    monkeypatch.setattr(SpineConfig, "load", classmethod(lambda cls, path=".spine/config.yaml": cfg))
    return cfg


WINDOWED = {
    "name": "local-60k",
    "model": "openai:model",
    "enabled": True,
    "context_window": 60000,
}
UNWINDOWED = {
    "name": "cloud",
    "model": "openrouter:some/model",
    "enabled": True,
}


# ── resolve_synthesis_budget ─────────────────────────────────────────────


def test_legacy_when_no_context_window(monkeypatch):
    _config_with_provider(monkeypatch, UNWINDOWED)
    budget = resolve_synthesis_budget("specify", fixed_texts=["hello"])
    assert budget.legacy
    assert budget.completion_cap == 0
    cfg = SpineConfig()
    assert budget.input_budget == (
        cfg.synthesize_findings_token_budget + cfg.specify_context_token_budget
    )


def test_ledger_arithmetic_invariant(monkeypatch):
    cfg = _config_with_provider(monkeypatch, WINDOWED)
    fixed = "word " * 1000
    reserve = 2000
    budget = resolve_synthesis_budget(
        "specify", fixed_texts=[fixed], tool_payload_reserve=reserve
    )
    assert not budget.legacy
    fixed_cost = count_tokens(fixed)
    assert (
        fixed_cost
        + budget.input_budget
        + budget.completion_cap
        + reserve
        + cfg.synthesize_overhead_tokens
        <= budget.window
    )


def test_completion_cap_takes_tightest(monkeypatch):
    _config_with_provider(
        monkeypatch,
        {**WINDOWED, "max_completion_tokens": 30000},
    )
    # synthesize_max_completion_tokens default (8000) is tighter than the
    # provider's 30000.
    assert synthesis_completion_cap("specify") == 8000


def test_completion_cap_zero_without_window(monkeypatch):
    _config_with_provider(monkeypatch, UNWINDOWED)
    assert synthesis_completion_cap("specify") == 0


def test_input_budget_floored_when_fixed_dominates(monkeypatch):
    _config_with_provider(
        monkeypatch, {**WINDOWED, "context_window": 12000}
    )
    huge_fixed = "word " * 20000  # ≫ the 12K window
    budget = resolve_synthesis_budget("specify", fixed_texts=[huge_fixed])
    assert budget.input_budget == MIN_INPUT_BUDGET


# ── allocate_evidence ────────────────────────────────────────────────────


def test_allocation_pass_through_when_fits():
    budget = SynthesisBudget(window=60000, completion_cap=8000, input_budget=40000, legacy=False)
    alloc = allocate_evidence(budget, findings_tokens=10000, recall_tokens=5000)
    assert alloc.findings == 10000
    assert alloc.recall == 5000


def test_allocation_squeeze_keeps_findings_floor():
    budget = SynthesisBudget(window=60000, completion_cap=8000, input_budget=10000, legacy=False)
    alloc = allocate_evidence(budget, findings_tokens=30000, recall_tokens=30000)
    assert alloc.findings + alloc.recall <= budget.input_budget
    assert alloc.findings >= int(budget.input_budget * 0.6)
    assert alloc.recall > 0


def test_allocation_findings_take_all_when_no_recall():
    budget = SynthesisBudget(window=60000, completion_cap=8000, input_budget=10000, legacy=False)
    alloc = allocate_evidence(budget, findings_tokens=30000, recall_tokens=0)
    assert alloc.findings == budget.input_budget
    assert alloc.recall == 0


def test_allocation_legacy_returns_historical_constants(monkeypatch):
    _config_with_provider(monkeypatch, UNWINDOWED)
    budget = SynthesisBudget(window=0, completion_cap=0, input_budget=50000, legacy=True)
    alloc = allocate_evidence(budget, findings_tokens=99999, recall_tokens=99999)
    cfg = SpineConfig()
    assert alloc.findings == cfg.synthesize_findings_token_budget
    assert alloc.recall == cfg.specify_context_token_budget


# ── estimate_tool_payload_reserve ────────────────────────────────────────


def test_tool_payload_reserve_counts_artifacts(tmp_path):
    spec_dir = tmp_path / ".spine" / "artifacts" / "w1" / "specify"
    spec_dir.mkdir(parents=True)
    (spec_dir / "specification.md").write_text("spec " * 500, encoding="utf-8")
    reserve = estimate_tool_payload_reserve(
        workspace_root=str(tmp_path),
        artifact_dirs=[".spine/artifacts/w1/specify"],
        description="add a flag",
        feedback=["fix the scope"],
    )
    assert reserve >= count_tokens("spec " * 500)
    # Missing dirs are fine (first run — no prior spec).
    assert (
        estimate_tool_payload_reserve(
            workspace_root=str(tmp_path),
            artifact_dirs=[".spine/artifacts/nope/specify"],
            description="d",
        )
        > 0
    )


# ── compress_recall_chunks ───────────────────────────────────────────────


def _chunk(symbol: str, raw_tokens: int, summary: str = "short summary") -> dict:
    return {
        "symbol_name": symbol,
        "file_path": f"src/{symbol}.py",
        "raw_code": "code " * raw_tokens,
        "enriched_summary": summary,
    }


def test_recall_compression_noop_under_budget():
    chunks = [_chunk("a", 10), _chunk("b", 10)]
    out = compress_recall_chunks(chunks, budget_tokens=10000)
    assert all(c["raw_code"] for c in out)


def test_recall_compression_swaps_largest_first():
    chunks = [_chunk("small", 50), _chunk("huge", 5000)]
    out = compress_recall_chunks(chunks, budget_tokens=500)
    by_symbol = {c["symbol_name"]: c for c in out}
    assert by_symbol["huge"]["raw_code"] == ""  # degraded to summary
    assert by_symbol["small"]["raw_code"]  # kept
    # input untouched
    assert chunks[1]["raw_code"]


# ── compress_findings ────────────────────────────────────────────────────


class _FakeModel:
    def __init__(self, responses=None, fail=False):
        self.calls = 0
        self.fail = fail
        self.responses = responses

    def model_copy(self, update=None):
        return self

    async def ainvoke(self, prompt):
        self.calls += 1
        if self.fail:
            raise RuntimeError("digest call failed")

        class _R:
            content = "- digest: spine/config.py SpineConfig.load preserved"

        return _R()


def _finding(topic: str, words: int) -> dict:
    return {"topic": topic, "summary": "fact " * words}


def _patch_model(monkeypatch, model):
    import spine.agents.helpers as helpers

    monkeypatch.setattr(helpers, "resolve_chat_model", lambda *a, **k: model)


def test_findings_unchanged_when_under_budget(monkeypatch):
    _config_with_provider(monkeypatch, WINDOWED)
    model = _FakeModel()
    _patch_model(monkeypatch, model)
    findings = [_finding("t1", 10)]
    out = asyncio.run(
        compress_findings(findings, budget_tokens=10000, phase="specify")
    )
    assert out == findings
    assert model.calls == 0  # no LLM work when it fits


def test_findings_compressed_when_over_budget(monkeypatch):
    _config_with_provider(monkeypatch, WINDOWED)
    model = _FakeModel()
    _patch_model(monkeypatch, model)
    findings = [_finding(f"t{i}", 3000) for i in range(4)]
    out = asyncio.run(
        compress_findings(findings, budget_tokens=2000, phase="specify")
    )
    assert model.calls >= 1
    assert len(out) < len(findings)
    assert all("digest" in str(f.get("topic", "")).lower() for f in out)


def test_findings_batch_failure_keeps_originals(monkeypatch):
    _config_with_provider(monkeypatch, WINDOWED)
    model = _FakeModel(fail=True)
    _patch_model(monkeypatch, model)
    findings = [_finding(f"t{i}", 3000) for i in range(2)]
    out = asyncio.run(
        compress_findings(findings, budget_tokens=2000, phase="specify")
    )
    assert out == findings  # all batches failed → originals preserved


def test_findings_kill_switch(monkeypatch):
    _config_with_provider(
        monkeypatch, WINDOWED, evidence_compression_enabled=False
    )
    model = _FakeModel()
    _patch_model(monkeypatch, model)
    findings = [_finding(f"t{i}", 3000) for i in range(4)]
    out = asyncio.run(
        compress_findings(findings, budget_tokens=2000, phase="specify")
    )
    assert out == findings
    assert model.calls == 0


# ── escalated_completion_cap (length-truncated synth retry, 019eb940) ────


def test_escalation_doubles_within_window_room(monkeypatch):
    _config_with_provider(monkeypatch, WINDOWED)
    budget = SynthesisBudget(
        window=60000, completion_cap=8000, input_budget=20000, legacy=False
    )
    # room = 60000 - 5000 - overhead(4000) = 51000 → min(16000, 51000)
    assert escalated_completion_cap(budget, prompt_tokens=5000) == 16000


def test_escalation_bounded_by_window_room(monkeypatch):
    _config_with_provider(monkeypatch, WINDOWED)
    budget = SynthesisBudget(
        window=60000, completion_cap=8000, input_budget=20000, legacy=False
    )
    # room = 60000 - 46000 - 4000 = 10000 < 16000 → clamp to room
    assert escalated_completion_cap(budget, prompt_tokens=46000) == 10000


def test_escalation_zero_when_no_headroom(monkeypatch):
    _config_with_provider(monkeypatch, WINDOWED)
    budget = SynthesisBudget(
        window=60000, completion_cap=8000, input_budget=20000, legacy=False
    )
    # room = 60000 - 50000 - 4000 = 6000 <= cap → escalation impossible
    assert escalated_completion_cap(budget, prompt_tokens=50000) == 0


def test_escalation_zero_for_legacy_and_uncapped():
    legacy = SynthesisBudget(window=0, completion_cap=0, input_budget=50000, legacy=True)
    assert escalated_completion_cap(legacy, prompt_tokens=1000) == 0
    uncapped = SynthesisBudget(
        window=60000, completion_cap=0, input_budget=20000, legacy=False
    )
    assert escalated_completion_cap(uncapped, prompt_tokens=1000) == 0
