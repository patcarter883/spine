"""Project-facts capture: side-index store semantics + gate-filtered writes.

The facts.jsonl side index is the authoritative intent log for the CAM memory
organ (the banks can't be enumerated), so its one-value-per-subject semantics
and the capture pipeline's fail-open behaviour are load-bearing: capture must
no-op without a cam provider, stop on an unreachable server without recording
phantom attempts, respect the capacity guard, and record gate-skipped attempts
(stored=False) alongside accepted ones.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import spine.agents.facts as facts_mod
from spine.agents.facts import _FactCandidate, capture_run_facts
from spine.models.types import ProjectFact
from spine.persistence.facts_store import FactsStore


def _fact(subject: str, obj: str = "main", ns: str | None = "proj", stored: bool = True) -> ProjectFact:
    return ProjectFact(
        id="x" * 12,
        work_id="w1",
        subject=subject,
        probe_prompt=f"The {subject} is",
        object=obj,
        namespace=ns,
        stored=stored,
        base_p=0.1,
        created_at="2026-07-08T00:00:00",
    )


# ── FactsStore ───────────────────────────────────────────────────────────────
def test_store_roundtrip_and_subject_replacement(tmp_path):
    store = FactsStore(str(tmp_path))
    assert store.add_many([_fact("default branch", "main")]) == 1
    # Same subject+namespace: replaces (one value per subject), not a new record.
    assert store.add_many([_fact("default branch", "develop")]) == 0
    all_facts = store.all()
    assert len(all_facts) == 1
    assert all_facts[0].object == "develop"


def test_store_stored_filter_and_namespace_scoping(tmp_path):
    store = FactsStore(str(tmp_path))
    store.add_many(
        [
            _fact("a", stored=True, ns="p1"),
            _fact("b", stored=False, ns="p1"),
            _fact("c", stored=True, ns="p2"),
        ]
    )
    assert {f.subject for f in store.stored()} == {"a", "c"}
    assert {f.subject for f in store.stored(namespace="p1")} == {"a"}


def test_store_delete_and_clear(tmp_path):
    store = FactsStore(str(tmp_path))
    store.add_many([_fact("a"), _fact("b")])
    assert store.delete("A") is True  # case-normalised
    assert store.delete("nope") is False
    assert store.clear() == 1


# ── capture_run_facts ────────────────────────────────────────────────────────
class _FakeCAMClient:
    """Scripted /cam/* client standing in for CAMClient."""

    def __init__(self, remember_responses, stats=None, facts=None, ask_text="ok"):
        self.remember_responses = list(remember_responses)
        self._stats = stats
        self._facts = facts
        self._ask_text = ask_text
        self.remember_calls: list[tuple] = []
        self.ask_calls: list[tuple] = []
        self.saved = False

    async def remember(self, subject, prompt, object_):
        self.remember_calls.append((subject, prompt, object_))
        return self.remember_responses.pop(0) if self.remember_responses else None

    async def ask(self, prompt, subject, max_tokens=32):
        self.ask_calls.append((prompt, subject))
        return self._ask_text

    async def stats(self):
        return self._stats

    async def facts(self):
        return self._facts

    async def save(self):
        self.saved = True
        return {}

    async def aclose(self):
        pass


def _config(tmp_path, cam: dict | bool | None = None) -> SimpleNamespace:
    provider = {"base_url": "http://h:1919/v1"}
    if cam is not None:
        provider["cam"] = cam
    return SimpleNamespace(
        experience_path=str(tmp_path),
        workspace_root=str(tmp_path),
        resolve_active_provider=lambda: provider,
    )


def _install_fake_client(monkeypatch, fake):
    monkeypatch.setattr(
        "spine.services.cam_client.CAMClient", lambda settings: fake
    )


def _install_candidates(monkeypatch, candidates):
    async def fake_distill(result, config):
        return candidates

    monkeypatch.setattr(facts_mod, "distill_run_facts", fake_distill)


_CANDS = [
    _FactCandidate(subject="default branch", probe_prompt="The default branch is", object="main"),
    _FactCandidate(subject="test runner", probe_prompt="Tests run with", object="pytest"),
]


@pytest.mark.asyncio
async def test_capture_noop_without_cam_provider(tmp_path, monkeypatch):
    _install_candidates(monkeypatch, _CANDS)
    assert await capture_run_facts({"work_id": "w"}, _config(tmp_path), "completed") == 0


@pytest.mark.asyncio
async def test_capture_noop_unless_write_distill(tmp_path, monkeypatch):
    _install_candidates(monkeypatch, _CANDS)
    cfg = _config(tmp_path, cam={"namespace": "p", "write": "off"})
    assert await capture_run_facts({"work_id": "w"}, cfg, "completed") == 0


@pytest.mark.asyncio
async def test_capture_records_accepted_and_gate_skipped(tmp_path, monkeypatch):
    fake = _FakeCAMClient(
        remember_responses=[
            {"stored": True, "base_p": 0.02},
            {"stored": False, "base_p": 0.91},
        ],
        stats={"total_facts": 3},
        ask_text="It is main, of course.",
    )
    _install_fake_client(monkeypatch, fake)
    _install_candidates(monkeypatch, _CANDS)
    cfg = _config(tmp_path, cam={"namespace": "p"})

    stored = await capture_run_facts({"work_id": "w1"}, cfg, "completed")

    assert stored == 1
    assert len(fake.remember_calls) == 2
    assert fake.saved is True  # /cam/save after an accepted write
    # Readback probe runs only for the accepted write.
    assert len(fake.ask_calls) == 1
    records = FactsStore(str(tmp_path)).all()
    assert len(records) == 2  # gate-skip is recorded too (stored=False)
    by_subject = {r.subject: r for r in records}
    assert by_subject["default branch"].stored is True
    assert by_subject["default branch"].verified is True  # "main" in ask text
    assert by_subject["test runner"].stored is False
    assert by_subject["test runner"].verified is None
    assert by_subject["test runner"].base_p == 0.91


@pytest.mark.asyncio
async def test_capture_flags_failed_readback_probe(tmp_path, monkeypatch):
    fake = _FakeCAMClient(
        remember_responses=[{"stored": True, "base_p": 0.02}],
        stats={"total_facts": 3},
        ask_text="something unrelated",  # store did not deliver the object
    )
    _install_fake_client(monkeypatch, fake)
    _install_candidates(monkeypatch, _CANDS[:1])
    cfg = _config(tmp_path, cam={"namespace": "p"})

    await capture_run_facts({"work_id": "w1"}, cfg, "completed")

    records = FactsStore(str(tmp_path)).all()
    assert records[0].verified is False


@pytest.mark.asyncio
async def test_capture_capacity_guard_blocks_writes(tmp_path, monkeypatch):
    fake = _FakeCAMClient(remember_responses=[], stats={"total_facts": 120})
    _install_fake_client(monkeypatch, fake)
    _install_candidates(monkeypatch, _CANDS)
    cfg = _config(tmp_path, cam={"namespace": "p", "capacity_alert": 100})

    assert await capture_run_facts({"work_id": "w"}, cfg, "completed") == 0
    assert fake.remember_calls == []  # nothing written past the alert threshold


@pytest.mark.asyncio
async def test_capture_server_down_records_nothing(tmp_path, monkeypatch):
    # remember() -> None means the server never saw the write; no phantom records.
    fake = _FakeCAMClient(remember_responses=[None], stats=None, facts=None)
    _install_fake_client(monkeypatch, fake)
    _install_candidates(monkeypatch, _CANDS)
    cfg = _config(tmp_path, cam={"namespace": "p"})

    assert await capture_run_facts({"work_id": "w"}, cfg, "completed") == 0
    assert FactsStore(str(tmp_path)).all() == []
    assert fake.saved is False


@pytest.mark.asyncio
async def test_capture_skips_crash_statuses(tmp_path, monkeypatch):
    _install_candidates(monkeypatch, _CANDS)
    cfg = _config(tmp_path, cam={"namespace": "p"})
    assert await capture_run_facts({"work_id": "w"}, cfg, "failed") == 0


def test_candidate_validation_rejects_long_objects():
    long_obj = _FactCandidate(
        subject="s", probe_prompt="p", object="a very long five word answer"
    )
    assert facts_mod._valid_candidate(long_obj) is False
    ok = _FactCandidate(subject="s", probe_prompt="p", object="main")
    assert facts_mod._valid_candidate(ok) is True


# ── known-facts prompt block (F1.3) ──────────────────────────────────────────
def test_known_facts_block_renders_stored_facts_for_namespace(tmp_path):
    FactsStore(str(tmp_path)).add_many(
        [
            _fact("default branch", "main", ns="p"),
            _fact("gate skipped", "x", ns="p", stored=False),  # never injected
            _fact("other project", "y", ns="q"),  # wrong namespace
        ]
    )
    cfg = _config(tmp_path, cam={"namespace": "p", "read": "facts_block"})
    block = facts_mod.resolve_known_facts_block(cfg)
    assert "<known_facts>" in block
    assert "default branch: main" in block
    assert "gate skipped" not in block
    assert "other project" not in block


def test_known_facts_block_empty_unless_facts_block_mode(tmp_path):
    FactsStore(str(tmp_path)).add_many([_fact("default branch", "main", ns="p")])
    # transparent mode: delivery is in-forward, no prompt block.
    cfg = _config(tmp_path, cam={"namespace": "p", "read": "transparent"})
    assert facts_mod.resolve_known_facts_block(cfg) == ""
    # no cam at all
    assert facts_mod.resolve_known_facts_block(_config(tmp_path)) == ""
    # `both` injects the deterministic block alongside transparent delivery.
    cfg = _config(tmp_path, cam={"namespace": "p", "read": "both"})
    assert "default branch: main" in facts_mod.resolve_known_facts_block(cfg)
