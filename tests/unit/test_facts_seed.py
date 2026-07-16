"""`spine facts seed`: docs → distilled candidates → gate-filtered writes.

Seeding populates the CAM store from curated onboarding docs so the first
run gets a useful <known_facts> block. Load-bearing behaviours: chunking,
cross-chunk subject dedupe (canonicalization), the total candidate cap, the
capacity-alert trim, dry-run writing nothing, and source="seeded" records.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import spine.agents.facts as facts_mod
from spine.agents.facts import _chunk_paragraphs, _FactCandidate, seed_project_facts
from spine.models.types import ProjectFact
from spine.persistence.facts_store import FactsStore


# ── chunking ─────────────────────────────────────────────────────────────────
def test_chunk_paragraphs_packs_and_hard_splits():
    text = "para one\n\npara two\n\n" + "x" * 50
    chunks = _chunk_paragraphs(text, size=20)
    # 8 + 2 + 8 chars pack into one chunk; a third paragraph would not fit.
    assert chunks[0] == "para one\n\npara two"
    # The 50-char paragraph is hard-split into <=20-char pieces, in order.
    assert all(len(c) <= 20 for c in chunks)
    assert "".join(chunks[1:]) == "x" * 50

    joined = _chunk_paragraphs("a\n\nb", size=100)
    assert joined == ["a\n\nb"]  # small paragraphs pack together


# ── fixtures ─────────────────────────────────────────────────────────────────
class _FakeCAMClient:
    def __init__(self, stats=None, gate=lambda subject: True):
        self._stats = stats
        self._gate = gate
        self.remember_calls: list[tuple] = []
        self.saved = False

    async def remember(self, subject, prompt, object_, mode=None):
        self.remember_calls.append((subject, prompt, object_, mode))
        return {"stored": self._gate(subject), "base_p": 0.0}

    async def ask_full(self, prompt, subject, max_tokens=32, mode=None):
        return {"text": "the answer is main indeed"}

    async def stats(self):
        return self._stats

    async def facts(self):
        return None

    async def save(self):
        self.saved = True
        return {}

    async def aclose(self):
        pass


def _config(tmp_path, cam: dict | bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        experience_path=str(tmp_path / "exp"),
        workspace_root=str(tmp_path),
        resolve_active_provider=lambda: {
            "base_url": "http://h:1919/v1",
            "cam": cam,
        },
    )


def _write_docs(tmp_path, texts: dict[str, str]):
    d = tmp_path / ".spine" / "onboarding"
    d.mkdir(parents=True)
    for name, text in texts.items():
        (d / name).write_text(text, encoding="utf-8")


def _install(monkeypatch, fake_client, candidates_per_call):
    monkeypatch.setattr(
        "spine.services.cam_client.CAMClient", lambda settings: fake_client
    )
    calls = []

    async def fake_distill(material, max_object_words=4, known_subjects=None, source="run"):
        calls.append(
            {"material": material, "known": list(known_subjects or []), "source": source}
        )
        return list(candidates_per_call)

    monkeypatch.setattr(facts_mod, "_distill_material", fake_distill)
    return calls


_CANDS = [
    _FactCandidate(subject="spine test runner", probe_prompt="Tests run with", object="pytest"),
    _FactCandidate(subject="spine ui port", probe_prompt="The spine UI serves on port", object="8501"),
]


# ── seed_project_facts ───────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_seed_reads_onboarding_dedupes_and_records(tmp_path, monkeypatch):
    _write_docs(tmp_path, {"A.md": "alpha doc", "B.md": "beta doc"})
    fake = _FakeCAMClient(stats={"total_edits": 0})
    # Both docs distil to the SAME candidates — dedupe must collapse them.
    calls = _install(monkeypatch, fake, _CANDS)
    cfg = _config(tmp_path)

    summary = await seed_project_facts(cfg, max_facts=20)

    assert len(summary["docs"]) == 2
    assert [c.subject for c in summary["candidates"]] == [
        "spine test runner",
        "spine ui port",
    ]
    assert summary["accepted"] == 2
    # Docs prompt variant + doc-two's call saw doc-one's accepted subjects.
    assert calls[0]["source"] == "docs"
    assert "spine test runner" in calls[1]["known"]
    records = FactsStore(str(tmp_path / "exp")).all()
    assert len(records) == 2
    assert all(r.source == "seeded" for r in records)
    assert fake.saved is True


@pytest.mark.asyncio
async def test_seed_dry_run_writes_nothing(tmp_path, monkeypatch):
    _write_docs(tmp_path, {"A.md": "alpha doc"})

    def _no_client(settings):
        raise AssertionError("dry run must not build a CAM client")

    monkeypatch.setattr("spine.services.cam_client.CAMClient", _no_client)

    async def fake_distill(material, **kwargs):
        return list(_CANDS)

    monkeypatch.setattr(facts_mod, "_distill_material", fake_distill)

    summary = await seed_project_facts(_config(tmp_path), dry_run=True)

    assert len(summary["candidates"]) == 2
    assert summary["records"] == [] and summary["accepted"] == 0
    assert FactsStore(str(tmp_path / "exp")).all() == []


@pytest.mark.asyncio
async def test_seed_respects_max_facts_and_side_index_dedupe(tmp_path, monkeypatch):
    _write_docs(tmp_path, {"A.md": "alpha doc"})
    # 'spine ui port' already in the side index -> not proposed again.
    FactsStore(str(tmp_path / "exp")).add_many(
        [
            ProjectFact(
                id="x" * 12,
                subject="spine ui port",
                probe_prompt="p",
                object="8501",
                namespace="spine-seed-t",  # same namespace the cam config pins
                stored=True,
                created_at="2026-07-16T00:00:00",
            )
        ]
    )
    fake = _FakeCAMClient(stats={"total_edits": 0})
    _install(monkeypatch, fake, _CANDS)
    cfg = _config(tmp_path, cam={"namespace": "spine-seed-t"})

    summary = await seed_project_facts(cfg, max_facts=1)

    # max_facts=1 caps the batch; the side-index subject is excluded first.
    assert [c.subject for c in summary["candidates"]] == ["spine test runner"]


@pytest.mark.asyncio
async def test_seed_capacity_trim_blocks_over_headroom(tmp_path, monkeypatch):
    _write_docs(tmp_path, {"A.md": "alpha doc"})
    # Store already at 99 of alert=100: headroom 1 -> second candidate blocked.
    fake = _FakeCAMClient(stats={"total_edits": 99})
    _install(monkeypatch, fake, _CANDS)
    cfg = _config(tmp_path, cam={"capacity_alert": 100})

    summary = await seed_project_facts(cfg)

    assert len(summary["records"]) == 1
    assert summary["blocked"] == 1
    assert len(fake.remember_calls) == 1


def test_is_alias_same_object_overlapping_subject():
    from spine.agents.facts import _is_alias

    cand = _FactCandidate(
        subject="spine agents default checkpoint path",
        probe_prompt="p",
        object=".spine/spine.db",
    )
    # The live 2026-07-16 seed duplicate: same object, shared 'checkpoint'/'path'.
    assert _is_alias(cand, [("spine checkpoint database path", ".spine/spine.db")])
    # Same object but disjoint meaningful tokens: NOT an alias.
    assert not _is_alias(cand, [("spine ui landing page", ".spine/spine.db")])
    # Different object: never an alias.
    assert not _is_alias(cand, [("spine checkpoint database path", "other.db")])


@pytest.mark.asyncio
async def test_seed_drops_near_alias_candidates(tmp_path, monkeypatch):
    _write_docs(tmp_path, {"A.md": "alpha doc"})
    fake = _FakeCAMClient(stats={"total_edits": 0})
    aliases = [
        _FactCandidate(
            subject="spine checkpoint database path",
            probe_prompt="The spine checkpoint db lives at",
            object=".spine/spine.db",
        ),
        _FactCandidate(
            subject="spine agents checkpoint file path",
            probe_prompt="The agents checkpoint file path is",
            object=".spine/spine.db",
        ),
    ]
    _install(monkeypatch, fake, aliases)

    summary = await seed_project_facts(_config(tmp_path))

    assert [c.subject for c in summary["candidates"]] == [
        "spine checkpoint database path"
    ]
    assert summary["aliases_dropped"] >= 1


@pytest.mark.asyncio
async def test_seed_replays_cached_candidates_without_distilling(
    tmp_path, monkeypatch
):
    _write_docs(tmp_path, {"A.md": "alpha doc"})
    fake = _FakeCAMClient(stats={"total_edits": 0})
    monkeypatch.setattr(
        "spine.services.cam_client.CAMClient", lambda settings: fake
    )

    async def must_not_distill(*a, **k):
        raise AssertionError("--from replay must not call the distiller")

    monkeypatch.setattr(facts_mod, "_distill_material", must_not_distill)
    lines: list[str] = []

    summary = await seed_project_facts(
        _config(tmp_path),
        candidates_override=[
            {"subject": "spine test runner", "probe_prompt": "Tests run with", "object": "pytest"}
        ],
        progress=lines.append,
    )

    assert summary["accepted"] == 1
    assert summary["records"][0].source == "seeded"
    assert any("replaying 1 cached" in ln for ln in lines)


@pytest.mark.asyncio
async def test_seed_without_cam_provider_raises(tmp_path):
    cfg = SimpleNamespace(
        experience_path=str(tmp_path),
        workspace_root=str(tmp_path),
        resolve_active_provider=lambda: {"base_url": "http://h:1/v1"},
    )
    with pytest.raises(RuntimeError):
        await seed_project_facts(cfg)


@pytest.mark.asyncio
async def test_seed_gate_skips_are_recorded_not_counted(tmp_path, monkeypatch):
    _write_docs(tmp_path, {"A.md": "alpha doc"})
    fake = _FakeCAMClient(
        stats={"total_edits": 0}, gate=lambda s: s == "spine test runner"
    )
    _install(monkeypatch, fake, _CANDS)

    summary = await seed_project_facts(_config(tmp_path))

    assert summary["accepted"] == 1
    assert len(summary["records"]) == 2  # skip recorded too
    by_subject = {r.subject: r for r in summary["records"]}
    assert by_subject["spine ui port"].stored is False
