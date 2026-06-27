"""Tests for the cross-run distilled-experience loop.

Covers the store (dedup, per-phase cap, relevance ranking), the distiller
(critic/adversarial feedback → lessons), and the capture/inject helpers.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from dataclasses import dataclass

import pytest

from spine.agents.experience import (
    capture_run_experience,
    distill_run_experience,
    experience_store_for,
    format_experience_block,
    resolve_experience_block,
)
from spine.models.types import ExperienceLesson
from spine.persistence.experience_store import (
    _MAX_LESSONS_PER_PHASE,
    ExperienceStore,
)


@dataclass
class _Cfg:
    """Minimal stand-in for SpineConfig (only the fields the helpers read)."""

    workspace_root: str
    experience_path: str = ".spine/experience"
    experience_capture: bool = True
    experience_injection: bool = True
    # Off in tests so capture never reaches out to an LLM. The generalisation
    # pass has its own dedicated test below.
    experience_generalize: bool = False


def _lesson(phase="plan", lesson="cover every requirement", **kw) -> ExperienceLesson:
    return ExperienceLesson(
        id=kw.pop("id", "abc123"),
        work_id=kw.pop("work_id", "w1"),
        phase=phase,
        lesson=lesson,
        trigger=kw.pop("trigger", "a requirement was uncovered"),
        salience=kw.pop("salience", 1),
        created_at=kw.pop("created_at", "2026-01-01T00:00:00"),
        **kw,
    )


# ── Store ────────────────────────────────────────────────────────────────────
def test_add_and_round_trip(tmp_path):
    store = ExperienceStore(base_path=str(tmp_path))
    assert store.add_many([_lesson()]) == 1
    got = store.all()
    assert len(got) == 1
    assert got[0].phase == "plan"
    assert got[0].lesson == "cover every requirement"


def test_dedup_same_lesson_not_duplicated(tmp_path):
    store = ExperienceStore(base_path=str(tmp_path))
    store.add_many([_lesson(lesson="Cover  every   requirement")])
    # Same text modulo whitespace/case → no new row.
    added = store.add_many([_lesson(lesson="cover every requirement", id="zzz")])
    assert added == 0
    assert len(store.all()) == 1


def test_dedup_stable_across_generalization_rewrite(tmp_path):
    # The same defect, generalized into different VISIBLE text on two runs, must
    # still dedup because dedup_basis (the pre-generalization text) is frozen.
    store = ExperienceStore(str(tmp_path / "exp"))
    first = _lesson(
        id="g0",
        lesson="Ensure every referenced symbol is produced by some slice",
        dedup_basis="slice-2 references foo() no slice provides",
    )
    assert store.add_many([first]) == 1
    # Run 2: same underlying defect, LLM paraphrased the visible lesson, same basis.
    second = _lesson(
        id="g1",
        work_id="w2",
        lesson="Check that all referenced symbols are defined by a slice",
        dedup_basis="slice-2 references foo() no slice provides",
    )
    assert store.add_many([second]) == 0
    assert len(store.all()) == 1


def test_dedup_keeps_higher_salience(tmp_path):
    store = ExperienceStore(base_path=str(tmp_path))
    store.add_many([_lesson(salience=1)])
    store.add_many([_lesson(salience=4, id="hot")])
    rows = store.all()
    assert len(rows) == 1
    assert rows[0].salience == 4


def test_per_phase_cap(tmp_path):
    store = ExperienceStore(base_path=str(tmp_path))
    many = [
        _lesson(lesson=f"lesson number {i}", id=f"id{i}", salience=i)
        for i in range(_MAX_LESSONS_PER_PHASE + 5)
    ]
    store.add_many(many)
    rows = store.all()
    assert len(rows) == _MAX_LESSONS_PER_PHASE
    # The highest-salience lessons survive the cap.
    assert min(le.salience for le in rows) == 5


def test_for_phase_ranks_category_then_salience(tmp_path):
    store = ExperienceStore(base_path=str(tmp_path))
    store.add_many(
        [
            _lesson(lesson="low salience no cat", id="a", salience=1),
            _lesson(lesson="high salience no cat", id="b", salience=9),
            _lesson(lesson="cat match low salience", id="c", salience=2, category="bugfix"),
        ]
    )
    ranked = store.for_phase("plan", category="bugfix", limit=3)
    # Category match wins the top slot even at lower salience.
    assert ranked[0].lesson == "cat match low salience"


def test_for_phase_filters_by_phase(tmp_path):
    store = ExperienceStore(base_path=str(tmp_path))
    store.add_many([_lesson(phase="plan", id="p"), _lesson(phase="specify", id="s")])
    assert {le.phase for le in store.for_phase("plan")} == {"plan"}


# ── Distiller ────────────────────────────────────────────────────────────────
def test_distill_from_terminal_escalation():
    result = {
        "work_id": "w9",
        "task_category": "feature",
        "feedback": [],
        "retry_count": {},
        "last_critic_review": {
            "phase": "plan",
            "status": "needs_review",
            "tier": "agent",
            "reason": "slice-2 references a method no slice provides",
            "suggestions": ["Ensure every referenced symbol is produced by a slice"],
            "attempt": 3,
        },
    }
    lessons = distill_run_experience(result, _Cfg(workspace_root="/x"))
    assert len(lessons) == 1
    le = lessons[0]
    assert le.phase == "plan"
    assert "referenced symbol" in le.lesson
    assert le.salience == 3
    assert le.category == "feature"


def test_distill_from_converged_rework_feedback():
    # Run passed (terminal review PASSED) but plan was reworked once — the
    # defect lives in the feedback list, attributed via retry_count.
    result = {
        "work_id": "w10",
        "feedback": [
            {
                "status": "needs_revision",
                "tier": "agent",
                "reason": "missing acceptance criteria coverage",
                "suggestions": ["Add a slice covering requirement R-003"],
            },
            {"status": "passed", "tier": "agent", "reason": "ok", "suggestions": []},
        ],
        "retry_count": {"plan": 1},
        "last_critic_review": {"phase": "plan", "status": "passed", "suggestions": []},
    }
    lessons = distill_run_experience(result, _Cfg(workspace_root="/x"))
    assert any("R-003" in le.lesson and le.phase == "plan" for le in lessons)


def test_distill_skips_structural_tier():
    result = {
        "work_id": "w11",
        "feedback": [
            {
                "status": "needs_revision",
                "tier": "structural",
                "reason": "artifact too short",
                "suggestions": ["expand it"],
            }
        ],
        "retry_count": {"plan": 1},
    }
    lessons = distill_run_experience(result, _Cfg(workspace_root="/x"))
    assert lessons == []


def test_distill_ambiguous_phase_not_guessed():
    # Two phases reworked, feedback entry has no phase hint → declined.
    result = {
        "work_id": "w12",
        "feedback": [
            {"status": "needs_revision", "tier": "agent", "reason": "x", "suggestions": ["y"]}
        ],
        "retry_count": {"plan": 1, "specify": 1},
    }
    lessons = distill_run_experience(result, _Cfg(workspace_root="/x"))
    assert lessons == []


def test_distill_attributes_explicit_phase_on_multiphase_run():
    # Two phases reworked, but the entry carries an explicit `phase` (stamped by
    # compose.py) → attributed, NOT dropped. Without the stamp this lesson would
    # be lost on any multi-phase run.
    result = {
        "work_id": "w12b",
        "feedback": [
            {
                "status": "needs_revision",
                "tier": "agent",
                "phase": "implement",
                "reason": "null deref on the cold-cache path",
                "suggestions": ["guard the optional field before deref"],
            }
        ],
        "retry_count": {"plan": 1, "implement": 2},
    }
    lessons = distill_run_experience(result, _Cfg(workspace_root="/x"))
    assert any(le.phase == "implement" for le in lessons)
    # Salience reflects the reworked phase's round count, not the other phase's.
    impl = next(le for le in lessons if le.phase == "implement")
    assert impl.salience == 2


# ── Capture + inject round trip ──────────────────────────────────────────────
@pytest.mark.asyncio
async def test_capture_then_resolve_block(tmp_path):
    cfg = _Cfg(workspace_root=str(tmp_path))
    result = {
        "work_id": "w13",
        "task_category": "feature",
        "feedback": [],
        "retry_count": {},
        "last_critic_review": {
            "phase": "plan",
            "status": "needs_revision",
            "tier": "agent",
            "reason": "dangling dependency on a removed slice",
            "suggestions": ["Validate every dependency id references an existing slice"],
            "attempt": 2,
        },
    }
    assert await capture_run_experience(result, cfg, "completed") == 1

    block = resolve_experience_block("plan", category="feature", config=cfg)
    assert "learned_experience" in block
    assert "Validate every dependency id" in block
    # A phase with no lessons yields an empty block.
    assert resolve_experience_block("implement", config=cfg) == ""


@pytest.mark.asyncio
async def test_capture_skipped_for_failed_status(tmp_path):
    cfg = _Cfg(workspace_root=str(tmp_path))
    result = {
        "work_id": "w14",
        "last_critic_review": {
            "phase": "plan",
            "status": "needs_revision",
            "tier": "agent",
            "reason": "x",
            "suggestions": ["y"],
        },
    }
    assert await capture_run_experience(result, cfg, "failed") == 0
    assert experience_store_for(cfg).all() == []


@pytest.mark.asyncio
async def test_capture_respects_disabled_flag(tmp_path):
    cfg = _Cfg(workspace_root=str(tmp_path), experience_capture=False)
    result = {
        "work_id": "w15",
        "last_critic_review": {
            "phase": "plan",
            "status": "needs_revision",
            "tier": "agent",
            "reason": "x",
            "suggestions": ["y"],
        },
    }
    assert await capture_run_experience(result, cfg, "completed") == 0


@pytest.mark.asyncio
async def test_resolve_block_respects_disabled_flag(tmp_path):
    cfg = _Cfg(workspace_root=str(tmp_path))
    await capture_run_experience(
        {
            "work_id": "w16",
            "last_critic_review": {
                "phase": "plan",
                "status": "needs_revision",
                "tier": "agent",
                "reason": "x",
                "suggestions": ["do y"],
            },
        },
        cfg,
        "completed",
    )
    disabled = _Cfg(workspace_root=str(tmp_path), experience_injection=False)
    assert resolve_experience_block("plan", config=disabled) == ""


def test_format_block_empty():
    assert format_experience_block([]) == ""


# ── Store delete / clear ─────────────────────────────────────────────────────
def test_delete_and_clear(tmp_path):
    store = ExperienceStore(base_path=str(tmp_path))
    store.add_many(
        [
            _lesson(phase="plan", lesson="a", id="1"),
            _lesson(phase="plan", lesson="b", id="2"),
            _lesson(phase="specify", lesson="c", id="3"),
        ]
    )
    assert store.delete("1") is True
    assert store.delete("nope") is False
    assert len(store.all()) == 2
    assert store.clear(phase="plan") == 1  # only id=2 left in plan
    assert {le.phase for le in store.all()} == {"specify"}
    assert store.clear() == 1
    assert store.all() == []


# ── LLM generalisation pass ──────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_generalize_rewrites_and_drops(monkeypatch):
    """The pass rewrites lesson text by index and drops flagged entries."""
    from spine.agents import experience as exp_mod

    raw = [
        _lesson(phase="plan", lesson="slice-2 references foo() no slice provides", id="r0"),
        _lesson(phase="plan", lesson="one-off junk", id="r1"),
    ]

    class _FakeBound:
        async def ainvoke(self, _messages):
            return exp_mod._GeneralizationResult(
                lessons=[
                    exp_mod._GeneralizedLesson(
                        index=0,
                        lesson="Ensure every referenced symbol is produced by some slice",
                    ),
                    exp_mod._GeneralizedLesson(index=1, drop=True),
                ]
            )

    from spine.agents import helpers

    monkeypatch.setattr(helpers, "resolve_chat_model", lambda *a, **k: object())
    monkeypatch.setattr(helpers, "bind_structured_output", lambda *a, **k: _FakeBound())

    out = await exp_mod.generalize_lessons(raw, _Cfg(workspace_root="/x"))
    assert len(out) == 1
    assert out[0].lesson == "Ensure every referenced symbol is produced by some slice"
    # Metadata is preserved through the rewrite.
    assert out[0].phase == "plan" and out[0].id == "r0"


@pytest.mark.asyncio
async def test_generalize_rejects_out_of_range_index(monkeypatch):
    """A 1-based renumbering must not graft a rule onto the wrong lesson.

    Model returns indices [1, 2] for inputs [0, 1]; index 1 is in range but would
    attach input-0's intended rule onto input-1, and index 2 is out of range.
    Both are rejected, so each input falls back to its original lesson unchanged.
    """
    from spine.agents import experience as exp_mod

    raw = [
        _lesson(phase="plan", lesson="original zero", id="r0"),
        _lesson(phase="implement", lesson="original one", id="r1"),
    ]

    class _FakeBound:
        async def ainvoke(self, _messages):
            return exp_mod._GeneralizationResult(
                lessons=[
                    exp_mod._GeneralizedLesson(index=1, lesson="rule meant for input 0"),
                    exp_mod._GeneralizedLesson(index=2, lesson="rule meant for input 1"),
                ]
            )

    from spine.agents import helpers

    monkeypatch.setattr(helpers, "resolve_chat_model", lambda *a, **k: object())
    monkeypatch.setattr(helpers, "bind_structured_output", lambda *a, **k: _FakeBound())

    out = await exp_mod.generalize_lessons(raw, _Cfg(workspace_root="/x"))
    # index 1 is in range, so it DOES rewrite input-1 (can't structurally tell a
    # same-position rewrite is "wrong"); index 2 is out of range and rejected, so
    # input-0 is untouched. The key guarantee: the out-of-range index never lands.
    assert out[0].lesson == "original zero"
    assert out[0].phase == "plan" and out[0].id == "r0"
    assert len(out) == 2


@pytest.mark.asyncio
async def test_generalize_rejects_duplicate_index(monkeypatch):
    """A duplicate index keeps only the first; it never silently overwrites twice."""
    from spine.agents import experience as exp_mod

    raw = [
        _lesson(phase="plan", lesson="zero", id="r0"),
        _lesson(phase="plan", lesson="one", id="r1"),
    ]

    class _FakeBound:
        async def ainvoke(self, _messages):
            return exp_mod._GeneralizationResult(
                lessons=[
                    exp_mod._GeneralizedLesson(index=0, lesson="first wins"),
                    exp_mod._GeneralizedLesson(index=0, lesson="second ignored"),
                ]
            )

    from spine.agents import helpers

    monkeypatch.setattr(helpers, "resolve_chat_model", lambda *a, **k: object())
    monkeypatch.setattr(helpers, "bind_structured_output", lambda *a, **k: _FakeBound())

    out = await exp_mod.generalize_lessons(raw, _Cfg(workspace_root="/x"))
    assert out[0].lesson == "first wins"
    assert out[1].lesson == "one"  # index 1 had no entry → original kept


@pytest.mark.asyncio
async def test_generalize_fails_open(monkeypatch):
    """Any failure in the pass returns the input lessons unchanged."""
    from spine.agents import experience as exp_mod

    raw = [_lesson(lesson="keep me", id="k")]

    def _boom(*a, **k):
        raise RuntimeError("no model")

    from spine.agents import helpers

    monkeypatch.setattr(helpers, "resolve_chat_model", _boom)
    out = await exp_mod.generalize_lessons(raw, _Cfg(workspace_root="/x"))
    assert out == raw
