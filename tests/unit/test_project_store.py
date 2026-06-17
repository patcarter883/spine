"""Tests for ProjectStore — atomic persistence + idempotent membership."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.models.types import ProjectSpec, RequirementRef, Roadmap, RoadmapPhase
from spine.persistence.project_store import ProjectStore


def _spec(pid: str = "demo", **kw) -> ProjectSpec:
    return ProjectSpec(
        id=pid,
        title=kw.pop("title", "Demo"),
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
        **kw,
    )


def test_save_load_round_trip(tmp_path):
    store = ProjectStore(base_path=str(tmp_path))
    spec = _spec(
        requirements=[RequirementRef(id="R-001", text="Do the thing")],
        member_work_ids=["w1", "w2"],
    )
    store.save_project(spec)

    loaded = store.load_project("demo")
    assert loaded is not None
    assert loaded.id == "demo"
    assert loaded.member_work_ids == ["w1", "w2"]
    assert loaded.requirements[0].id == "R-001"


def test_load_missing_returns_none(tmp_path):
    store = ProjectStore(base_path=str(tmp_path))
    assert store.load_project("nope") is None


def test_meta_sidecar_written(tmp_path):
    store = ProjectStore(base_path=str(tmp_path))
    store.save_project(_spec(member_work_ids=["w1"]))
    meta_path = tmp_path / "demo" / "spec.json.meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["project_id"] == "demo"
    assert meta["member_count"] == 1


def test_save_is_atomic_no_tmp_left(tmp_path):
    store = ProjectStore(base_path=str(tmp_path))
    store.save_project(_spec())
    assert not (tmp_path / "demo" / "spec.json.tmp").exists()
    assert (tmp_path / "demo" / "spec.json").exists()


def test_add_members_idempotent_set_union(tmp_path):
    store = ProjectStore(base_path=str(tmp_path))
    store.save_project(_spec(member_work_ids=["w1"]))

    spec = store.add_members("demo", ["w2", "w1", "w3"])
    assert spec.member_work_ids == ["w1", "w2", "w3"]  # order preserved, no dupes

    # Re-adding existing members is a no-op for the set.
    spec2 = store.add_members("demo", ["w2"])
    assert spec2.member_work_ids == ["w1", "w2", "w3"]


def test_add_members_unknown_project_raises(tmp_path):
    store = ProjectStore(base_path=str(tmp_path))
    try:
        store.add_members("ghost", ["w1"])
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown project")


def test_list_projects_sorted(tmp_path):
    store = ProjectStore(base_path=str(tmp_path))
    store.save_project(_spec("beta"))
    store.save_project(_spec("alpha"))
    assert store.list_projects() == ["alpha", "beta"]


def test_remove_members_set_difference(tmp_path):
    store = ProjectStore(base_path=str(tmp_path))
    store.save_project(_spec(member_work_ids=["w1", "w2", "w3"]))

    spec = store.remove_members("demo", ["w2", "missing"])
    assert spec.member_work_ids == ["w1", "w3"]


def test_remove_members_strips_phase_membership(tmp_path):
    store = ProjectStore(base_path=str(tmp_path))
    store.save_project(
        _spec(
            member_work_ids=["w1", "w2"],
            roadmap=Roadmap(
                phases=[RoadmapPhase(id="M-001", title="P1", member_work_ids=["w1", "w2"])]
            ),
        )
    )

    spec = store.remove_members("demo", ["w1"])
    assert spec.member_work_ids == ["w2"]
    assert spec.roadmap.phases[0].member_work_ids == ["w2"]


def test_remove_members_unknown_project_raises(tmp_path):
    store = ProjectStore(base_path=str(tmp_path))
    try:
        store.remove_members("ghost", ["w1"])
    except KeyError:
        return
    raise AssertionError("expected KeyError for unknown project")


def test_delete_project(tmp_path):
    store = ProjectStore(base_path=str(tmp_path))
    store.save_project(_spec())
    assert store.delete_project("demo") is True
    assert store.load_project("demo") is None
    assert store.delete_project("demo") is False


# ── Concurrency: the inter-process lock prevents lost updates ──


def _add_member_worker(base_path: str, work_id: str) -> None:
    """Module-level worker so it is importable by spawned processes."""
    ProjectStore(base_path=base_path).add_members("demo", [work_id])


def test_concurrent_add_members_no_lost_update(tmp_path):
    """Parallel add_members from separate processes must preserve every member.

    Without the per-project flock, the read-modify-write on spec.json races:
    two processes read the same membership and the later write clobbers the
    earlier one's new member (a lost-update anomaly). The lock serialises the
    cycle so all members survive.
    """
    import multiprocessing as mp

    store = ProjectStore(base_path=str(tmp_path))
    store.save_project(_spec(member_work_ids=[]))

    work_ids = [f"w{i}" for i in range(20)]
    ctx = mp.get_context("fork")
    procs = [
        ctx.Process(target=_add_member_worker, args=(str(tmp_path), wid))
        for wid in work_ids
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)

    loaded = store.load_project("demo")
    assert loaded is not None
    assert sorted(loaded.member_work_ids) == sorted(work_ids)


def test_lock_file_not_listed_as_project(tmp_path):
    """The per-project lock file must not pollute list_projects()."""
    store = ProjectStore(base_path=str(tmp_path))
    store.save_project(_spec("demo", member_work_ids=[]))
    store.add_members("demo", ["w1"])  # creates demo.lock
    assert (tmp_path / "demo.lock").exists()
    assert store.list_projects() == ["demo"]
