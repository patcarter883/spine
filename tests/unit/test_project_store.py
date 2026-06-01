"""Tests for ProjectStore — atomic persistence + idempotent membership."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.models.types import ProjectSpec, RequirementRef
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
