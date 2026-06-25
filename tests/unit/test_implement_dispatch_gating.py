"""IMPLEMENT dispatch: DAG dependency gating (A) and the dispatch backstop (C).

The flat ``pending_slices`` seed (compose._implement_state_mapper) discards the
plan's dependency DAG, so ``_route_slices`` must re-impose ordering itself —
otherwise dependent slices (and slices sharing a target file) race and the loop
explodes (trace 019efd92: 3 same-file slices → ~687 executions / 1.33M tokens).
"""

from __future__ import annotations

from spine.workflow.subgraphs.implement_subgraph import _ready_slices, _route_slices


def _sl(sid: str, deps: list[str] | None = None, **extra: object) -> dict:
    d: dict = {"id": sid, "target_files": ["x.py"]}
    if deps is not None:
        d["dependencies"] = deps
    d.update(extra)
    return d


def _state(**kw: object) -> dict:
    base: dict = {
        "work_id": "t",
        "workspace_root": ".",
        "pending_slices": [],
        "failed_slices": [],
        "completed_slices": [],
    }
    base.update(kw)
    return base


# ── A: _ready_slices ────────────────────────────────────────────────────


def test_no_deps_is_ready() -> None:
    p = [_sl("a"), _sl("b")]
    assert _ready_slices(p, []) == p


def test_unmet_dep_is_not_ready() -> None:
    assert _ready_slices([_sl("b", deps=["a"])], []) == []


def test_met_dep_is_ready() -> None:
    p = [_sl("b", deps=["a"])]
    assert _ready_slices(p, [{"id": "a"}]) == p


def test_dep_satisfied_by_parent_id() -> None:
    # 'a' was split into a sub-slice chain; its completed entry references the
    # parent via _parent_slice_id, which must satisfy a dependency named 'a'.
    p = [_sl("b", deps=["a"])]
    completed = [{"id": "a-file1", "_parent_slice_id": "a"}]
    assert _ready_slices(p, completed) == p


def test_dep_still_pending_blocks_dependent() -> None:
    # 'a' has one sub-slice completed but another still pending → 'a' is NOT done,
    # so 'b' must wait; the still-pending sub-slice (no deps) is itself ready.
    p = [_sl("a-file2", _parent_slice_id="a"), _sl("b", deps=["a"])]
    completed = [{"id": "a-file1", "_parent_slice_id": "a"}]
    ready_ids = {s["id"] for s in _ready_slices(p, completed)}
    assert "a-file2" in ready_ids
    assert "b" not in ready_ids


# ── C / routing: _route_slices ──────────────────────────────────────────


def test_empty_routes_to_synthesis() -> None:
    assert _route_slices(_state()) == "synthesize_implementation"


def test_dispatch_cap_aborts_to_synthesis() -> None:
    # cap defaults to 100; once reached the loop must stop fanning out.
    st = _state(pending_slices=[_sl("a")], slice_dispatch_count=100)
    assert _route_slices(st) == "synthesize_implementation"


def test_gating_dispatches_only_ready() -> None:
    st = _state(pending_slices=[_sl("a"), _sl("b", deps=["a"])])
    out = _route_slices(st)
    assert isinstance(out, list)
    dispatched = {s.arg["active_slice"]["id"] for s in out}
    assert dispatched == {"a"}  # b is gated behind a


def test_deadlock_guard_dispatches_all() -> None:
    # 'b' depends on a dangling id and nothing is decomposing → dispatch anyway
    # so the phase terminates instead of stalling forever.
    st = _state(pending_slices=[_sl("b", deps=["does-not-exist"])])
    out = _route_slices(st)
    assert isinstance(out, list) and len(out) == 1
