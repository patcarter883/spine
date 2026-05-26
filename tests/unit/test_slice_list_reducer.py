"""Unit tests for ``_slice_list_reducer`` in spine.workflow.subgraph_state."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.workflow.subgraph_state import _slice_list_reducer


def _s(slice_id: str) -> dict:
    return {"id": slice_id, "title": f"slice {slice_id}"}


class TestSliceListReducer:
    def test_initial_seed_appends_list_update(self):
        result = _slice_list_reducer([], [_s("a"), _s("b")])
        assert [s["id"] for s in result] == ["a", "b"]

    def test_remove_directive(self):
        existing = [_s("a"), _s("b")]
        result = _slice_list_reducer(existing, {"remove": ["a"]})
        assert [s["id"] for s in result] == ["b"]

    def test_combined_add_and_remove_directive(self):
        existing = [_s("a")]
        result = _slice_list_reducer(existing, {"add": [_s("b")], "remove": ["a"]})
        assert [s["id"] for s in result] == ["b"]

    def test_sequential_remove_directives_simulate_parallel_sends(self):
        existing = [_s("a"), _s("b"), _s("c")]
        step1 = _slice_list_reducer(existing, {"remove": ["a"]})
        step2 = _slice_list_reducer(step1, {"remove": ["b"]})
        assert [s["id"] for s in step2] == ["c"]

        existing = [_s("a"), _s("b"), _s("c")]
        step1 = _slice_list_reducer(existing, {"remove": ["b"]})
        step2 = _slice_list_reducer(step1, {"remove": ["a"]})
        assert [s["id"] for s in step2] == ["c"]

    def test_remove_missing_id_is_noop(self):
        existing = [_s("a"), _s("b")]
        result = _slice_list_reducer(existing, {"remove": ["does-not-exist"]})
        assert [s["id"] for s in result] == ["a", "b"]

    def test_none_update_returns_copy_of_existing(self):
        existing = [_s("a")]
        result = _slice_list_reducer(existing, None)
        assert result == existing
        assert result is not existing

    def test_none_existing_treated_as_empty(self):
        assert _slice_list_reducer(None, [_s("x")]) == [_s("x")]
        assert _slice_list_reducer(None, {"add": [_s("y")]}) == [_s("y")]

    def test_list_update_appends_does_not_replace(self):
        existing = [_s("a")]
        result = _slice_list_reducer(existing, [_s("b"), _s("c")])
        assert [s["id"] for s in result] == ["a", "b", "c"]
