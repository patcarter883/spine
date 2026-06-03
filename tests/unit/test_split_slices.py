"""Unit tests for single-file slice decomposition in the IMPLEMENT subgraph.

Covers the ``split_slices`` node (proactive PER_FILE decomposition), the
sequential per-parent handoff in ``slice_implementer``, and the reducer
composition that makes the dispatch loop advance one file at a time.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from spine.workflow.subgraph_state import _slice_list_reducer
from spine.workflow.subgraphs.implement_subgraph import (
    _build_subslice_chain,
    _slice_implementer_node,
    _split_slices_node,
    _subslice_context,
)


def _base_state() -> dict:
    return {
        "phase": "implement",
        "work_id": "test-work",
        "work_type": "feature",
        "workspace_root": "/tmp/test",
        "plan_path": ".spine/artifacts/test-work/plan",
    }


_PARENT = {
    "id": "add-auth",
    "title": "Add auth",
    "target_files": ["src/models.py", "src/login.py", "tests/test_auth.py"],
    "acceptance_criteria": ["pytest passes"],
}


def _subs(*files: str) -> list[dict]:
    return [
        {
            "id": f"add-auth::{i}-{f.rsplit('/', 1)[-1]}",
            "title": f"Add auth — {f}",
            "description": f"work on {f}",
            "target_files": [f],
            "acceptance_criteria": ["pytest passes"],
        }
        for i, f in enumerate(files, start=1)
    ]


# ── _build_subslice_chain ────────────────────────────────────────────────


def test_build_subslice_chain_sets_metadata_and_queue():
    subs = _subs("src/models.py", "src/login.py", "tests/test_auth.py")
    head = _build_subslice_chain(_PARENT, subs)

    assert head["target_files"] == ["src/models.py"]
    assert head["_parent_slice_id"] == "add-auth"
    assert head["_file_index"] == 1
    assert head["_file_total"] == 3
    assert head["_validate_slice_criteria"] is False
    assert head["_all_files"] == _PARENT["target_files"]

    queue = head["_sibling_queue"]
    assert [q["target_files"][0] for q in queue] == ["src/login.py", "tests/test_auth.py"]
    # Only the very last file validates slice-level criteria.
    assert queue[-1]["_validate_slice_criteria"] is True
    assert queue[0]["_validate_slice_criteria"] is False


# ── _split_slices_node ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_split_node_replaces_multi_file_slice_with_head():
    state = {**_base_state(), "pending_slices": [dict(_PARENT)]}
    fake = AsyncMock(return_value=_subs("src/models.py", "src/login.py", "tests/test_auth.py"))
    with patch("spine.agents.decomposer.run_decomposer", fake):
        out = await _split_slices_node(state, None)

    assert out["pending_slices"]["remove"] == ["add-auth"]
    adds = out["pending_slices"]["add"]
    assert len(adds) == 1
    assert adds[0]["target_files"] == ["src/models.py"]
    assert len(adds[0]["_sibling_queue"]) == 2


@pytest.mark.asyncio
async def test_split_node_passes_single_file_slice_through():
    state = {**_base_state(), "pending_slices": [{"id": "solo", "target_files": ["only.py"]}]}
    # No multi-file slices → no decomposition, no state change.
    out = await _split_slices_node(state, None)
    assert out == {}


@pytest.mark.asyncio
async def test_split_node_keeps_slice_whole_on_decomposer_failure():
    state = {**_base_state(), "pending_slices": [dict(_PARENT)]}
    fake = AsyncMock(side_effect=RuntimeError("model down"))
    with patch("spine.agents.decomposer.run_decomposer", fake):
        out = await _split_slices_node(state, None)
    # Graceful degradation: nothing removed, slice stays whole in pending.
    assert out == {}


# ── _subslice_context ────────────────────────────────────────────────────


def test_subslice_context_empty_for_ordinary_slice():
    assert _subslice_context({"id": "x", "target_files": ["a.py"]}) == ""


def test_subslice_context_lists_siblings_and_pending_note():
    sub = {
        "_parent_slice_id": "add-auth",
        "_all_files": ["src/models.py", "src/login.py"],
        "target_files": ["src/models.py"],
        "_file_index": 1,
        "_file_total": 2,
        "_validate_slice_criteria": False,
    }
    text = _subslice_context(sub)
    assert "only this one file: src/models.py" in text
    assert "src/login.py" in text  # sibling listed for context
    assert "do NOT expect" in text.lower() or "do not expect" in text.lower()


def test_subslice_context_last_file_validates():
    sub = {
        "_parent_slice_id": "add-auth",
        "_all_files": ["src/models.py", "tests/test_auth.py"],
        "target_files": ["tests/test_auth.py"],
        "_file_index": 2,
        "_file_total": 2,
        "_validate_slice_criteria": True,
    }
    text = _subslice_context(sub)
    assert "LAST file" in text


# ── Sequential handoff in _slice_implementer_node ────────────────────────


def _patch_implementer(monkeypatch, status: str, files_modified=None):
    """Stub the subagent machinery so the node returns a fixed slice result."""
    monkeypatch.setattr(
        "spine.agents.subagents.build_subagent_spec",
        lambda **kw: {"system_prompt": "x", "tools": [], "response_format": None},
    )
    monkeypatch.setattr(
        "spine.agents.factory.build_phase_agent",
        lambda **kw: MagicMock(),
    )

    async def _fake_ainvoke(agent, payload, **kw):
        return {
            "structured_response": {
                "status": status,
                "files_modified": files_modified or [],
                "files_created": [],
                "test_results": "ok",
                "issues": [],
            }
        }

    monkeypatch.setattr(
        "spine.workflow.subgraphs.implement_subgraph.ainvoke_with_retry", _fake_ainvoke
    )


@pytest.mark.asyncio
async def test_success_promotes_next_sibling(monkeypatch):
    _patch_implementer(monkeypatch, "implemented", files_modified=["src/models.py"])
    head = _build_subslice_chain(_PARENT, _subs("src/models.py", "src/login.py", "tests/test_auth.py"))
    state = {**_base_state(), "active_slice": head}

    out = await _slice_implementer_node(state, None)

    assert out["pending_slices"]["remove"] == [head["id"]]
    promoted = out["pending_slices"]["add"]
    assert len(promoted) == 1
    assert promoted[0]["target_files"] == ["src/login.py"]
    # The promoted file carries the remaining queue (the last file).
    assert [q["target_files"][0] for q in promoted[0]["_sibling_queue"]] == ["tests/test_auth.py"]
    # completed_slices entry is tidied of the queue.
    assert "_sibling_queue" not in out["completed_slices"]["add"][0]


@pytest.mark.asyncio
async def test_last_file_success_adds_nothing(monkeypatch):
    _patch_implementer(monkeypatch, "implemented", files_modified=["tests/test_auth.py"])
    last = {
        **_subs("tests/test_auth.py")[0],
        "_parent_slice_id": "add-auth",
        "_sibling_queue": [],
        "_validate_slice_criteria": True,
    }
    state = {**_base_state(), "active_slice": last}

    out = await _slice_implementer_node(state, None)
    assert "add" not in out["pending_slices"]
    assert out["pending_slices"]["remove"] == [last["id"]]


@pytest.mark.asyncio
async def test_failure_drops_queue_and_notes_skipped(monkeypatch):
    _patch_implementer(monkeypatch, "blocked")
    head = _build_subslice_chain(_PARENT, _subs("src/models.py", "src/login.py", "tests/test_auth.py"))
    state = {**_base_state(), "active_slice": head}

    out = await _slice_implementer_node(state, None)

    tagged = out["failed_slices"]["add"][0]
    assert "_sibling_queue" not in tagged
    assert any("skipped after failure" in i for i in tagged["issues"])
    # The two later files are named in the skip note.
    note = next(i for i in tagged["issues"] if "skipped after failure" in i)
    assert "add-auth::2-login.py" in note and "add-auth::3-test_auth.py" in note


# ── Reducer composition drives the loop forward ──────────────────────────


def test_reducer_advances_one_file_at_a_time():
    chain = _subs("a.py", "b.py")
    head = _build_subslice_chain(_PARENT, chain)
    pending = _slice_list_reducer([], [head])  # initial seed
    assert [s["id"] for s in pending] == [head["id"]]

    # Head completes → remove it, add the next sibling.
    nxt = {**head["_sibling_queue"][0], "_sibling_queue": head["_sibling_queue"][1:]}
    pending = _slice_list_reducer(pending, {"remove": [head["id"]], "add": [nxt]})
    assert [s["id"] for s in pending] == [nxt["id"]]

    # Last sibling completes → remove it, nothing added. Loop drains.
    pending = _slice_list_reducer(pending, {"remove": [nxt["id"]]})
    assert pending == []
