"""Tests for per-slice convergence in the verify→gap_plan→implement loop.

Run 019f20e0: with gap feedback flowing, one slice reached 1 remaining gap
while both slices were re-implemented AND re-verified from scratch every
cycle — wasting budget and (trace 019f2040) risking regression of passing
work. Two filters fix that:

* `_implement_state_mapper` — on a gap rework, pending_slices excludes slices
  whose last verdict was VERIFIED.
* `seed_prior_results` + `_verify_router` — a VERIFIED slice whose targets
  were not rewritten keeps its verdict without a re-verification; one whose
  files WERE touched is re-verified (the regression case skipping must not
  mask).
"""

from __future__ import annotations

import pytest

from spine.workflow.compose import _implement_state_mapper, _verify_state_mapper
from spine.workflow.subgraphs.verify_subgraph import (
    _seed_prior_results_node,
    _verify_router,
)

WAVES = [
    [
        {
            "id": "api-slice",
            "target_files": ["spine/ui_api/api.py"],
            "acceptance_criteria": ["x"],
        }
    ],
    [
        {
            "id": "view-slice",
            "target_files": ["spine/ui/_pages/config_view.py"],
            "acceptance_criteria": ["y"],
        }
    ],
]


def _findings(**verdicts: str) -> list[dict]:
    return [
        {"slice_name": sid, "verdict": v, "gaps": []}
        for sid, v in verdicts.items()
    ]


# ── implement-side filter ─────────────────────────────────────────────────────


def test_first_implement_dispatches_all_slices() -> None:
    mapped = _implement_state_mapper(
        {"work_id": "w", "execution_waves": WAVES, "verify_attempts": 0,
         "retry_count": {}},
        None,
    )
    assert [s["id"] for s in mapped["pending_slices"]] == ["api-slice", "view-slice"]


def test_gap_rework_drops_verified_slices() -> None:
    mapped = _implement_state_mapper(
        {
            "work_id": "w",
            "execution_waves": WAVES,
            "verify_attempts": 1,
            "retry_count": {},
            "verification_findings": _findings(
                **{"api-slice": "NOT_VERIFIED", "view-slice": "VERIFIED"}
            ),
        },
        None,
    )
    assert [s["id"] for s in mapped["pending_slices"]] == ["api-slice"]


def test_gap_rework_keeps_all_when_every_slice_verified() -> None:
    # Defensive: an empty IMPLEMENT dispatch is a contract violation, so a
    # (should-not-happen) rework where everything is VERIFIED keeps the list.
    mapped = _implement_state_mapper(
        {
            "work_id": "w",
            "execution_waves": WAVES,
            "verify_attempts": 1,
            "retry_count": {},
            "verification_findings": _findings(
                **{"api-slice": "VERIFIED", "view-slice": "VERIFIED"}
            ),
        },
        None,
    )
    assert len(mapped["pending_slices"]) == 2


def test_gap_rework_keeps_slices_with_no_verdict() -> None:
    mapped = _implement_state_mapper(
        {
            "work_id": "w",
            "execution_waves": WAVES,
            "verify_attempts": 1,
            "retry_count": {},
            "verification_findings": _findings(**{"view-slice": "VERIFIED"}),
        },
        None,
    )
    # api-slice has no verdict → unknown ≠ passed → stays in.
    assert [s["id"] for s in mapped["pending_slices"]] == ["api-slice"]


# ── verify-side carry-forward ─────────────────────────────────────────────────


def test_verify_mapper_forwards_prior_findings_only_on_rework() -> None:
    parent = {
        "work_id": "w",
        "retry_count": {},
        "verification_findings": _findings(**{"api-slice": "VERIFIED"}),
        "files_written": ["a.py"],
        "execution_waves": WAVES,
    }
    first = _verify_state_mapper({**parent, "verify_attempts": 0}, None)
    assert first["prior_verification_findings"] == []
    rework = _verify_state_mapper({**parent, "verify_attempts": 1}, None)
    assert rework["prior_verification_findings"] == parent["verification_findings"]
    assert rework["files_written"] == ["a.py"]


@pytest.mark.asyncio
async def test_seed_carries_forward_untouched_verified_slice() -> None:
    out = await _seed_prior_results_node(
        {
            "execution_waves": WAVES,
            "prior_verification_findings": _findings(
                **{"api-slice": "VERIFIED", "view-slice": "NOT_VERIFIED"}
            ),
            "files_written": ["spine/ui/_pages/config_view.py"],
        }
    )
    assert out["reverify_skipped_ids"] == ["api-slice"]
    seeded = out["verification_results"]
    assert len(seeded) == 1
    assert seeded[0]["slice_name"] == "api-slice"
    assert seeded[0]["reused_from_prior_cycle"] is True


@pytest.mark.asyncio
async def test_seed_reverifies_verified_slice_whose_files_were_touched() -> None:
    out = await _seed_prior_results_node(
        {
            "execution_waves": WAVES,
            "prior_verification_findings": _findings(
                **{"api-slice": "VERIFIED", "view-slice": "NOT_VERIFIED"}
            ),
            # The rework rewrote the verified slice's own target file — the
            # regression case (trace 019f2040) that must be re-verified.
            "files_written": ["spine/ui_api/api.py"],
        }
    )
    assert out == {}


@pytest.mark.asyncio
async def test_seed_noop_on_first_cycle() -> None:
    out = await _seed_prior_results_node(
        {"execution_waves": WAVES, "prior_verification_findings": []}
    )
    assert out == {}


def test_router_excludes_skipped_and_dispatches_rest() -> None:
    sends = _verify_router(
        {
            "execution_waves": WAVES,
            "reverify_skipped_ids": ["api-slice"],
            "work_id": "w",
        }
    )
    assert isinstance(sends, list)
    dispatched = [s.arg["slice"]["id"] for s in sends]
    assert dispatched == ["view-slice"]


def test_router_synthesizes_when_all_carried_forward() -> None:
    result = _verify_router(
        {
            "execution_waves": WAVES,
            "reverify_skipped_ids": ["api-slice", "view-slice"],
            "work_id": "w",
        }
    )
    assert result == "synthesize_verification"


# ── cross-slice reattribution (probe 21 / run ad237d70) ───────────────────────


def _write_gap_plan(tmp_path, items):
    import json as _json
    from pathlib import Path as _Path

    d = _Path(tmp_path) / ".spine" / "artifacts" / "w" / "gap_plan"
    d.mkdir(parents=True, exist_ok=True)
    (d / "gap_plan.json").write_text(
        _json.dumps({"remediation_items": items}), encoding="utf-8"
    )


def test_gap_rework_reopens_verified_slice_implicated_by_id(tmp_path) -> None:
    """A VERIFIED slice the gap plan names by slice_id is re-dispatched."""
    _write_gap_plan(tmp_path, [{"slice_id": "view-slice", "fixes": []}])
    mapped = _implement_state_mapper(
        {
            "work_id": "w",
            "workspace_root": str(tmp_path),
            "execution_waves": WAVES,
            "verify_attempts": 1,
            "retry_count": {},
            "verification_findings": _findings(
                **{"api-slice": "NOT_VERIFIED", "view-slice": "VERIFIED"}
            ),
        },
        None,
    )
    assert [s["id"] for s in mapped["pending_slices"]] == ["api-slice", "view-slice"]


def test_gap_rework_reopens_verified_slice_implicated_by_file(tmp_path) -> None:
    """A VERIFIED slice is re-opened when a fix names ITS file even though the
    remediation item is keyed to the slice where the failure surfaced."""
    _write_gap_plan(tmp_path, [{
        "slice_id": "api-slice",
        "fixes": [{"file_path": "spine/ui/_pages/config_view.py",
                   "issue_description": "table name mismatch"}],
    }])
    mapped = _implement_state_mapper(
        {
            "work_id": "w",
            "workspace_root": str(tmp_path),
            "execution_waves": WAVES,
            "verify_attempts": 1,
            "retry_count": {},
            "verification_findings": _findings(
                **{"api-slice": "NOT_VERIFIED", "view-slice": "VERIFIED"}
            ),
        },
        None,
    )
    assert [s["id"] for s in mapped["pending_slices"]] == ["api-slice", "view-slice"]


def test_gap_rework_without_gap_plan_filters_as_before(tmp_path) -> None:
    """No gap_plan.json on disk → the convergence filter is unchanged."""
    mapped = _implement_state_mapper(
        {
            "work_id": "w",
            "workspace_root": str(tmp_path),
            "execution_waves": WAVES,
            "verify_attempts": 1,
            "retry_count": {},
            "verification_findings": _findings(
                **{"api-slice": "NOT_VERIFIED", "view-slice": "VERIFIED"}
            ),
        },
        None,
    )
    assert [s["id"] for s in mapped["pending_slices"]] == ["api-slice"]
