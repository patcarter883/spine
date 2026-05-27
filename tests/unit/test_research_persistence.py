"""Prior research must survive phase re-entry, not just retry_count > 0.

The SPECIFY / PLAN state mappers used to seed prior research into the
sub-graph state only when ``retry_count > 0``. CriticalContractFailure
on a critic gate does not increment retry_count but DOES re-enter the
phase, so the research_manager started from scratch and re-issued the
same architectural questions every time. The mappers now always attempt
to load ``research_log.json`` and seed it when present.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from spine.workflow.compose import (
    _plan_state_mapper,
    _specify_state_mapper,
)


def _write_research_log(
    workspace_root: Path,
    work_id: str,
    phase: str,
    topics: list[str],
    findings: list[dict],
) -> None:
    log_dir = workspace_root / ".spine" / "artifacts" / work_id / phase
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "research_log.json").write_text(
        json.dumps({"topics": topics, "findings": findings}),
        encoding="utf-8",
    )


def _base_parent_state(workspace_root: Path, work_id: str) -> dict:
    return {
        "work_id": work_id,
        "work_type": "critical_reviewed_task",
        "description": "Add --verbose flag",
        "workspace_root": str(workspace_root),
        "feedback": [],
        "last_critic_review": None,
        "retry_count": {},
        "scratchpad": "",
    }


@pytest.mark.parametrize(
    "mapper,phase_name",
    [
        (_specify_state_mapper, "specify"),
        (_plan_state_mapper, "plan"),
    ],
)
def test_state_mapper_seeds_prior_research_with_retry_count_zero(
    tmp_path, mapper, phase_name
):
    work_id = "test123"
    topics = ["How is CLI logging configured?"]
    findings = [
        {
            "topic": "How is CLI logging configured?",
            "summary": "logging configured via spine.logging",
            "file_map": {"spine/logging.py": "main logger setup"},
            "patterns": [],
            "dependencies": [],
        }
    ]
    _write_research_log(tmp_path, work_id, phase_name, topics, findings)

    parent = _base_parent_state(tmp_path, work_id)
    out = mapper(parent, config=None)  # retry_count is empty dict → 0

    assert out["topics"] == topics
    assert out["findings"] == findings


@pytest.mark.parametrize(
    "mapper,phase_name",
    [
        (_specify_state_mapper, "specify"),
        (_plan_state_mapper, "plan"),
    ],
)
def test_state_mapper_omits_topics_when_log_missing(tmp_path, mapper, phase_name):
    parent = _base_parent_state(tmp_path, "fresh")
    out = mapper(parent, config=None)

    # When no research_log.json exists, topics/findings should not be seeded
    # (so the schema reducer default applies). The base_state_mapper does
    # NOT set them, so they must be absent.
    assert "topics" not in out
    assert "findings" not in out


def test_specify_state_mapper_preserves_retry_count_when_present(tmp_path):
    parent = _base_parent_state(tmp_path, "wid")
    parent["retry_count"] = {"specify": 2}
    out = _specify_state_mapper(parent, config=None)
    assert out["retry_count"] == 2
