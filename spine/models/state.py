"""SPINE workflow state — LangGraph state schema and reducers."""

from __future__ import annotations

import operator
from typing import Annotated

from typing_extensions import TypedDict


def _merge_dicts(left: dict, right: dict) -> dict:
    """Merge two dicts, with right overwriting left for overlapping keys.

    Used as a LangGraph reducer for dict-typed state fields that should
    accumulate across phases (retry_count, etc.).
    """
    merged = {**left, **right}
    return merged


def _merge_artifacts(left: dict, right: dict) -> dict:
    """Deep-merge artifacts dicts so per-file entries aren't lost.

    Artifacts have a two-level structure: ``{phase: {filename: content}}``.
    A shallow merge would replace the entire inner dict for a phase key,
    destroying any files that weren't re-produced (e.g. individual slice
    files from the tasks phase).  This reducer merges at the file level
    instead, so returning ``{"tasks": {"tasks.md": summary}}`` only
    overwrites ``tasks.md`` — any ``slice-1.md``, ``slice-2.md``, etc.
    from a prior run are preserved.
    """
    merged = {**left}
    for phase_key, phase_artifacts in right.items():
        if not phase_artifacts or not isinstance(phase_artifacts, dict):
            # New value is empty or not a dict — overwrite the key
            merged[phase_key] = phase_artifacts
        elif phase_key in merged and isinstance(merged.get(phase_key), dict):
            # Both sides are dicts — merge at the file level
            merged[phase_key] = {**merged[phase_key], **phase_artifacts}
        else:
            # Left side missing or not a dict — use right's value
            merged[phase_key] = phase_artifacts
    return merged


class PhaseResult(TypedDict, total=False):
    """Lightweight summary of a phase subgraph's output.

    Stored in ``WorkflowState.phase_results`` so the parent graph can
    track progress without carrying full artifact content.
    """

    phase: str
    status: str  # "success" | "needs_review" | "error"
    artifact_count: int
    artifact_names: list[str]
    error: str | None


class WorkflowState(TypedDict, total=False):
    """State schema for the SPINE workflow StateGraph.

    Fields with reducers accumulate across node executions:
    - artifacts: phase output documents merge by key
    - feedback: review feedback appends to a list
    - retry_count: per-phase retry counts merge by phase name
    - phase_results: per-phase summary dicts merge by key
    """

    work_id: str
    work_type: str
    description: str  # Original work description — only used by SPECIFY (first spec
                      # phase) and TASKS (first quick-workflow phase). Downstream
                      # phases work from artifacts on disk, not the raw description.
    current_phase: str
    phase_index: int
    retry_count: Annotated[dict, _merge_dicts]
    max_retries: int
    artifacts: Annotated[dict, _merge_artifacts]
    feedback: Annotated[list, operator.add]
    status: str
    prompt_request: dict | None
    critic_reviewing: str  # Phase the current critic node is reviewing
    workspace_root: str  # Project root directory for deep agent backends
    phase_results: Annotated[dict, _merge_dicts]  # phase → PhaseResult
    needs_review_phase: str | None  # Which phase triggered human review
    plan_id: str | None  # Optional reference to an approved planning work item.
    # For execution work items spawned from a plan: references the planning work
    # item that spawned this item. None for standalone quick/critical_quick items.
    spawned_work_ids: Annotated[list[str], operator.add]  # IDs of execution work
    # items spawned from this planning item. Empty for standalone execution items
    # and for planning items that haven't been approved yet.
