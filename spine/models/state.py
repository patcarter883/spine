"""SPINE workflow state — LangGraph state schema and reducers."""

from __future__ import annotations

import operator
from typing import Annotated

from typing_extensions import TypedDict


def _merge_dicts(left: dict, right: dict) -> dict:
    """Merge two dicts, with right overwriting left for overlapping keys.

    Used as a LangGraph reducer for dict-typed state fields that should
    accumulate across phases (artifacts, retry_count, etc.).
    """
    merged = {**left, **right}
    return merged


class WorkflowState(TypedDict, total=False):
    """State schema for the SPINE workflow StateGraph.

    Fields with reducers accumulate across node executions:
    - artifacts: phase output documents merge by key
    - feedback: review feedback appends to a list
    - retry_count: per-phase retry counts merge by phase name
    """

    work_id: str
    work_type: str
    description: str
    current_phase: str
    phase_index: int
    retry_count: Annotated[dict, _merge_dicts]
    max_retries: int
    artifacts: Annotated[dict, _merge_dicts]
    feedback: Annotated[list, operator.add]
    status: str
    prompt_request: dict | None
    critic_reviewing: str  # Phase the current critic node is reviewing
    workspace_root: str  # Project root directory for deep agent backends
