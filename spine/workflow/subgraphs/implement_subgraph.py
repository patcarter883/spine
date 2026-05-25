"""IMPLEMENT phase as a LangGraph subgraph with Send API dispatch.

The subgraph uses the same manager/router/call/aggregate pattern as the
exploration subgraph, replacing the old ``eval`` + ``tools.task()`` +
``Promise.allSettled`` approach with native LangGraph ``Send`` API
parallel dispatch.

Nodes:
- ``implement_router``: conditional edge — reads ``execution_waves`` from
  state, returns ``[Send("run_slice_implementer", ...)]`` or ``"synthesize_implementation"``
- ``run_slice_implementer``: builds a ``slice-implementer`` subagent per
  slice and invokes it. Runs in parallel via Send API.
- ``aggregate_implementation``: deterministic fan-in point after all
  parallel slice-implementer nodes complete.
- ``synthesize_implementation``: writes ``implementation.md`` and
  ``implementation.json`` from accumulated slice results.
- ``save_artifacts``: scans disk, materializes to state.

Edges::

    START → implement_router
    implement_router → Send("run_slice_implementer", {slice}) × N  OR  → synthesize_implementation
    run_slice_implementer → aggregate_implementation
    aggregate_implementation → synthesize_implementation → save_artifacts → END
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from spine.agents.artifacts import (
    artifact_path,
    materialize_phase_artifacts,
    scan_artifact_dir,
)
from spine.agents.retry import ainvoke_with_retry
from spine.exceptions import CriticalContractFailure
from spine.models.enums import PhaseName
from spine.workflow.subgraph_state import ImplementSubgraphState

logger = logging.getLogger(__name__)
_MAX_ARTIFACT_STATE_CHARS = 500


# ── Router: START → run_slice_implementer (Send) or synthesize ──────────


def _implement_router(
    state: ImplementSubgraphState,
) -> list[Send] | Literal["synthesize_implementation"]:
    """Fan-out to slice-implementer nodes via Send API.

    Reads ``execution_waves`` from state, flattens all waves into a single
    dispatch list (the scheduler guarantees dependency ordering within
    waves, and wave ordering is satisfied by topological sort).

    Raises ``CriticalContractFailure`` if ``execution_waves`` is missing
    or empty — this is a structural invariant violation that means PLAN
    did not produce the required structured data transfer.
    """
    execution_waves = state.get("execution_waves")

    if not execution_waves:
        raise CriticalContractFailure(
            phase="implement",
            reason="execution_waves is missing or empty in state — "
                   "the PLAN phase did not produce structured data transfer. "
                   "The prerequisite gate should have caught this before "
                   "IMPLEMENT ran; check artifact_gate.py.",
        )

    all_slices: list[dict] = []
    for wave in execution_waves:
        if isinstance(wave, list):
            for sl in wave:
                if isinstance(sl, dict) and sl.get("id"):
                    all_slices.append(sl)

    if not all_slices:
        raise CriticalContractFailure(
            phase="implement",
            reason="execution_waves is present but contains zero valid "
                   "slice dicts with 'id' fields. The PLAN phase produced "
                   "malformed structured data.",
        )

    logger.info(
        "IMPLEMENT router: dispatching %d slice-implementer(s): %s",
        len(all_slices),
        [s.get("id", "?") for s in all_slices],
    )

    base_state = {
        "phase": state.get("phase", "implement"),
        "work_id": state.get("work_id", "unknown"),
        "work_type": state.get("work_type", ""),
        "workspace_root": state.get("workspace_root", "."),
    }
    return [
        Send("run_slice_implementer", {**base_state, "slice": s})
        for s in all_slices
    ]


# ── Node: run_slice_implementer ─────────────────────────────────────────


async def _run_slice_implementer_node(
    state: ImplementSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Run a slice-implementer subagent for one feature slice.

    The slice is injected into state by the Send API via
    ``Send("run_slice_implementer", {"slice": {...}})`` — it arrives as
    a state key, not a keyword argument.  The state merger injects
    ``slice`` alongside the normal subgraph state fields.
    """
    from spine.agents.factory import build_phase_agent
    from spine.agents.subagents import build_subagent_spec

    work_id = state.get("work_id", "unknown")
    slice_data: dict = state.get("slice", {})
    slice_id = slice_data.get("id", "unknown")

    logger.info(
        "[%s] Slice-implementer node: slice=%r (title=%r)",
        work_id,
        slice_id,
        slice_data.get("title", ""),
    )

    try:
        subagent_spec = build_subagent_spec(
            name="slice-implementer",
            phase=PhaseName.IMPLEMENT,
            state=state,
            config=config,
        )

        extra_tools = list(subagent_spec.get("tools", []))
        agent = build_phase_agent(
            state=state,
            config=config,
            phase=PhaseName.IMPLEMENT,
            system_prompt=subagent_spec["system_prompt"],
            is_subagent=True,
            extra_tools=extra_tools,
            response_format=subagent_spec.get("response_format"),
            skip_filesystem_middleware=True,
        )

        slice_json = json.dumps(slice_data, indent=2, ensure_ascii=False)
        prompt = (
            f"## Implement Slice: {slice_id}\n\n"
            f"The full slice definition is in the JSON below. "
            f"Read it carefully — it specifies target_files, "
            f"execution_requirements, dependencies, and acceptance_criteria. "
            f"Make only the changes described in the slice.\n\n"
            f"```json\n{slice_json}\n```"
        )

        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name="implement-slice",
            work_id=work_id,
        )

        slice_result = _extract_slice_result(result, slice_id)

    except Exception as e:
        logger.error(
            "[%s] Slice-implementer failed for %r: %s",
            work_id,
            slice_id,
            e,
            exc_info=True,
        )
        slice_result = {
            "slice_name": slice_id,
            "status": "blocked",
            "files_modified": [],
            "files_created": [],
            "test_results": f"Subagent error: {e}",
            "issues": [str(e)],
        }

    return {"slice_results": [slice_result]}


def _normalize_status(status: str) -> str:
    """Normalize status to a valid value: implemented, partial, or blocked."""
    if not isinstance(status, str):
        return "implemented"
    status_lower = status.lower().strip()
    if status_lower in ("implemented", "partial", "blocked"):
        return status_lower
    if status_lower in ("in_progress", "in", "running", "done"):
        return "implemented"
    if status_lower in ("failed", "error", "not_implemented"):
        return "blocked"
    return "implemented"


def _extract_slice_result(result: dict, slice_id: str) -> dict:
    """Extract a SliceResult dict from an agent result.

    If the agent returned structured output via ``response_format``,
    it'll be in the ``structured_response`` key.  Falls back to the
    last assistant message content.

    The ``slice_name`` field is overridden with the actual slice_id
    from the router to guarantee consistency — the subagent may not
    include the slice name in its response.
    """
    structured = result.get("structured_response")
    if structured:
        if isinstance(structured, dict):
            structured["slice_name"] = slice_id
            if "status" in structured:
                structured["status"] = _normalize_status(structured["status"])
            return structured
        if hasattr(structured, "model_dump"):
            d = structured.model_dump()
            d["slice_name"] = slice_id
            if "status" in d:
                d["status"] = _normalize_status(d["status"])
            return d

    messages = result.get("messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content.strip():
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict):
                    parsed["slice_name"] = slice_id
                    if "status" in parsed:
                        parsed["status"] = _normalize_status(parsed["status"])
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
            return {
                "slice_name": slice_id,
                "status": "implemented",
                "files_modified": [],
                "files_created": [],
                "test_results": "",
                "issues": [],
            }

    return {
        "slice_name": slice_id,
        "status": "implemented",
        "files_modified": [],
        "files_created": [],
        "test_results": "(no output from subagent)",
        "issues": ["Subagent produced no output"],
    }


# ── Node: aggregate_implementation ──────────────────────────────────────


async def _aggregate_implementation_node(
    state: ImplementSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Fan-in point after all parallel slice-implementer nodes complete.

    Results are already accumulated via ``operator.add`` on the
    ``slice_results`` field — no manual merging needed. This node
    exists as a routing checkpoint.
    """
    results = state.get("slice_results", [])
    statuses = [r.get("status", "?") for r in results]
    logger.info(
        "IMPLEMENT aggregate: %d slice result(s) — statuses: %s",
        len(results),
        statuses,
    )
    return {}


# ── Node: synthesize_implementation ──────────────────────────────────────


async def _synthesize_implementation_node(
    state: ImplementSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Write implementation.md and implementation.json from accumulated results.

    Uses the existing ``WriteImplementationReportTool`` logic (via a
    utility function) to produce both the human-readable markdown and
    the structured JSON artifact.
    """
    from spine.agents.implement_tools import write_implementation_files

    work_id = state.get("work_id", "unknown")
    workspace_root = state.get("workspace_root", ".")
    slice_results = state.get("slice_results", [])

    if not slice_results:
        logger.warning("[%s] IMPLEMENT synthesize: zero slice results", work_id)
        return {
            "agent_response": "",
            "artifacts_output": {},
            "phase_status": "needs_review",
            "slices_dispatched": False,
            "implementation_files_written": False,
        }

    impl_dir = artifact_path(work_id, PhaseName.IMPLEMENT.value)
    summary = _build_implementation_summary(slice_results)

    try:
        write_implementation_files(slice_results, summary, workspace_root, impl_dir)
    except Exception as e:
        logger.error(
            "[%s] IMPLEMENT synthesize: failed to write artifacts: %s",
            work_id,
            e,
        )
        return {
            "agent_response": summary,
            "artifacts_output": {},
            "phase_status": "error",
            "slices_dispatched": True,
            "implementation_files_written": False,
        }

    logger.info(
        "[%s] IMPLEMENT synthesize: wrote %d slice results to %s/",
        work_id,
        len(slice_results),
        impl_dir,
    )

    return {
        "agent_response": summary,
        "artifacts_output": {"implementation.md": summary[:_MAX_ARTIFACT_STATE_CHARS]},
        "phase_status": "success",
        "slices_dispatched": True,
        "implementation_files_written": True,
    }


def _build_implementation_summary(slice_results: list[dict]) -> str:
    """Build a human-readable implementation summary from slice results."""
    total = len(slice_results)
    statuses: dict[str, int] = {}
    for r in slice_results:
        s = r.get("status", "unknown")
        statuses[s] = statuses.get(s, 0) + 1

    parts = [
        f"Implemented {total} feature slice(s).",
    ]
    for status, count in sorted(statuses.items()):
        parts.append(f"- {status}: {count}")

    implemented = statuses.get("implemented", 0)
    blocked = statuses.get("blocked", 0)
    if implemented == total:
        parts.append("All slices implemented successfully.")
    elif blocked > 0:
        parts.append(f"{blocked} slice(s) blocked — see implementation.md for details.")

    return "\n".join(parts)


# ── Node: save_artifacts ────────────────────────────────────────────────


async def _save_implement_artifacts(
    state: ImplementSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Save artifacts from the implement phase to disk and state."""
    workspace_root = state.get("workspace_root", ".")
    work_id = state.get("work_id", "unknown")
    agent_response = state.get("agent_response", "")
    existing_phase_status = state.get("phase_status", "")

    if existing_phase_status in ("error", "needs_review"):
        return {
            "artifacts_output": {},
            "phase_status": existing_phase_status,
        }

    disk_artifacts = scan_artifact_dir(
        workspace_root,
        work_id,
        PhaseName.IMPLEMENT.value,
        max_preview_chars=_MAX_ARTIFACT_STATE_CHARS,
    )

    if not disk_artifacts and agent_response.strip():
        materialize_phase_artifacts(
            PhaseName.IMPLEMENT.value,
            {"implementation.md": agent_response},
            workspace_root,
            work_id=work_id,
        )
        disk_artifacts = {"implementation.md": agent_response[:_MAX_ARTIFACT_STATE_CHARS]}

    return {
        "artifacts_output": disk_artifacts,
        "phase_status": "success" if disk_artifacts else "needs_review",
    }


# ── Builder ──────────────────────────────────────────────────────────────


def build_implement_subgraph() -> Any:
    """Build the IMPLEMENT phase subgraph with Send API dispatch.

    Returns a compiled StateGraph with five nodes:
    1. implement_router — conditional edge dispatching Send objects
    2. run_slice_implementer — per-slice subagent invocation (parallel)
    3. aggregate_implementation — fan-in checkpoint
    4. synthesize_implementation — writes implementation artifacts
    5. save_artifacts — scans disk, materializes to state
    """
    builder = StateGraph(ImplementSubgraphState)

    builder.add_node("run_slice_implementer", _run_slice_implementer_node)
    builder.add_node("aggregate_implementation", _aggregate_implementation_node)
    builder.add_node("synthesize_implementation", _synthesize_implementation_node)
    builder.add_node("save_artifacts", _save_implement_artifacts)

    builder.add_conditional_edges(
        START,
        _implement_router,
        {
            "run_slice_implementer": "run_slice_implementer",
            "synthesize_implementation": "synthesize_implementation",
        },
    )

    builder.add_edge("run_slice_implementer", "aggregate_implementation")
    builder.add_edge("aggregate_implementation", "synthesize_implementation")
    builder.add_edge("synthesize_implementation", "save_artifacts")
    builder.add_edge("save_artifacts", END)

    return builder