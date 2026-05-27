"""IMPLEMENT phase subgraph — SmallCode refactor.

The slice-implementer's filesystem tool surface is narrowed to a single
Compound Tool ``ReadEditLintTool`` (read -> edit -> lint, atomic with
revert on lint failure) plus the MCP codebase-index tools. Removing
``read_file`` / ``write_file`` / ``edit_file`` / ``ls`` / ``glob`` /
``grep`` / ``execute`` / ``search_codebase`` stops the Qwen3-class
local model from looping on raw reads and surfaces lint errors before
synthesis declares a slice "implemented".

Dispatch is a loop driven by a single conditional edge ``_route_slices``:
``pending_slices`` -> ``slice_implementer`` (per-slice Sends),
``failed_slices`` -> ``fallback_decomposer`` (per-slice Sends), and once
both lists are empty -> ``synthesize_implementation``. The custom
``_slice_list_reducer`` (see ``spine.workflow.subgraph_state``) lets a
node atomically remove the slice it was handed and add new ones in a
single state update, so the loop actually terminates.
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
from spine.models.enums import PhaseName
from spine.workflow.subgraph_state import ImplementSubgraphState

logger = logging.getLogger(__name__)
_MAX_ARTIFACT_STATE_CHARS = 500
_MAX_DECOMPOSE_DEPTH = 2


# ── Router ──────────────────────────────────────────────────────────────


def _base_send_payload(state: ImplementSubgraphState) -> dict[str, Any]:
    """Propagated context shared by every Send.

    The reducer-managed slice lists are intentionally omitted — each
    fan-out node only needs the active slice plus the surrounding
    work/phase context.
    """
    return {
        "phase": state.get("phase", "implement"),
        "work_id": state.get("work_id", "unknown"),
        "work_type": state.get("work_type", ""),
        "workspace_root": state.get("workspace_root", "."),
        "plan_path": state.get("plan_path", ""),
        "gap_plan_path": state.get("gap_plan_path"),
    }


def _route_slices(
    state: ImplementSubgraphState,
) -> list[Send] | Literal["synthesize_implementation"]:
    """Conditional edge from START / slice_implementer / fallback_decomposer.

    Fans out one Send per pending slice (to ``slice_implementer``) and
    one Send per failed slice (to ``fallback_decomposer``). When both
    lists are empty, routes to synthesis.
    """
    pending = state.get("pending_slices", []) or []
    failed = state.get("failed_slices", []) or []

    if not pending and not failed:
        logger.info(
            "[%s] IMPLEMENT route: pending=0 failed=0 — routing to synthesis",
            state.get("work_id", "?"),
        )
        return "synthesize_implementation"

    base = _base_send_payload(state)
    sends: list[Send] = []
    for sl in pending:
        sends.append(Send("slice_implementer", {**base, "active_slice": sl}))
    for sl in failed:
        sends.append(Send("fallback_decomposer", {**base, "active_slice": sl}))

    logger.info(
        "[%s] IMPLEMENT route: pending=%d failed=%d — dispatching %d Send(s)",
        state.get("work_id", "?"),
        len(pending),
        len(failed),
        len(sends),
    )
    return sends


# ── slice_implementer node ──────────────────────────────────────────────


async def _slice_implementer_node(
    state: ImplementSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Run a slice-implementer subagent on the active slice.

    Tools available to the subagent are restricted to
    ``ReadEditLintTool`` plus MCP codebase-index tools — raw filesystem
    tools are stripped. Returns a state update that removes the active
    slice from ``pending_slices`` and adds it (with outcome merged in)
    to either ``completed_slices`` or ``failed_slices``.
    """
    from spine.agents.factory import build_phase_agent
    from spine.agents.subagents import build_subagent_spec
    from spine.agents.tools.read_edit_lint import ReadEditLintTool

    work_id = state.get("work_id", "unknown")
    workspace_root = state.get("workspace_root", ".")
    active_slice: dict = state.get("active_slice") or {}
    slice_id = active_slice.get("id", "unknown")

    logger.info(
        "[%s] slice_implementer: slice=%r title=%r",
        work_id,
        slice_id,
        active_slice.get("title", ""),
    )

    try:
        subagent_spec = build_subagent_spec(
            name="slice-implementer",
            phase=PhaseName.IMPLEMENT,
            state=state,
            config=config,
        )

        spec_tools = list(subagent_spec.get("tools", []))
        mcp_tools = [t for t in spec_tools if getattr(t, "name", "").startswith("mcp_")]
        restricted_tools: list[Any] = [
            ReadEditLintTool(workspace_root=workspace_root),
            *mcp_tools,
        ]
        logger.info(
            "[%s] slice_implementer: tool surface = read_edit_lint + %d MCP tool(s)",
            work_id,
            len(mcp_tools),
        )

        agent = build_phase_agent(
            state=state,
            config=config,
            phase=PhaseName.IMPLEMENT,
            system_prompt=subagent_spec["system_prompt"],
            is_subagent=True,
            extra_tools=restricted_tools,
            response_format=subagent_spec.get("response_format"),
            skip_filesystem_middleware=True,
        )

        slice_json = json.dumps(active_slice, indent=2, ensure_ascii=False, default=str)
        prompt = (
            f"## Implement Slice: {slice_id}\n\n"
            f"The full slice definition is in the JSON below. "
            f"Read it carefully — it specifies target_files, "
            f"acceptance_criteria, and (optionally) a description. "
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
            "[%s] slice_implementer failed for %r: %s",
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

    status = slice_result.get("status", "blocked")
    if status in ("implemented", "partial"):
        merged = {**active_slice, **slice_result}
        return {
            "pending_slices": {"remove": [slice_id]},
            "completed_slices": {"add": [merged]},
            "slices_dispatched": True,
            "implementation_files_written": True,
        }

    failure_traceback = "\n".join(
        part
        for part in (
            slice_result.get("test_results", ""),
            "\n".join(slice_result.get("issues", []) or []),
        )
        if part
    )
    tagged = {
        **active_slice,
        **slice_result,
        "_failure_traceback": failure_traceback,
        "_decompose_depth": active_slice.get("_decompose_depth", 0),
    }
    return {
        "pending_slices": {"remove": [slice_id]},
        "failed_slices": {"add": [tagged]},
        "slices_dispatched": True,
    }


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


# ── fallback_decomposer node ────────────────────────────────────────────


async def _fallback_decomposer_node(
    state: ImplementSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Decompose a failed slice into 2–3 micro-slices via a structured LLM call.

    Bounded by ``_decompose_depth`` so a chronically-unrecoverable slice
    surfaces as permanently blocked rather than looping forever.
    """
    from spine.agents.decomposer import run_decomposer

    work_id = state.get("work_id", "unknown")
    active_slice: dict = state.get("active_slice") or {}
    slice_id = active_slice.get("id", "unknown")
    depth = active_slice.get("_decompose_depth", 0)

    if depth >= _MAX_DECOMPOSE_DEPTH:
        logger.warning(
            "[%s] fallback_decomposer: slice=%r hit depth cap (%d) — leaving as blocked",
            work_id,
            slice_id,
            depth,
        )
        return {}

    try:
        micro_slices = await run_decomposer(
            mode="FALLBACK",
            failed_slice=active_slice,
            error_traceback=active_slice.get("_failure_traceback", ""),
            config=config,
            session_id=work_id,
        )
    except Exception as e:
        logger.error(
            "[%s] fallback_decomposer failed for %r: %s — dropping slice",
            work_id,
            slice_id,
            e,
            exc_info=True,
        )
        return {"failed_slices": {"remove": [slice_id]}}

    if not micro_slices:
        logger.warning(
            "[%s] fallback_decomposer: slice=%r produced 0 micro-slices — dropping",
            work_id,
            slice_id,
        )
        return {"failed_slices": {"remove": [slice_id]}}

    next_depth = depth + 1
    for sl in micro_slices:
        sl["_decompose_depth"] = next_depth

    logger.info(
        "[%s] fallback_decomposer: slice=%r -> %d micro-slice(s) at depth=%d",
        work_id,
        slice_id,
        len(micro_slices),
        next_depth,
    )
    return {
        "failed_slices": {"remove": [slice_id]},
        "pending_slices": {"add": micro_slices},
    }


# ── synthesize_implementation node ──────────────────────────────────────


def _failed_to_blocked(s: dict) -> dict:
    """Coerce a failed-slice dict into the SliceResult shape expected by synthesis."""
    issues = (
        ["exceeded fallback depth"]
        if s.get("_decompose_depth", 0) >= _MAX_DECOMPOSE_DEPTH
        else ["implementer failed"]
    )
    return {
        "slice_name": s.get("id", "?"),
        "status": "blocked",
        "files_modified": s.get("files_modified", []) or [],
        "files_created": s.get("files_created", []) or [],
        "test_results": (s.get("_failure_traceback", "") or "")[:1000],
        "issues": issues,
    }


async def _synthesize_implementation_node(
    state: ImplementSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Write implementation.md / implementation.json from accumulated results."""
    from spine.agents.implement_tools import write_implementation_files

    work_id = state.get("work_id", "unknown")
    workspace_root = state.get("workspace_root", ".")
    completed = state.get("completed_slices", []) or []
    failed = state.get("failed_slices", []) or []

    slice_results: list[dict] = []
    slice_results.extend(completed)
    slice_results.extend(_failed_to_blocked(s) for s in failed)

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
            "[%s] IMPLEMENT synthesize: failed to write artifacts: %s", work_id, e
        )
        return {
            "agent_response": summary,
            "artifacts_output": {},
            "phase_status": "error",
            "slices_dispatched": True,
            "implementation_files_written": False,
        }

    logger.info(
        "[%s] IMPLEMENT synthesize: wrote %d slice result(s) to %s/",
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

    parts = [f"Implemented {total} feature slice(s)."]
    for status, count in sorted(statuses.items()):
        parts.append(f"- {status}: {count}")

    implemented = statuses.get("implemented", 0)
    blocked = statuses.get("blocked", 0)
    if implemented == total:
        parts.append("All slices implemented successfully.")
    elif blocked > 0:
        parts.append(f"{blocked} slice(s) blocked — see implementation.md for details.")

    return "\n".join(parts)


# ── save_artifacts node ─────────────────────────────────────────────────


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

_ROUTE_MAP = {
    "slice_implementer": "slice_implementer",
    "fallback_decomposer": "fallback_decomposer",
    "synthesize_implementation": "synthesize_implementation",
}


def build_implement_subgraph() -> Any:
    """Build the IMPLEMENT phase subgraph (SmallCode dispatch loop).

    Nodes:
    1. ``slice_implementer`` — per-slice subagent with restricted tools.
    2. ``fallback_decomposer`` — micro-slices a failed slice.
    3. ``synthesize_implementation`` — writes implementation artifacts.
    4. ``save_artifacts`` — scans disk and materializes to state.

    The conditional edge ``_route_slices`` is wired three times — from
    ``START`` and from each fan-out node — so the loop re-evaluates after
    every super-step until both ``pending_slices`` and ``failed_slices``
    are empty.
    """
    builder = StateGraph(ImplementSubgraphState)

    builder.add_node("slice_implementer", _slice_implementer_node)
    builder.add_node("fallback_decomposer", _fallback_decomposer_node)
    builder.add_node("synthesize_implementation", _synthesize_implementation_node)
    builder.add_node("save_artifacts", _save_implement_artifacts)

    builder.add_conditional_edges(START, _route_slices, _ROUTE_MAP)
    builder.add_conditional_edges("slice_implementer", _route_slices, _ROUTE_MAP)
    builder.add_conditional_edges("fallback_decomposer", _route_slices, _ROUTE_MAP)
    builder.add_edge("synthesize_implementation", "save_artifacts")
    builder.add_edge("save_artifacts", END)

    return builder
