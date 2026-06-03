"""IMPLEMENT phase subgraph — SmallCode refactor.

The slice-implementer's write surface is a single Compound Tool
``ReadEditLintTool`` (read -> edit -> lint, atomic with revert on lint
failure), the only tool that can mutate files. For navigation it carries
the curated read/search surface — the ``codebase_query`` index wrapper
plus ``read_file`` / ``search_codebase`` / ``ast_extract_symbol`` — and
``execute`` for tests/linters. Generic ``write_file`` / ``edit_file`` and
raw ``mcp_``-prefixed tools are never exposed, which surfaces lint errors
before synthesis declares a slice "implemented".

Dispatch is a loop driven by a single conditional edge ``_route_slices``:
``pending_slices`` -> ``slice_implementer`` (per-slice Sends),
``failed_slices`` -> ``fallback_decomposer`` (per-slice Sends), and once
both lists are empty -> ``synthesize_implementation``. The custom
``_slice_list_reducer`` (see ``spine.workflow.subgraph_state``) lets a
node atomically remove the slice it was handed and add new ones in a
single state update, so the loop actually terminates.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from spine.agents.artifacts import (
    artifact_path,
    materialize_phase_artifacts,
    scan_artifact_dir,
)
from spine.agents.prompt_format import Tag, hostage_layout, xml_blocks
from spine.agents.plan_do import (
    directive_from_state,
    format_directive_for_prompt,
    run_plan_node,
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
        # Each parallel slice goes through plan_slice_implementer → slice_implementer
        # so the model splits "plan approach" from "execute with tools" — a
        # mitigation for smaller models that fixate on the slice JSON.
        sends.append(Send("plan_slice_implementer", {**base, "active_slice": sl}))
    for sl in failed:
        # The decomposer is already a no-tool structured-output call, so it
        # doesn't get a plan-before-do step.
        sends.append(Send("fallback_decomposer", {**base, "active_slice": sl}))

    logger.info(
        "[%s] IMPLEMENT route: pending=%d failed=%d — dispatching %d Send(s)",
        state.get("work_id", "?"),
        len(pending),
        len(failed),
        len(sends),
    )
    return sends


# ── split_slices node (per-file decomposition) ──────────────────────────


def _build_subslice_chain(parent: dict, subs: list[dict]) -> dict:
    """Turn an ordered list of single-file sub-slices into a dispatchable head.

    Every sub-slice is tagged with sequencing metadata; the head carries the
    remainder as a flat ``_sibling_queue`` (the slice_implementer promotes the
    next entry on success). All chain members share ``_all_files`` so each
    implementer can list its siblings for read-only context.
    """
    total = len(subs)
    depth = parent.get("_decompose_depth", 0)
    all_files = [f for f in (parent.get("target_files") or []) if f]
    chain: list[dict] = []
    for i, sub in enumerate(subs, start=1):
        chain.append(
            {
                **sub,
                "_parent_slice_id": parent.get("id"),
                "_all_files": all_files,
                "_file_index": i,
                "_file_total": total,
                "_validate_slice_criteria": i == total,
                "_decompose_depth": depth,
            }
        )
    head = chain[0]
    head["_sibling_queue"] = chain[1:]
    return head


async def _split_slices_node(
    state: ImplementSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Proactively split each multi-file slice into single-file sub-slices.

    Runs once at START, before the dispatch loop. For every seeded slice with
    ≥2 target files we call the PER_FILE decomposer (all concurrently) and
    replace the slice with the **head** of its single-file chain; the rest of
    the chain rides on the head's ``_sibling_queue`` and is promoted one file
    at a time by ``_slice_implementer_node``. Single-file slices — and any
    slice whose decomposition fails — pass through unchanged, so a decomposer
    outage never strands work.
    """
    work_id = state.get("work_id", "unknown")
    pending = state.get("pending_slices", []) or []
    multi = [
        sl
        for sl in pending
        if len([f for f in (sl.get("target_files") or []) if f]) >= 2
    ]
    if not multi:
        return {}

    from spine.agents.decomposer import run_decomposer

    async def _decompose(parent: dict) -> list[dict] | None:
        try:
            subs = await run_decomposer(
                mode="PER_FILE",
                source_slice=parent,
                config=config,
                session_id=work_id,
            )
        except Exception as e:  # noqa: BLE001 — graceful degradation
            logger.warning(
                "[%s] split_slices: PER_FILE failed for %r: %s — keeping slice whole",
                work_id,
                parent.get("id"),
                e,
            )
            return None
        return subs or None

    results = await asyncio.gather(*[_decompose(p) for p in multi])

    remove_ids: list[str] = []
    adds: list[dict] = []
    for parent, subs in zip(multi, results):
        if not subs:
            continue
        remove_ids.append(parent.get("id"))
        adds.append(_build_subslice_chain(parent, subs))
        logger.info(
            "[%s] split_slices: slice=%r -> %d single-file sub-slice(s)",
            work_id,
            parent.get("id"),
            len(subs),
        )

    if not remove_ids:
        return {}
    return {"pending_slices": {"remove": remove_ids, "add": adds}}


def _subslice_context(active_slice: dict) -> str:
    """Build a single-file directive block, or '' for an ordinary slice.

    Lists the slice's other files as read-only context and states whether
    this implementer is the one expected to satisfy slice-level criteria
    (the last file) or should treat sibling-dependent tests as pending.
    """
    if not active_slice.get("_parent_slice_id"):
        return ""
    target_files = active_slice.get("target_files") or []
    my_file = target_files[0] if target_files else "(unknown)"
    siblings = [f for f in (active_slice.get("_all_files") or []) if f != my_file]
    idx = active_slice.get("_file_index", 1)
    total = active_slice.get("_file_total", 1)
    sib_lines = (
        "\n".join(f"- {p}" for p in siblings) if siblings else "(none)"
    )
    if active_slice.get("_validate_slice_criteria"):
        validate = (
            "This is the LAST file of the slice — every sibling file is now in "
            "place, so run the slice's acceptance-criteria checks (tests/lint) "
            "and make them pass."
        )
    else:
        validate = (
            "Sibling files are still being built in later steps — do NOT expect "
            "slice-level/integration tests to pass yet. Make your one file "
            "correct and self-consistent; note any sibling-dependent check as "
            "pending rather than failing."
        )
    return (
        f"\n\n## Single-file scope (file {idx}/{total} of slice "
        f"{active_slice.get('_parent_slice_id')!r})\n"
        f"CREATE or MODIFY only this one file: {my_file}\n"
        f"You MAY READ these sibling files for context but MUST NOT edit them:\n"
        f"{sib_lines}\n\n{validate}"
    )


# ── plan_slice_implementer node (no tools) ──────────────────────────────


async def _plan_slice_implementer_node(
    state: ImplementSubgraphState,
    config: RunnableConfig | None = None,
) -> Command:
    """No-tool planning step for a single slice's implementation.

    Produces a per-branch SubagentDirective and dispatches a Send to
    the slice_implementer carrying both the slice and the directive on
    the per-branch payload. Returning ``Command(goto=Send(...))`` —
    rather than writing the directive to a shared channel — is required
    because parallel Send branches share the subgraph's channel space,
    so N concurrent writes to ``active_slice_directive`` would crash
    apply_writes with ``InvalidUpdateError``.
    """
    work_id = state.get("work_id", "unknown")
    active_slice: dict = state.get("active_slice") or {}
    slice_id = active_slice.get("id", "unknown")
    title = active_slice.get("title", "")
    target_files = active_slice.get("target_files") or []
    criteria = active_slice.get("acceptance_criteria") or []
    description = active_slice.get("description", "")

    crit_lines = "\n".join(f"- {c}" for c in criteria) if criteria else "(none)"
    file_lines = "\n".join(f"- {p}" for p in target_files) if target_files else "(none listed)"

    task = (
        f"Plan how to implement slice {slice_id!r} (title: {title!r}). The do "
        "node will use read_edit_lint and MCP tools to make atomic edits.\n\n"
        f"## Description\n{description or '(none)'}\n\n"
        f"## Target files\n{file_lines}\n\n"
        f"## Acceptance criteria\n{crit_lines}"
        f"{_subslice_context(active_slice)}"
    )
    directive = await run_plan_node(
        state=dict(state),
        config=config,
        phase_path=f"{PhaseName.IMPLEMENT.value}/subagents/slice-implementer",
        task_description=task,
        role_hint=f"slice-implementer for slice {slice_id!r}",
    )
    logger.info(
        "[%s] plan_slice_implementer: slice=%r approach=%r",
        work_id, slice_id, directive.approach[:80],
    )
    send_payload: dict[str, Any] = {
        **_base_send_payload(state),
        "active_slice": active_slice,
        "active_slice_directive": directive.model_dump(),
    }
    return Command(goto=Send("slice_implementer", send_payload))


# ── slice_implementer node ──────────────────────────────────────────────


async def _slice_implementer_node(
    state: ImplementSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Run a slice-implementer subagent on the active slice.

    The subagent's tool surface is the curated set built by
    ``build_subagent_spec`` — ``read_edit_lint`` (the only write tool),
    the ``codebase_query`` index wrapper, plus ``read_file`` /
    ``search_codebase`` / ``ast_extract_symbol`` / ``execute``. No raw
    ``mcp_``-prefixed tool is exposed. Returns a state update that removes
    the active slice from ``pending_slices`` and adds it (with outcome
    merged in) to either ``completed_slices`` or ``failed_slices``.
    """
    from spine.agents.factory import build_phase_agent
    from spine.agents.subagents import build_subagent_spec

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

        extra_tools = list(subagent_spec.get("tools", []))
        logger.info(
            "[%s] slice_implementer: tool surface = %s",
            work_id,
            [getattr(t, "name", repr(t)) for t in extra_tools],
        )

        agent = build_phase_agent(
            state=state,
            config=config,
            phase=PhaseName.IMPLEMENT,
            system_prompt=subagent_spec["system_prompt"],
            is_subagent=True,
            extra_tools=extra_tools,
            response_format=subagent_spec.get("response_format"),
            skip_filesystem_middleware=True,
            # The subagent_spec already curated the implementer's tool
            # surface (read_edit_lint + codebase_query wrapper + read/search/
            # execute via subagents.py) — they live in ``extra_tools`` above.
        )

        # Strip private sequencing keys (e.g. _sibling_queue, which nests the
        # remaining sub-slices) from the JSON block so the prompt stays small;
        # the single-file scope is rendered as plain text by _subslice_context.
        public_slice = {
            k: v for k, v in active_slice.items() if not k.startswith("_")
        }
        slice_json = json.dumps(public_slice, indent=2, ensure_ascii=False, default=str)
        directive_block = format_directive_for_prompt(
            directive_from_state(dict(state), "active_slice_directive")
        )
        # Hostage layout: data blocks first, plain-text directive at the
        # absolute tail. The directive_block from format_directive_for_prompt
        # is already wrapped in <directive> — splice it after xml_blocks
        # rather than re-wrapping.
        prompt = hostage_layout(
            xml_blocks(
                (Tag.OBJECTIVE, f"Implement slice: {slice_id}"),
                (Tag.FINDINGS, f"```json\n{slice_json}\n```"),
            )
            + ("\n\n" + directive_block if directive_block else ""),
            (
                "Read the slice JSON above carefully — it specifies "
                "target_files, acceptance_criteria, and (optionally) a "
                "description. Make only the changes described in the slice."
                + _subslice_context(active_slice)
            ),
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

    # The remaining single-file sub-slices of this parent (empty for an
    # ordinary slice). Promoted one at a time so a parent's files land
    # sequentially while other parents proceed in parallel.
    sibling_queue = active_slice.get("_sibling_queue") or []

    status = slice_result.get("status", "blocked")
    if status in ("implemented", "partial"):
        merged = {**active_slice, **slice_result}
        merged.pop("_sibling_queue", None)  # keep completed_slices tidy
        update: dict[str, Any] = {
            "pending_slices": {"remove": [slice_id]},
            "completed_slices": {"add": [merged]},
            "slices_dispatched": True,
            "implementation_files_written": True,
        }
        if sibling_queue:
            # Promote the next file; thread the rest onto its own queue.
            nxt = {**sibling_queue[0], "_sibling_queue": sibling_queue[1:]}
            update["pending_slices"]["add"] = [nxt]
        return update

    failure_traceback = "\n".join(
        part
        for part in (
            slice_result.get("test_results", ""),
            "\n".join(slice_result.get("issues", []) or []),
        )
        if part
    )
    issues = list(slice_result.get("issues") or [])
    if sibling_queue:
        skipped = [q.get("id", "?") for q in sibling_queue]
        issues.append(
            "Remaining slice files skipped after failure: " + ", ".join(skipped)
        )
    tagged = {
        **active_slice,
        **slice_result,
        "issues": issues,
        "_failure_traceback": failure_traceback,
        "_decompose_depth": active_slice.get("_decompose_depth", 0),
    }
    # Drop the queue: the fallback decomposer micro-slices the failed FILE,
    # it does not resume the parent's later files.
    tagged.pop("_sibling_queue", None)
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

    # Deterministic record of every file the implementer reported touching —
    # consumed by the scope-boundary gate in _implement_result_mapper.
    files_written = _collect_files_written(slice_results)

    if not slice_results:
        logger.warning("[%s] IMPLEMENT synthesize: zero slice results", work_id)
        return {
            "agent_response": "",
            "artifacts_output": {},
            "phase_status": "needs_review",
            "slices_dispatched": False,
            "implementation_files_written": False,
            "files_written": [],
        }

    # ── Honesty guard ────────────────────────────────────────────────
    # Trace 019e6974 showed the slice-implementer returning
    # status="implemented" with empty files_modified/files_created
    # arrays despite the actual edits being applied on disk — a
    # reporting bug masquerading as success. Downstream gates and
    # verify treat "success + no files" as a working implementation,
    # so verify burns its whole budget chasing an empty diff.
    #
    # If every slice claims success but reports no file activity,
    # demote to needs_review so a human can reconcile the gap between
    # the report and reality.
    non_failed = [
        r for r in slice_results
        if r.get("status") in ("implemented", "partial")
    ]
    touched = [
        r for r in non_failed
        if (r.get("files_modified") or r.get("files_created"))
    ]
    all_claimed_no_files = bool(non_failed) and not touched

    impl_dir = artifact_path(work_id, PhaseName.IMPLEMENT.value)
    summary = _build_implementation_summary(slice_results)

    if all_claimed_no_files:
        warning_line = (
            f"WARNING: {len(non_failed)} slice(s) reported success but "
            "files_modified/files_created are empty — the implementer's "
            "self-report disagrees with what was written. Manual review "
            "required to reconcile the implementation report against disk."
        )
        summary = f"{summary}\n\n{warning_line}"
        logger.warning(
            "[%s] IMPLEMENT synthesize: %d slice(s) success-with-no-files — "
            "demoting phase_status to needs_review",
            work_id, len(non_failed),
        )

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
            "files_written": files_written,
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
        "phase_status": "needs_review" if all_claimed_no_files else "success",
        "slices_dispatched": True,
        "implementation_files_written": True,
        "files_written": files_written,
    }


def _collect_files_written(slice_results: list[dict]) -> list[str]:
    """Aggregate every file path the implementer reported creating or modifying.

    Returns a sorted, de-duplicated list across all slice results. Non-string
    entries are skipped defensively so a malformed subagent report cannot crash
    the scope-boundary gate downstream.
    """
    seen: set[str] = set()
    for r in slice_results:
        for key in ("files_modified", "files_created"):
            for path in r.get(key, []) or []:
                if isinstance(path, str) and path.strip():
                    seen.add(path.strip())
    return sorted(seen)


def _build_implementation_summary(slice_results: list[dict]) -> str:
    """Build a human-readable implementation summary from slice results.

    Results may be per-file sub-slices (from PER_FILE decomposition). They are
    counted both per file and grouped back to their logical parent slice (via
    ``_parent_slice_id``) so the report still reads at slice granularity.
    """
    total = len(slice_results)
    statuses: dict[str, int] = {}
    for r in slice_results:
        s = r.get("status", "unknown")
        statuses[s] = statuses.get(s, 0) + 1

    # Distinct logical slices: the parent id when present, else the result's id.
    logical = {r.get("_parent_slice_id") or r.get("slice_name") or r.get("id") for r in slice_results}
    logical.discard(None)

    if logical and len(logical) != total:
        parts = [
            f"Implemented {len(logical)} feature slice(s) across {total} "
            "single-file step(s)."
        ]
    else:
        parts = [f"Implemented {total} feature slice(s)."]
    for status, count in sorted(statuses.items()):
        parts.append(f"- {status}: {count} file-step(s)")

    implemented = statuses.get("implemented", 0)
    blocked = statuses.get("blocked", 0)
    if implemented == total:
        parts.append("All slices implemented successfully.")
    elif blocked > 0:
        parts.append(f"{blocked} step(s) blocked — see implementation.md for details.")

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
    "plan_slice_implementer": "plan_slice_implementer",
    "slice_implementer": "slice_implementer",
    "fallback_decomposer": "fallback_decomposer",
    "synthesize_implementation": "synthesize_implementation",
}


def build_implement_subgraph() -> Any:
    """Build the IMPLEMENT phase subgraph (SmallCode dispatch loop).

    Nodes:
    1. ``split_slices`` — runs once at START, replacing each multi-file slice
       with the head of a single-file sub-slice chain (PER_FILE decomposition).
    2. ``slice_implementer`` — per-slice subagent with restricted tools; on
       success it promotes the parent's next file from ``_sibling_queue``.
    3. ``fallback_decomposer`` — micro-slices a failed slice.
    4. ``synthesize_implementation`` — writes implementation artifacts.
    5. ``save_artifacts`` — scans disk and materializes to state.

    The conditional edge ``_route_slices`` is wired three times — from
    ``split_slices`` and from each fan-out node — so the loop re-evaluates
    after every super-step until both ``pending_slices`` and ``failed_slices``
    are empty.
    """
    builder = StateGraph(ImplementSubgraphState)

    builder.add_node("split_slices", _split_slices_node)
    builder.add_node("plan_slice_implementer", _plan_slice_implementer_node)
    builder.add_node("slice_implementer", _slice_implementer_node)
    builder.add_node("fallback_decomposer", _fallback_decomposer_node)
    builder.add_node("synthesize_implementation", _synthesize_implementation_node)
    builder.add_node("save_artifacts", _save_implement_artifacts)

    # START → split_slices (per-file decomposition) → route. The split node
    # runs once, replacing multi-file slices with single-file chains before
    # any dispatch; the router then fans out as before.
    builder.add_edge(START, "split_slices")
    builder.add_conditional_edges("split_slices", _route_slices, _ROUTE_MAP)
    # plan_slice_implementer dispatches to slice_implementer dynamically
    # via Command(goto=Send) (see the node itself) so each parallel
    # branch carries its own directive without colliding on a shared
    # LastValue channel. slice_implementer's outgoing conditional edge
    # re-enters the router so the dispatch loop terminates correctly.
    builder.add_conditional_edges("slice_implementer", _route_slices, _ROUTE_MAP)
    builder.add_conditional_edges("fallback_decomposer", _route_slices, _ROUTE_MAP)
    builder.add_edge("synthesize_implementation", "save_artifacts")
    builder.add_edge("save_artifacts", END)

    return builder
