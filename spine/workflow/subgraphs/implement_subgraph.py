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
import ast
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
    SubagentDirective,
    directive_from_state,
    format_directive_for_prompt,
    run_plan_node,
)
from spine.agents.retry import (
    MaxTokenBudgetExceeded,
    ServerUnreachable,
    ainvoke_with_retry,
)

# Run-level abort signals must NOT be swallowed into a "blocked" slice — they
# have to propagate past the per-slice handlers to the subgraph wrapper, which
# converts them into a clean needs_review (token budget exhausted, or the LLM
# endpoint is unreachable). Catching them here would let a down server keep
# fanning out dead Sends (trace 019ece87).
_ABORT_EXCEPTIONS = (MaxTokenBudgetExceeded, ServerUnreachable)
from spine.models.enums import PhaseName
from spine.workflow.subgraph_state import ImplementSubgraphState

logger = logging.getLogger(__name__)
_MAX_ARTIFACT_STATE_CHARS = 500


def _max_decompose_depth() -> int:
    """Configured cap on the fallback-decompose recursion.

    Read from SpineConfig (``implement_max_decompose_depth``) so deployments on
    weaker local models can fail a stubborn slice fast instead of letting it
    fan out into 1 + 3 + 9 = 13 implementer attempts (trace 019ed3dc). Fails
    open to 1 if config can't be loaded.
    """
    try:
        from spine.config import SpineConfig

        return max(0, int(SpineConfig.load().implement_max_decompose_depth))
    except Exception:  # noqa: BLE001 — never let config break the dispatch loop
        return 1

# Substrings that mark a slice failure as a context-window overflow rather than
# a logic error. A finite-window local model (llama.cpp/vLLM GGUF) rejects the
# request when prompt + requested completion exceeds n_ctx; re-running the same
# whole-file work overflows identically, so the fallback decomposer must react
# by producing narrower, region-scoped micro-slices (trace 019ece87).
_OVERFLOW_MARKERS = (
    "context size has been exceeded",
    "context length",
    "context window",
    "maximum context",
    "exceeds the maximum",
    "too many tokens",
    "reduce the length",
)


def _is_context_overflow(text: str) -> bool:
    """True when ``text`` looks like a context-window overflow error."""
    if not text:
        return False
    low = text.lower()
    return any(marker in low for marker in _OVERFLOW_MARKERS)


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
        "verification_findings": state.get("verification_findings") or [],
    }


def _slice_marker_ids(slices: list[dict]) -> set[str]:
    """All ids a slice is known by — its own id/slice_name plus its parent's.

    Sub-slices produced by ``split_slices`` / ``fallback_decomposer`` carry a
    synthetic ``id`` but reference the originating plan slice via
    ``_parent_slice_id``. A dependency named after the parent is satisfied once
    the parent's sub-slices land, so dependency matching considers all three
    keys.
    """
    ids: set[str] = set()
    for s in slices:
        for k in ("id", "slice_name", "_parent_slice_id"):
            v = s.get(k)
            if v:
                ids.add(v)
    return ids


# Final-mile mode: when a slice is down to a HANDFUL of failing criteria
# (with a substantial passing majority), wholesale re-synthesis risks more
# than it fixes — run 019f25f5 reached 3/25 failing and then two full
# regenerations scored worse (6, 9) until the budget expired. Below this
# threshold the editor is constrained to the smallest edit set that fixes
# exactly the failing criteria, and placement prefers the smallest clean
# candidate.
_FINAL_MILE_MAX_FAILS = 5


def _final_mile_fails(state: ImplementSubgraphState, active_slice: dict) -> list[str]:
    """This slice's failing criteria when it qualifies for final-mile mode.

    Qualifies when the last verification shows 1..=_FINAL_MILE_MAX_FAILS
    failing checklist criteria AND more passing than failing (real progress
    to protect). Returns the failing criterion texts, or [] (not a rework,
    no findings, too many fails, or no passing majority).
    """
    ids = _slice_marker_ids([active_slice])
    fails: list[str] = []
    passing = 0
    seen_any = False
    for f in state.get("verification_findings") or []:
        if not isinstance(f, dict) or f.get("slice_name") not in ids:
            continue
        for item in f.get("checklist") or []:
            if not isinstance(item, dict) or not item.get("criterion"):
                continue
            seen_any = True
            if item.get("passed"):
                passing += 1
            else:
                fails.append(str(item["criterion"]))
    if not seen_any or not fails:
        return []
    if len(fails) > _FINAL_MILE_MAX_FAILS or passing <= len(fails):
        return []
    return fails


# Bound on the rendered gap-remediation block. A gap plan covers every failed
# slice; the per-slice extract is small, but a degenerate gap plan must not
# blow the editor's prompt.
_GAP_BODY_CHAR_CAP = 7000

# Bound on the do-not-break (currently passing criteria) block within the
# gap body — large merged slices carry 40+ criteria.
_GAP_PASSING_CHAR_CAP = 2500


def _gap_fixes_body(state: ImplementSubgraphState, active_slice: dict) -> str:
    """This slice's verify-gap remediation, rendered for the editor prompt.

    On a gap-fix rework (verify failed → gap_plan → implement) the gap plan's
    per-slice fixes are the ONLY channel telling the editor what its previous
    output got wrong. Nothing consumed ``gap_plan_path`` before this: the
    no-tool synthesis editor cannot read gap_plan.json from disk, so it
    regenerated the exact failing code every cycle (run 019f20a5 — three
    byte-identical wrong attempts → needs_review). Matching uses
    ``_slice_marker_ids`` so a split sub-slice still collects its parent's
    fixes. Returns '' outside gap reworks or on any load/shape problem.
    """
    gap_dir = state.get("gap_plan_path")
    if not gap_dir:
        return ""
    from pathlib import Path

    path = Path(state.get("workspace_root", ".")) / gap_dir / "gap_plan.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    items = data.get("remediation_items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return ""

    ids = _slice_marker_ids([active_slice])
    own_files = {str(t) for t in (active_slice.get("target_files") or []) if t}
    lines: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        own_item = item.get("slice_id") in ids
        fixes = [fx for fx in item.get("fixes") or [] if isinstance(fx, dict)]
        if not own_item:
            # Cross-slice reattribution: a fix can target THIS slice's file
            # while the remediation item is keyed to the slice where the
            # failure SURFACED (probe 21: the table-name fix belonged in the
            # model slice but the item named the crashing test slice). A
            # reopened slice must still see the fix meant for its file —
            # and ONLY that fix; the item's other fixes belong elsewhere.
            fixes = [fx for fx in fixes if str(fx.get("file_path")) in own_files]
            if not fixes:
                continue
        lines.append(
            f"A previous implementation of slice '{item.get('slice_id')}' "
            f"FAILED verification (priority: {item.get('priority', 'medium')})."
        )
        for failure in item.get("failures") or []:
            lines.append(f"- FAILED: {failure}")
        root_cause = str(item.get("root_cause") or "").strip()
        if root_cause:
            lines.append(f"Root cause: {root_cause}")
        for fix in fixes:
            lines.append(f"Required fix in {fix.get('file_path', '?')}:")
            issue = str(fix.get("issue_description") or "").strip()
            if issue:
                lines.append(f"  Issue: {issue}")
            suggested = str(fix.get("suggested_fix") or "").strip()
            if suggested:
                lines.append(f"  Fix: {suggested}")
            for ac in fix.get("acceptance_criteria") or []:
                lines.append(f"  Must satisfy: {ac}")
    if not lines:
        return ""
    # Criteria this slice ALREADY satisfies, from last cycle's verification
    # checklist. The no-tool editor re-synthesizes wholesale, so without an
    # explicit do-not-break list a rework is free to trade passing criteria
    # for failing ones (run 019f2579: cycle 6 regressed 9→23 open gaps).
    passing: list[str] = []
    for f in state.get("verification_findings") or []:
        if not isinstance(f, dict) or f.get("slice_name") not in ids:
            continue
        for item in f.get("checklist") or []:
            if isinstance(item, dict) and item.get("passed") and item.get("criterion"):
                passing.append(str(item["criterion"]))
    if passing:
        block = ["", "CURRENTLY PASSING — your edits MUST NOT break these behaviors:"]
        block.extend(f"  [pass] {p}" for p in passing)
        passing_body = "\n".join(block)
        if len(passing_body) > _GAP_PASSING_CHAR_CAP:
            passing_body = (
                passing_body[:_GAP_PASSING_CHAR_CAP].rstrip()
                + f" … [+{len(passing)} passing criteria total — preserve ALL current behavior you are not explicitly fixing]"
            )
        lines.append(passing_body)
    body = "\n".join(lines)
    if len(body) > _GAP_BODY_CHAR_CAP:
        body = body[:_GAP_BODY_CHAR_CAP].rstrip() + " … [gap plan truncated]"
    return body


def _ready_slices(pending: list[dict], completed: list[dict]) -> list[dict]:
    """Pending slices whose declared ``dependencies`` have all completed.

    Enforces the plan's dependency DAG at dispatch — it is otherwise discarded
    by the flat ``pending_slices`` seed in ``compose._implement_state_mapper``,
    which lets every slice (including ones editing the SAME file) fan out
    concurrently and race (the 019efd92 dispatch explosion: 3 slices all editing
    config_view.py → ~687 executions / 1.33M tokens). A slice is ready only when
    every id in its ``dependencies`` is present in ``completed_slices`` AND no
    longer anywhere in ``pending`` — so a dependency that was split into a
    sub-slice chain counts as done only after its LAST file lands, not its first.
    A blocked slice lands in ``completed_slices`` too, so a dead dependency
    unblocks its dependents rather than deadlocking the loop.
    """
    done = _slice_marker_ids(completed)
    in_pending = _slice_marker_ids(pending)
    ready: list[dict] = []
    for sl in pending:
        deps = [d for d in (sl.get("dependencies") or []) if d]
        if all(d in done and d not in in_pending for d in deps):
            ready.append(sl)
    return ready


def _route_slices(
    state: ImplementSubgraphState,
) -> list[Send] | Literal["synthesize_implementation"]:
    """Conditional edge from START / slice_implementer / fallback_decomposer.

    Fans out one Send per *dependency-ready* pending slice (to the editor) and
    one Send per failed slice (to ``fallback_decomposer``). Dispatch respects
    the plan DAG (see :func:`_ready_slices`) so dependent slices — and slices
    sharing a target file — are serialized instead of racing. When pending and
    failed are both empty, or a hard dispatch ceiling is hit, routes to synthesis.
    """
    pending = state.get("pending_slices", []) or []
    failed = state.get("failed_slices", []) or []
    work_id = state.get("work_id", "?")

    if not pending and not failed:
        logger.info(
            "[%s] IMPLEMENT route: pending=0 failed=0 — routing to synthesis",
            work_id,
        )
        return "synthesize_implementation"

    # ── Backstop (C): bound total slice executions ──────────────────────
    # Every implementer/decomposer execution increments ``slice_dispatch_count``.
    # If a runaway slips past the DAG gating (e.g. a same-file race the gating
    # can't serialize because two slices are genuinely independent), abort the
    # loop to synthesis instead of dispatching hundreds of Sends and burning
    # the token budget (trace 019efd92).
    try:
        from spine.config import SpineConfig

        cfg = SpineConfig.load()
        cap = int(cfg.implement_max_slice_dispatches)
    except Exception:  # noqa: BLE001
        cfg, cap = None, 100
    dispatched = int(state.get("slice_dispatch_count", 0) or 0)
    if cap > 0 and dispatched >= cap:
        logger.error(
            "[%s] IMPLEMENT route: dispatch ceiling hit (%d>=%d) — aborting to "
            "synthesis with pending=%d failed=%d. Likely a decompose / same-file "
            "runaway; remaining slices surface as incomplete.",
            work_id, dispatched, cap, len(pending), len(failed),
        )
        return "synthesize_implementation"

    base = _base_send_payload(state)
    sends: list[Send] = []

    # ── Dependency gating (A): only dispatch slices whose deps are done ──
    ready = _ready_slices(pending, state.get("completed_slices", []) or [])
    if pending and not ready:
        # No pending slice is dependency-ready. Distinguish two cases:
        #   * Legitimately waiting: a slice's unsatisfied dep is in the failed
        #     set, so the fallback_decomposer may yet complete it — keep waiting.
        #   * Permanently blocked: every unsatisfied dep is neither completed nor
        #     being decomposed — a dependency cycle among pending slices or a
        #     dangling dep the plan critic missed. No future round can make these
        #     ready, so dispatch them NOW to break the deadlock instead of
        #     churning failed slices until the dispatch ceiling (finding #5).
        done_ids = _slice_marker_ids(state.get("completed_slices", []) or [])
        failed_ids = _slice_marker_ids(failed)
        deadlocked = []
        for sl in pending:
            unsatisfied = [
                d for d in (sl.get("dependencies") or []) if d and d not in done_ids
            ]
            if unsatisfied and not any(d in failed_ids for d in unsatisfied):
                deadlocked.append(sl)
        if deadlocked:
            logger.warning(
                "[%s] IMPLEMENT route: %d pending slice(s) permanently blocked "
                "(dependency cycle or dangling dep, none resolvable by the "
                "decomposer) — dispatching to break deadlock rather than churn to "
                "the dispatch ceiling.",
                work_id, len(deadlocked),
            )
            ready = deadlocked

    # Flag-gate the editor architecture. The synthesis path is a single no-tool
    # node (synthesize → place), so it skips the plan_slice_implementer
    # "plan approach" pre-step the tool path uses; the tool path keeps the
    # two-step split that helps smaller models stop fixating on the slice JSON.
    pending_target = "plan_slice_implementer"
    try:
        if (cfg or SpineConfig.load()).implement_synthesis_placement:
            pending_target = "synthesis_implementer"
    except Exception:  # noqa: BLE001 — config issues fall back to the tool path
        pass

    for sl in ready:
        sends.append(Send(pending_target, {**base, "active_slice": sl}))
    for sl in failed:
        # The decomposer is already a no-tool structured-output call, so it
        # doesn't get a plan-before-do step.
        sends.append(Send("fallback_decomposer", {**base, "active_slice": sl}))

    logger.info(
        "[%s] IMPLEMENT route: pending=%d ready=%d failed=%d dispatched_total=%d "
        "— %d Send(s)",
        work_id, len(pending), len(ready), len(failed), dispatched, len(sends),
    )
    return sends


async def _dispatch_gate_node(
    state: ImplementSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Single barrier between the parallel slice fan-out and the router.

    Every dispatch node (``split_slices`` and each implementer/decomposer) feeds
    into this node via a *static* edge, so LangGraph collapses all parallel
    branches of a super-step into **one** ``dispatch_gate`` task. ``_route_slices``
    is wired only here, so the routing decision is computed exactly once per
    super-step on the fully-merged state.

    Without this barrier the conditional edge was attached to each fan-out node,
    so it was re-evaluated once per parallel branch. Sibling branches saw only a
    subset of each other's slice-list removes (partially-merged state), producing
    divergent decisions: one branch re-dispatched a slice another had already
    claimed, and — when the last slices completed together — one branch routed to
    ``synthesize_implementation`` while another re-dispatched, scheduling
    ``synthesize_implementation`` and ``save_artifacts`` into a colliding
    super-step. Both write the un-reduced ``artifacts_output`` channel, tripping
    ``InvalidUpdateError`` ("Can receive only one value per step", trace
    019f0193). This mirrors VERIFY's ``aggregate_verification`` barrier.
    """
    return {}


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
    single_no_plan = [
        sl
        for sl in pending
        if len([f for f in (sl.get("target_files") or []) if f]) < 2
        and not (sl.get("edit_plan") or [])
    ]
    if not multi and not single_no_plan:
        return {}

    from spine.agents.decomposer import enrich_slice, run_decomposer

    async def _decompose(parent: dict) -> list[dict] | None:
        try:
            subs = await run_decomposer(
                mode="PER_FILE",
                source_slice=parent,
                config=config,
                session_id=work_id,
            )
        except _ABORT_EXCEPTIONS:
            raise
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

    if single_no_plan:
        async def _enrich(sl: dict) -> dict:
            try:
                return await enrich_slice(
                    source_slice=sl,
                    config=config,
                    session_id=work_id,
                )
            except _ABORT_EXCEPTIONS:
                raise
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[%s] split_slices: enrich failed for %r: %s",
                    work_id,
                    sl.get("id"),
                    e,
                )
                return sl

        enriched_singles = await asyncio.gather(*[_enrich(sl) for sl in single_no_plan])
        for orig, enriched in zip(single_no_plan, enriched_singles):
            if enriched is not orig and (enriched.get("edit_plan") or []):
                remove_ids.append(orig.get("id"))
                adds.append(enriched)
                logger.info(
                    "[%s] split_slices: enriched slice=%r -> %d edit_plan entry(ies)",
                    work_id,
                    orig.get("id"),
                    len(enriched.get("edit_plan") or []),
                )

    if not remove_ids:
        return {}
    return {"pending_slices": {"remove": remove_ids, "add": adds}}


# A target file estimated above this many tokens is "large": reading it whole
# into the slice-implementer loop dominates a finite window, so we proactively
# steer the implementer to read narrow ranges. DynamicCompletionCap + eviction
# still enforce the hard window limit; this just avoids the costly first read.
_LARGE_FILE_TOKEN_BUDGET = 6000


# Per-symbol inline cap. A reference symbol can be a whole class (SpineConfig
# is ~1000 lines); inlining it verbatim would crowd the window worse than the
# survey we are replacing. Beyond this we inline the head and point the model at
# read_symbol for the rest — still anchored, still no blind survey.
_MAX_INLINE_SYMBOL_CHARS = 4000
_MAX_INLINE_SYMBOL_LINES = 60


def _index_ctx(state_or_root: Any) -> tuple[str | None, str]:
    """Return ``(db_path, workspace_root)`` for index-backed source inlining.

    Accepts either the subgraph state dict (reads ``workspace_root``) or a bare
    workspace-root string. ``db_path`` is ``None`` when the index is
    unavailable, in which case callers fall back to name-only references.
    """
    if isinstance(state_or_root, dict):
        workspace_root = state_or_root.get("workspace_root", ".")
    else:
        workspace_root = state_or_root or "."
    try:
        from spine.config import SpineConfig

        return SpineConfig.load().checkpoint_path, workspace_root
    except Exception:  # noqa: BLE001 — degrade to name-only references
        return None, workspace_root


def _inline_symbol_source(db_path: str | None, workspace_root: str, symbol: str) -> str:
    """Fenced current source of *symbol* from the index, or '' if unavailable.

    Truncated to a head slice for very large symbols so a whole class never
    dominates the prompt (the model can ``read_symbol`` for the remainder).
    """
    if not db_path or not symbol:
        return ""
    try:
        from spine.agents.tools.codebase_query import get_symbol_source

        src = get_symbol_source(db_path, workspace_root, symbol)
    except Exception:  # noqa: BLE001
        return ""
    if not src:
        return ""
    lines = src.splitlines()
    truncated = len(src) > _MAX_INLINE_SYMBOL_CHARS or len(lines) > _MAX_INLINE_SYMBOL_LINES
    if truncated:
        lines = lines[:_MAX_INLINE_SYMBOL_LINES]
        src = "\n".join(lines) + (
            f"\n# … truncated — read_symbol '{symbol}' for the full body."
        )
    return f"```python\n{src}\n```"


_MAX_INLINE_FILE_LINES = 500
_MAX_INLINE_FILE_CHARS = 28000


def _file_top_level_outline(text: str) -> str:
    """Top-level ``def``/``class`` anchors of a python file, one per line.

    Used when a target file is too large to inline whole: the synthesizer still
    needs to know which definitions ALREADY exist so it inserts against them
    instead of recreating (and clobbering) the file. '' for non-python or
    unparseable source.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return ""
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            out.append(f"class {node.name}  # line {node.lineno}")
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append(f"    {node.name}.{sub.name}  # line {sub.lineno}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(f"def {node.name}  # line {node.lineno}")
    return "\n".join(out)


_EXEMPLAR_MAX_LINES = 120
_EXEMPLAR_MAX_CHARS = 4500

_TEST_PATH_SEGMENTS = {"tests", "test", "__tests__"}


def _looks_like_test_file(rel: str) -> bool:
    parts = rel.replace("\\", "/").split("/")
    if any(p in _TEST_PATH_SEGMENTS for p in parts[:-1]):
        return True
    name = parts[-1]
    stem = name.rsplit(".", 1)[0]
    return (
        name.startswith("test_")
        or stem.endswith("Test")
        or stem.endswith(".test")
        or stem.endswith(".spec")
    )


def _sibling_exemplar_block(root, rel: str) -> str:
    """An EXISTING sibling file inlined as a style exemplar, or ''.

    When a slice CREATES a file, the editor has nothing concrete to imitate
    and local models hallucinate framework API shapes — run 984f9c8e
    invented `use Pest\\Pest; Pest::test(...)` (Pest tests are global
    ``test()`` calls); run 3313775a declared ``static $model`` on a Laravel
    factory (fatal redeclaration against the parent class) three identical
    gap cycles running, with sibling factories showing the correct shape
    two files away. The repo's own files are the ground truth for style:
    inline the smallest non-empty same-suffix sibling, bounded. Test files
    search up the whole test root; other files stay in their own directory
    (a factory mimics a factory, a migration a migration).
    """
    target = root / rel
    suffix = target.suffix
    if not suffix:
        return ""
    is_test = _looks_like_test_file(rel)
    directories = (
        [target.parent, *target.parent.parents] if is_test else [target.parent]
    )
    for directory in directories:
        try:
            if not directory.is_dir() or not directory.is_relative_to(root):
                break
            pool = (
                directory.rglob(f"*{suffix}") if is_test
                else directory.glob(f"*{suffix}")
            )
            candidates = sorted(
                (
                    f for f in pool
                    if f.is_file()
                    and f != target
                    and (
                        not is_test
                        or _looks_like_test_file(str(f.relative_to(root)))
                    )
                ),
                key=lambda f: f.stat().st_size,
            )
        except (OSError, ValueError):
            return ""
        for cand in candidates:
            try:
                text = cand.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            lines = text.splitlines()
            if len(lines) > _EXEMPLAR_MAX_LINES or len(text) > _EXEMPLAR_MAX_CHARS:
                continue
            if not text.strip():
                continue
            kind = (
                "PASSING test from this suite" if is_test
                else "file from this directory"
            )
            return (
                f"\nAn EXISTING {kind} — mimic its imports, structure, and "
                f"framework idioms EXACTLY "
                f"(`{cand.relative_to(root)}`):\n```\n{text}\n```"
            )
        if is_test and directory.name in _TEST_PATH_SEGMENTS:
            break  # searched the whole test root — stop walking up
    return ""


def _target_files_body(target_files: list[str], workspace_root: str) -> str:
    """Fenced CURRENT content of each target file, read live from the sandbox.

    The synthesis editor has no filesystem access, so unless the current file is
    put in front of it, it cannot see what the file already contains — including
    edits an EARLIER same-file slice in this feature just wrote. Same-file slices
    are serialized into consecutive waves (``serialize_file_overlapping_slices``)
    sharing one sandbox, so this happens routinely. Blind, the synthesizer
    regenerates the whole file and drops the prior slice's work: trace 019f2005
    had three ``config_view.py`` slices each rewrite the file — the embedding
    section came back as a reranker section, the phase-subagent section vanished,
    and one pass replaced the file with an unrelated page. Inlining the live
    content makes each serialized slice edit ADDITIVELY.

    Files under the cap are inlined whole; larger files degrade to a top-level
    symbol outline (existing anchors to insert against, never recreate) so a
    god-class cannot blow the token budget (memory: implement-godclass-*).
    Fully defensive: any read failure is reported as a to-be-created file.
    """
    from pathlib import Path

    files = [str(f).strip() for f in (target_files or []) if str(f).strip()]
    if not files:
        return ""
    root = Path(workspace_root or ".")
    blocks: list[str] = []
    for rel in files:
        try:
            text = (root / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            exemplar = _sibling_exemplar_block(root, rel)
            blocks.append(
                f"### `{rel}`\n(does not exist yet — this slice creates it; "
                f"synthesize the new file's full content)" + exemplar
            )
            continue
        if not text.strip():
            blocks.append(f"### `{rel}`\n(exists but is empty)")
            continue
        lines = text.splitlines()
        if len(text) <= _MAX_INLINE_FILE_CHARS and len(lines) <= _MAX_INLINE_FILE_LINES:
            blocks.append(
                f"### `{rel}` — current content ({len(lines)} lines); "
                f"PRESERVE it, edit in place\n```python\n{text}\n```"
            )
            continue
        outline = _file_top_level_outline(text)
        if outline:
            blocks.append(
                f"### `{rel}` — {len(lines)} lines, too large to inline. These "
                f"top-level definitions ALREADY EXIST; do NOT recreate them, "
                f"insert_after / replace them by name:\n```\n{outline}\n```"
            )
        else:
            head = "\n".join(lines[:_MAX_INLINE_FILE_LINES])
            blocks.append(
                f"### `{rel}` — head of {len(lines)}-line file (rest omitted; "
                f"PRESERVE it):\n```python\n{head}\n# … truncated\n```"
            )
    return "\n\n".join(blocks)


def _edit_plan_body(active_slice: dict, db_path: str | None, workspace_root: str) -> str:
    """Inner body for the targeted-edit block — '' when the slice has no plan.

    Each entry states the file, the anchor symbol, the action, and the precise
    intent, followed by the symbol's CURRENT source inlined from the index so
    the implementer edits from what is in front of it instead of surveying to
    rediscover the edit site (the api.py 89×-read spiral, trace 019ef2ae).
    """
    plan = active_slice.get("edit_plan") or []
    if not plan:
        return ""
    blocks: list[str] = []
    for i, h in enumerate(plan, start=1):
        if not isinstance(h, dict):
            h = getattr(h, "__dict__", {}) or {}
        file = h.get("file", "")
        symbol = h.get("symbol", "")
        action = h.get("action", "") or h.get("mode", "")
        intent = h.get("intent", "")
        header = f"{i}. {file}"
        if symbol:
            header += f" — `{symbol}`"
        if action:
            header += f" ({action})"
        parts = [header]
        if intent:
            parts.append(f"   intent: {intent}")
        src = _inline_symbol_source(db_path, workspace_root, symbol)
        if src:
            label = (
                "current source (edit this with ast_edit replace)"
                if action == "replace"
                else "current source of the anchor"
            )
            parts.append(f"   {label}:\n{src}")
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


def _reference_symbols_body(
    active_slice: dict, db_path: str | None, workspace_root: str
) -> str:
    """Inner body for the reference-symbols block — '' when the slice has none.

    Existing definitions the slice's code calls/extends/mimics, each with its
    current source inlined so the implementer never surveys to find them. A
    symbol whose source cannot be resolved degrades to a name the model may
    ``read_symbol`` on demand.
    """
    refs = active_slice.get("reference_symbols") or []
    if not refs:
        return ""
    blocks: list[str] = []
    rendered: set[str] = set()
    for r in refs:
        if r in rendered:
            continue
        rendered.add(r)
        src = _inline_symbol_source(db_path, workspace_root, r)
        if src:
            blocks.append(f"### `{r}`\n{src}")
        else:
            blocks.append(f"### `{r}`\n(source unavailable — read_symbol it if needed)")
    # Inline the constructor of every referenced CLASS as well. A slice that
    # references `Class.method` almost always has to INSTANTIATE the class
    # (tests especially), and with only the method's source in the prompt the
    # editor guesses constructor kwargs — run 86a3ab17 burned three verify
    # cycles inventing root_dir/base_dir/root for ArtifactStore(base_path).
    owners: dict[str, str] = {}
    for r in refs:
        parts = (r or "").strip().split("(", 1)[0].split(".")
        for i, seg in enumerate(parts):
            if seg and seg[0].isupper():
                owners.setdefault(".".join(parts[: i + 1]), seg)
                break
    for owner, bare in owners.items():
        init_sym = f"{bare}.__init__"
        if init_sym in rendered or owner in rendered or bare in rendered:
            continue
        src = _inline_symbol_source(
            db_path, workspace_root, f"{owner}.__init__"
        ) or _inline_symbol_source(db_path, workspace_root, init_sym)
        if src:
            rendered.add(init_sym)
            blocks.append(
                f"### `{init_sym}` (constructor — call it with EXACTLY these "
                f"parameters)\n{src}"
            )
    return "\n\n".join(blocks)


_DEP_FILES_MAX = 3
_DEP_FILE_MAX_CHARS = 4000


def _dependency_files_body(
    state: ImplementSubgraphState, active_slice: dict, workspace_root: str
) -> str:
    """LIVE source of files created by the slices this slice depends on.

    Files a sibling slice created THIS RUN are not in the codebase index, so
    their reference_symbols are scrubbed as phantoms and nothing inlines
    their real API — the editor invents one instead (probe 22/f9aac445: the
    test called generateName()/generateUuid() on a factory, two waves
    earlier, that defines neither; probe 21's model/migration table mismatch
    is the same blindness). Waves are dependency-ordered, so a dependency's
    files exist on disk by the time this slice runs — inline them, bounded.
    Returns '' when the slice has no dependencies or on any load problem.
    """
    dep_ids = {str(d) for d in (active_slice.get("dependencies") or []) if d}
    if not dep_ids:
        return ""
    from pathlib import Path

    plan_dir = state.get("plan_path")
    if not plan_dir:
        return ""
    try:
        plan = json.loads(
            (Path(workspace_root) / plan_dir / "plan.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, ValueError):
        return ""
    own_files = {str(t) for t in (active_slice.get("target_files") or []) if t}
    blocks: list[str] = []
    for sl in plan.get("feature_slices") or []:
        if not isinstance(sl, dict) or str(sl.get("id")) not in dep_ids:
            continue
        for rel in sl.get("target_files") or []:
            rel = str(rel).strip()
            if not rel or rel in own_files:
                continue  # own target files are inlined by _target_files_body
            try:
                text = (Path(workspace_root) / rel).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if len(text) > _DEP_FILE_MAX_CHARS:
                text = text[:_DEP_FILE_MAX_CHARS] + "\n… (truncated)"
            blocks.append(
                f"### `{rel}` (created by dependency slice '{sl.get('id')}' "
                "THIS RUN — call EXACTLY the API defined here; do NOT invent "
                f"methods on it)\n```\n{text}\n```"
            )
            if len(blocks) >= _DEP_FILES_MAX:
                return "\n\n".join(blocks)
    return "\n\n".join(blocks)


def _scrub_phantom_refs(active_slice: dict, work_id: str = "?") -> dict:
    """Return a copy of *active_slice* with non-existent reference_symbols removed.

    Also clears edit_plan[].symbol values not found in the local codebase index
    and demotes their mode from 'ast_edit' to '' so the implementer falls back
    to patch/full_replace rather than a guaranteed-failing symbol lookup.

    Queries only the local spine.db (no MCP round-trip). No-op when the index
    is unavailable or the slice has no reference_symbols / edit_plan.
    """
    needs_refs = bool(active_slice.get("reference_symbols"))
    needs_plan = any(h.get("symbol") for h in (active_slice.get("edit_plan") or []))
    if not needs_refs and not needs_plan:
        return active_slice

    try:
        from spine.agents.tools.codebase_query import find_symbol
        from spine.config import SpineConfig
        db_path = SpineConfig.load().checkpoint_path
    except Exception:
        return active_slice

    slice_id = active_slice.get("id", "?")
    result = dict(active_slice)

    if needs_refs:
        good: list[str] = []
        for sym in active_slice.get("reference_symbols") or []:
            try:
                exists = find_symbol(db_path, sym) is not None
            except Exception:
                good.append(sym)
                continue
            if exists:
                good.append(sym)
            else:
                logger.warning(
                    "[%s] scrub_phantom_refs: dropping reference_symbol %r from slice %r "
                    "(not in codebase index — new method or planner forward-reference)",
                    work_id, sym, slice_id,
                )
        result["reference_symbols"] = good

    if needs_plan:
        new_plan = []
        for hint in active_slice.get("edit_plan") or []:
            h = dict(hint)
            sym = h.get("symbol", "")
            if sym:
                try:
                    exists = find_symbol(db_path, sym) is not None
                except Exception:
                    exists = True
                if not exists:
                    logger.warning(
                        "[%s] scrub_phantom_refs: clearing edit_plan symbol %r in slice %r "
                        "(not in codebase index)",
                        work_id, sym, slice_id,
                    )
                    h["symbol"] = ""
                    if h.get("mode") == "ast_edit":
                        h["mode"] = ""
            new_plan.append(h)
        result["edit_plan"] = new_plan

    return result


def _large_file_directive(active_slice: dict, workspace_root: str) -> str:
    """Read-narrow directive when a slice's target file is large, else ''.

    Best-effort: measures each target file's token size on disk and, if any is
    large, returns a directive telling the implementer to navigate with
    ``codebase_query`` and read only the relevant line ranges (offset/limit)
    rather than the whole file. Any measurement failure returns '' silently —
    the window safety nets (DynamicCompletionCap, TokenBudgetCompactor) still
    apply regardless.
    """
    from pathlib import Path

    from spine.agents._tokens import count_tokens

    # When the planner supplied an edit_plan, the implementer applies edits by
    # symbol via `ast_edit` (which locates the symbol itself — no whole-file or
    # ranged reading needed), and `codebase_query` has been withheld from its
    # tool surface. This read-narrow directive steers toward ranged reads + a
    # `codebase_query` the agent no longer has, which is exactly the
    # read-the-file-12-times survey loop we want to kill — so suppress it
    # entirely once targeting is in hand.
    if active_slice.get("edit_plan"):
        return ""

    targets = [f for f in (active_slice.get("target_files") or []) if f]
    big: list[tuple[str, int]] = []
    for rel in targets:
        try:
            text = (Path(workspace_root) / rel).read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001 — file may be new/binary/unreadable
            continue
        toks = count_tokens(text)
        if toks > _LARGE_FILE_TOKEN_BUDGET:
            big.append((rel, toks))
    if not big:
        return ""
    listed = "\n".join(f"- {p} (~{t:,} tokens)" for p, t in big)
    return (
        "\n\n## Large file(s) — read in RANGES, not whole\n"
        "These target files are large; reading them whole will crowd the "
        "context window and can stall the run:\n"
        f"{listed}\n"
        "Use `codebase_query` (find_symbol / get_source) to locate the exact "
        "symbol you need, then read ONLY that line range with start_line/"
        "end_line. Do NOT read these files from line 1 in full."
    )


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
            "Sibling files are still being built in later steps, so slice-level/"
            "integration tests may not pass yet — that is expected and is NOT a "
            "reason to skip your work. You MUST still apply this file's edits "
            "now and make it correct and self-consistent; only a "
            "sibling-dependent check may be noted as pending."
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
    # Strip reference_symbols that don't exist in the codebase index before
    # they reach the implementer. Plan-phase outputs routinely include forward
    # references to methods that will be created by a sibling slice (e.g.
    # UIApi.update_phase_providers before slice 0 adds it). The implementer
    # treats reference_symbols as "read these NOW" — a missing symbol triggers
    # a futile 70-turn search loop that consumes the entire token budget
    # (GLM-5.2 trace 019eec68). Scrubbing them here keeps the menu honest.
    active_slice = _scrub_phantom_refs(active_slice, work_id)
    slice_id = active_slice.get("id", "unknown")
    title = active_slice.get("title", "")
    target_files = active_slice.get("target_files") or []
    criteria = active_slice.get("acceptance_criteria") or []
    description = active_slice.get("description", "")

    crit_lines = "\n".join(f"- {c}" for c in criteria) if criteria else "(none)"
    file_lines = "\n".join(f"- {p}" for p in target_files) if target_files else "(none listed)"

    # When the slice already has an edit_plan, the work is fully specified — the
    # ordered edits plus the inlined source ARE the plan. Running the LLM planner
    # here adds nothing and actively harms: it does not know the editor receives
    # pre-loaded source, so its `approach` routinely opens with "explore the
    # repository structure to locate X" — the exact mass-survey the inlining was
    # meant to eliminate. Synthesise a deterministic edit-first directive instead
    # (no LLM call, no exploration foot-gun, one fewer slow local round-trip).
    if active_slice.get("edit_plan"):
        directive = SubagentDirective(
            approach=(
                "Apply the edits in <edit_plan> in order with read_edit_lint — "
                "ast_edit by symbol for replace/insert, patch for snippet edits. "
                "The current source of every target and reference symbol is "
                "inlined in your prompt; do NOT explore, search, or read files to "
                "locate them. One edit per entry; you are done only when each "
                'edit returns status="ok".'
            ),
            target_files=list(target_files),
            acceptance=list(criteria),
        )
        logger.info(
            "[%s] plan_slice_implementer: slice=%r — deterministic edit-first "
            "directive (edit_plan has %d entr(ies))",
            work_id, slice_id, len(active_slice.get("edit_plan") or []),
        )
    else:
        # No edit_plan: the planner may legitimately suggest a read of an
        # inlined reference, but must not send the editor on a broad survey —
        # arbitrary/whole-file reads are disabled and reference source is inlined.
        db_path, workspace_root = _index_ctx(state)
        refs_body = _reference_symbols_body(active_slice, db_path, workspace_root)
        plan_body = _edit_plan_body(active_slice, db_path, workspace_root)
        refs_md = f"\n\n## Reference symbols (source inlined)\n{refs_body}" if refs_body else ""
        plan_md = f"\n\n## Targeted edits (from the planner)\n{plan_body}" if plan_body else ""
        guidance = (
            "The implementer ALREADY has the target file(s) and every reference "
            "symbol's current SOURCE inlined in its prompt, and arbitrary "
            "whole-file reads are DISABLED — it cannot and must not survey the "
            "repo. Make your `approach` edit-first: which symbols to change and "
            "in what order. Do NOT tell it to explore, locate, examine, search, "
            "or read files to orient — that is already done; do NOT put "
            "read/search/explore steps in tool_calls_to_make."
            if refs_md
            else "Keep the approach concrete and edit-focused; avoid broad exploration."
        )
        task = (
            f"Plan how to implement slice {slice_id!r} (title: {title!r}). The do "
            "node applies atomic edits with read_edit_lint.\n\n"
            f"{guidance}\n\n"
            f"## Description\n{description or '(none)'}\n\n"
            f"## Target files\n{file_lines}\n\n"
            f"## Acceptance criteria\n{crit_lines}"
            f"{refs_md}"
            f"{plan_md}"
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

    # Failure-driven escalation: a slice that has survived one or more FALLBACK
    # re-decompositions carries a higher ``_decompose_depth``. Reuse it as the
    # escalation rung so the existing decompose-on-failure loop doubles as the
    # model ladder (depth 0 → primary, 1 → medium, 2 → strong). No new state or
    # control loop — purely a function of the checkpointed counter, so replay
    # stays deterministic. No-op unless an ``escalation`` ladder is configured.
    escalation_level = int(active_slice.get("_decompose_depth", 0) or 0)

    logger.info(
        "[%s] slice_implementer: slice=%r title=%r escalation_level=%d",
        work_id,
        slice_id,
        active_slice.get("title", ""),
        escalation_level,
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

        from spine.agents.synthesis_budget import synthesis_completion_cap
        from spine.config import SpineConfig

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
            # surface (read_edit_lint + codebase_query wrapper via
            # subagents.py) — they live in ``extra_tools`` above.
            #
            # Implement turns are tool calls (edit payloads), not essays.
            # Without a clamp the request inherits the global
            # max_completion_tokens (30K) and a finite-window model 400s
            # once the conversation grows past window - 30K (trace
            # 019eb502: 30,001-token prompt + 30K requested vs 60K window).
            completion_token_cap=synthesis_completion_cap(
                PhaseName.IMPLEMENT.value,
                phase_cap=SpineConfig.load().implement_max_completion_tokens,
            ),
            escalation_level=escalation_level,
        )

        # Trim the slice JSON to scannable metadata only. The reference symbols
        # and edit_plan get dedicated blocks below (with source inlined), so
        # repeating them here would render each datum THREE times (JSON +
        # directive + section) — the triplication that bloated every turn and
        # let the three renderings drift. The run-on `execution_requirements`
        # string is dropped: the edit_plan is the authoritative "what to change".
        title = active_slice.get("title", "")
        public_slice = {
            k: active_slice.get(k)
            for k in ("id", "title", "description", "target_files", "acceptance_criteria")
            if active_slice.get(k) not in (None, "", [], {})
        }
        slice_json = json.dumps(public_slice, indent=2, ensure_ascii=False, default=str)

        db_path, workspace_root = _index_ctx(state)
        refs_body = _reference_symbols_body(active_slice, db_path, workspace_root)
        dep_body = _dependency_files_body(state, active_slice, workspace_root)
        if dep_body:
            refs_body = (refs_body + "\n\n" + dep_body) if refs_body else dep_body
        plan_body = _edit_plan_body(active_slice, db_path, workspace_root)

        # Directive: approach + notes only — target_files / acceptance / tool
        # calls already live in <findings> / <edit_plan>, so the full directive
        # would state them a third time.
        directive_block = format_directive_for_prompt(
            directive_from_state(dict(state), "active_slice_directive"),
            compact=True,
        )

        # On a gap-fix rework, the slice's verify-gap remediation rides along —
        # same channel as the synthesis path (run 019f20a5: without it the
        # rework regenerates the code that already failed verification).
        gaps_body = _gap_fixes_body(state, active_slice)

        # Hostage layout: data blocks first, plain-text instruction at the tail.
        objective = f"Implement slice: {slice_id} — {title}" if title else f"Implement slice: {slice_id}"
        blocks = xml_blocks(
            (Tag.OBJECTIVE, objective),
            (Tag.FINDINGS, f"```json\n{slice_json}\n```"),
            (Tag.REFERENCE_SYMBOLS, refs_body),
            (Tag.EDIT_PLAN, plan_body),
            (Tag.CRITIC_FEEDBACK, gaps_body),
        ) + ("\n\n" + directive_block if directive_block else "")

        if plan_body:
            instruction = (
                "Apply the edits in <edit_plan> NOW with read_edit_lint — "
                "ast_edit (by symbol) to replace or insert at an anchor, patch "
                "for snippet edits, full_replace for a new file. The current "
                "source of every target and reference symbol is inlined above, "
                "so do NOT survey or re-read the files. Make one edit per "
                "<edit_plan> entry; you are NOT done until each returns "
                'status="ok". Never report success without an applied edit.'
            )
        else:
            instruction = (
                "Implement the slice in <findings>. The reference symbols above "
                "have their source inlined — read_symbol one only if you need a "
                "definition not shown. Apply edits with read_edit_lint "
                "(ast_edit / patch / full_replace); make only the changes the "
                'slice describes, and you are NOT done until each edit returns '
                'status="ok".'
            )
        if gaps_body:
            instruction += (
                " This is a VERIFICATION REWORK: a previous implementation of "
                "this slice failed the checks in <critic_feedback> — your "
                "edits MUST resolve exactly those failures."
            )
        instruction += _subslice_context(active_slice)
        instruction += _large_file_directive(active_slice, workspace_root)

        prompt = hostage_layout(blocks, instruction)

        result = await ainvoke_with_retry(
            agent,
            {"messages": [{"role": "user", "content": prompt}]},
            phase_name="implement-slice",
            work_id=work_id,
        )
        slice_result = _extract_slice_result(result, slice_id)

    except _ABORT_EXCEPTIONS:
        raise
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

    # Tick the dispatch counter so _route_slices can enforce its ceiling.
    return {**_slice_result_to_update(active_slice, slice_result), "slice_dispatch_count": 1}


def _slice_result_to_update(
    active_slice: dict, slice_result: dict
) -> dict[str, Any]:
    """Map a SliceResult onto the dispatch-loop state update.

    Shared by the tool-using ``_slice_implementer_node`` and the
    ``_synthesis_implementer_node`` so both honour the same
    pending→completed/failed contract and the same ``_sibling_queue`` promotion
    (one parent file lands, the next is re-queued) — a divergence here would let
    one editor strand sub-slices the other handles.
    """
    slice_id = active_slice.get("id", "unknown")
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
        # Distinguish a context-overflow from a logic failure so the fallback
        # decomposer narrows scope (region-scoped reads) instead of re-running
        # the same whole-file work that overflowed (trace 019ece87).
        "_overflow": _is_context_overflow(failure_traceback)
        or _is_context_overflow(" ".join(issues)),
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
    """Normalize status to a valid value: implemented, partial, or blocked.

    An unrecognized or non-string status falls through to ``blocked`` (NOT
    ``implemented``): a slice whose status we cannot trust must not be recorded
    as silently done — the verify phase / orchestrator needs to see it was not
    completed (finding #5b).
    """
    if not isinstance(status, str):
        return "blocked"
    status_lower = status.lower().strip()
    if status_lower in ("implemented", "partial", "blocked"):
        return status_lower
    if status_lower in ("in_progress", "in", "running", "done"):
        return "implemented"
    if status_lower in ("failed", "error", "not_implemented"):
        return "blocked"
    return "blocked"


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
            # The subagent emitted a final message that is not parseable as a
            # SliceResult. We cannot confirm the slice was done — record it as
            # blocked rather than fabricating success (finding #5b).
            return {
                "slice_name": slice_id,
                "status": "blocked",
                "files_modified": [],
                "files_created": [],
                "test_results": "",
                "issues": ["Subagent output was not a parseable SliceResult"],
            }

    return {
        "slice_name": slice_id,
        "status": "blocked",
        "files_modified": [],
        "files_created": [],
        "test_results": "(no output from subagent)",
        "issues": ["Subagent produced no output"],
    }


# ── synthesis_implementer node (no tools: synthesize → place) ───────────


def _format_placement_failure(f: dict) -> str:
    """Render one placement failure, preserving ``target``/``next_action``.

    The ast_edit creation-anchor guard builds these fields so a weak model can
    self-correct in one step instead of re-emitting the same broken anchor.
    Both consumers of failure text — the same-cycle synthesis retry
    (``_placement_feedback``) and the persisted slice ``issues`` that reach
    later rework cycles via implementation.md/gap_plan (``_placement_to_slice_result``)
    — must go through this one formatter, or the two drift: 019f1c10 fixed the
    retry path, but 019f3f7d showed the SAME anchor mistake recur across three
    rework cycles because ``_placement_to_slice_result`` had its own separate
    (and older) format that silently dropped target/next_action, leaving later
    cycles no way to learn the fix.
    """
    loc = f.get("file", "?")
    sym = f.get("symbol", "")
    if sym:
        loc += f" `{sym}`"
    detail = (f.get("detail", "") or "").strip().replace("\n", " ")
    line = f"{loc} ({f.get('status', 'error')}): {detail[:300]}"
    if f.get("target"):
        line += f" [target: {f['target']}]"
    if f.get("next_action"):
        line += f" [DO THIS: {f['next_action']}]"
    return line


def _placement_feedback(placement: Any) -> str:
    """Render placement failures as compact feedback for the synthesis retry."""
    return "\n".join(f"- {_format_placement_failure(f)}" for f in placement.failures)


def _placement_to_slice_result(
    slice_id: str, placement: Any, summary: str = ""
) -> dict:
    """Convert a PlacementResult into the SliceResult dict the contract expects.

    ``implemented`` when every edit placed, ``partial`` when some did and some
    failed, ``blocked`` when nothing placed. Files touched come from the applied
    edits; failures become ``issues`` so synthesis (and any fallback decompose)
    can see exactly what the linter rejected — including the anchor guard's
    ``target``/``next_action`` (via ``_format_placement_failure``), so a later
    rework cycle can learn the fix instead of repeating the same broken anchor.
    """
    n_ok = placement.n_applied
    n_fail = placement.n_failures
    if n_ok and not n_fail:
        status = "implemented"
    elif n_ok and n_fail:
        status = "partial"
    else:
        status = "blocked"
    files = sorted({a.get("file") for a in placement.applied if a.get("file")})
    issues = [_format_placement_failure(f) for f in placement.failures]
    return {
        "slice_name": slice_id,
        "status": status,
        "files_modified": files,
        "files_created": [],
        "test_results": summary or f"synthesis placed {n_ok} edit(s), {n_fail} failed",
        "issues": issues,
    }


async def _synthesis_implementer_node(
    state: ImplementSubgraphState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """Synthesis + placement editor — the no-tool IMPLEMENT path (flag-gated).

    Replaces the tool-using ``_slice_implementer_node`` when
    ``implement_synthesis_placement`` is set. A structured-output call with no
    filesystem tools synthesizes complete, symbol-anchored edits from the
    inlined source already in the prompt; placement applies them deterministically
    through ``ReadEditLintTool`` (lint is the oracle). The editor cannot read, so
    it cannot survey-spiral. Synthesis is side-effect-free, so
    ``implement_synthesis_variants > 1`` samples N candidates and keeps the one
    that applies + lints cleanest (Score + KeepBest). Honours the same
    pending→completed/failed contract and ``_sibling_queue`` promotion as the
    tool path via ``_slice_result_to_update``.
    """
    from spine.agents.synthesis_implementer import (
        place_best_candidate,
        synthesize_slice_code,
    )
    from spine.config import SpineConfig

    work_id = state.get("work_id", "unknown")
    active_slice: dict = state.get("active_slice") or {}
    slice_id = active_slice.get("id", "unknown")
    title = active_slice.get("title", "")

    # Failure-driven escalation rung — see _slice_implementer_node. Reuses the
    # FALLBACK re-decomposition depth so the synthesis path escalates its model
    # on the same schedule (no-op unless an escalation ladder is configured).
    escalation_level = int(active_slice.get("_decompose_depth", 0) or 0)

    logger.info(
        "[%s] synthesis_implementer: slice=%r title=%r escalation_level=%d",
        work_id, slice_id, title, escalation_level,
    )

    try:
        cfg = SpineConfig.load()
        variants = max(1, int(cfg.implement_synthesis_variants))

        # Same grounding as the tool path: drop phantom anchors, then inline the
        # current source of every target + reference symbol so synthesis rewrites
        # from what is in front of it (never surveys to rediscover it).
        active_slice = _scrub_phantom_refs(active_slice, work_id)
        db_path, workspace_root = _index_ctx(state)
        public_slice = {
            k: active_slice.get(k)
            for k in ("id", "title", "description", "target_files", "acceptance_criteria")
            if active_slice.get(k) not in (None, "", [], {})
        }
        slice_json = json.dumps(public_slice, indent=2, ensure_ascii=False, default=str)
        refs_body = _reference_symbols_body(active_slice, db_path, workspace_root)
        dep_body = _dependency_files_body(state, active_slice, workspace_root)
        if dep_body:
            refs_body = (refs_body + "\n\n" + dep_body) if refs_body else dep_body
        plan_body = _edit_plan_body(active_slice, db_path, workspace_root)
        target_files = [f for f in (active_slice.get("target_files") or []) if f]
        # Inline the CURRENT (live) content of every target file so a serialized
        # same-file slice edits ADDITIVELY on top of the prior slice's work
        # instead of blindly regenerating the file and clobbering it (trace
        # 019f2005). Without this the no-tool synthesizer cannot see what the
        # file already holds.
        files_body = _target_files_body(target_files, workspace_root)
        # On a gap-fix rework, inline this slice's verify-gap remediation —
        # the no-tool editor cannot read gap_plan.json, and without the gaps
        # in-prompt it regenerates the exact failing code (run 019f20a5).
        gaps_body = _gap_fixes_body(state, active_slice)
        # Final mile: a handful of failing criteria on a mostly-green slice ->
        # smallest-possible edits only, and placement prefers the smallest
        # clean candidate (run 019f25f5: wholesale regeneration at 3/25
        # failing scored worse twice and the budget expired).
        final_mile = _final_mile_fails(state, active_slice)
        if final_mile:
            # WARNING, not info: the run log surfaces WARNING+, and mode
            # engagement is exactly what a convergence post-mortem needs
            # (run 019f82b1 was mis-diagnosed as "final-mile never engaged"
            # because this line was invisible at info level).
            logger.warning(
                "[%s] synthesis_implementer: FINAL MILE for %r - %d failing "
                "criteria, minimal-edit mode",
                work_id, slice_id, len(final_mile),
            )

        candidates = await synthesize_slice_code(
            final_mile_fails=final_mile,
            slice_json=slice_json,
            refs_body=refs_body,
            plan_body=plan_body,
            files_body=files_body,
            gaps_body=gaps_body,
            config=config,
            session_id=work_id,
            n=variants,
            escalation_level=escalation_level,
        )
        winner, placement = place_best_candidate(
            candidates,
            workspace_root=workspace_root,
            target_files=target_files,
            prefer_minimal=bool(final_mile),
        )

        # One corrective retry: feed the exact lint failures back to synthesis so
        # it repairs them, then re-place. Keep whichever attempt placed cleaner.
        if placement.n_failures:
            retry = await synthesize_slice_code(
                final_mile_fails=final_mile,
                slice_json=slice_json,
                refs_body=refs_body,
                plan_body=plan_body,
                files_body=files_body,
                gaps_body=gaps_body,
                config=config,
                session_id=work_id,
                n=1,
                feedback=_placement_feedback(placement),
                escalation_level=escalation_level,
            )
            if retry:
                winner2, placement2 = place_best_candidate(
                    retry, workspace_root=workspace_root, target_files=target_files
                )
                if placement2.score() > placement.score():
                    winner, placement = winner2, placement2

        summary = getattr(winner, "summary", "") if winner else ""
        slice_result = _placement_to_slice_result(slice_id, placement, summary)
        logger.info(
            "[%s] synthesis_implementer: slice=%r status=%s applied=%d failed=%d",
            work_id, slice_id, slice_result["status"],
            placement.n_applied, placement.n_failures,
        )

    except _ABORT_EXCEPTIONS:
        raise
    except Exception as e:
        logger.error(
            "[%s] synthesis_implementer failed for %r: %s",
            work_id, slice_id, e, exc_info=True,
        )
        slice_result = {
            "slice_name": slice_id,
            "status": "blocked",
            "files_modified": [],
            "files_created": [],
            "test_results": f"Synthesis error: {e}",
            "issues": [str(e)],
        }

    # Tick the dispatch counter so _route_slices can enforce its ceiling.
    return {**_slice_result_to_update(active_slice, slice_result), "slice_dispatch_count": 1}


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
    max_depth = _max_decompose_depth()

    # Every terminal path below removes the slice from ``failed_slices`` and
    # records a blocked SliceResult in ``completed_slices``. The removal is
    # mandatory for termination: ``_route_slices`` re-dispatches every failed
    # slice and only reaches synthesis once ``failed_slices`` is empty, so a
    # slice left in place loops back into this node forever. Recording the
    # blocked result keeps the failure visible in synthesis instead of silently
    # vanishing (the pre-fix cap path returned ``{}`` — neither terminating nor
    # reporting).
    if depth >= max_depth:
        logger.warning(
            "[%s] fallback_decomposer: slice=%r hit depth cap (%d/%d) — marking blocked",
            work_id,
            slice_id,
            depth,
            max_depth,
        )
        return {
            "failed_slices": {"remove": [slice_id]},
            "completed_slices": {"add": [_failed_to_blocked(active_slice)]},
        }

    try:
        overflow = bool(active_slice.get("_overflow"))
        micro_slices = await run_decomposer(
            mode="FALLBACK",
            failed_slice=active_slice,
            error_traceback=active_slice.get("_failure_traceback", ""),
            overflow_hint=overflow,
            config=config,
            session_id=work_id,
        )
    except _ABORT_EXCEPTIONS:
        raise
    except Exception as e:
        logger.error(
            "[%s] fallback_decomposer failed for %r: %s — marking blocked",
            work_id,
            slice_id,
            e,
            exc_info=True,
        )
        blocked = _failed_to_blocked({**active_slice, "_failure_traceback": str(e)})
        return {
            "failed_slices": {"remove": [slice_id]},
            "completed_slices": {"add": [blocked]},
        }

    if not micro_slices:
        logger.warning(
            "[%s] fallback_decomposer: slice=%r produced 0 micro-slices — marking blocked",
            work_id,
            slice_id,
        )
        return {
            "failed_slices": {"remove": [slice_id]},
            "completed_slices": {"add": [_failed_to_blocked(active_slice)]},
        }

    next_depth = depth + 1
    for sl in micro_slices:
        sl["_decompose_depth"] = next_depth

    # When the micro-slices all target the SAME single file (the common case —
    # a failed single-file slice re-sliced into region-scoped pieces), dispatch
    # them through one sibling-queue chain instead of fanning out N parallel
    # implementers. N parallel branches on one file each re-read the whole file
    # into a fresh context (trace 019ed3dc: one 1.6k-line file read 4× in 60s)
    # and race on edits to it; a sequential chain reads per step and lets each
    # micro build on the last.
    micro_files = {
        f for sl in micro_slices for f in (sl.get("target_files") or []) if f
    }
    if len(micro_files) == 1 and len(micro_slices) > 1:
        head = _build_subslice_chain(active_slice, micro_slices)
        # _build_subslice_chain stamps the PARENT's depth onto every member;
        # re-stamp the incremented depth so the fallback cap still advances.
        head["_decompose_depth"] = next_depth
        for q in head.get("_sibling_queue", []):
            q["_decompose_depth"] = next_depth
        adds: list[dict] = [head]
        logger.info(
            "[%s] fallback_decomposer: slice=%r -> %d micro-slice(s) chained "
            "sequentially on %s at depth=%d",
            work_id,
            slice_id,
            len(micro_slices),
            next(iter(micro_files)),
            next_depth,
        )
    else:
        adds = micro_slices
        logger.info(
            "[%s] fallback_decomposer: slice=%r -> %d micro-slice(s) at depth=%d",
            work_id,
            slice_id,
            len(micro_slices),
            next_depth,
        )
    return {
        "failed_slices": {"remove": [slice_id]},
        "pending_slices": {"add": adds},
    }


# ── synthesize_implementation node ──────────────────────────────────────


def _failed_to_blocked(s: dict) -> dict:
    """Coerce a failed-slice dict into the SliceResult shape expected by synthesis."""
    issues = (
        ["exceeded fallback depth"]
        if s.get("_decompose_depth", 0) >= _max_decompose_depth()
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

# The implementation report must cross the sandbox→durable boundary whole:
# gap_plan reads implementation.md, restart/UI read both files, and the sandbox
# worktree holding the full copy is torn down at finalize. A 500-char preview
# leaves the persisted report truncated mid-record. The mapper's
# _FULL_PERSIST_ARTIFACTS keeps these untruncated through parent state.
_FULL_REPORT_FILES = ("implementation.md", "implementation.json")
_MAX_FULL_REPORT_CHARS = 200_000


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
        full_fidelity=_FULL_REPORT_FILES,
        max_full_chars=_MAX_FULL_REPORT_CHARS,
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
    "synthesis_implementer": "synthesis_implementer",
    "fallback_decomposer": "fallback_decomposer",
    "synthesize_implementation": "synthesize_implementation",
}


def build_implement_subgraph() -> Any:
    """Build the IMPLEMENT phase subgraph (SmallCode dispatch loop).

    Nodes:
    1. ``split_slices`` — runs once at START, replacing each multi-file slice
       with the head of a single-file sub-slice chain (PER_FILE decomposition).
    2. ``dispatch_gate`` — single barrier; the sole holder of ``_route_slices``.
    3. ``slice_implementer`` — per-slice subagent with restricted tools; on
       success it promotes the parent's next file from ``_sibling_queue``.
    4. ``fallback_decomposer`` — micro-slices a failed slice.
    5. ``synthesize_implementation`` — writes implementation artifacts.
    6. ``save_artifacts`` — scans disk and materializes to state.

    ``split_slices`` and every fan-out node feed ``dispatch_gate`` via STATIC
    edges, so LangGraph collapses each super-step's parallel branches into one
    gate task. ``_route_slices`` is wired ONLY on the gate, so it re-evaluates
    exactly once per super-step on fully-merged state until both
    ``pending_slices`` and ``failed_slices`` are empty. (Wiring the router on
    each fan-out node instead re-evaluated it per parallel branch on
    partially-merged state — the duplicate-dispatch / artifacts_output-collision
    bug in trace 019f0193.)
    """
    builder = StateGraph(ImplementSubgraphState)

    builder.add_node("split_slices", _split_slices_node)
    builder.add_node("dispatch_gate", _dispatch_gate_node)
    builder.add_node("plan_slice_implementer", _plan_slice_implementer_node)
    builder.add_node("slice_implementer", _slice_implementer_node)
    builder.add_node("synthesis_implementer", _synthesis_implementer_node)
    builder.add_node("fallback_decomposer", _fallback_decomposer_node)
    builder.add_node("synthesize_implementation", _synthesize_implementation_node)
    builder.add_node("save_artifacts", _save_implement_artifacts)

    # START → split_slices (per-file decomposition) → dispatch_gate → route.
    # The split node runs once, replacing multi-file slices with single-file
    # chains before any dispatch.
    builder.add_edge(START, "split_slices")

    # Every dispatch node feeds the single ``dispatch_gate`` barrier via a STATIC
    # edge. LangGraph collapses the parallel branches of a super-step into one
    # gate task, so ``_route_slices`` (wired ONLY on the gate) is evaluated
    # exactly once per super-step on the fully-merged state. Attaching the router
    # to each fan-out node instead re-evaluated it per parallel branch on
    # partially-merged state, causing duplicate dispatch and the
    # synthesize/save artifacts_output collision (trace 019f0193). Mirrors
    # VERIFY's ``aggregate_verification`` barrier.
    builder.add_edge("split_slices", "dispatch_gate")
    # plan_slice_implementer dispatches to slice_implementer dynamically
    # via Command(goto=Send) (see the node itself) so each parallel
    # branch carries its own directive without colliding on a shared
    # LastValue channel.
    builder.add_edge("slice_implementer", "dispatch_gate")
    # The flag-gated synthesis editor re-enters the gate exactly like the tool
    # path, so the dispatch loop terminates the same way.
    builder.add_edge("synthesis_implementer", "dispatch_gate")
    builder.add_edge("fallback_decomposer", "dispatch_gate")
    builder.add_conditional_edges("dispatch_gate", _route_slices, _ROUTE_MAP)
    builder.add_edge("synthesize_implementation", "save_artifacts")
    builder.add_edge("save_artifacts", END)

    return builder
