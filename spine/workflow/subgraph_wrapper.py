"""Factory for wrapping phase subgraphs as parent graph nodes.

Each wrapper:
1. Maps ParentState → SubgraphState
2. Invokes the subgraph with its own checkpointer + timeout
3. Catches CancelledError and other exceptions
4. Maps subgraph output → ParentState update
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from spine.agents.artifacts import scan_artifact_dir
from langgraph.errors import GraphRecursionError

from spine.agents.retry import (
    MaxTokenBudgetExceeded,
    ServerUnreachable,
    reset_token_budget,
)
from spine.exceptions import CriticalContractFailure
from spine.models.state import WorkflowState
from spine.workflow.phase_progress import mark_phase_started

logger = logging.getLogger(__name__)

# Structural contract failures (e.g. a synthesizer that emitted the spec as
# text instead of calling its write tool) get this many fresh-thread re-runs
# before escalating to human review. One retry covers the common case of a
# weak model failing the structured-output contract on the first roll.
_MAX_STRUCTURAL_RETRIES = 1

# Floor on the wall-clock that must remain (when a phase timeout is set) before
# a structural retry is allowed to start. A retry launched with less budget than
# the prior attempt consumed would be cancelled mid-synthesis and discard all of
# its work (trace 019ec997), so we salvage the prior phases instead. Only
# consulted when a non-zero phase timeout is configured.
_MIN_RETRY_BUDGET_SECS = 30.0

# ── Per-phase checkpoint store ─────────────────────────────────────────
# Cache to avoid opening the same phase DB multiple times per workflow.
_phase_checkpointer_cache: dict[str, AsyncSqliteSaver] = {}

# ``AsyncSqliteSaver.from_conn_string`` is an async context manager that opens
# the aiosqlite connection inside its own ``async with`` and yields the saver.
# The saver holds the connection but NOT the context-manager object, so if the
# ctx is dropped its ``__aexit__`` runs on GC and CLOSES the connection out from
# under a still-running graph ("Connection closed"). Hold every entered ctx here
# for the process lifetime so the connection thread survives until exit.
_phase_checkpointer_ctx_cache: dict[str, Any] = {}


def _ensure_checkpoint_dir(base_path: str) -> bool:
    """Create the checkpoint base directory if it doesn't exist."""
    try:
        Path(base_path).mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        logger.error(f"Failed to create checkpoint directory: {base_path}")
        return False


def _saver_connection_alive(saver: AsyncSqliteSaver) -> bool:
    """Best-effort check that a cached saver's aiosqlite connection is usable.

    A cached :class:`AsyncSqliteSaver` is only reusable while its underlying
    aiosqlite connection (and the worker thread bound to the event loop that
    opened it) is still alive. When the same ``work_id`` is processed again in a
    NEW event loop — e.g. a re-run under a fresh ``asyncio.run`` — the old
    connection is dead, so reusing it raises "Connection closed" /
    "no active connection". We detect that here so the caller can transparently
    reopen. On any uncertainty we report "not alive" so the caller rebuilds (a
    fresh connection is always safe; a stale one is not).
    """
    conn = getattr(saver, "conn", None)
    if conn is None:
        return False
    # aiosqlite sets ``_connection`` to None on close, and runs the real sqlite3
    # connection on a dedicated ``_thread``. Treat the saver as alive only when
    # both are present and the worker thread is still running.
    if getattr(conn, "_connection", None) is None:
        return False
    thread = getattr(conn, "_thread", None)
    if thread is not None and not thread.is_alive():
        return False
    return True


async def _get_phase_checkpointer(
    work_id: str,
    phase: str,
    checkpoint_base_dir: str | None = None,
    workspace_root: str | None = None,
) -> AsyncSqliteSaver:
    """Get a phase-specific checkpointer for per-phase checkpoint isolation.

    Each phase subgraph writes to its own SQLite database so that a crash
    or CancelledError in one phase doesn't corrupt another phase's checkpoints.

    Args:
        work_id: The work item ID.
        phase: Phase name (e.g. "verify").
        checkpoint_base_dir: Base directory for checkpoint files.
            Defaults to <workspace_root>/.spine/checkpoints/ (or the
            package-based fallback if workspace_root is also unset).
        workspace_root: The project workspace root. Used to resolve
            the default checkpoint_base_dir when not explicitly set.

    Returns:
        An AsyncSqliteSaver scoped to this phase's database.
    """
    if checkpoint_base_dir is None:
        if workspace_root:
            checkpoint_base_dir = str(Path(workspace_root) / ".spine" / "checkpoints")
        else:
            # Fallback: resolve relative to the spine package directory
            # so that /root or /tmp CWD doesn't cause permission errors.
            pkg_dir = Path(__file__).resolve().parent.parent
            checkpoint_base_dir = str(pkg_dir / ".spine" / "checkpoints")

    cache_key = f"{work_id}/{phase}"
    cached = _phase_checkpointer_cache.get(cache_key)
    if cached is not None and _saver_connection_alive(cached):
        return cached
    if cached is not None:
        # A cached saver whose aiosqlite connection was bound to an
        # already-closed event loop (e.g. a re-run of the same work_id in a
        # fresh ``asyncio.run`` loop) is unusable — drop it and reopen below so
        # the second run gets a live connection instead of "Connection closed".
        _phase_checkpointer_cache.pop(cache_key, None)
        _phase_checkpointer_ctx_cache.pop(cache_key, None)

    db_path = Path(checkpoint_base_dir) / work_id / f"{phase}.db"
    _ensure_checkpoint_dir(str(db_path.parent))

    ctx = AsyncSqliteSaver.from_conn_string(str(db_path))
    saver = await ctx.__aenter__()
    # Hold the context manager so its ``__aexit__`` does not run on GC and close
    # the aiosqlite connection while the graph is still using it.
    _phase_checkpointer_ctx_cache[cache_key] = ctx
    _phase_checkpointer_cache[cache_key] = saver
    logger.debug(f"[{work_id}] [{phase}] per-phase checkpointer: {db_path}")
    return saver


# Per-phase timeout overrides (seconds). Default: _DEFAULT_PHASE_TIMEOUT.
# These are fallbacks — the subgraph wrapper reads SpineConfig.phase_timeouts
# at invocation time so users can override via config.yaml.
_PHASE_TIMEOUTS: dict[str, int] = {
    "specify": 0,
    "plan": 0,
    "tasks": 0,
    "implement": 0,
    "verify": 0,
    "critic": 0,
}
_DEFAULT_PHASE_TIMEOUT = 0


def _resolve_timeout(phase: str, config: RunnableConfig | None = None) -> int:
    """Resolve timeout for a phase from config or fallback defaults.

    Args:
        phase: Phase name (e.g. "verify").
        config: Optional LangGraph config dict that may contain
            ``configurable.spine_config`` with ``phase_timeouts``.

    Returns:
        Timeout in seconds.
    """
    # Try config first
    if config and isinstance(config, dict):
        spine_cfg = config.get("configurable", {}).get("spine_config")
        if spine_cfg and hasattr(spine_cfg, "phase_timeouts"):
            timeouts = spine_cfg.phase_timeouts
            if isinstance(timeouts, dict) and phase in timeouts:
                return int(timeouts[phase])
            if hasattr(spine_cfg, "default_timeout"):
                return int(spine_cfg.default_timeout)
    # Fallback to defaults
    return _PHASE_TIMEOUTS.get(phase, _DEFAULT_PHASE_TIMEOUT)


# Max chars of artifact content to store in parent WorkflowState.
_MAX_ARTIFACT_STATE_CHARS = 500


def _error_update(
    state: WorkflowState,
    phase: str,
    error: str,
) -> dict[str, Any]:
    """Build a parent state update for subgraph errors."""
    return {
        "current_phase": phase,
        "status": "needs_review",
        "prompt_request": None,
        "feedback": [
            {
                "status": "needs_review",
                "tier": "structural",
                "reason": f"[{phase}] subgraph error: {error}",
                "suggestions": ["Review logs and retry", "Reduce scope"],
            }
        ],
        "phase_results": {
            phase: {
                "phase": phase,
                "status": "error",
                "artifact_count": 0,
                "artifact_names": [],
                "error": error,
            }
        },
    }


def _salvaged_artifact_names(state: WorkflowState, phase: str) -> list[str]:
    """Names of any artifacts the phase already wrote to disk before aborting.

    The structured-write tools (``write_specification`` /
    ``write_structured_plan``) persist their output to
    ``.spine/artifacts/{work_id}/{phase}/`` *themselves*, before the subgraph's
    save node runs. So a cancellation, timeout, or budget abort that lands after
    the write but before the phase completes leaves valid artifacts stranded on
    disk. Reporting ``artifact_count=0`` in that case is misleading (trace
    019ec997: plan.json was on disk yet the phase claimed zero artifacts) and
    hides reusable work. This best-effort scan lets the abort handlers tell the
    truth so a human/retry can reuse what's already there.
    """
    work_id = state.get("work_id", "unknown")
    workspace_root = state.get("workspace_root", ".")
    try:
        disk = scan_artifact_dir(workspace_root, work_id, phase)
    except Exception:  # never let salvage reporting mask the original abort
        logger.debug("[%s] [%s] artifact salvage scan failed", work_id, phase, exc_info=True)
        return []
    return sorted(disk.keys())


def _needs_review_update(
    state: WorkflowState,
    phase: str,
    reason: str,
    suggestions: list[str] | None = None,
) -> dict[str, Any]:
    """Build a parent state update for needs_review.

    If the phase already persisted artifacts to disk before aborting, surface
    them honestly (``artifact_count``/``artifact_names``) and note that they
    were preserved, rather than reporting zero artifacts.
    """
    salvaged = _salvaged_artifact_names(state, phase)
    if salvaged:
        reason = f"{reason} (Artifacts preserved on disk: {', '.join(salvaged)}.)"
    return {
        "current_phase": phase,
        "status": "needs_review",
        "prompt_request": None,
        "feedback": [
            {
                "status": "needs_review",
                "tier": "structural",
                "reason": reason,
                "suggestions": suggestions or [],
            }
        ],
        "phase_results": {
            phase: {
                "phase": phase,
                "status": "needs_review",
                "artifact_count": len(salvaged),
                "artifact_names": salvaged,
                "error": reason,
            }
        },
        "needs_review_phase": phase,
    }


def make_subgraph_node(
    subgraph: Any,
    phase_name: str,
    state_mapper: Callable[[WorkflowState, RunnableConfig | None], dict],
    result_mapper: Callable[[dict, WorkflowState], dict[str, Any]],
    checkpoint_base_dir: str | None = None,
    use_per_phase_checkpointer: bool = False,
) -> Callable:
    """Create a LangGraph node that wraps a phase subgraph.

    Args:
        subgraph: Compiled StateGraph for this phase, or a callable
            that returns one when invoked with a checkpointer (used by
            builders that need per-phase isolation). NOTE: when using
            use_per_phase_checkpointer=True, the builder from the registry
            must return an uncompiled StateGraph so it can be recompiled
            with a per-phase checkpointer.
        phase_name: Phase enum value (e.g. "verify").
        state_mapper: Function (parent_state, config) → subgraph_input.
        result_mapper: Function (subgraph_output, parent_state) → parent_state_update.
        checkpoint_base_dir: Base directory for per-phase checkpoint files.
            Only used when ``use_per_phase_checkpointer`` is True.
        use_per_phase_checkpointer: If True, compile a fresh subgraph with
            a phase-specific SQLite checkpointer on each graph build.

    Returns:
        An async node function with signature (WorkflowState, config) → dict.
    """

    async def subgraph_node(
        parent_state: WorkflowState,
        config: RunnableConfig = None,
    ) -> dict[str, Any]:
        work_id = parent_state.get("work_id", "unknown")
        timeout = _resolve_timeout(phase_name, config)

        # Per-PHASE token budget. The budget is a cumulative-per-work_id counter
        # whose purpose is to abort a SPIRAL — an unbounded loop within one phase.
        # Left cumulative across the whole pipeline it instead starves the tail:
        # a legitimately expensive IMPLEMENT (best-of-N synthesis) consumed ~1.2M
        # across specify+plan+implement, so VERIFY's first (cheap, one-shot) judge
        # call tripped the 1M ceiling and every downstream phase crashed with a
        # verdict already in hand (trace spine-bench-verify-judge-0630). Resetting
        # at each phase boundary makes the ceiling per-phase: each phase still
        # aborts if IT spirals, but one phase's honest spend no longer steals the
        # next phase's allowance. Total spend stays bounded by per-phase timeouts,
        # recursion limits, and dispatch caps.
        reset_token_budget(work_id)

        # Mark the phase as started so the UI shows the correct phase
        # immediately, before the subgraph begins executing.
        mark_phase_started(parent_state, phase_name)

        timeout_label = f"timeout={timeout}s" if timeout > 0 else "no timeout"
        logger.info(f"[{work_id}] [{phase_name}] subgraph starting ({timeout_label})")

        try:
            # Map parent state to subgraph input
            subgraph_input = state_mapper(parent_state, config)

            # Build subgraph config with its own thread_id for independent checkpointing
            base_configurable: dict[str, Any] = {"thread_id": f"{work_id}_{phase_name}"}
            if config and isinstance(config, dict):
                cfg = config.get("configurable", {})
                if cfg:
                    base_configurable.update(cfg)

            # When using per-phase checkpointers, recompile the subgraph with its own
            # phase-specific checkpointer before invoking.  This gives each phase
            # an isolated SQLite DB so a crash in one phase doesn't corrupt another.
            active_subgraph = subgraph
            if use_per_phase_checkpointer:
                from spine.workflow.compose import _SUBGRAPH_BUILDER_REGISTRY as builders

                builder_fn = builders.get(phase_name)
                # Critic subgraphs are parameterized (need reviewed_phase arg)
                # and are lightweight — skip per-phase checkpointer for them.
                if builder_fn is not None and not phase_name.startswith("critic"):
                    try:
                        checkpointer = await _get_phase_checkpointer(
                            work_id,
                            phase_name,
                            checkpoint_base_dir,
                            workspace_root=parent_state.get("workspace_root"),
                        )
                        active_subgraph = builder_fn().compile(checkpointer=checkpointer)
                    except Exception as exc:
                        logger.warning(
                            f"[{work_id}] [{phase_name}] per-phase checkpointer failed, "
                            f"falling back to parent checkpointer: {exc!r}",
                            exc_info=True,
                        )

            # Invoke, retrying structural contract failures on a fresh thread.
            # Each attempt uses a distinct thread_id so the subgraph re-runs from
            # START — a same-thread re-invoke would resume at the failed save
            # node (with the bad agent output checkpointed) and re-raise without
            # re-running the agent.
            max_attempts = 1 + _MAX_STRUCTURAL_RETRIES
            # When a timeout is configured it is a budget for the WHOLE phase,
            # not a fresh allowance per attempt — otherwise a 2-attempt phase
            # could run for 2× the timeout. Track elapsed wall-clock so each
            # attempt sees only the remaining budget and a doomed retry is
            # skipped before it starts (trace 019ec997). When timeout == 0
            # (the default) all of this is inert and behaviour is unchanged.
            phase_start = time.monotonic()
            last_attempt_secs = 0.0
            for attempt in range(max_attempts):
                attempt_configurable = dict(base_configurable)
                if attempt > 0:
                    attempt_configurable["thread_id"] = (
                        f"{work_id}_{phase_name}_retry{attempt}"
                    )
                subgraph_config: dict[str, Any] = {"configurable": attempt_configurable}

                # Hard super-step ceiling so a non-converging dispatch loop (or a
                # 0-token server-crash spin the token breaker can't catch) aborts
                # fast instead of running to LangGraph's runaway backstop
                # (trace 019ece87). Read from the injected SpineConfig; fall back
                # to a fresh load only if it is absent so config never breaks dispatch.
                _rlim = 0
                _sc = base_configurable.get("spine_config")
                if _sc is None:
                    try:
                        from spine.config import SpineConfig

                        _sc = SpineConfig.load()
                    except Exception:  # noqa: BLE001 — never let config break dispatch
                        _sc = None
                if _sc is not None:
                    try:
                        _rlim = int(getattr(_sc, "subgraph_recursion_limit", 0) or 0)
                    except (TypeError, ValueError):
                        _rlim = 0
                if _rlim > 0:
                    subgraph_config["recursion_limit"] = _rlim

                attempt_timeout = 0.0
                if timeout > 0:
                    remaining = timeout - (time.monotonic() - phase_start)
                    # A retry that can't plausibly finish in the time left would
                    # be cancelled mid-synthesis, discarding its work and the
                    # prior attempt's. Salvage instead of burning the budget.
                    if attempt > 0:
                        needed = max(last_attempt_secs, _MIN_RETRY_BUDGET_SECS)
                        if remaining < needed:
                            logger.warning(
                                f"[{work_id}] [{phase_name}] skipping structural "
                                f"retry: {remaining:.0f}s of the {timeout}s budget "
                                f"left, prior attempt needed {last_attempt_secs:.0f}s"
                            )
                            return _needs_review_update(
                                parent_state,
                                phase_name,
                                (
                                    f"Structural retry skipped — only {remaining:.0f}s "
                                    f"of the {timeout}s phase budget remained after the "
                                    "first attempt. Prior phases preserved."
                                ),
                                suggestions=[
                                    "Raise the phase timeout if the scope warrants it",
                                    "Reduce scope or split into smaller work items",
                                ],
                            )
                    attempt_timeout = max(remaining, 0.0)

                attempt_start = time.monotonic()
                try:
                    # Invoke with or without timeout (0 = no timeout)
                    raw_coro = active_subgraph.ainvoke(subgraph_input, subgraph_config)
                    if timeout > 0:
                        result = await asyncio.wait_for(raw_coro, timeout=attempt_timeout)
                    else:
                        result = await raw_coro
                except CriticalContractFailure as cf:
                    last_attempt_secs = time.monotonic() - attempt_start
                    if attempt + 1 < max_attempts:
                        # Seed whatever the failed attempt salvaged (e.g.
                        # exploration findings that preceded a failed
                        # synthesis) so the fresh thread retries only the
                        # broken step instead of redoing completed work
                        # (trace 019eb940).
                        carry = getattr(cf, "carryover", None) or {}
                        if carry:
                            subgraph_input = {**subgraph_input, **carry}
                            logger.info(
                                f"[{work_id}] [{phase_name}] carrying "
                                f"{len(carry.get('findings') or [])} findings "
                                f"into the structural retry"
                            )
                        logger.warning(
                            f"[{work_id}] [{phase_name}] structural contract failure "
                            f"({cf.reason}) — auto-retrying on a clean thread "
                            f"(attempt {attempt + 2}/{max_attempts})"
                        )
                        continue
                    logger.error(
                        f"[{work_id}] [{phase_name}] structural contract failure "
                        f"persisted after {max_attempts} attempt(s): {cf.reason}"
                    )
                    return _error_update(parent_state, phase_name, str(cf))

                # Map subgraph result back to parent state
                parent_update = result_mapper(result, parent_state)
                logger.info(
                    f"[{work_id}] [{phase_name}] subgraph completed: "
                    f"phase_status={result.get('phase_status', 'unknown')}"
                )
                return parent_update

        except GraphRecursionError as rec_exc:
            logger.error(
                f"[{work_id}] [{phase_name}] hit recursion ceiling "
                f"({_rlim} super-steps): {rec_exc}"
            )
            return _needs_review_update(
                parent_state,
                phase_name,
                (
                    f"Dispatch loop hit the recursion ceiling ({_rlim} super-steps) "
                    "— likely a non-converging slice or a backend outage. Surfaced "
                    "for review instead of spinning. Prior phases preserved."
                ),
                suggestions=[
                    "Check the model backend is healthy and converging",
                    "Reduce scope or split into smaller work items",
                    "Raise subgraph_recursion_limit if the scope genuinely warrants it",
                ],
            )

        except asyncio.TimeoutError:
            logger.error(f"[{work_id}] [{phase_name}] subgraph timed out after {timeout}s")
            return _needs_review_update(
                parent_state,
                phase_name,
                f"Timed out after {timeout}s",
            )

        except asyncio.CancelledError:
            logger.error(f"[{work_id}] [{phase_name}] subgraph cancelled")
            return _needs_review_update(
                parent_state,
                phase_name,
                "Cancelled — subgraph did not complete. Prior phases preserved.",
            )

        except MaxTokenBudgetExceeded as budget_exc:
            logger.error(
                f"[{work_id}] [{phase_name}] token budget exceeded: "
                f"{budget_exc.cumulative:,} / {budget_exc.budget:,} tokens"
            )
            return _needs_review_update(
                parent_state,
                phase_name,
                (
                    f"Token budget exceeded "
                    f"({budget_exc.cumulative:,}/{budget_exc.budget:,} tokens). "
                    "Phase aborted to prevent unbounded spend."
                ),
                suggestions=[
                    "Reduce scope or split into smaller work items",
                    "Raise the per-work-type budget if the scope genuinely warrants it",
                ],
            )

        except ServerUnreachable as conn_exc:
            logger.error(
                f"[{work_id}] [{phase_name}] LLM endpoint unreachable: {conn_exc}"
            )
            return _needs_review_update(
                parent_state,
                phase_name,
                (
                    f"LLM endpoint unreachable after {conn_exc.count} consecutive "
                    "connection failures. Phase aborted instead of hammering a "
                    "down server."
                ),
                suggestions=[
                    "Check the local model server is running and reachable",
                    "Resume the work item once the endpoint is back up",
                ],
            )

        except Exception as e:
            logger.error(f"[{work_id}] [{phase_name}] subgraph failed: {e}", exc_info=True)
            return _error_update(parent_state, phase_name, str(e))

    # Name for LangSmith Studio / debug
    subgraph_node.__name__ = f"{phase_name}_subgraph"
    return subgraph_node


def make_success_result_mapper(phase: str) -> Callable:
    """Create a standard result mapper for a successful phase completion.

    Extracts artifacts_output from the subgraph result and maps to the
    parent graph's phase_results and status fields.
    """

    def map_success(
        subgraph_result: dict,
        parent_state: dict,
    ) -> dict[str, Any]:
        artifacts = subgraph_result.get("artifacts_output", {})
        artifact_names = list(artifacts.keys()) if isinstance(artifacts, dict) else []

        # Build artifact previews for parent state (truncated)
        artifact_previews = {}
        if isinstance(artifacts, dict):
            for name, content in artifacts.items():
                if isinstance(content, str):
                    artifact_previews[name] = content[:_MAX_ARTIFACT_STATE_CHARS]

        # Mirror the subgraph's reported phase_status into the per-phase
        # record. Hardcoding "success" here mislabels phases that completed
        # without raising but reported phase_status="error"/"needs_review"
        # (e.g. a swallowed synthesis failure): callers patch the top-level
        # ``status`` to failed/needs_review while phase_results still claimed
        # success/error=None, so the summary contradicted the run outcome
        # (trace 019ec90d). PhaseResult.status is "success" | "needs_review"
        # | "error" — keep it honest.
        phase_status = subgraph_result.get("phase_status", "")
        if phase_status in ("error", "needs_review"):
            phase_result_status = phase_status
            phase_result_error = (
                subgraph_result.get("agent_response")
                or f"phase ended with status {phase_status!r}"
            )[:_MAX_ARTIFACT_STATE_CHARS]
        else:
            phase_result_status = "success"
            phase_result_error = None

        result: dict[str, Any] = {
            "current_phase": phase,
            "status": "running",
            "prompt_request": None,
            "phase_results": {
                phase: {
                    "phase": phase,
                    "status": phase_result_status,
                    "artifact_count": len(artifact_names),
                    "artifact_names": artifact_names,
                    "error": phase_result_error,
                }
            },
            "artifacts": {phase: artifact_previews},
        }
        # Bubble the subgraph's accumulated dedupe cache back to WorkflowState
        # so downstream phases (and rework cycles) start with a warm cache.
        cache = subgraph_result.get("read_cache")
        if cache:
            result["read_cache"] = cache
        return result

    return map_success
