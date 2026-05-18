"""Factory for wrapping phase subgraphs as parent graph nodes.

Each wrapper:
1. Maps ParentState → SubgraphState
2. Invokes the subgraph with its own checkpointer + timeout
3. Catches CancelledError and other exceptions
4. Maps subgraph output → ParentState update
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from spine.models.state import WorkflowState

logger = logging.getLogger(__name__)

# ── Per-phase checkpoint store ─────────────────────────────────────────
# Cache to avoid opening the same phase DB multiple times per workflow.
_phase_checkpointer_cache: dict[str, AsyncSqliteSaver] = {}


def _ensure_checkpoint_dir(base_path: str) -> bool:
    """Create the checkpoint base directory if it doesn't exist."""
    try:
        Path(base_path).mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        logger.error(f"Failed to create checkpoint directory: {base_path}")
        return False


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
    if cache_key in _phase_checkpointer_cache:
        return _phase_checkpointer_cache[cache_key]

    db_path = Path(checkpoint_base_dir) / work_id / f"{phase}.db"
    _ensure_checkpoint_dir(str(db_path.parent))

    ctx = AsyncSqliteSaver.from_conn_string(str(db_path))
    saver = await ctx.__aenter__()
    _phase_checkpointer_cache[cache_key] = saver
    logger.debug(f"[{work_id}] [{phase}] per-phase checkpointer: {db_path}")
    return saver

# Per-phase timeout overrides (seconds). Default: _DEFAULT_PHASE_TIMEOUT.
# These are fallbacks — the subgraph wrapper reads SpineConfig.phase_timeouts
# at invocation time so users can override via config.yaml.
_PHASE_TIMEOUTS: dict[str, int] = {
    "specify": 600,
    "plan": 600,
    "tasks": 900,
    "implement": 1800,
    "verify": 600,
    "critic": 300,
}
_DEFAULT_PHASE_TIMEOUT = 900


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


def _needs_review_update(
    state: WorkflowState,
    phase: str,
    reason: str,
    suggestions: list[str] | None = None,
) -> dict[str, Any]:
    """Build a parent state update for needs_review."""
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
                "artifact_count": 0,
                "artifact_names": [],
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

        logger.info(
            f"[{work_id}] [{phase_name}] subgraph starting (timeout={timeout}s)"
        )

        try:
            # Map parent state to subgraph input
            subgraph_input = state_mapper(parent_state, config)

            # Build subgraph config with its own thread_id for independent checkpointing
            subgraph_config = {
                "configurable": {
                    "thread_id": f"{work_id}_{phase_name}",
                },
            }
            if config and isinstance(config, dict):
                cfg = config.get("configurable", {})
                if cfg:
                    subgraph_config["configurable"].update(cfg)

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

            # Invoke with timeout
            result = await asyncio.wait_for(
                active_subgraph.ainvoke(subgraph_input, subgraph_config),
                timeout=timeout,
            )

            # Map subgraph result back to parent state
            parent_update = result_mapper(result, parent_state)
            logger.info(
                f"[{work_id}] [{phase_name}] subgraph completed: "
                f"phase_status={result.get('phase_status', 'unknown')}"
            )
            return parent_update

        except asyncio.TimeoutError:
            logger.error(
                f"[{work_id}] [{phase_name}] subgraph timed out after {timeout}s"
            )
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

        except Exception as e:
            logger.error(
                f"[{work_id}] [{phase_name}] subgraph failed: {e}", exc_info=True
            )
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
        artifact_names = (
            list(artifacts.keys()) if isinstance(artifacts, dict) else []
        )

        # Build artifact previews for parent state (truncated)
        artifact_previews = {}
        if isinstance(artifacts, dict):
            for name, content in artifacts.items():
                if isinstance(content, str):
                    artifact_previews[name] = content[:_MAX_ARTIFACT_STATE_CHARS]

        return {
            "current_phase": phase,
            "status": "running",
            "prompt_request": None,
            "phase_results": {
                phase: {
                    "phase": phase,
                    "status": "success",
                    "artifact_count": len(artifact_names),
                    "artifact_names": artifact_names,
                    "error": None,
                }
            },
            "artifacts": {phase: artifact_previews},
        }

    return map_success
