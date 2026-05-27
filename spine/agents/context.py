"""SPINE runtime context — structured per-run context for Deep Agents.

Deep Agents support ``context_schema`` — a typed dataclass that gets passed
at invoke time and propagates automatically to subagents.  Tools read it
via ``ToolRuntime.context``.  This is more efficient than baking values into
the system prompt because:

- It's typed and validated at invoke time.
- Tools can read it programmatically (no string parsing).
- Subagents inherit it automatically.
- It doesn't consume prompt tokens until a tool reads it.

This module defines ``SpineContext`` — the per-run context for all SPINE
phase agents — and a helper to build one from a ``WorkflowState``.

.. note::

   ``SpineContext`` is a Pydantic ``BaseModel`` (not a plain dataclass)
   because LangGraph's config schema uses Pydantic to serialise the
   ``context_schema`` field during checkpointing.  A plain dataclass
   produces ``PydanticSerializationUnexpectedValue`` warnings because
   the Pydantic model field defaults to ``None`` but receives a
   dataclass instance.  Making it a ``BaseModel`` lets Pydantic
   serialise it natively, eliminating the warnings.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from spine.models.enums import PhaseName
from spine.agents.artifacts import artifact_path


class SpineContext(BaseModel):
    """Runtime context for a SPINE phase agent invocation.

    Passed via the ``context=`` kwarg to ``agent.invoke()`` / ``agent.ainvoke()``
    and available inside tools as ``runtime.context``.  Propagates to subagents.

    Attributes:
        work_id: Unique work item identifier.
        phase: Current phase name (e.g. "specify", "implement").
        active_subagent: Name of the subagent the interpreter should target
            with the ``task`` tool (e.g. ``"researcher"`` for SPECIFY,
            ``"slice-implementer"`` for IMPLEMENT, ``"slice-verifier"``
            for VERIFY).  Empty for phases without subagents.
        workspace_root: Absolute path to the project directory.
        retry_count: How many times this phase has been retried.
        is_rework: True if this is a rework (retry_count > 0).
        critic_feedback: Feedback from prior critic reviews (if reworking).
        artifact_paths: Mapping of phase name -> artifact file path on disk.
        read_cache: Short-term read-file cache shared across subagents.
            Maps relative file paths to ``{"n_lines": int, "symbols": str}``.
            Populated by ``ReadCacheMiddleware`` on first read; subsequent
            reads of the same path return a compact summary instead of
            re-reading.
        read_cache_turn: Monotonic turn counter incremented by the
            ``ReadCacheMiddleware`` on every read_file call (hit or miss).
            Used to stamp cache entries.
    """

    model_config = {"arbitrary_types_allowed": True}

    work_id: str = ""
    phase: str = ""
    active_subagent: str = ""
    workspace_root: str = "."
    retry_count: int = 0
    is_rework: bool = False
    critic_feedback: list[str] = Field(default_factory=list)
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    read_cache: dict[str, dict[str, object]] = Field(default_factory=dict)
    read_cache_turn: int = 0


def build_context(
    state: dict,
    phase: PhaseName | str,
) -> SpineContext:
    """Build a SpineContext from a WorkflowState and the current phase name.

    Args:
        state: The current workflow state dict.
        phase: The phase being executed.

    Returns:
        A populated SpineContext instance.
    """
    phase_name = phase.value if isinstance(phase, PhaseName) else phase
    # retry_count is a dict {phase: count} in parent WorkflowState but
    # a plain int inside subgraphs (state mapper already extracts it).
    rc = state.get("retry_count", 0)
    if isinstance(rc, dict):
        retry_count = rc.get(phase_name, 0)
    elif isinstance(rc, int):
        retry_count = rc
    else:
        retry_count = 0

    # Extract critic feedback as a flat list of reason strings
    feedback = state.get("feedback", [])
    critic_feedback: list[str] = []
    for f in feedback:
        if isinstance(f, dict):
            reason = f.get("reason", "")
            if reason:
                critic_feedback.append(f"[{f.get('tier', 'unknown')}] {reason}")

    # Build artifact paths — these are where prior phase outputs live on disk
    # (materialized by the phase functions via the artifact materializer).
    workspace_root = state.get("workspace_root", ".")
    work_id = state.get("work_id", "")
    artifacts = state.get("artifacts", {})
    artifact_paths: dict[str, str] = {}
    for phase_key in (
        PhaseName.SPECIFY.value,
        PhaseName.PLAN.value,
        PhaseName.TASKS.value,
        PhaseName.IMPLEMENT.value,
    ):
        if artifacts.get(phase_key):
            artifact_paths[phase_key] = artifact_path(work_id, phase_key)

    # Resolve the named subagent for this phase (for interpreter PTC)
    from spine.agents.subagents import PHASE_SUBAGENTS

    subagent_names = PHASE_SUBAGENTS.get(phase_name, [])
    active_subagent = subagent_names[0] if len(subagent_names) == 1 else ""

    # Seed the dedupe cache from prior phases / rework cycles. The mapping
    # is checkpointed in WorkflowState.read_cache; we hand the middleware a
    # mutable copy so its in-place writes during this invocation can be
    # snapshotted back into state without leaking writes onto the parent
    # dict before the LangGraph reducer runs.
    seeded_cache = dict(state.get("read_cache") or {})

    return SpineContext(
        work_id=state.get("work_id", "unknown"),
        phase=phase_name,
        active_subagent=active_subagent,
        workspace_root=workspace_root,
        retry_count=retry_count,
        is_rework=retry_count > 0,
        critic_feedback=critic_feedback,
        artifact_paths=artifact_paths,
        read_cache=seeded_cache,
    )
