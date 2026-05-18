"""SPINE dispatcher — unified entry point for work submission.

``submit_work()`` is the single entry point for:
- CLI commands (``spine run``)
- Streamlit UI (submit page)
- RalphLoopWorker (background queue processor)

All reads go through ``UIApi``. UI pages never import from
workflow/ or phases/ directly.

Work items are tracked in a SQLite database at ``.spine/work_entries.db``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import sqlite_utils

from spine.config import SpineConfig
from spine.models.enums import TaskStatus
from spine.persistence.artifacts import ArtifactStore
from spine.services.audit_service import AuditService

logger = logging.getLogger(__name__)

# ── Stall detection ─────────────────────────────────────────────────────
# Maximum time to wait for a single stream event (update or LLM token)
# before declaring the workflow stalled.  With stream_mode=["updates",
# "messages"] and subgraphs=True, token-level LLM output keeps this timer
# alive during long agent runs.  Only a genuine connection drop or hung
# LLM call will trigger the stall.
# Default: 2 minutes — generous enough for brief pauses between agent
# turns, but catches genuine hangs quickly.
_STALL_TIMEOUT_SECONDS = int(__import__("os").environ.get("SPINE_STALL_TIMEOUT", "120"))


# ── Work entries database ──


def _get_work_db(config: SpineConfig) -> sqlite_utils.Database:
    """Get or create the work entries database."""
    db_path = Path(config.queue_path).parent / "work_entries.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite_utils.Database(str(db_path))

    if "work_entries" not in db.table_names():
        db["work_entries"].create(
            {
                "id": str,
                "description": str,
                "work_type": str,
                "status": str,
                "current_phase": str,
                "created_at": str,
                "updated_at": str,
                "result": str,  # JSON
            },
            pk="id",
        )

    return db


def _update_work_progress(
    db: sqlite_utils.Database,
    work_id: str,
    current_phase: str,
    status: str,
) -> None:
    """Update the work entry's current_phase and status mid-workflow.

    Called after each phase node completes so the UI can show progress
    without waiting for the entire workflow to finish.  Also publishes
    a WebSocket event so connected UI clients get live push updates.

    Args:
        db: The work entries database.
        work_id: The work item ID.
        current_phase: The phase that just completed.
        status: The current status string.
    """
    try:
        db["work_entries"].update(
            work_id,
            {
                "current_phase": current_phase,
                "status": status,
                "updated_at": datetime.now().isoformat(),
            },
        )
    except Exception:
        # Don't let a DB update failure crash the workflow
        logger.warning(f"Failed to update progress for {work_id}", exc_info=True)

    # ── Push event to WebSocket bus ──
    try:
        from spine.ui.ws_bus import get_bus

        get_bus().publish_sync(
            "work_progress",
            {"work_id": work_id, "current_phase": current_phase, "status": status},
        )
    except Exception:
        # Bus may not be initialised (CLI-only mode) — that's fine.
        pass


# ── Submit work ──


async def submit_work(
    description: str,
    work_type: str = "spec",
    config: SpineConfig | None = None,
    created_at: str | None = None,
    work_id: str | None = None,
    plan_id: str | None = None,
) -> dict[str, Any]:
    """Submit a new work item for processing.

    This is the unified entry point for CLI, UI, and worker. It:
    1. Creates a work entry with a unique ID
    2. Builds the workflow graph for the given work type
    3. Invokes the graph with checkpoint persistence
    4. Returns the work ID and initial state

    When called from the background queue worker (RalphLoopWorker),
    ``created_at`` should be the original ``enqueued_at`` timestamp so
    the dashboard shows items in submission order, not processing order.

    Args:
        description: The work description / prompt.
        work_type: One of "quick", "critical_quick", "spec", "critical_spec",
            "plan", or "plan_spec".
        config: Optional SpineConfig (loads from default if not provided).
        created_at: Optional ISO timestamp for the work entry's ``created_at``
            field.  When ``None`` (default), uses the current time.
        work_id: Optional pre-generated work item ID.  When ``None``, a new
            8-char UUID prefix is generated.  The queue worker pre-generates
            the ID so the queue row can display the correct work_id while
            the job is still running, instead of falling back to the queue
            sequence number.
        plan_id: Optional reference to an approved planning work item
            whose spec/plan this execution derives from.

    Returns:
        A dict with keys: ``work_id``, ``status``, ``work_type``.
    """
    if config is None:
        config = SpineConfig.load()
    config.ensure_dirs()

    work_id = work_id or str(uuid.uuid4())[:8]
    audit = AuditService(db_path=str(Path(config.queue_path).parent / "audit.db"))
    artifacts = ArtifactStore(base_path=config.artifact_path)

    audit.log_event(
        work_id,
        "work_submitted",
        "dispatcher",
        {
            "description": description[:200],
            "work_type": work_type,
        },
    )

    now = created_at or datetime.now().isoformat()

    # Record the work entry
    db = _get_work_db(config)

    # Ensure the plan_id column exists (migration for existing databases)
    if "plan_id" not in db["work_entries"].columns_dict:
        try:
            db["work_entries"].add_column("plan_id", str)
        except Exception:
            logger.warning("Could not add plan_id column — may already exist", exc_info=True)

    db["work_entries"].insert(
        {
            "id": work_id,
            "description": description,
            "work_type": work_type,
            "status": TaskStatus.RUNNING.value,
            "current_phase": "",
            "created_at": now,
            "updated_at": now,
            "result": "{}",
            "plan_id": plan_id,
        }
    )

    # Build and run the workflow graph
    try:
        from spine.persistence.checkpoint import CheckpointStore
        from spine.workflow.compose import build_workflow_graph

        checkpoint_store = CheckpointStore(db_path=config.checkpoint_path)
        checkpointer = await checkpoint_store.get_checkpointer()

        graph = build_workflow_graph(work_type, checkpointer=checkpointer)

        initial_state = {
            "work_id": work_id,
            "work_type": work_type,
            "description": description,
            "current_phase": "",
            "phase_index": 0,
            "retry_count": {},
            "max_retries": config.max_critic_retries,
            "artifacts": {},
            "feedback": [],
            "status": "running",
            "prompt_request": None,
            "critic_reviewing": "",
            "workspace_root": config.workspace_root,
            "plan_id": plan_id,
        }

        thread_config = {
            "configurable": {
                "thread_id": work_id,
                # Per-phase model resolution happens inside each phase's
                # build_phase_agent() via resolve_model(phase=...).  Do NOT
                # inject a "model" key here — it would short-circuit
                # _model_spec_from_config() in helpers.py and force every
                # phase to use the default provider, ignoring
                # providers.phases.<phase> overrides.
                "spine_config": config,
            }
        }

        # Stream the graph so we can update the work entry after each phase.
        # This lets the UI see progress (current_phase, status) while the
        # workflow is still running, instead of only getting the final result.
        #
        # Stream mode: ["updates", "messages"] with subgraphs=True, version="v2".
        #   - "updates" yields {node_name: output} on each node completion.
        #   - "messages" yields token-level LLM output from inside nodes,
        #     keeping the stall timer alive during long agent runs.
        #   - subgraphs=True reaches into Deep Agent subgraph LLM calls.
        #   - version="v2" gives a consistent dict-based StreamPart format:
        #       {"type": "updates"|"messages"|..., "ns": (...), "data": ...}
        #     The v1 format changes shape depending on stream_mode / subgraph
        #     settings (2-tuple vs 3-tuple), which caused all chunks to be
        #     silently dropped when subgraphs=True (len != 2 check failed).
        #
        # Stall detection: wrap the astream iterator with a per-chunk timeout.
        # If no chunk (update OR message token) arrives within
        # _STALL_TIMEOUT_SECONDS, the workflow is considered stalled
        # (e.g. the LLM connection dropped silently).  Token-level streaming
        # means the timer resets on every LLM token, so only a genuine
        # hang triggers the stall — not a legitimately long agent run.
        result: dict[str, Any] = dict(initial_state)
        stream_iter = graph.astream(
            initial_state,
            thread_config,
            stream_mode=["updates", "messages"],
            subgraphs=True,
            version="v2",
        )
        stalled = False
        while True:
            try:
                chunk = await asyncio.wait_for(
                    stream_iter.__anext__(),
                    timeout=_STALL_TIMEOUT_SECONDS,
                )
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                last_phase = result.get("current_phase", "")
                logger.error(
                    f"[{work_id}] Workflow stalled — no chunk received for "
                    f"{_STALL_TIMEOUT_SECONDS}s (last phase: {last_phase}). "
                    f"Marking as stalled."
                )
                stalled = True
                _update_work_progress(db, work_id, last_phase, "stalled")
                audit.log_event(
                    work_id,
                    "work_stalled",
                    "dispatcher",
                    {"last_phase": last_phase, "timeout": _STALL_TIMEOUT_SECONDS},
                )
                break

            # V2 format: each chunk is a StreamPart dict:
            #   {"type": "updates"|"messages", "ns": (...), "data": ...}
            # Skip non-update chunks — "messages" tokens only serve to keep
            # the stall timer alive.  Process "updates" chunks (node
            # completions) for state tracking and artifact persistence.
            if not isinstance(chunk, dict) or chunk.get("type") != "updates":
                continue

            # Only process root-level updates (ns == ()).
            # Subgraph updates come from Deep Agent internals — we don't
            # need to track them at this level.
            ns = chunk.get("ns", ())
            if ns != ():
                continue

            data = chunk.get("data", {})
            if not isinstance(data, dict):
                continue
            # data is {node_name: partial_state_update}
            # With subgraphs=True, some entries may be non-dict (e.g. tuples
            # from subgraph-internal routing).  Skip those — only process
            # dict outputs that carry state updates we can merge.
            for node_name, node_output in data.items():
                if not isinstance(node_output, dict):
                    continue
                # Deep-merge artifacts so phase outputs accumulate instead of
                # getting overwritten.  The LangGraph state reducer does this
                # inside the graph, but our local `result` dict needs the same
                # logic — a plain result.update() would replace the entire
                # artifacts dict with the latest phase's output, losing all
                # prior phase artifacts.
                node_artifacts = node_output.get("artifacts")
                if node_artifacts and isinstance(node_artifacts, dict):
                    existing = result.get("artifacts", {})
                    if not isinstance(existing, dict):
                        existing = {}
                    merged = {**existing, **node_artifacts}
                    # Deep-merge nested dicts (phase → {name: content})
                    for key in set(existing) & set(node_artifacts):
                        if isinstance(existing[key], dict) and isinstance(
                            node_artifacts[key], dict
                        ):
                            merged[key] = {**existing[key], **node_artifacts[key]}
                    node_output = {**node_output, "artifacts": merged}

                # Accumulate feedback list instead of overwriting.
                # The LangGraph state reducer (operator.add) appends inside
                # the graph, but result.update() would replace the whole list
                # with just the latest node's entries.
                node_feedback = node_output.get("feedback")
                if node_feedback and isinstance(node_feedback, list):
                    existing_fb = result.get("feedback", [])
                    if not isinstance(existing_fb, list):
                        existing_fb = []
                    node_output = {
                        **node_output,
                        "feedback": existing_fb + node_feedback,
                    }

                # Merge the node output into our running result
                result.update(node_output)
                # Update the work entry DB so the UI can see progress
                phase = node_output.get("current_phase", "")
                status = node_output.get("status", "")
                if phase or status:
                    _update_work_progress(db, work_id, phase, status)
                    logger.info(f"[{work_id}] Phase {phase or node_name} → {status}")
                    audit.log_event(
                        work_id,
                        "phase_completed",
                        node_name,
                        {"phase": phase, "status": status},
                    )

                # Persist artifacts to disk immediately after each phase
                # completes, so they're visible on disk without waiting for
                # the entire workflow to finish.  We use ArtifactStore (with
                # work_id) and also materialize to the agent-readable path.
                node_artifacts = node_output.get("artifacts")
                if node_artifacts and isinstance(node_artifacts, dict):
                    for art_phase, phase_arts in node_artifacts.items():
                        if not isinstance(phase_arts, dict):
                            continue
                        for art_name, art_content in phase_arts.items():
                            if art_content is not None:
                                artifacts.save_artifact(
                                    work_id, art_phase, art_name, str(art_content)
                                )
                                logger.debug(f"[{work_id}] Saved artifact {art_phase}/{art_name}")

        # Update work entry with final results.
        # Derive the terminal status from the graph state:
        #   - If stalled, the status was already set to "stalled" above.
        #   - If the critic routed to needs_review → END, the feedback
        #     contains a needs_review entry.
        #   - If the graph completed naturally (all phases ran), status
        #     should be "completed" or the verify phase's final status.
        #   - The top-level except (below) catches true failures.
        if stalled:
            final_status = "stalled"
        else:
            final_status = result.get("status", "completed")
        final_phase = result.get("current_phase", "")
        result_artifacts = result.get("artifacts", {})
        feedback = result.get("feedback", [])

        # If status is still "running" after graph completes, the graph
        # ended normally — treat as completed.
        if final_status == "running":
            final_status = "completed"

        # Check if any feedback entry indicates needs_review (the critic
        # router sends to END when max retries are exceeded).
        if any(
            isinstance(f, dict) and f.get("status") == "needs_review"
            for f in feedback
        ):
            final_status = "needs_review"

        # Save artifacts to disk
        for phase, phase_artifacts in result_artifacts.items():
            for name, content in phase_artifacts.items():
                if content is not None:
                    artifacts.save_artifact(work_id, phase, name, str(content))

        db["work_entries"].update(
            work_id,
            {
                "status": final_status,
                "current_phase": final_phase,
                "updated_at": datetime.now().isoformat(),
                "result": json.dumps(
                    {
                        "artifacts": {k: list(v.keys()) for k, v in result_artifacts.items()},
                        "feedback_count": len(feedback),
                        "prompt_request": result.get("prompt_request"),
                    }
                ),
            },
        )

        audit.log_event(
            work_id,
            "work_completed",
            final_phase,
            {
                "status": final_status,
            },
        )

        # ── Push completion event to WebSocket bus ──
        try:
            from spine.ui.ws_bus import get_bus

            get_bus().publish_sync(
                "work_completed",
                {"work_id": work_id, "status": final_status},
            )
        except Exception:
            pass

        return {
            "work_id": work_id,
            "status": final_status,
            "work_type": work_type,
        }

    except Exception as e:
        logger.error(f"Work {work_id} failed: {e}", exc_info=True)
        last_phase = ""
        if isinstance(locals().get("result"), dict):
            last_phase = result.get("current_phase", "")  # type: ignore[possibly-undefined]
        db["work_entries"].update(
            work_id,
            {
                "status": TaskStatus.FAILED.value,
                "current_phase": last_phase,
                "updated_at": datetime.now().isoformat(),
                "result": json.dumps({"error": str(e)}),
            },
        )
        audit.log_event(work_id, "work_failed", "dispatcher", {"error": str(e)})

        # ── Push failure event to WebSocket bus ──
        try:
            from spine.ui.ws_bus import get_bus

            get_bus().publish_sync(
                "work_failed",
                {"work_id": work_id, "error": str(e)},
            )
        except Exception:
            pass
        return {
            "work_id": work_id,
            "status": TaskStatus.FAILED.value,
            "work_type": work_type,
            "error": str(e),
        }


# ── Resume work ──


async def resume_work(
    work_id: str,
    human_feedback: str,
    action: str = "rework",
    config: SpineConfig | None = None,
) -> dict[str, Any]:
    """Resume a work item that is in ``needs_review`` status.

    Restarts the workflow from the beginning with the full accumulated
    state (prior artifacts, critic feedback, and the new human feedback)
    injected into the initial state.  Phases that have already produced
    artifacts will see them on disk and refine them based on the feedback
    rather than generating from scratch.

    Args:
        work_id: The work item ID to resume.
        human_feedback: The human's review input / decision.
        action: Resume action — ``"rework"`` (default) reruns from the
            phase that was flagged, ``"approve"`` forces the workflow
            to proceed without rework.
        config: Optional SpineConfig.

    Returns:
        A dict with keys: ``work_id``, ``status``, ``work_type``.

    Raises:
        ValueError: If the work item is not in ``needs_review`` status.
    """
    if config is None:
        config = SpineConfig.load()
    config.ensure_dirs()

    db = _get_work_db(config)

    # Validate the work item exists and is in needs_review
    try:
        entry = db["work_entries"].get(work_id)
    except sqlite_utils.db.NotFoundError:
        raise ValueError(f"Work item '{work_id}' not found")

    if entry.get("status") != "needs_review":
        raise ValueError(
            f"Work item '{work_id}' is in '{entry.get('status')}' status, "
            f"not 'needs_review'. Only needs_review items can be resumed."
        )

    work_type = entry.get("work_type", "spec")
    description = entry.get("description", "")

    audit = AuditService(db_path=str(Path(config.queue_path).parent / "audit.db"))
    artifacts = ArtifactStore(base_path=config.artifact_path)

    audit.log_event(
        work_id,
        "work_resumed",
        "dispatcher",
        {
            "human_feedback": human_feedback[:200],
            "action": action,
        },
    )

    # Load the existing checkpoint to recover accumulated state
    from spine.persistence.checkpoint import CheckpointStore

    checkpoint_store = CheckpointStore(db_path=config.checkpoint_path)
    saved_state = await checkpoint_store.get_state(work_id)

    # Build initial state for the resumed run, seeded with the
    # accumulated artifacts and feedback from the previous run.
    if saved_state:
        prior_artifacts = saved_state.get("artifacts", {})
        prior_feedback = saved_state.get("feedback", [])
        prior_retry_count = saved_state.get("retry_count", {})
    else:
        # Fallback: reconstruct from work entry result
        result_data = entry.get("result", {})
        if isinstance(result_data, str):
            try:
                result_data = json.loads(result_data)
            except json.JSONDecodeError:
                result_data = {}
        prior_artifacts = {}
        prior_feedback = []
        prior_retry_count = {}

    # Append the human feedback to the accumulated feedback list
    human_review_entry = {
        "status": "needs_revision" if action == "rework" else "passed",
        "tier": "human",
        "reason": human_feedback,
        "suggestions": [],
    }
    all_feedback = list(prior_feedback) + [human_review_entry]

    # Mark the work entry as running again
    db["work_entries"].update(
        work_id,
        {
            "status": TaskStatus.RUNNING.value,
            "current_phase": "",
            "updated_at": datetime.now().isoformat(),
        },
    )

    # ── Rebuild and re-run the workflow graph ──
    try:
        from spine.workflow.compose import build_workflow_graph

        checkpointer = await checkpoint_store.get_checkpointer()
        graph = build_workflow_graph(work_type, checkpointer=checkpointer)

        # Seed the new run with all prior state plus the human feedback
        resume_state: dict[str, Any] = {
            "work_id": work_id,
            "work_type": work_type,
            "description": description,
            "current_phase": "",
            "phase_index": 0,
            "retry_count": prior_retry_count,
            "max_retries": config.max_critic_retries,
            "artifacts": prior_artifacts,
            "feedback": all_feedback,
            "status": "running",
            "prompt_request": None,
            "critic_reviewing": "",
            "workspace_root": config.workspace_root,
        }

        thread_config = {
            "configurable": {
                "thread_id": work_id,
                # Per-phase model resolution happens inside each phase's
                # build_phase_agent() via resolve_model(phase=...).  Do NOT
                # inject a "model" key here — it would short-circuit
                # _model_spec_from_config() in helpers.py and force every
                # phase to use the default provider, ignoring
                # providers.phases.<phase> overrides.
                "spine_config": config,
            }
        }

        # Stream the graph, updating the work entry after each phase
        # (same pattern as submit_work — uses stream_mode=["updates", "messages"]
        # with subgraphs=True and version="v2" for consistent StreamPart format)
        result: dict[str, Any] = dict(resume_state)
        async for chunk in graph.astream(
            resume_state,
            thread_config,
            stream_mode=["updates", "messages"],
            subgraphs=True,
            version="v2",
        ):
            # V2 format: each chunk is a StreamPart dict:
            #   {"type": "updates"|"messages", "ns": (...), "data": ...}
            # Skip non-update chunks.  Only process root-level updates
            # (ns == ()) — subgraph updates come from Deep Agent internals.
            if not isinstance(chunk, dict) or chunk.get("type") != "updates":
                continue
            ns = chunk.get("ns", ())
            if ns != ():
                continue

            data = chunk.get("data", {})
            for node_name, node_output in data.items():
                # Deep-merge artifacts
                node_artifacts = node_output.get("artifacts")
                if node_artifacts and isinstance(node_artifacts, dict):
                    existing = result.get("artifacts", {})
                    if not isinstance(existing, dict):
                        existing = {}
                    merged = {**existing, **node_artifacts}
                    for key in set(existing) & set(node_artifacts):
                        if isinstance(existing[key], dict) and isinstance(
                            node_artifacts[key], dict
                        ):
                            merged[key] = {**existing[key], **node_artifacts[key]}
                    node_output = {**node_output, "artifacts": merged}

                # Accumulate feedback
                node_feedback = node_output.get("feedback")
                if node_feedback and isinstance(node_feedback, list):
                    existing_fb = result.get("feedback", [])
                    if not isinstance(existing_fb, list):
                        existing_fb = []
                    node_output = {
                        **node_output,
                        "feedback": existing_fb + node_feedback,
                    }

                result.update(node_output)
                phase = node_output.get("current_phase", "")
                status = node_output.get("status", "")
                if phase or status:
                    _update_work_progress(db, work_id, phase, status)
                    logger.info(
                        f"[{work_id}] Resume: Phase {phase or node_name} → {status}"
                    )
                    audit.log_event(
                        work_id,
                        "phase_completed",
                        node_name,
                        {"phase": phase, "status": status},
                    )

                # Persist artifacts
                node_artifacts = node_output.get("artifacts")
                if node_artifacts and isinstance(node_artifacts, dict):
                    for art_phase, phase_arts in node_artifacts.items():
                        if not isinstance(phase_arts, dict):
                            continue
                        for art_name, art_content in phase_arts.items():
                            if art_content is not None:
                                artifacts.save_artifact(
                                    work_id, art_phase, art_name, str(art_content)
                                )

        # Derive final status (same logic as submit_work)
        final_status = result.get("status", "completed")
        final_phase = result.get("current_phase", "")
        result_artifacts = result.get("artifacts", {})
        feedback = result.get("feedback", [])

        if final_status == "running":
            final_status = "completed"

        if any(
            isinstance(f, dict) and f.get("status") == "needs_review"
            for f in feedback
        ):
            final_status = "needs_review"

        db["work_entries"].update(
            work_id,
            {
                "status": final_status,
                "current_phase": final_phase,
                "updated_at": datetime.now().isoformat(),
                "result": json.dumps(
                    {
                        "artifacts": {
                            k: list(v.keys()) for k, v in result_artifacts.items()
                        },
                        "feedback_count": len(feedback),
                        "prompt_request": result.get("prompt_request"),
                    }
                ),
            },
        )

        audit.log_event(
            work_id,
            "work_completed",
            final_phase,
            {"status": final_status, "resumed": True},
        )

        try:
            from spine.ui.ws_bus import get_bus

            get_bus().publish_sync(
                "work_completed",
                {"work_id": work_id, "status": final_status},
            )
        except Exception:
            pass

        return {
            "work_id": work_id,
            "status": final_status,
            "work_type": work_type,
        }

    except Exception as e:
        logger.error(f"Resume of work {work_id} failed: {e}", exc_info=True)
        last_phase = ""
        if isinstance(locals().get("result"), dict):
            last_phase = result.get("current_phase", "")
        db["work_entries"].update(
            work_id,
            {
                "status": TaskStatus.FAILED.value,
                "current_phase": last_phase,
                "updated_at": datetime.now().isoformat(),
                "result": json.dumps({"error": str(e), "resumed": True}),
            },
        )
        audit.log_event(
            work_id, "work_failed", "dispatcher", {"error": str(e), "resumed": True}
        )

        try:
            from spine.ui.ws_bus import get_bus

            get_bus().publish_sync(
                "work_failed",
                {"work_id": work_id, "error": str(e)},
            )
        except Exception:
            pass

        return {
            "work_id": work_id,
            "status": TaskStatus.FAILED.value,
            "work_type": work_type,
            "error": str(e),
        }



# ── Resume interrupted work (using Command + interrupt) ──

async def resume_interrupted_work(
    work_id: str,
    action: str,
    feedback: str,
    config: SpineConfig | None = None,
) -> dict[str, Any]:
    """Resume a workflow that hit an ``interrupt()`` for human review.

    Uses LangGraph's ``Command(resume=...)`` to continue from the
    interrupt point without restarting the entire graph.

    Args:
        work_id: The work item ID.
        action: ``"rework"``, ``"approve"``, or ``"abort"``.
        feedback: Human review text.
        config: Optional SpineConfig.

    Returns:
        Dict with ``work_id``, ``status``, ``work_type``.
    """
    if config is None:
        config = SpineConfig.load()
    config.ensure_dirs()

    db = _get_work_db(config)

    try:
        entry = db["work_entries"].get(work_id)
    except sqlite_utils.db.NotFoundError:
        raise ValueError(f"Work item '{work_id}' not found")

    work_type = entry.get("work_type", "spec")

    from spine.persistence.checkpoint import CheckpointStore
    from spine.workflow.compose import build_workflow_graph
    from langgraph.types import Command

    checkpoint_store = CheckpointStore(db_path=config.checkpoint_path)
    checkpointer = await checkpoint_store.get_checkpointer()
    graph = build_workflow_graph(work_type, checkpointer=checkpointer)

    thread_config = {
        "configurable": {
            "thread_id": work_id,
            # Per-phase model resolution happens inside each phase's
            # build_phase_agent() via resolve_model(phase=...).  Do NOT
            # inject a "model" key here — it would short-circuit
            # _model_spec_from_config() in helpers.py and force every
            # phase to use the default provider, ignoring
            # providers.phases.<phase> overrides.
            "spine_config": config,
        }
    }

    command = Command(resume={"action": action, "feedback": feedback})

    audit = AuditService(db_path=str(Path(config.queue_path).parent / "audit.db"))
    audit.log_event(
        work_id,
        "work_resumed_interrupt",
        "dispatcher",
        {"action": action, "feedback": feedback[:200]},
    )

    # Stream the rest of the graph from the interrupt point
    result: dict[str, Any] = {}
    async for chunk in graph.astream(
        command,
        thread_config,
        stream_mode=["updates", "messages"],
        subgraphs=True,
        version="v2",
    ):
        if not isinstance(chunk, dict) or chunk.get("type") != "updates":
            continue
        if chunk.get("ns", ()) != ():
            continue

        data = chunk.get("data", {})
        for _node_name, node_output in data.items():
            result.update(node_output)
            phase = node_output.get("current_phase", "")
            status = node_output.get("status", "")
            if phase or status:
                _update_work_progress(db, work_id, phase, status)

    final_status = result.get("status", "completed")
    final_phase = result.get("current_phase", "")

    db["work_entries"].update(
        work_id,
        {
            "status": final_status,
            "current_phase": final_phase,
            "updated_at": datetime.now().isoformat(),
        },
    )

    audit.log_event(
        work_id,
        "work_completed",
        final_phase,
        {"status": final_status, "resumed_from_interrupt": True},
    )

    return {
        "work_id": work_id,
        "status": final_status,
        "work_type": work_type,
    }


# ── Restart work ──


async def restart_work(
    work_id: str,
    config: SpineConfig | None = None,
    *,
    clear_artifacts: bool = False,
) -> dict[str, Any]:
    """Restart a work item that is running, stalled, or needs_review.

    Unlike ``resume_work`` (which continues from a checkpoint with human
    feedback), ``restart_work`` re-runs the workflow from phase 0.  It
    is intended for items whose worker or UI died mid-execution.

    Steps:
      1. Validates the work item exists and is in a restartable status.
      2. Optionally clears on-disk artifacts and the checkpoint.
      3. Resets the work entry status to "running".
      4. Rebuilds the workflow graph and re-invokes it with fresh initial state.

    Args:
        work_id: The work item ID to restart.
        config: Optional SpineConfig.
        clear_artifacts: If True, delete on-disk artifacts before restarting.
            Default False preserves them so downstream phases can re-use
            what was already produced.

    Returns:
        A dict with keys ``work_id``, ``status``, ``work_type``.

    Raises:
        ValueError: If the work item is not in a restartable status.
    """
    if config is None:
        config = SpineConfig.load()
    config.ensure_dirs()

    db = _get_work_db(config)

    # ── Validate work item ──
    try:
        entry = db["work_entries"].get(work_id)
    except sqlite_utils.db.NotFoundError:
        raise ValueError(f"Work item '{work_id}' not found")

    status = entry.get("status", "")
    restartable = (
        TaskStatus.RUNNING.value,
        TaskStatus.STALLED.value,
        TaskStatus.NEEDS_REVIEW.value,
    )
    if status not in restartable:
        raise ValueError(
            f"Work item '{work_id}' is in '{status}' status — "
            f"only {restartable} items can be restarted."
        )

    work_type = entry.get("work_type", "spec")
    description = entry.get("description", "")

    audit = AuditService(db_path=str(Path(config.queue_path).parent / "audit.db"))
    artifact_store = ArtifactStore(base_path=config.artifact_path)

    audit.log_event(
        work_id,
        "work_restarted",
        "dispatcher",
        {"previous_status": status, "clear_artifacts": clear_artifacts},
    )

    # ── Optionally wipe on-disk artifacts ──
    if clear_artifacts:
        work_dir = Path(artifact_store._base) / work_id
        if work_dir.exists():
            for f in work_dir.rglob("*"):
                if f.is_file():
                    f.unlink()
            logger.info(f"[{work_id}] Cleared on-disk artifacts")

    # Purge LangGraph checkpoint so the graph starts from phase 0
    from spine.persistence.checkpoint import CheckpointStore

    checkpoint_store = CheckpointStore(db_path=config.checkpoint_path)
    saver = await checkpoint_store.get_checkpointer()
    await saver.adelete_thread(work_id)
    logger.info(f"[{work_id}] Purged checkpoint")

    # ── Rebuild initial state (fresh start) ──
    initial_state: dict[str, Any] = {
        "work_id": work_id,
        "work_type": work_type,
        "description": description,
        "current_phase": "",
        "phase_index": 0,
        "retry_count": {},
        "max_retries": config.max_critic_retries,
        "artifacts": {},
        "feedback": [],
        "status": "running",
        "prompt_request": None,
        "critic_reviewing": "",
        "workspace_root": config.workspace_root,
    }

    # Mark as running in the work entries DB
    db["work_entries"].update(
        work_id,
        {
            "status": TaskStatus.RUNNING.value,
            "current_phase": "",
            "updated_at": datetime.now().isoformat(),
            "result": "{}",
        },
    )

    # ── Rebuild and re-run the workflow graph ──
    return await _run_workflow_graph(
        work_id=work_id,
        work_type=work_type,
        config=config,
        db=db,
        audit=audit,
        artifact_store=artifact_store,
        initial_state=initial_state,
        checkpoint_store=checkpoint_store,
        is_restart=True,
    )


# ── Shared workflow execution logic (used by submit_work, resume_work, restart_work) ──


async def _run_workflow_graph(
    *,
    work_id: str,
    work_type: str,
    config: SpineConfig,
    db: sqlite_utils.Database,
    audit: AuditService,
    artifact_store: ArtifactStore,
    initial_state: dict[str, Any],
    checkpoint_store: CheckpointStore,
    is_restart: bool = False,
) -> dict[str, Any]:
    """Run a workflow graph to completion, streaming updates.

    This shared helper avoids the ~100-line duplication between
    ``submit_work``, ``resume_work``, and ``restart_work``.
    """
    from spine.workflow.compose import build_workflow_graph

    graph = build_workflow_graph(work_type, checkpointer=await checkpoint_store.get_checkpointer())

    thread_config = {
        "configurable": {
            "thread_id": work_id,
            # Per-phase model resolution happens inside each phase's
            # build_phase_agent() via resolve_model(phase=...).  Do NOT
            # inject a "model" key here — it would short-circuit
            # _model_spec_from_config() in helpers.py and force every
            # phase to use the default provider, ignoring
            # providers.phases.<phase> overrides.
            "spine_config": config,
        }
    }

    result: dict[str, Any] = dict(initial_state)
    stream_iter = graph.astream(
        initial_state,
        thread_config,
        stream_mode=["updates", "messages"],
        subgraphs=True,
        version="v2",
    )
    stalled = False

    while True:
        try:
            chunk = await asyncio.wait_for(
                stream_iter.__anext__(),
                timeout=_STALL_TIMEOUT_SECONDS,
            )
        except StopAsyncIteration:
            break
        except asyncio.TimeoutError:
            last_phase = result.get("current_phase", "")
            logger.error(
                f"[{work_id}] Workflow stalled — no chunk received for "
                f"{_STALL_TIMEOUT_SECONDS}s (last phase: {last_phase}). "
                f"Marking as stalled."
            )
            stalled = True
            _update_work_progress(db, work_id, last_phase, "stalled")
            audit.log_event(
                work_id,
                "work_stalled",
                "dispatcher",
                {"last_phase": last_phase, "timeout": _STALL_TIMEOUT_SECONDS},
            )
            break

        if not isinstance(chunk, dict) or chunk.get("type") != "updates":
            continue

        ns = chunk.get("ns", ())
        if ns != ():
            continue

        data = chunk.get("data", {})
        for node_name, node_output in data.items():
            node_artifacts = node_output.get("artifacts")
            if node_artifacts and isinstance(node_artifacts, dict):
                existing = result.get("artifacts", {})
                if not isinstance(existing, dict):
                    existing = {}
                merged = {**existing, **node_artifacts}
                for key in set(existing) & set(node_artifacts):
                    if isinstance(existing[key], dict) and isinstance(
                        node_artifacts[key], dict
                    ):
                        merged[key] = {**existing[key], **node_artifacts[key]}
                node_output = {**node_output, "artifacts": merged}

            node_feedback = node_output.get("feedback")
            if node_feedback and isinstance(node_feedback, list):
                existing_fb = result.get("feedback", [])
                if not isinstance(existing_fb, list):
                    existing_fb = []
                node_output = {
                    **node_output,
                    "feedback": existing_fb + node_feedback,
                }

            result.update(node_output)
            phase = node_output.get("current_phase", "")
            status = node_output.get("status", "")
            if phase or status:
                _update_work_progress(db, work_id, phase, status)
                logger.info(f"[{work_id}] Phase {phase or node_name} → {status}")
                audit.log_event(
                    work_id,
                    "phase_completed",
                    node_name,
                    {"phase": phase, "status": status},
                )

            node_artifacts = node_output.get("artifacts")
            if node_artifacts and isinstance(node_artifacts, dict):
                for art_phase, phase_arts in node_artifacts.items():
                    if not isinstance(phase_arts, dict):
                        continue
                    for art_name, art_content in phase_arts.items():
                        if art_content is not None:
                            artifact_store.save_artifact(
                                work_id, art_phase, art_name, str(art_content)
                            )

    # ── Final status ──
    if stalled:
        final_status = "stalled"
    else:
        final_status = result.get("status", "completed")

    final_phase = result.get("current_phase", "")
    result_artifacts = result.get("artifacts", {})
    feedback = result.get("feedback", [])

    if final_status == "running":
        final_status = "completed"

    if any(
        isinstance(f, dict) and f.get("status") == "needs_review"
        for f in feedback
    ):
        final_status = "needs_review"

    result_payload = {
        "artifacts": {k: list(v.keys()) for k, v in result_artifacts.items()},
        "feedback_count": len(feedback),
        "prompt_request": result.get("prompt_request"),
    }
    if is_restart:
        result_payload["restarted"] = True

    db["work_entries"].update(
        work_id,
        {
            "status": final_status,
            "current_phase": final_phase,
            "updated_at": datetime.now().isoformat(),
            "result": json.dumps(result_payload),
        },
    )

    audit.log_event(
        work_id,
        "work_completed",
        final_phase,
        {"status": final_status, "restarted": is_restart},
    )

    try:
        from spine.ui.ws_bus import get_bus

        get_bus().publish_sync(
            "work_completed",
            {"work_id": work_id, "status": final_status},
        )
    except Exception:
        pass

    return {
        "work_id": work_id,
        "status": final_status,
        "work_type": work_type,
        **({"restarted": True} if is_restart else {}),
    }


# ── Query work ──


def get_work_status(work_id: str, config: SpineConfig | None = None) -> dict[str, Any] | None:
    """Get the status of a work item.

    Args:
        work_id: The work item ID.
        config: Optional SpineConfig.

    Returns:
        A dict with work entry fields, or None if not found.
    """
    if config is None:
        config = SpineConfig.load()

    db = _get_work_db(config)
    try:
        row = db["work_entries"].get(work_id)
        if row and row.get("result"):
            row["result"] = json.loads(row["result"])
        return row
    except sqlite_utils.db.NotFoundError:
        return None


def list_work(
    status: str | None = None,
    limit: int = 50,
    config: SpineConfig | None = None,
) -> list[dict[str, Any]]:
    """List work items, optionally filtered by status.

    Args:
        status: Filter by status (e.g. "running", "completed", "needs_review").
        limit: Maximum number of items to return.
        config: Optional SpineConfig.

    Returns:
        A list of work entry dicts, newest first.
    """
    if config is None:
        config = SpineConfig.load()

    db = _get_work_db(config)

    # Use raw SQL so we can include `rowid` for the tiebreaker and use
    # NULLS LAST (SQLite 3.30+).  The `rows_where` API doesn't expose
    # rowid or support NULLS LAST natively.
    if status:
        rows = list(
            db.query(
                "SELECT rowid, * FROM work_entries "
                "WHERE status = ? "
                "ORDER BY created_at DESC NULLS LAST, rowid DESC "
                "LIMIT ?",
                [status, limit],
            )
        )
    else:
        rows = list(
            db.query(
                "SELECT rowid, * FROM work_entries "
                "ORDER BY created_at DESC NULLS LAST, rowid DESC "
                "LIMIT ?",
                [limit],
            )
        )

    results = []
    for row in rows:
        if row.get("result"):
            try:
                row["result"] = json.loads(row["result"])
            except json.JSONDecodeError:
                pass
        results.append(row)

    # ── Post-query Python safety-net sort ─────────────────────────────────
    # Handles edge cases where SQLite ordering behaves unexpectedly
    # (e.g. different sqlite-utils version, column type mismatch).
    #
    # Sort key: (group, created_at, rowid)
    #   - group 1 for entries with a valid created_at
    #   - group 0 for NULL created_at (sorts after group 1 in DESC)
    #   - created_at as-is for ISO-8601 lexicographic ordering
    #   - rowid as tiebreaker (higher = newer = first in DESC)
    # With reverse=True, larger keys sort first.
    def _sort_key(r: dict[str, Any]) -> tuple[int, str, int]:
        """Sort key: newest-first, NULL timestamps at end."""
        ts = r.get("created_at")
        rid = r.get("rowid", 0) or 0
        if ts is None:
            return (0, "", rid)  # group 0 → last in descending order
        return (1, ts, rid)  # group 1 → newest-first descending

    results.sort(key=_sort_key, reverse=True)

    return results


def update_work_status(
    work_id: str,
    status: str,
    current_phase: str | None = None,
    config: SpineConfig | None = None,
) -> None:
    """Update the status of a work item.

    Args:
        work_id: The work item ID.
        status: New status value.
        current_phase: Optional updated phase name.
        config: Optional SpineConfig.
    """
    if config is None:
        config = SpineConfig.load()

    db = _get_work_db(config)
    updates: dict[str, Any] = {
        "status": status,
        "updated_at": datetime.now().isoformat(),
    }
    if current_phase is not None:
        updates["current_phase"] = current_phase

    db["work_entries"].update(work_id, updates)


# ── Split a plan into execution work items ──


async def split_work_plan(
    plan_id: str,
    tasks_text: str | None = None,
    description_override: str | None = None,
    work_type_override: str = "quick",
    config: SpineConfig | None = None,
) -> list[dict[str, Any]]:
    """Split an approved planning work item into execution tasks.

    Reads the tasks artifact produced by the plan workflow (or the
    caller-supplied ``tasks_text``), parses it into individual work items,
    and submits each one as a new execution work item with ``plan_id`` set
    to the source plan's ID.

    Args:
        plan_id: The ID of the approved planning work item.
        tasks_text: Optional raw tasks text.  When ``None``, reads the
            tasks.md artifact from disk.
        description_override: If provided, all spawned tasks use this
            description instead of the description extracted from sections.
        work_type_override: The work_type for spawned items (default "quick").
        config: Optional SpineConfig.

    Returns:
        A list of dicts, each with ``work_id``, ``status``, ``work_type``,
        and ``description`` for the spawned items.
    """
    if config is None:
        config = SpineConfig.load()

    # Get tasks text from artifact if not provided
    if tasks_text is None:
        artifacts = ArtifactStore(base_path=config.artifact_path)
        tasks_path = artifacts.artifact_path(plan_id, "tasks", "tasks.md")
        try:
            tasks_text = Path(tasks_path).read_text(encoding="utf-8")
        except FileNotFoundError:
            tasks_path = artifacts.artifact_path(plan_id, "tasks", "tasks.txt")
            tasks_text = Path(tasks_path).read_text(encoding="utf-8")

    # Parse tasks — each section starting with ## or # followed by text
    # becomes a separate work item. Simple heuristic: split on
    # lines that look like "# Section" or "## Section".
    sections: list[tuple[str, str]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for line in tasks_text.splitlines():
        if line.startswith(("### ", "## ", "# ")) and not line.startswith("# Slice") and not line.startswith("# Tasks"):
            # Save previous section
            if current_title is not None:
                sections.append((current_title, "\n".join(current_lines).strip()))
            # Extract new section title
            current_title = line.lstrip("# ").strip().split("\n")[0]
            current_lines = []
        elif current_title is not None:
            current_lines.append(line)

    # Don't forget the last section
    if current_title is not None and current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    spawned: list[dict[str, Any]] = []
    for title, content in sections:
        description = description_override or f"{title}: {content[:500]}"
        result = await submit_work(
            description=description,
            work_type=work_type_override,
            config=config,
            plan_id=plan_id,
        )
        spawned.append({
            **result,
            "description": description[:200],
        })

    # Update the plan work entry with spawned task IDs
    db = _get_work_db(config)
    db["work_entries"].update(
        plan_id,
        {
            "result": json.dumps({
                "split": True,
                "spawned_ids": [s["work_id"] for s in spawned],
            }),
            "updated_at": datetime.now().isoformat(),
        },
    )

    return spawned


def list_plans(
    status: str | None = None,
    limit: int = 50,
    config: SpineConfig | None = None,
) -> list[dict[str, Any]]:
    """List planning work items (work_type = 'plan' or 'plan_spec').

    Args:
        status: Optional filter by status (e.g. 'completed', 'needs_review').
        limit: Maximum number of results to return.
        config: Optional SpineConfig.

    Returns:
        List of planning work item dicts.
    """
    if config is None:
        config = SpineConfig.load()

    db = _get_work_db(config)
    conditions = "work_type IN ('plan', 'plan_spec')"
    params: dict[str, Any] = {"limit": limit}

    if status:
        conditions += " AND status = :status"
        params["status"] = status

    sql = f"SELECT * FROM work_entries WHERE {conditions} ORDER BY created_at DESC LIMIT :limit"
    return list(db.query(sql, params))


# ── Resume work ──

# (resume_work is defined above)