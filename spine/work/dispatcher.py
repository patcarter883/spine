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
        work_type: One of "quick", "critical_quick", "spec", "critical_spec".
        config: Optional SpineConfig (loads from default if not provided).
        created_at: Optional ISO timestamp for the work entry's ``created_at``
            field.  When ``None`` (default), uses the current time.

    Returns:
        A dict with keys: ``work_id``, ``status``, ``work_type``.
    """
    if config is None:
        config = SpineConfig.load()
    config.ensure_dirs()

    work_id = str(uuid.uuid4())[:8]
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
        }

        thread_config = {
            "configurable": {
                "thread_id": work_id,
                "model": config.resolve_model(),
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
            # data is {node_name: partial_state_update}
            for node_name, node_output in data.items():
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
                "model": config.resolve_model(),
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
