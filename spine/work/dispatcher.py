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


# ── Submit work ──


async def submit_work(
    description: str,
    work_type: str = "spec",
    config: SpineConfig | None = None,
) -> dict[str, Any]:
    """Submit a new work item for processing.

    This is the unified entry point for CLI, UI, and worker. It:
    1. Creates a work entry with a unique ID
    2. Builds the workflow graph for the given work type
    3. Invokes the graph with checkpoint persistence
    4. Returns the work ID and initial state

    Args:
        description: The work description / prompt.
        work_type: One of "quick", "critical_quick", "spec", "critical_spec".
        config: Optional SpineConfig (loads from default if not provided).

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

    # Record the work entry
    db = _get_work_db(config)
    db["work_entries"].insert(
        {
            "id": work_id,
            "description": description,
            "work_type": work_type,
            "status": TaskStatus.RUNNING.value,
            "current_phase": "",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
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
        result = await graph.ainvoke(initial_state, thread_config)

        # Update work entry with results
        final_status = result.get("status", "completed")
        final_phase = result.get("current_phase", "")
        result_artifacts = result.get("artifacts", {})

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
                        "feedback_count": len(result.get("feedback", [])),
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

        return {
            "work_id": work_id,
            "status": final_status,
            "work_type": work_type,
        }

    except Exception as e:
        logger.error(f"Work {work_id} failed: {e}", exc_info=True)
        db["work_entries"].update(
            work_id,
            {
                "status": TaskStatus.FAILED.value,
                "updated_at": datetime.now().isoformat(),
                "result": json.dumps({"error": str(e)}),
            },
        )
        audit.log_event(work_id, "work_failed", "dispatcher", {"error": str(e)})
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
    table = db["work_entries"]

    if status:
        rows = table.rows_where(
            "status = ?",
            [status],
            order_by="-created_at",
            limit=limit,
        )
    else:
        rows = table.rows_where(order_by="-created_at", limit=limit)

    results = []
    for row in rows:
        if row.get("result"):
            try:
                row["result"] = json.loads(row["result"])
            except json.JSONDecodeError:
                pass
        results.append(row)
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
