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
from contextlib import aclosing
from datetime import datetime
from pathlib import Path
from typing import Any

import sqlite_utils

from spine.config import SpineConfig
from spine.models.enums import PhaseName, TaskStatus, WorkType
from spine.observability import traced_astream
from spine.persistence.artifacts import ArtifactStore
from spine.persistence.sqlite_tuning import tune_connection
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
    db = tune_connection(sqlite_utils.Database(str(db_path)))

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


def _work_is_cancelled(db: sqlite_utils.Database, work_id: str) -> bool:
    """Return True if the work entry has been marked ``cancelled``.

    ``UIApi.stop_work`` marks the ``work_entries`` row ``cancelled`` as a
    cooperative-cancellation signal.  The dispatcher polls this at each stream
    boundary so a Stop Work click halts the run — and closes the LangSmith
    trace — promptly, instead of letting the graph stream (and its trace) run
    on until the stall timeout fires.  Mirrors the onboarding engine's
    ``_is_cancelled`` poll.
    """
    try:
        row = db["work_entries"].get(work_id)
    except Exception:
        return False
    return bool(row) and row.get("status") == TaskStatus.CANCELLED.value


# ── Phase-start update (public) ──────────────────────────────────────────
# Called at the START of each phase node so the UI shows the correct phase
# while work is in progress.


def get_work_db(config: SpineConfig) -> sqlite_utils.Database:
    """Get or create the work entries database.

    Public alias for ``_get_work_db`` so other modules can open the DB
    without importing a private function.

    Args:
        config: The SPINE configuration.

    Returns:
        A ``sqlite_utils.Database`` handle for the ``work_entries`` table.
    """
    return _get_work_db(config)


def update_work_phase_started(
    db: sqlite_utils.Database,
    work_id: str,
    current_phase: str,
) -> None:
    """Update work_entries to mark a phase as started (status='running').

    Called at the START of each phase node so the UI shows the correct
    phase while work is in progress.

    Args:
        db: The work database handle.
        work_id: The work item ID.
        current_phase: The phase that is about to start.
    """
    _update_work_progress(db, work_id, current_phase, "running")

    # Publish a dedicated phase_started event so UI can distinguish
    # start from completion.
    try:
        from spine.ui.ws_bus import get_bus

        get_bus().publish_sync(
            "phase_started",
            {"work_id": work_id, "current_phase": current_phase, "status": "running"},
        )
    except Exception:
        # Bus may not be initialised (CLI-only mode) — that's fine.
        pass


# Planning work types pause for human approval after critic_plan, before any
# execution tasks are spawned.
PLAN_TYPES = ("reviewed_task", "critical_reviewed_task")


def _derive_final_status(
    result: dict[str, Any],
    *,
    stalled: bool,
    feedback: list[Any],
    work_type: str,
) -> str:
    """Derive the terminal work status from a completed graph result.

    Shared by ``submit_work``, ``resume_interrupted_work``, and
    ``_run_workflow_graph_inner`` so reviewed-plan types consistently end at
    ``awaiting_approval`` regardless of which entry point ran the graph.
    """
    if stalled:
        return "stalled"
    status = result.get("status", "completed")
    # A graph that ends normally leaves status "running" — treat as completed.
    if status == "running":
        status = "completed"
    # The critic router sends to END with a needs_review feedback entry when
    # max retries are exceeded.
    if any(isinstance(f, dict) and f.get("status") == "needs_review" for f in feedback):
        status = "needs_review"
    # Planning work types must end as "awaiting_approval" so users can review
    # before execution tasks are spawned.
    if work_type in PLAN_TYPES and status == "completed":
        status = "awaiting_approval"
    return status


# ── Submit work ──


async def submit_work(
    description: str,
    work_type: str = "spec",
    config: SpineConfig | None = None,
    created_at: str | None = None,
    work_id: str | None = None,
    plan_id: str | None = None,
    project_id: str | None = None,
    phase_id: str | None = None,
    start: bool = True,
) -> dict[str, Any]:
    """Submit a new work item for processing.

    This is the unified entry point for CLI, UI, and worker. It:
    1. Creates a work entry with a unique ID
    2. Builds the workflow graph for the given work type
    3. Invokes the graph with checkpoint persistence
    4. Returns the work ID and initial state

    When ``start`` is False the work item is only *created*: the entry is
    recorded as ``pending`` and (if given) registered as a project member,
    but the workflow graph is NOT built or run. This lets a batch of project
    tasks be reviewed in the UI before any of them executes; each is later
    launched from phase 0 via ``restart_work`` (the UI "Start" button).

    When called from the background queue worker (RalphLoopWorker),
    ``created_at`` should be the original ``enqueued_at`` timestamp so
    the dashboard shows items in submission order, not processing order.

    Args:
        description: The work description / prompt.
        work_type: One of "task", "critical_task", "reviewed_task",
            "critical_reviewed_task".
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
        project_id: Optional project this work item belongs to. Orthogonal to
            ``plan_id`` — it is a project-membership back-reference, NOT a
            planning-source pointer, and the two never imply a hierarchy. When
            set, the work item is registered as a member of the project.
        phase_id: Optional roadmap phase (within ``project_id``) to assign this
            work item to. When set, the work item is added to that phase's
            ``member_work_ids`` in addition to the project membership. Requires
            ``project_id``; a missing project or unknown phase is logged and
            skipped rather than failing the submission.
        start: When True (default), build and run the workflow graph inline.
            When False, only create the work entry (status ``pending``) and
            register project membership, then return without running — the
            item is launched later via ``restart_work``.

    Returns:
        A dict with keys: ``work_id``, ``status``, ``work_type``.
    """
    if config is None:
        config = SpineConfig.load()
    config.ensure_dirs()

    work_id = work_id or str(uuid.uuid4())[:8]

    # ── Onboarding work routes off the LangGraph phase sequence ──
    # The onboarding engine is not a workflow graph — it analyses/scaffolds a
    # repo and synthesises docs directly.  The JSON description carries its
    # parameters.  run_onboarding returns the same {work_id, status, work_type}
    # dict the queue worker's _loop already consumes, so queue-row finalisation
    # is unchanged.
    if work_type == "onboarding":
        try:
            params = json.loads(description)
        except (json.JSONDecodeError, TypeError):
            params = {}
        from spine.work.onboarding.engine import run_onboarding

        return await run_onboarding(
            workspace_root=params.get("workspace_root", config.workspace_root),
            mode=params.get("mode", "brownfield"),
            tech_stack=params.get("tech_stack"),
            config=config,
            work_id=work_id,
        )

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

    # Ensure the project_id column exists (migration). project_id is a project-
    # membership back-reference, orthogonal to plan_id — see submit_work docstring.
    if "project_id" not in db["work_entries"].columns_dict:
        try:
            db["work_entries"].add_column("project_id", str)
        except Exception:
            logger.warning("Could not add project_id column — may already exist", exc_info=True)

    db["work_entries"].insert(
        {
            "id": work_id,
            "description": description,
            "work_type": work_type,
            "status": (TaskStatus.RUNNING if start else TaskStatus.PENDING).value,
            "current_phase": "",
            "created_at": now,
            "updated_at": now,
            "result": "{}",
            "plan_id": plan_id,
            "project_id": project_id,
        }
    )

    # Register membership in the project store (source of truth). The DB column
    # above is only a reverse-lookup convenience. Missing project → warn, don't
    # fail the submission. When a phase_id is given, route through
    # add_phase_members so the item joins both the project and the phase in a
    # single locked read-modify-write.
    if project_id:
        try:
            from spine.persistence.project_store import ProjectStore

            store = ProjectStore(base_path=config.project_path)
            if phase_id:
                store.add_phase_members(project_id, phase_id, [work_id])
            else:
                store.add_members(project_id, [work_id])
        except KeyError:
            logger.warning(
                "Work %s submitted with project_id=%s but no such project exists; "
                "membership not recorded in the project store.",
                work_id,
                project_id,
            )
        except ValueError:
            # Project exists but the phase does not — still record plain
            # membership so the work isn't orphaned, then warn about the phase.
            try:
                store.add_members(project_id, [work_id])
            except KeyError:
                pass
            logger.warning(
                "Work %s submitted with project_id=%s phase_id=%s but that "
                "project has no such phase; recorded project membership only.",
                work_id,
                project_id,
                phase_id,
            )
    elif phase_id:
        logger.warning(
            "Work %s submitted with phase_id=%s but no project_id; phase "
            "assignment ignored (a phase belongs to a project).",
            work_id,
            phase_id,
        )

    # ── Create-only mode ──
    # When start is False the item is parked as ``pending`` for review in the
    # UI; do NOT build or run the graph. It is launched later from phase 0 via
    # restart_work (the UI "Start" button).
    if not start:
        audit.log_event(
            work_id,
            "work_created",
            "dispatcher",
            {"project_id": project_id, "deferred": True},
        )
        return {
            "work_id": work_id,
            "status": TaskStatus.PENDING.value,
            "work_type": work_type,
        }

    # Build and run the workflow graph
    try:
        from spine.persistence.checkpoint import CheckpointStore
        from spine.workflow.compose import build_workflow_graph
        from spine.agents.retry import reset_conn_breaker, reset_token_budget

        from spine.git import WorktreeSandbox

        checkpoint_store = CheckpointStore(db_path=config.checkpoint_path)
        checkpointer = await checkpoint_store.get_checkpointer()

        # Clear any stale cumulative token count from a prior run of the
        # same work_id (e.g. a restart) so the budget enforcer starts fresh.
        reset_token_budget(work_id)
        # Likewise clear the connection-failure circuit breaker so a prior
        # run's down-server failures don't trip a fresh run on its first calls.
        reset_conn_breaker()

        # ── Mandatory worktree sandbox for code-producing work ──
        # Work types that run IMPLEMENT edit the repo, so the graph runs
        # against an isolated git worktree and is fast-forward merged to
        # main only on success (rolled back on any other outcome). For
        # planning / onboarding work types this is a no-op and run_config
        # is the original config.
        sandbox = WorktreeSandbox(config, work_type)
        run_config = sandbox.enter()

        graph = build_workflow_graph(work_type, checkpointer=checkpointer)

        initial_state = {
            "work_id": work_id,
            "work_type": work_type,
            "description": description,
            "current_phase": "",
            "phase_index": 0,
            "retry_count": {},
            "max_retries": config.max_critic_retries,
            "max_adversarial_retries": config.max_adversarial_retries,
            "adversarial_retry_count": 0,
            "artifacts": {},
            "feedback": [],
            "status": "running",
            "prompt_request": None,
            "critic_reviewing": "",
            "workspace_root": run_config.workspace_root,
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
                "spine_config": run_config,
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
        # Trace this work run to LangSmith.  Tracing is off process-wide by
        # default (see spine.config / spine.observability); only genuine work
        # task runs opt in by streaming through traced_astream.
        result: dict[str, Any] = dict(initial_state)
        stream_iter = traced_astream(
            graph.astream(
                initial_state,
                thread_config,
                stream_mode=["updates", "messages"],
                subgraphs=True,
                version="v2",
            ),
            work_id,
            work_type,
        )
        stalled = False
        cancelled = False
        while True:
            # Cooperative cancellation: a Stop Work click marks the work entry
            # ``cancelled``.  Poll at each stream boundary so we break promptly
            # (within one node) and close the trace, instead of streaming on
            # until the stall timeout.
            if _work_is_cancelled(db, work_id):
                cancelled = True
                logger.info(f"[{work_id}] Work cancelled — halting stream")
                audit.log_event(
                    work_id,
                    "work_cancelled",
                    "dispatcher",
                    {"last_phase": result.get("current_phase", "")},
                )
                break
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

        # Close the stream generator so the LangSmith trace tears down now.
        # On natural completion it's already exhausted (no-op); on a cancel or
        # stall break it's left suspended at a ``yield`` — ``aclose()`` runs the
        # ``traced_astream`` teardown immediately so the trace is closed at stop
        # time instead of lingering until garbage collection.
        try:
            await stream_iter.aclose()
        except Exception:
            logger.debug(f"[{work_id}] stream aclose raised", exc_info=True)

        # Update work entry with final results.
        # Derive the terminal status from the graph state:
        #   - If cancelled (Stop Work), the status stays "cancelled" — a
        #     user-requested stop is terminal and must not be overwritten.
        #   - If stalled, the status was already set to "stalled" above.
        #   - If the critic routed to needs_review → END, the feedback
        #     contains a needs_review entry.
        #   - If the graph completed naturally (all phases ran), status
        #     should be "completed" or the verify phase's final status.
        #   - The top-level except (below) catches true failures.
        final_phase = result.get("current_phase", "")
        result_artifacts = result.get("artifacts", {})
        feedback = result.get("feedback", [])
        final_status = (
            TaskStatus.CANCELLED.value
            if cancelled
            else _derive_final_status(
                result, stalled=stalled, feedback=feedback, work_type=work_type
            )
        )

        # Save artifacts to disk
        for phase, phase_artifacts in result_artifacts.items():
            for name, content in phase_artifacts.items():
                if content is not None:
                    artifacts.save_artifact(work_id, phase, name, str(content))

        # Filter feedback to only include needs_review entries for display
        needs_review_feedback = [
            f for f in feedback if isinstance(f, dict) and f.get("status") == "needs_review"
        ]

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
                        "feedback": needs_review_feedback,
                        # The critic's final verdict — kept regardless of status
                        # so reviewers can see it for awaiting_approval plans
                        # (which PASS, and so leave needs_review_feedback empty).
                        "last_critic_review": result.get("last_critic_review"),
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

        # ── Capture cross-run experience ──
        # Distil this run's critic feedback into reusable lessons, persisted to
        # the MAIN repo root (base `config`, not the worktree) so they survive
        # the sandbox rollback below. Best-effort — never raises.
        from spine.agents.experience import capture_run_experience

        await capture_run_experience(result, config, final_status)

        # ── Land or discard the sandbox patch ──
        # Code-producing work ran against an isolated worktree; merge it to
        # main on success, roll it back on any other terminal status. No-op
        # for non-code work types.
        sandbox.finalize(final_status)

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
        # Discard the isolated worktree (if one was created) so a crashed
        # run never leaves the main tree dirty. Best-effort; never masks the
        # original error.
        _sandbox = locals().get("sandbox")
        if _sandbox is not None:
            _sandbox.abort()
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


def _preflight_worktree_clean(config: SpineConfig, work_type: str) -> None:
    """Fail fast (and retryably) if a code-producing run can't be isolated.

    Code-producing work types run against an isolated git worktree, which
    requires a clean main tree. Re-run entry points (resume / restart /
    restart-from-phase / approved-plan continuation) mutate the work entry's
    status to ``running`` — and perform other destructive setup like wiping
    artifacts and purging the checkpoint — *before* the worktree is created
    deep inside the graph runner. If the tree is dirty, that creation fails
    and the entry is finalised to ``failed``, which is no longer in the
    status set its own retry path accepts: the work is stranded and the user
    cannot retry even after cleaning the tree.

    Calling this *before* any such side effect makes a dirty tree raise while
    the entry is still in its original, retryable status — so the user can
    stash/commit and re-issue the same command. No-op for non-code work
    types.

    Raises:
        SandboxPreparationError: If the working tree is dirty.
    """
    from spine.git import WorktreeSandbox

    WorktreeSandbox(config, work_type).preflight()


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

    # A dirty tree here would fail worktree creation and finalise the entry to
    # 'failed' — out of the 'needs_review' status resume requires. Check first
    # so the entry stays resumable and the user can retry after cleaning up.
    _preflight_worktree_clean(config, work_type)

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
        from spine.workflow.compose import WORKFLOW_SEQUENCES, build_workflow_graph
        from spine.git import WorktreeSandbox

        checkpointer = await checkpoint_store.get_checkpointer()
        graph = build_workflow_graph(work_type, checkpointer=checkpointer)

        # Resuming code-producing work re-runs IMPLEMENT, so isolate it in a
        # worktree just like a fresh submission. No-op for planning types.
        sandbox = WorktreeSandbox(config, work_type)
        run_config = sandbox.enter()

        # Delete the old checkpoint so the graph starts fresh with our resume_state
        # instead of restoring the previous checkpoint state
        await checkpoint_store.delete_state(work_id)

        # Determine the starting phase based on action and prior state
        # For "rework" action, restart from the phase that needs review
        # For "approve" action, start from the phase after the review
        phase_seq = WORKFLOW_SEQUENCES.get(work_type, [])
        phase_index = 0
        current_phase = ""

        if action == "rework" and saved_state:
            needs_review_phase = saved_state.get("needs_review_phase", "")
            if needs_review_phase:
                for idx, (name, _) in enumerate(phase_seq):
                    if name == needs_review_phase:
                        phase_index = idx
                        current_phase = needs_review_phase
                        break

        # Seed the new run with all prior state plus the human feedback
        resume_state: dict[str, Any] = {
            "work_id": work_id,
            "work_type": work_type,
            "description": description,
            "current_phase": current_phase,
            "phase_index": phase_index,
            "retry_count": prior_retry_count,
            "max_retries": config.max_critic_retries,
            "max_adversarial_retries": config.max_adversarial_retries,
            "adversarial_retry_count": 0,
            "artifacts": prior_artifacts,
            "feedback": all_feedback,
            "status": "running",
            "prompt_request": None,
            "critic_reviewing": "",
            "workspace_root": run_config.workspace_root,
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
                "spine_config": run_config,
            }
        }

        # Stream the graph, updating the work entry after each phase
        # (same pattern as submit_work — uses stream_mode=["updates", "messages"]
        # with subgraphs=True and version="v2" for consistent StreamPart format)
        result: dict[str, Any] = dict(resume_state)
        # aclosing guarantees the generator (and the tracing context inside
        # traced_astream) is finalised even on an exception/early exit, so
        # the LangSmith root span is closed deterministically instead of
        # waiting on GC.
        async with aclosing(
            traced_astream(
                graph.astream(
                    resume_state,
                    thread_config,
                    stream_mode=["updates", "messages"],
                    subgraphs=True,
                    version="v2",
                ),
                work_id,
                work_type,
            )
        ) as stream:
            async for chunk in stream:
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
                    # Only dict outputs carry mergeable state; non-dict payloads
                    # (e.g. tuples from subgraph routing / multi-update super-
                    # steps) would crash on .get() — skip them.
                    if not isinstance(node_output, dict):
                        continue
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
                        logger.info(f"[{work_id}] Resume: Phase {phase or node_name} → {status}")
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

        # Derive final status (resume has no stall tracking).
        final_phase = result.get("current_phase", "")
        result_artifacts = result.get("artifacts", {})
        feedback = result.get("feedback", [])
        final_status = _derive_final_status(
            result, stalled=False, feedback=feedback, work_type=work_type
        )

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
            {"status": final_status, "resumed": True},
        )

        # Capture cross-run experience from the resumed run (best-effort).
        from spine.agents.experience import capture_run_experience

        await capture_run_experience(result, config, final_status)

        # Land or discard the worktree patch based on the resumed outcome.
        sandbox.finalize(final_status)

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
        _sandbox = locals().get("sandbox")
        if _sandbox is not None:
            _sandbox.abort()
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
        audit.log_event(work_id, "work_failed", "dispatcher", {"error": str(e), "resumed": True})

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


async def _resume_flagged_without_interrupt(
    *,
    work_id: str,
    work_type: str,
    action: str,
    flagged_phase: str,
    feedback: str,
    config: SpineConfig,
    db: sqlite_utils.Database,
    audit: AuditService,
) -> dict[str, Any]:
    """Resume a needs_review item whose graph never paused at an interrupt().

    Autonomous work types (``task`` / ``critical_task``) route a
    ``needs_review`` verdict to the ``flag_needs_review`` terminal node, not
    to the ``human_review`` interrupt — so their thread is already at END and
    there is no checkpoint to resume via ``Command(resume=...)``. Translate
    the human's action into the equivalent real operation:

    - ``rework``  → re-run from the flagged phase (``restart_from_phase``).
    - ``approve`` → re-run from the phase *after* the flagged one; if the
      flagged phase was the last, finalise as completed.
    - ``abort``   → finalise the entry as cancelled.

    ``restart_from_phase`` carries the prior run's ``execution_waves`` /
    ``plan_json`` forward from the checkpoint, so the re-run has the slices to
    implement rather than starting empty.
    """
    from spine.workflow.compose import WORKFLOW_SEQUENCES

    audit.log_event(
        work_id,
        "work_resumed_no_interrupt",
        "dispatcher",
        {"action": action, "flagged_phase": flagged_phase, "feedback": feedback[:200]},
    )

    if action == "abort":
        db["work_entries"].update(
            work_id,
            {
                "status": TaskStatus.CANCELLED.value,
                "updated_at": datetime.now().isoformat(),
            },
        )
        return {
            "work_id": work_id,
            "status": TaskStatus.CANCELLED.value,
            "work_type": work_type,
        }

    # Ordered non-critic phases for this work type (restart targets).
    ordered_phases = [
        name
        for name, _ in WORKFLOW_SEQUENCES.get(work_type, [])
        if not name.startswith(PhaseName.CRITIC.value)
    ]

    target_phase = flagged_phase
    if action == "approve":
        # Approving the flagged phase means advancing past it.
        if flagged_phase in ordered_phases:
            idx = ordered_phases.index(flagged_phase)
            if idx + 1 < len(ordered_phases):
                target_phase = ordered_phases[idx + 1]
            else:
                # Flagged phase was the last — nothing left to run.
                db["work_entries"].update(
                    work_id,
                    {
                        "status": TaskStatus.COMPLETED.value,
                        "updated_at": datetime.now().isoformat(),
                    },
                )
                return {
                    "work_id": work_id,
                    "status": TaskStatus.COMPLETED.value,
                    "work_type": work_type,
                }

    if not target_phase or target_phase not in ordered_phases:
        # No usable phase to resume from — surface the error instead of
        # silently completing (the very failure mode this guard exists for).
        raise ValueError(
            f"Cannot resume work '{work_id}' for action '{action}': "
            f"no valid flagged phase recorded in state (got '{flagged_phase!r}')."
        )

    return await restart_from_phase(work_id, target_phase, config)


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

    audit = AuditService(db_path=str(Path(config.queue_path).parent / "audit.db"))

    # ── Guard: is there actually an interrupt to resume from? ──
    # Command(resume=...) only advances a thread that paused at an interrupt()
    # awaiting human input. Reviewed work types (reviewed_task,
    # critical_reviewed_task) do pause there. Autonomous types (task,
    # critical_task) instead route a needs_review verdict to the
    # flag_needs_review TERMINAL node — their thread is already at END. A
    # resume Command against an ended thread executes no nodes, streams zero
    # updates, and the empty result then defaults to "completed" in
    # _derive_final_status — silently marking the work done with nothing run
    # (trace 019ed36f). Detect that and translate the action into a real
    # rerun instead of a no-op resume.
    snapshot = await graph.aget_state(thread_config)
    has_pending_interrupt = bool(snapshot.next) and any(
        getattr(task, "interrupts", None) for task in snapshot.tasks
    )
    if not has_pending_interrupt:
        values = snapshot.values or {}
        flagged_phase = values.get("needs_review_phase") or values.get(
            "current_phase", ""
        )
        logger.info(
            f"[{work_id}] resume_interrupted_work: thread has no pending "
            f"interrupt (work type '{work_type}', flagged at "
            f"'{flagged_phase or 'unknown'}'); translating action '{action}' "
            f"into a real rerun."
        )
        return await _resume_flagged_without_interrupt(
            work_id=work_id,
            work_type=work_type,
            action=action,
            flagged_phase=flagged_phase,
            feedback=feedback,
            config=config,
            db=db,
            audit=audit,
        )

    command = Command(resume={"action": action, "feedback": feedback})

    audit.log_event(
        work_id,
        "work_resumed_interrupt",
        "dispatcher",
        {"action": action, "feedback": feedback[:200]},
    )

    # Stream the rest of the graph from the interrupt point. Guard the run
    # so an unhandled error finalises the work entry to "failed" instead of
    # leaving it stuck "running" as a ghost — this path runs on the detached
    # resume executor thread, where the exception is otherwise just dropped.
    result: dict[str, Any] = {}
    stream = traced_astream(
        graph.astream(
            command,
            thread_config,
            stream_mode=["updates", "messages"],
            subgraphs=True,
            version="v2",
        ),
        work_id,
        work_type,
    )
    try:
        async for chunk in stream:
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
    except Exception as e:
        _finalise_failed_work(db, work_id, audit, e)
        raise
    finally:
        # Deterministically finalise the generator (and the tracing context
        # inside traced_astream) so the LangSmith root span closes even on
        # an exception/early exit instead of waiting on GC.
        await stream.aclose()

    # Defense in depth: the pending-interrupt guard above should ensure at
    # least one node runs, but if the resume still streamed zero updates the
    # result is empty — and _derive_final_status would default that to
    # "completed", silently marking the work done with nothing run. Preserve
    # needs_review instead so the item stays actionable.
    if not result:
        logger.warning(
            f"[{work_id}] resume_interrupted_work streamed no updates — "
            f"preserving needs_review instead of defaulting to completed."
        )
        db["work_entries"].update(
            work_id,
            {
                "status": TaskStatus.NEEDS_REVIEW.value,
                "updated_at": datetime.now().isoformat(),
            },
        )
        return {
            "work_id": work_id,
            "status": TaskStatus.NEEDS_REVIEW.value,
            "work_type": work_type,
        }

    final_phase = result.get("current_phase", "")
    final_status = _derive_final_status(
        result, stalled=False, feedback=result.get("feedback", []), work_type=work_type
    )

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
    """Restart a work item that is pending, running, stalled, or needs_review.

    Unlike ``resume_work`` (which continues from a checkpoint with human
    feedback), ``restart_work`` re-runs the workflow from phase 0.  It
    is intended for items whose worker or UI died mid-execution, and to
    launch a ``pending`` item that was created (e.g. ``spine run --project``)
    but never started.

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
        TaskStatus.PENDING.value,
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

    # Bail before any destructive setup (artifact wipe, checkpoint purge,
    # status→running) if the tree is dirty: otherwise worktree creation fails
    # mid-restart and finalises the entry to 'failed', which is not in this
    # path's restartable set — stranding the work. Failing here leaves the
    # original restartable status intact so the user can retry once clean.
    _preflight_worktree_clean(config, work_type)

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
        "max_adversarial_retries": config.max_adversarial_retries,
        "adversarial_retry_count": 0,
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


async def restart_from_phase(
    work_id: str,
    phase_name: str,
    config: SpineConfig | None = None,
    clear_artifacts: bool = False,
) -> dict[str, Any]:
    """Restart a stalled/failed/running work item from a specific phase.

    Unlike ``restart_work`` (which always starts from phase 0), this
    rebuilds the graph so that ``START`` routes directly to the requested
    phase.  Earlier phases are skipped; their artifacts are preserved
    unless ``clear_artifacts`` is set.

    Workflow:
        1. Validate that the work item exists and is in a restartable status.
        2. Validate that ``phase_name`` is a valid node for the work type.
        3. Optionally clear on-disk artifacts for phases at or after the
           target phase (earlier artifacts are always preserved).
        4. Purge the LangGraph checkpoint so the graph starts fresh.
        5. Rebuild the workflow graph with ``start_from_phase`` routing.
        6. Re-invoke the graph with accumulated state from prior phases.

    Args:
        work_id: The work item ID to restart.
        phase_name: The phase node to start from (e.g. ``"implement"``).
        config: Optional SpineConfig.
        clear_artifacts: If True, delete on-disk artifacts for the target
            phase and all subsequent phases. Artifacts from earlier phases
            are always preserved so they can be reused.

    Returns:
        A dict with keys ``work_id``, ``status``, ``work_type``, ``phase_name``.

    Raises:
        ValueError: If the work item is not in a restartable status,
            the phase name is invalid, or the work item is not found.
    """
    if config is None:
        config = SpineConfig.load()
    config.ensure_dirs()

    db = _get_work_db(config)

    # ── Check if this task is already running ──
    # Only block restart if the active task is the same work_id being restarted.
    # This allows other tasks to proceed while one is already running.
    from spine.work.ralph_worker import get_worker

    worker = get_worker(config)
    active = worker.get_active()
    if active is not None and active.get("id") == work_id:
        return {"status": "skipped", "message": "This task is already running"}

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
        TaskStatus.FAILED.value,
    )
    if status not in restartable:
        raise ValueError(
            f"Work item '{work_id}' is in '{status}' status — "
            f"only {restartable} items can be restarted from a phase."
        )

    work_type = entry.get("work_type", "spec")
    description = entry.get("description", "")

    # ── Validate phase name ──
    from spine.workflow.compose import get_restart_phases

    valid_phases = get_restart_phases(work_type)
    if phase_name not in valid_phases:
        raise ValueError(
            f"Phase '{phase_name}' is not valid for work type '{work_type}'. "
            f"Valid phases: {valid_phases}"
        )

    # Check the tree is clean before clearing artifacts / purging the
    # checkpoint / marking running — a dirty tree fails worktree creation
    # later, and doing the destructive setup first would discard recoverable
    # state for a run that can't start. Raising here keeps the entry retryable.
    _preflight_worktree_clean(config, work_type)

    audit = AuditService(db_path=str(Path(config.queue_path).parent / "audit.db"))
    artifact_store = ArtifactStore(base_path=config.artifact_path)

    audit.log_event(
        work_id,
        "work_restarted_from_phase",
        "dispatcher",
        {
            "previous_status": status,
            "phase_name": phase_name,
            "clear_artifacts": clear_artifacts,
        },
    )

    # ── Optionally clear on-disk artifacts for target phase onwards ──
    if clear_artifacts:
        from spine.workflow.compose import WORKFLOW_SEQUENCES

        sequence = WORKFLOW_SEQUENCES.get(work_type, [])
        # Find the index of the target phase and clear from there onward
        target_idx = next(
            (i for i, (name, _) in enumerate(sequence) if name == phase_name),
            len(sequence),
        )
        phases_to_clear = {name for i, (name, _) in enumerate(sequence) if i >= target_idx}
        work_dir = Path(artifact_store._base) / work_id
        if work_dir.exists():
            for phase_dir in work_dir.iterdir():
                if phase_dir.is_dir() and phase_dir.name in phases_to_clear:
                    for f in phase_dir.rglob("*"):
                        if f.is_file():
                            f.unlink()
                    logger.info(f"[{work_id}] Cleared artifacts for phase: {phase_dir.name}")

    # Purge LangGraph checkpoint so the graph starts fresh
    from spine.persistence.checkpoint import CheckpointStore

    checkpoint_store = CheckpointStore(db_path=config.checkpoint_path)
    # Recover the prior run's execution state BEFORE purging — IMPLEMENT/VERIFY
    # consume execution_waves / plan_json from checkpoint state (not disk), so a
    # restart at those phases would otherwise have nothing to implement.
    prior_state = await checkpoint_store.get_state(work_id) or {}
    saver = await checkpoint_store.get_checkpointer()
    await saver.adelete_thread(work_id)
    logger.info(f"[{work_id}] Purged checkpoint for restart from phase '{phase_name}'")

    # ── Rebuild initial state (preserving prior artifacts) ──
    # Load existing artifacts from disk so downstream phases can reuse them
    existing_artifacts: dict[str, dict[str, str]] = {}
    work_dir = Path(artifact_store._base) / work_id
    if work_dir.exists():
        for phase_dir in work_dir.iterdir():
            if phase_dir.is_dir():
                phase_artifacts: dict[str, str] = {}
                for f in phase_dir.rglob("*"):
                    if f.is_file():
                        try:
                            phase_artifacts[f.name] = f.read_text(encoding="utf-8")
                        except (UnicodeDecodeError, OSError):
                            # Skip binary/unreadable artifacts
                            pass
                if phase_artifacts:
                    existing_artifacts[phase_dir.name] = phase_artifacts

    # Find the phase_index for the target phase
    from spine.workflow.compose import WORKFLOW_SEQUENCES as _SEQ

    phase_seq = _SEQ.get(work_type, [])
    phase_index = next(
        (i for i, (name, _) in enumerate(phase_seq) if name == phase_name),
        0,
    )

    initial_state: dict[str, Any] = {
        "work_id": work_id,
        "work_type": work_type,
        "description": description,
        "current_phase": phase_name,
        "phase_index": phase_index,
        "retry_count": {},
        "max_retries": config.max_critic_retries,
        "max_adversarial_retries": config.max_adversarial_retries,
        "adversarial_retry_count": 0,
        "artifacts": existing_artifacts,
        # Carry execution state forward so a restart at IMPLEMENT/VERIFY has the
        # slices to run; harmless for restarts at SPECIFY/PLAN (recomputed there).
        "execution_waves": prior_state.get("execution_waves", []),
        "plan_json": prior_state.get("plan_json"),
        "specification_json": prior_state.get("specification_json"),
        "read_cache": prior_state.get("read_cache", {}),
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
            "current_phase": phase_name,
            "updated_at": datetime.now().isoformat(),
            "result": "{}",
        },
    )

    # ── Rebuild and re-run the workflow graph from the target phase ──
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
        start_from_phase=phase_name,
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
    start_from_phase: str | None = None,
) -> dict[str, Any]:
    """Run a workflow graph, guaranteeing the work entry is finalised.

    Thin guard around :func:`_run_workflow_graph_inner`. If execution
    raises, the work entry is finalised to ``failed`` *before* the error
    propagates. Without this, a restart/resume job — which runs on a
    detached executor thread where the exception is simply logged and
    dropped — would leave its work entry stuck ``running`` forever and
    show as a phantom "active" job in the queue UI (a ghost). The normal
    queue path's ``submit_work`` already had its own handler; this makes
    the guarantee hold for every caller.

    This guard is also the mandatory-sandbox chokepoint for every path
    that re-runs the graph (restart, restart-from-phase, approved-plan
    continuation): code-producing work types run against an isolated git
    worktree, which is fast-forward merged to main only on success and
    rolled back on every other outcome.
    """
    from spine.git import WorktreeSandbox

    # Isolate code-producing runs in a throwaway worktree. enter() swaps
    # workspace_root to the sandbox (and is a no-op for planning work types,
    # returning the original config). enter() is inside the guard so that a
    # sandbox-preparation failure (e.g. a dirty tree) finalises the work
    # entry to 'failed' rather than escaping as a phantom 'running' ghost.
    sandbox = WorktreeSandbox(config, work_type)
    try:
        run_config = sandbox.enter()
        if run_config is not config:
            initial_state = {**initial_state, "workspace_root": run_config.workspace_root}
        result = await _run_workflow_graph_inner(
            work_id=work_id,
            work_type=work_type,
            config=run_config,
            db=db,
            audit=audit,
            artifact_store=artifact_store,
            initial_state=initial_state,
            checkpoint_store=checkpoint_store,
            is_restart=is_restart,
            start_from_phase=start_from_phase,
        )
    except Exception as e:
        sandbox.abort()
        _finalise_failed_work(db, work_id, audit, e)
        raise
    # Land the patch on main (success) or discard the worktree (otherwise).
    sandbox.finalize(result.get("status", ""))
    return result


def _finalise_failed_work(
    db: sqlite_utils.Database,
    work_id: str,
    audit: AuditService | None,
    error: BaseException,
) -> None:
    """Mark a work entry ``failed`` and emit failure events.

    The ghost-prevention backstop: an in-flight work entry must never be
    left ``running`` when its execution raises, or it shows as a phantom
    "active" job in the queue UI (which sources active jobs from in-flight
    work entries). Best-effort throughout — finalising must not itself
    raise and mask the original error.
    """
    logger.error(f"[{work_id}] Workflow execution failed: {error}", exc_info=True)
    try:
        db["work_entries"].update(
            work_id,
            {
                "status": TaskStatus.FAILED.value,
                "updated_at": datetime.now().isoformat(),
                "result": json.dumps({"error": str(error)}),
            },
        )
    except Exception:
        logger.exception(f"[{work_id}] Could not finalise work entry after failure")
    if audit is not None:
        try:
            audit.log_event(work_id, "work_failed", "", {"error": str(error)})
        except Exception:
            pass
    try:
        from spine.ui.ws_bus import get_bus

        get_bus().publish_sync("work_failed", {"work_id": work_id, "error": str(error)})
    except Exception:
        pass


async def _run_workflow_graph_inner(
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
    start_from_phase: str | None = None,
) -> dict[str, Any]:
    """Run a workflow graph to completion, streaming updates.

    This shared helper avoids the ~100-line duplication between
    ``submit_work``, ``resume_work``, ``restart_work``, and
    ``restart_from_phase``.
    """
    from spine.workflow.compose import build_workflow_graph

    graph = build_workflow_graph(
        work_type,
        checkpointer=await checkpoint_store.get_checkpointer(),
        start_from_phase=start_from_phase,
    )

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
    stream_iter = traced_astream(
        graph.astream(
            initial_state,
            thread_config,
            stream_mode=["updates", "messages"],
            subgraphs=True,
            version="v2",
        ),
        work_id,
        work_type,
    )
    stalled = False
    cancelled = False

    while True:
        # Cooperative cancellation: a Stop Work click marks the work entry
        # ``cancelled``.  Poll at each stream boundary so we break promptly
        # (within one node) and close the trace, instead of streaming on
        # until the stall timeout.
        if _work_is_cancelled(db, work_id):
            cancelled = True
            logger.info(f"[{work_id}] Work cancelled — halting stream")
            audit.log_event(
                work_id,
                "work_cancelled",
                "dispatcher",
                {"last_phase": result.get("current_phase", "")},
            )
            break
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
            # Only dict outputs carry state updates we can merge. A node can also
            # surface non-dict payloads in the "updates" stream (e.g. tuples from
            # subgraph-internal routing / multi-update super-steps) — skip those
            # rather than calling .get() on them (critic_plan emitted a tuple →
            # 'tuple' object has no attribute 'get' at this line). Mirrors the
            # guard in the primary streaming loop above.
            if not isinstance(node_output, dict):
                continue
            node_artifacts = node_output.get("artifacts")
            if node_artifacts and isinstance(node_artifacts, dict):
                existing = result.get("artifacts", {})
                if not isinstance(existing, dict):
                    existing = {}
                merged = {**existing, **node_artifacts}
                for key in set(existing) & set(node_artifacts):
                    if isinstance(existing[key], dict) and isinstance(node_artifacts[key], dict):
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

    # Close the stream generator so the LangSmith trace tears down now (no-op
    # on natural exhaustion; runs teardown promptly on a cancel/stall break).
    try:
        await stream_iter.aclose()
    except Exception:
        logger.debug(f"[{work_id}] stream aclose raised", exc_info=True)

    # ── Final status ──
    # A user-requested cancel is terminal — it must not be overwritten by the
    # graph-state derivation.
    final_phase = result.get("current_phase", "")
    result_artifacts = result.get("artifacts", {})
    feedback = result.get("feedback", [])
    final_status = (
        TaskStatus.CANCELLED.value
        if cancelled
        else _derive_final_status(
            result, stalled=stalled, feedback=feedback, work_type=work_type
        )
    )

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

    # Capture cross-run experience from this run's critic feedback (best-effort).
    from spine.agents.experience import capture_run_experience

    await capture_run_experience(result, config, final_status)

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
    work_type_override: str = "task",
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
        work_type_override: The work_type for spawned items (default "task").
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
        tasks_text = (
            artifacts.load_artifact(plan_id, "tasks", "tasks.md")
            or artifacts.load_artifact(plan_id, "tasks", "tasks.txt")
        )
        if tasks_text is None:
            raise FileNotFoundError(
                f"No tasks artifact found for {plan_id} (looked for tasks.md / tasks.txt)"
            )

    # Parse tasks — each section starting with ## or # followed by text
    # becomes a separate work item. Simple heuristic: split on
    # lines that look like "# Section" or "## Section".
    sections: list[tuple[str, str]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for line in tasks_text.splitlines():
        if (
            line.startswith(("### ", "## ", "# "))
            and not line.startswith("# Slice")
            and not line.startswith("# Tasks")
        ):
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
        spawned.append(
            {
                **result,
                "description": description[:200],
            }
        )

    # Update the plan work entry with spawned task IDs
    db = _get_work_db(config)
    db["work_entries"].update(
        plan_id,
        {
            "result": json.dumps(
                {
                    "split": True,
                    "spawned_ids": [s["work_id"] for s in spawned],
                }
            ),
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
    conditions = "work_type IN ('plan', 'plan_spec', 'plan_only', 'critical_plan_only')"
    params: dict[str, Any] = {"limit": limit}

    if status:
        conditions += " AND status = :status"
        params["status"] = status

    sql = f"SELECT * FROM work_entries WHERE {conditions} ORDER BY created_at DESC LIMIT :limit"
    return list(db.query(sql, params))


# ── Plan Approval & Spawning ──


async def approve_and_spawn(
    plan_id: str,
    action: str = "approve",
    feedback: str | None = None,
    config: SpineConfig | None = None,
) -> dict[str, Any]:
    """Approve a planning work item and spawn execution tasks.

    This is the bridge between the planning workflow and execution. When a plan
    is approved via the spec & planning UI, this function:
    1. Loads the plan's artifacts (spec.md, plan.md)
    2. Resolves the plan into work units using the plan resolver
    3. Spawns execution work items for each unit
    4. Returns the spawned work IDs

    Args:
        plan_id: The ID of the approved planning work item.
        action: One of "approve", "request_revision", or "reject".
        feedback: Optional feedback text for revision requests.
        config: Optional SpineConfig.

    Returns:
        For "approve": ``{plan_id, work_id, status, spawned_ids (empty),
        continued_from, work_type}`` — the SAME work item is re-keyed to its
        execution work type and continued from IMPLEMENT. For "reject" /
        "request_revision": ``{plan_id, status, spawned_ids}``.
    """
    if config is None:
        config = SpineConfig.load()

    db = _get_work_db(config)
    artifacts = ArtifactStore(base_path=config.artifact_path)
    audit = AuditService(db_path=str(Path(config.queue_path).parent / "audit.db"))

    # Validate and fetch the plan work entry
    try:
        entry = db["work_entries"].get(plan_id)
    except sqlite_utils.db.NotFoundError:
        raise ValueError(f"Plan '{plan_id}' not found")

    work_type = entry.get("work_type", "")
    if work_type not in ("reviewed_task", "critical_reviewed_task"):
        raise ValueError(f"Work item '{plan_id}' is not a planning work type (got '{work_type}')")

    # Validate status is awaiting_approval before allowing approval
    if entry.get("status") != "awaiting_approval":
        raise ValueError(
            f"Work item '{plan_id}' is in '{entry.get('status')}' status, "
            f"not 'awaiting_approval'. Only awaiting_approval items can be approved."
        )

    # Handle rejection
    if action == "reject":
        db["work_entries"].update(
            plan_id,
            {
                "status": TaskStatus.REJECTED.value,
                "updated_at": datetime.now().isoformat(),
                "result": json.dumps({"action": "rejected", "feedback": feedback}),
            },
        )
        audit.log_event(plan_id, "plan_rejected", "dispatcher", {"feedback": feedback})
        return {"plan_id": plan_id, "status": "rejected", "spawned_ids": []}

    # Handle revision request
    if action == "request_revision":
        db["work_entries"].update(
            plan_id,
            {
                "status": TaskStatus.NEEDS_REVIEW.value,
                "updated_at": datetime.now().isoformat(),
                "result": json.dumps({"action": "revision_requested", "feedback": feedback}),
            },
        )
        audit.log_event(plan_id, "plan_revision_requested", "dispatcher", {"feedback": feedback})

        # Re-run the plan workflow with the user's feedback
        from spine.workflow.compose import build_workflow_graph
        from spine.persistence.checkpoint import CheckpointStore

        work_type = entry.get("work_type", "plan")
        description = entry.get("description", "")
        artifacts_store = ArtifactStore(base_path=config.artifact_path)
        checkpoint_store = CheckpointStore(db_path=config.checkpoint_path)
        checkpointer = await checkpoint_store.get_checkpointer()
        graph = build_workflow_graph(work_type, checkpointer=checkpointer)

        saved_state = await checkpoint_store.get_state(plan_id)
        if saved_state:
            prior_artifacts = saved_state.get("artifacts", {})
            prior_feedback = saved_state.get("feedback", [])
            prior_retry_count = saved_state.get("retry_count", {})
        else:
            prior_artifacts, prior_feedback, prior_retry_count = {}, [], {}

        human_review_entry = {
            "status": "needs_revision",
            "tier": "human",
            "reason": feedback or "Revision requested",
            "suggestions": [],
        }
        all_feedback = list(prior_feedback) + [human_review_entry]

        resume_state: dict[str, Any] = {
            "work_id": plan_id,
            "work_type": work_type,
            "description": description,
            "current_phase": "",
            "phase_index": 0,
            "retry_count": prior_retry_count,
            "max_retries": config.max_critic_retries,
            "max_adversarial_retries": config.max_adversarial_retries,
            "adversarial_retry_count": 0,
            "artifacts": prior_artifacts,
            "feedback": all_feedback,
            "status": "running",
            "prompt_request": None,
            "critic_reviewing": "",
            "workspace_root": config.workspace_root,
        }

        thread_config = {
            "configurable": {
                "thread_id": plan_id,
                "spine_config": config,
            }
        }

        stalled = False
        import os as _os

        stall_timeout = _os.environ.get("SPINE_STALL_TIMEOUT")
        stall_timeout_val = int(stall_timeout or "120")

        result: dict[str, Any] = dict(resume_state)

        # Progress-aware stall detection: time out each chunk individually rather
        # than imposing a flat wall-clock cap on the whole stream. A long but
        # progressing generation (e.g. a local model emitting thousands of critic
        # tokens) runs to completion, while a genuinely hung backend that emits
        # nothing for stall_timeout_val seconds is caught and marked stalled. The
        # previous `wait_for(_stream_graph(), stall_timeout_val + 10)` was a flat
        # deadline that killed healthy long generations mid-flight even while they
        # streamed steadily (trace 019ed38f cancelled a live critic at exactly
        # 130s). This mirrors the per-chunk pattern used by the other resume paths.
        stream_iter = traced_astream(
            graph.astream(
                resume_state,
                thread_config,
                stream_mode=["updates", "messages"],
                subgraphs=True,
                version="v2",
            ),
            plan_id,
            work_type,
        ).__aiter__()

        while True:
            try:
                if stall_timeout_val > 0:
                    chunk = await asyncio.wait_for(
                        stream_iter.__anext__(), timeout=stall_timeout_val
                    )
                else:
                    chunk = await stream_iter.__anext__()
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                stalled = True
                last_phase = result.get("current_phase", "")
                logger.error(
                    f"Resume of work {plan_id} stalled — no chunk received for "
                    f"{stall_timeout_val}s (last phase: {last_phase})"
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
                    node_output = {**node_output, "feedback": existing_fb + node_feedback}
                result.update(node_output)
                phase = node_output.get("current_phase", "")
                status = node_output.get("status", "")
                if phase or status:
                    _update_work_progress(db, plan_id, phase, status)
                    logger.info(f"[{plan_id}] Resume: Phase {phase or node_name} → {status}")
                    audit.log_event(
                        plan_id,
                        "phase_completed",
                        node_name,
                        {"phase": phase, "status": status},
                    )
                if node_artifacts and isinstance(node_artifacts, dict):
                    for art_phase, phase_arts in node_artifacts.items():
                        if not isinstance(phase_arts, dict):
                            continue
                        for art_name, art_content in phase_arts.items():
                            if art_content is not None:
                                artifacts_store.save_artifact(
                                    plan_id, art_phase, art_name, str(art_content)
                                )

        if stalled:
            db["work_entries"].update(
                plan_id,
                {
                    "status": TaskStatus.STALLED.value,
                    "current_phase": result.get("current_phase", ""),
                    "updated_at": datetime.now().isoformat(),
                    "result": json.dumps(
                        {"error": f"stalled after {stall_timeout_val}s", "stalled": True}
                    ),
                },
            )
            audit.log_event(
                plan_id, "work_failed", "dispatcher", {"error": "stalled", "stalled": True}
            )
            try:
                from spine.ui.ws_bus import get_bus

                get_bus().publish_sync("work_failed", {"work_id": plan_id, "error": "stalled"})
            except Exception:
                pass
            return {
                "plan_id": plan_id,
                "status": TaskStatus.STALLED.value,
                "spawned_ids": [],
            }

        final_phase = result.get("current_phase", "")
        result_artifacts = result.get("artifacts", {})
        feedback = result.get("feedback", [])
        final_status = _derive_final_status(
            result, stalled=False, feedback=feedback, work_type=work_type
        )

        db["work_entries"].update(
            plan_id,
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
            plan_id, "work_completed", final_phase, {"status": final_status, "resumed": True}
        )
        try:
            from spine.ui.ws_bus import get_bus

            get_bus().publish_sync("work_completed", {"work_id": plan_id, "status": final_status})
        except Exception:
            pass
        return {"plan_id": plan_id, "status": final_status, "spawned_ids": []}

    # Approve the plan — CONTINUE this same work item from IMPLEMENT, reusing the
    # already-approved spec + plan, instead of spawning fresh tasks that re-run
    # SPECIFY/PLAN. The reviewed (gated) work type is re-keyed to its execution
    # counterpart (whose WORKFLOW_SEQUENCES includes IMPLEMENT + VERIFY), then the
    # graph is re-run from IMPLEMENT against the persisted plan state.
    from spine.persistence.checkpoint import CheckpointStore
    from spine.workflow.compose import WORKFLOW_SEQUENCES

    exec_work_type = {
        WorkType.REVIEWED_TASK.value: WorkType.TASK.value,
        WorkType.CRITICAL_REVIEWED_TASK.value: WorkType.CRITICAL_TASK.value,
    }.get(work_type, WorkType.TASK.value)

    phase_seq = WORKFLOW_SEQUENCES.get(exec_work_type, [])
    impl_index = next(
        (i for i, (name, _) in enumerate(phase_seq) if name == PhaseName.IMPLEMENT.value),
        None,
    )
    if impl_index is None:
        raise ValueError(
            f"No '{PhaseName.IMPLEMENT.value}' phase in WORKFLOW_SEQUENCES for "
            f"'{exec_work_type}' — cannot continue {plan_id} from implement."
        )

    # Recover the plan-phase state IMPLEMENT consumes (execution_waves / plan_json
    # / specification_json / artifacts). The reviewed run's checkpoint still holds
    # it; if the checkpoint is gone, rebuild execution_waves from the approved
    # plan.json's feature_slices.
    checkpoint_store = CheckpointStore(db_path=config.checkpoint_path)
    saved_state = await checkpoint_store.get_state(plan_id) or {}

    execution_waves = saved_state.get("execution_waves") or []
    plan_json = saved_state.get("plan_json")
    specification_json = saved_state.get("specification_json")
    prior_artifacts = saved_state.get("artifacts") or {}
    read_cache = saved_state.get("read_cache") or {}

    if not execution_waves:
        # Rebuild the waves from the approved plan's feature_slices. plan_json is
        # persisted as a serialized JSON STRING in checkpoint state (not a dict),
        # so the old isinstance(dict) guard never matched and this fallback never
        # fired — dead-ending the resume even though plan.json had slices. Parse
        # the string, and fall back to the on-disk plan.json artifact if state
        # has no usable plan_json at all.
        plan_obj: dict[str, Any] | None = None
        if isinstance(plan_json, dict):
            plan_obj = plan_json
        elif isinstance(plan_json, str) and plan_json.strip():
            try:
                parsed = json.loads(plan_json)
                plan_obj = parsed if isinstance(parsed, dict) else None
            except (ValueError, TypeError):
                plan_obj = None
        if not (plan_obj and plan_obj.get("feature_slices")):
            disk_plan = (
                Path(config.workspace_root)
                / ".spine" / "artifacts" / plan_id / "plan" / "plan.json"
            )
            try:
                loaded = json.loads(disk_plan.read_text(encoding="utf-8"))
                if isinstance(loaded, dict) and loaded.get("feature_slices"):
                    plan_obj = loaded
            except (OSError, ValueError, TypeError):
                pass

        if plan_obj and plan_obj.get("feature_slices"):
            from spine.models.types import FeatureSlice
            from spine.workflow.slice_scheduler import (
                compute_execution_waves,
                slices_to_state_dict,
            )

            slices = [FeatureSlice(**s) for s in plan_obj["feature_slices"]]
            execution_waves = slices_to_state_dict(compute_execution_waves(slices))

    if not execution_waves:
        raise ValueError(
            f"Cannot continue {plan_id} from IMPLEMENT: the approved plan has no "
            f"execution_waves in its checkpoint state and none could be rebuilt "
            f"from plan.json feature_slices."
        )

    # Preflight the worktree precondition BEFORE touching status. The
    # IMPLEMENT-onward run is code-producing, so it needs a clean tree to
    # create its sandbox worktree. Checking here means a dirty tree raises
    # while the entry is still awaiting_approval — so the user can stash /
    # commit and re-approve. Done after the status mutation below, the same
    # failure would have left the entry stuck running→failed, stranding the
    # plan (re-approval is rejected once status leaves awaiting_approval).
    _preflight_worktree_clean(config, exec_work_type)

    # Re-key the work item to its execution type so status/restart logic uses the
    # IMPLEMENT/VERIFY sequence and it will NOT relabel back to awaiting_approval.
    db["work_entries"].update(
        plan_id,
        {
            "work_type": exec_work_type,
            "status": TaskStatus.RUNNING.value,
            "current_phase": PhaseName.IMPLEMENT.value,
            "updated_at": datetime.now().isoformat(),
            "result": json.dumps(
                {"approved": True, "continued_from": PhaseName.IMPLEMENT.value}
            ),
        },
    )
    audit.log_event(
        plan_id,
        "plan_approved",
        "dispatcher",
        {"continued_work_type": exec_work_type, "from_phase": PhaseName.IMPLEMENT.value},
    )

    # Seed the IMPLEMENT-onward run with the approved artifacts + plan state so
    # SPECIFY/PLAN are not re-run.
    initial_state: dict[str, Any] = {
        "work_id": plan_id,
        "work_type": exec_work_type,
        "description": entry.get("description", ""),
        "current_phase": PhaseName.IMPLEMENT.value,
        "phase_index": impl_index,
        "retry_count": {},
        "max_retries": config.max_critic_retries,
        "max_adversarial_retries": config.max_adversarial_retries,
        "adversarial_retry_count": 0,
        "artifacts": prior_artifacts,
        "execution_waves": execution_waves,
        "plan_json": plan_json,
        "specification_json": specification_json,
        "read_cache": read_cache,
        "feedback": [],
        "status": "running",
        "prompt_request": None,
        "critic_reviewing": "",
        "workspace_root": config.workspace_root,
    }

    # Purge the gated graph's checkpoint so the execution graph starts cleanly at
    # IMPLEMENT from initial_state instead of restoring the critic_plan stop point.
    await checkpoint_store.delete_state(plan_id)

    run_result = await _run_workflow_graph(
        work_id=plan_id,
        work_type=exec_work_type,
        config=config,
        db=db,
        audit=audit,
        artifact_store=artifacts,
        initial_state=initial_state,
        checkpoint_store=checkpoint_store,
        is_restart=True,
        start_from_phase=PhaseName.IMPLEMENT.value,
    )

    # No fresh work is spawned — the same work_id continued. `spawned_ids` stays
    # empty for backward compatibility with the UI/result consumers.
    return {
        "plan_id": plan_id,
        "work_id": plan_id,
        "status": run_result.get("status", "completed"),
        "spawned_ids": [],
        "continued_from": PhaseName.IMPLEMENT.value,
        "work_type": exec_work_type,
    }


# ── Resume work ──

# (resume_work is defined above)
