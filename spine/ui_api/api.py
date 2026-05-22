"""SPINE UI API — sole read/write interface for Streamlit.

UI pages MUST use UIApi for all data access. Never import directly from
workflow/, phases/, or work.dispatcher. This maintains the zero-duplication
principle: CLI and UI share the same backend code paths.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from spine.config import SpineConfig
from spine.models.enums import TaskStatus
from spine.persistence.artifacts import ArtifactStore
from spine.services.audit_service import AuditService
from spine.work.dispatcher import get_work_status, list_work
from spine.work.ralph_worker import get_worker

logger = logging.getLogger(__name__)


class UIApi:
    """The sole read/write interface for Streamlit UI pages.

    Wraps the dispatcher, artifact store, and audit service
    to provide a unified API for all UI operations.

    Usage::

        api = UIApi()
        items = api.list_work(status="running")
        artifacts = api.get_artifacts(work_id="abc123")
    """

    def __init__(self, config: SpineConfig | None = None) -> None:
        self._config = config or SpineConfig.load()
        self._artifacts = ArtifactStore(base_path=self._config.artifact_path)
        self._audit = AuditService(
            db_path=str(__import__("pathlib").Path(self._config.queue_path).parent / "audit.db")
        )

    # ── Work operations ──

    def get_work(self, work_id: str) -> dict[str, Any] | None:
        """Get details for a specific work item.

        Args:
            work_id: The work item ID.

        Returns:
            Work entry dict, or None.
        """
        return get_work_status(work_id, self._config)

    def list_work(self, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """List work items.

        Args:
            status: Optional status filter.
            limit: Maximum items to return.

        Returns:
            List of work entry dicts.
        """
        return list_work(status=status, limit=limit, config=self._config)

    async def submit_work(self, description: str, work_type: str = "spec") -> dict[str, Any]:
        """Submit new work via the dispatcher (blocking — prefer enqueue_work for UI).

        This method blocks until the entire workflow completes. Use
        ``enqueue_work()`` from UI pages to avoid blocking Streamlit.

        Args:
            description: Work description.
            work_type: Workflow type.

        Returns:
            Result dict with work_id and status.
        """
        from spine.work.dispatcher import submit_work

        return await submit_work(description, work_type, self._config)

    def enqueue_work(self, description: str, work_type: str = "spec") -> dict[str, Any]:
        """Enqueue work for async processing via the RalphLoopWorker.

        Non-blocking: adds the item to the persistent SQLite queue and
        starts the background worker if not already running, then returns
        immediately with a queue reference. The work_status page can poll
        for progress using the returned queue_id.

        Args:
            description: Work description.
            work_type: Workflow type.

        Returns:
            Dict with queue_id, status, and work_type.
        """
        from spine.work.ralph_worker import get_worker

        worker = get_worker(self._config)
        queue_id = worker.enqueue(description=description, work_type=work_type)
        worker.start()  # no-op if already running
        logger.info(f"Enqueued work via RalphLoopWorker: queue_id={queue_id}")
        return {
            "queue_id": queue_id,
            "status": "pending",
            "work_type": work_type,
        }

    # ── Artifact operations ──

    def get_artifacts(self, work_id: str) -> list[dict[str, Any]]:
        """List all artifacts for a work item.

        Args:
            work_id: The work item ID.

        Returns:
            List of artifact metadata dicts.
        """
        return self._artifacts.list_artifacts(work_id)

    def read_artifact(self, work_id: str, phase: str, name: str) -> str | None:
        """Read the content of a specific artifact.

        Args:
            work_id: The work item ID.
            phase: The phase name.
            name: The artifact filename.

        Returns:
            The artifact content, or None.
        """
        return self._artifacts.load_artifact(work_id, phase, name)

    def get_feedback(self, work_id: str) -> list[dict[str, Any]]:
        """Get feedback entries for a work item.

        Returns feedback entries that indicate why review is needed.
        Only returns entries with status "needs_review".

        Args:
            work_id: The work item ID.

        Returns:
            List of feedback dicts with keys: status, tier, reason, suggestions.
        """
        entry = self.get_work(work_id)
        if entry is None:
            return []
        result = entry.get("result", {})
        if isinstance(result, dict):
            feedback = result.get("feedback", [])
            if isinstance(feedback, list):
                return feedback
        return []

    # ── Audit operations ──

    def get_audit_log(
        self,
        work_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query audit events.

        Args:
            work_id: Optional work item filter.
            event_type: Optional event type filter.
            limit: Maximum events to return.

        Returns:
            List of audit event dicts.
        """
        return self._audit.query_events(work_id=work_id, event_type=event_type, limit=limit)

    # ── Config ──

    def get_config(self) -> dict[str, Any]:
        """Return the current configuration as a dict."""
        return {
            "checkpoint_path": self._config.checkpoint_path,
            "artifact_path": self._config.artifact_path,
            "max_critic_retries": self._config.max_critic_retries,
            "work_type": self._config.work_type,
            "queue_backend": self._config.queue_backend,
            "workspace_root": self._config.workspace_root,
            "mcp_servers": self._config.mcp_servers,
        }

    def update_mcp_server(self, server_name: str, config: dict[str, Any]) -> bool:
        """Update or add an MCP server configuration.

        Writes the updated ``mcp_servers`` section back to
        ``.spine/config.yaml``, preserving all other config keys.

        Args:
            server_name: MCP server name (e.g. ``"codebase-index"``).
            config: Server config dict with keys ``command``, ``args``,
                ``env``, ``timeout``, ``connect_timeout``.

        Returns:
            ``True`` if the save succeeded, ``False`` otherwise.
        """
        import yaml

        config_path = ".spine/config.yaml"
        try:
            if os.path.exists(config_path):
                with open(config_path) as f:
                    all_config = yaml.safe_load(f) or {}
            else:
                all_config = {}
            mcp_servers = all_config.get("mcp_servers", {})
            mcp_servers[server_name] = config
            all_config["mcp_servers"] = mcp_servers
            with open(config_path, "w") as f:
                yaml.dump(all_config, f, default_flow_style=False, sort_keys=False)
            # Reload config so the in-memory instance reflects changes
            self._config = SpineConfig.load()
            return True
        except Exception:
            logger.exception("Failed to save MCP server config for '%s'", server_name)
            return False

    def test_mcp_connection(self, server_name: str) -> dict[str, Any]:
        """Test connection to an MCP server and return tool info.

        Uses ``MultiServerMCPClient`` from ``langchain-mcp-adapters``
        for stateless MCP tool discovery.

        Args:
            server_name: MCP server name as configured.

        Returns:
            Dict with ``connected`` (bool), ``tool_count`` (int),
            ``tool_names`` (list[str]), and ``error`` (str or None).
        """
        cfg = self._config.mcp_servers.get(server_name)
        if not cfg:
            return {
                "connected": False,
                "tool_count": 0,
                "tool_names": [],
                "error": "No config found",
            }

        try:
            import asyncio

            from langchain_mcp_adapters.client import MultiServerMCPClient

            adapter_cfg = {
                "transport": cfg.get("transport", "stdio"),
                "command": cfg["command"],
            }
            if cfg.get("args"):
                adapter_cfg["args"] = cfg["args"]
            if cfg.get("env"):
                adapter_cfg["env"] = cfg["env"]

            client = MultiServerMCPClient({server_name: adapter_cfg})

            async def _discover():
                return await client.get_tools()

            tools = asyncio.run(_discover())
            tool_names = [t.name for t in tools]
            return {
                "connected": True,
                "tool_count": len(tools),
                "tool_names": tool_names,
                "error": None,
            }
        except Exception as e:
            return {
                "connected": False,
                "tool_count": 0,
                "tool_names": [],
                "error": str(e),
            }

    def remove_mcp_server(self, server_name: str) -> bool:
        """Remove an MCP server from the configuration.

        Args:
            server_name: MCP server name to remove.

        Returns:
            ``True`` if the save succeeded, ``False`` otherwise.
        """
        import yaml

        config_path = ".spine/config.yaml"
        try:
            if os.path.exists(config_path):
                with open(config_path) as f:
                    all_config = yaml.safe_load(f) or {}
            else:
                all_config = {}
            mcp_servers = all_config.get("mcp_servers", {})
            if server_name in mcp_servers:
                del mcp_servers[server_name]
                all_config["mcp_servers"] = mcp_servers
                with open(config_path, "w") as f:
                    yaml.dump(all_config, f, default_flow_style=False, sort_keys=False)
                self._config = SpineConfig.load()
            return True
        except Exception:
            logger.exception("Failed to remove MCP server '%s'", server_name)
            return False

    # ── Worker ──

    def get_worker_status(self) -> dict[str, Any]:
        """Get the RalphLoopWorker queue status.

        Returns:
            Dict with queue item counts by status.
        """
        worker = get_worker(self._config)
        return {
            "running": worker.running,
            "queue": worker.queue_status(),
        }

    # ── Queue operations ──

    def get_queue_overview(self) -> dict[str, Any]:
        """Get a combined view of the queue: pending items, active job with
        phase and timing, and recent history.

        The active job is sourced from the **queue** table (via
        ``RalphLoopWorker.get_active()``) rather than the ``work_entries``
        table.  The ``work_entries`` status can go stale because
        ``dispatcher.py`` finalises it independently of the worker's queue
        lifecycle.  The queue table is the authoritative source for the
        currently-running item.

        Phase-timing details (``current_phase``, ``created_at``,
        ``updated_at``) are enriched from the corresponding ``work_entries``
        record, so the UI can display a live phase progress bar and timing
        captions.

        Returns:
            Dict with keys: pending, active, recent, status_summary.
        """
        worker = get_worker(self._config)
        pending = worker.list_pending()
        recent = worker.list_recent_completed()

        # Active job from the queue table (authoritative source).
        active = worker.get_active()

        # Enrich with phase-timing details from work_entries.
        if active is not None:
            self._enrich_active_with_work_entry(active)

        return {
            "pending": pending,
            "active": active,
            "recent": recent,
            "status_summary": worker.queue_status(),
        }

    def _enrich_active_with_work_entry(self, active: dict[str, Any]) -> None:
        """Merge ``current_phase``, ``created_at``, ``updated_at`` from the
        corresponding ``work_entries`` record into *active* (mutated in
        place).

        Correlation logic:
        1. If the queue row already has a ``work_id`` (stored by the worker
           after ``submit_work()`` returns), use it directly.
        2. Otherwise, fall back to finding the most recently-created
           ``work_entries`` row with ``status = "running"``.  Since the
           worker processes items sequentially, there should be at most
           one running work entry at any time.
        """
        work_id = active.get("work_id")
        if not work_id:
            # Fallback: find the running work entry by status.
            entries = list_work(status="running", limit=1, config=self._config)
            if entries:
                work_id = entries[0].get("id")
                if work_id:
                    # Surface the work_id so the UI can display it.
                    active["work_id"] = work_id

        if not work_id:
            return

        entry = get_work_status(work_id, self._config)
        if entry is None:
            return

        for key in ("current_phase", "created_at", "updated_at"):
            if key in entry and entry[key]:
                active[key] = entry[key]

    # ── Resume operations ──

    def resume_work(
        self,
        work_id: str,
        human_feedback: str,
        action: str = "rework",
    ) -> dict[str, Any]:
        """Resume a work item in ``needs_review`` status.

        Non-blocking: enqueues the resume for async processing via
        RalphLoopWorker and returns immediately. The work_detail page
        can poll for progress.

        Args:
            work_id: The work item ID.
            human_feedback: The human's review input.
            action: ``"rework"`` to rerun from the flagged phase,
                ``"approve"`` to proceed without rework.

        Returns:
            Dict with queue_id, status, work_id.
        """
        from spine.work.dispatcher import resume_work as _async_resume

        # Mark the work entry as running immediately so the UI
        # shows progress right away.
        self._mark_running(work_id)

        # Run resume in the background via RalphLoopWorker thread pool
        import asyncio
        import concurrent.futures

        def _run():
            asyncio.run(_async_resume(work_id, human_feedback, action, self._config))

        # Submit to the worker's executor (or a fresh one)
        executor = getattr(get_worker(self._config), "_executor", None)
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        executor.submit(_run)

        return {
            "work_id": work_id,
            "status": "running",
            "action": action,
        }

    def resume_interrupted_work(
        self,
        work_id: str,
        action: str,
        feedback: str = "",
    ) -> dict[str, Any]:
        """Resume a work item that hit an interrupt() for human review.

        Uses LangGraph's Command(resume=...) to continue from the interrupt
        point without restarting the entire graph.  This is the preferred
        resume path for subgraph-based workflows — the legacy resume_work()
        restarts the full graph from scratch.

        Non-blocking: enqueues for async processing via RalphLoopWorker.

        Args:
            work_id: The work item ID.
            action: ``"rework"``, ``"approve"``, or ``"abort"``.
            feedback: Human review text.

        Returns:
            Dict with work_id, status, and action.
        """
        from spine.work.dispatcher import resume_interrupted_work as _async_resume

        # Mark the work entry as running immediately so the UI
        # shows progress right away.
        self._mark_running(work_id)

        # Run resume in the background via RalphLoopWorker thread pool
        import asyncio
        import concurrent.futures

        def _run():
            asyncio.run(_async_resume(work_id, action, feedback, self._config))

        executor = getattr(get_worker(self._config), "_executor", None)
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        executor.submit(_run)

        return {
            "work_id": work_id,
            "status": "running",
            "action": action,
        }

    def _mark_running(self, work_id: str) -> None:
        """Transition a needs_review work entry back to running."""
        from spine.work.dispatcher import update_work_status

        update_work_status(work_id, TaskStatus.RUNNING.value, config=self._config)

    def restart_work(
        self,
        work_id: str,
        clear_artifacts: bool = False,
    ) -> dict[str, Any]:
        """Restart a running, stalled, or needs_review work item from phase 0.

        Non-blocking: runs the restart in the background via
        RalphLoopWorker's executor and returns immediately. The work_detail
        page can poll for progress.

        Args:
            work_id: The work item ID to restart.
            clear_artifacts: If True, delete on-disk artifacts before
                restarting so all phases regenerate from scratch.

        Returns:
            Dict with work_id, status, and work_type.
        """
        from spine.work.dispatcher import restart_work

        import concurrent.futures

        def _run() -> None:
            import asyncio

            asyncio.run(restart_work(work_id, self._config, clear_artifacts=clear_artifacts))

        # Submit to the worker's executor (or a fresh one)
        executor = getattr(get_worker(self._config), "_executor", None)
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        executor.submit(_run)

        return {
            "work_id": work_id,
            "status": TaskStatus.RUNNING.value,
            "action": "restart",
        }

    def restart_from_phase(
        self,
        work_id: str,
        phase_name: str,
        clear_artifacts: bool = False,
    ) -> dict[str, Any]:
        """Restart a work item from a specific phase.

        Unlike ``restart_work`` (which starts from phase 0), this
        rebuilds the graph so that the START edge routes directly
        to the requested phase. Earlier phases and their artifacts
        are preserved.

        Non-blocking: runs the restart in the background via
        RalphLoopWorker's executor and returns immediately.

        Args:
            work_id: The work item ID to restart.
            phase_name: The phase to start from (e.g. ``"implement"``).
            clear_artifacts: If True, delete on-disk artifacts for the
                target phase and all subsequent phases. Earlier artifacts
                are always preserved.

        Returns:
            Dict with work_id, status, phase_name, action, and optionally
            message. When status is "skipped", the message explains why
            the restart was not initiated (e.g., work already running).
        """
        from spine.work.dispatcher import restart_from_phase as _async_restart

        import concurrent.futures

        # Mark the work entry as running immediately so the UI
        # shows progress right away.
        self._mark_running(work_id)

        def _run() -> None:
            import asyncio

            asyncio.run(
                _async_restart(work_id, phase_name, self._config, clear_artifacts=clear_artifacts)
            )

        # Submit to the worker's executor (or a fresh one)
        executor = getattr(get_worker(self._config), "_executor", None)
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        executor.submit(_run)

        return {
            "work_id": work_id,
            "status": TaskStatus.RUNNING.value,
            "phase_name": phase_name,
            "action": "restart_from_phase",
        }

    def get_restart_phases(self, work_type: str) -> list[str]:
        """Return valid phase names for restart_from_phase for a work type.

        Filters out critic nodes since restarting into a critic doesn't
        make sense. Used by the UI to populate the phase dropdown.

        Args:
            work_type: One of the valid WorkType values.

        Returns:
            Sorted list of non-critic phase names.
        """
        from spine.workflow.compose import get_restart_phases as _get_phases

        return _get_phases(work_type)

    def stop_work(self, work_id: str) -> dict[str, Any]:
        """Stop a running work item.

        Cancels the queue item (if pending) or marks the running work as
        cancelled. Also purges the LangGraph checkpoint so the work can
        be restarted cleanly if needed.

        Non-blocking: runs the stop in the background via RalphLoopWorker's
        executor and returns immediately.

        Args:
            work_id: The work item ID to stop.

        Returns:
            Dict with work_id, status, and action.
        """
        import concurrent.futures

        def _run() -> None:
            worker = get_worker(self._config)
            # Try to cancel a pending item first
            active = worker.get_active()
            if active and active.get("work_id") == work_id:
                # Running item — cancel by work_id
                worker.cancel_running(work_id)
            else:
                # Try to find and cancel pending item by work_id
                # (work_id is stored in queue items after submission)
                db = worker._get_db()
                item = db["queue"].rows_where(
                    "work_id = ? AND status = ?",
                    [work_id, "pending"],
                    limit=1,
                )
                item = item[0] if item else None
                if item:
                    worker.cancel_item(item["id"])

        executor = getattr(get_worker(self._config), "_executor", None)
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        executor.submit(_run)

        return {
            "work_id": work_id,
            "status": TaskStatus.RUNNING.value,
            "action": "stop",
        }

    def reset_stuck_items(self) -> int:
        """Reset any queue items stuck in 'running' back to 'pending'.

        Delegates to RalphLoopWorker.reset_stuck_items().  Use this when
        the worker or UI died mid-execution and items are permanently
        stuck in the 'running' state.

        Returns:
            The number of items that were reset.
        """
        worker = get_worker(self._config)
        return worker.reset_stuck_items()

    # ── Planning operations ──

    def list_planning_sessions(
        self,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List planning work items.

        Args:
            status: Optional status filter (e.g. 'completed', 'needs_review', 'awaiting_approval').
            limit: Maximum number of results to return.

        Returns:
            List of planning work item dicts.
        """
        from spine.work.dispatcher import list_plans

        return list_plans(status=status, limit=limit, config=self._config)

    def get_planning_detail(self, plan_id: str) -> dict[str, Any] | None:
        """Get details for a planning work item.

        Args:
            plan_id: The planning work item ID.

        Returns:
            Dict with work entry fields plus spec/plan artifacts, or None if not found.
        """
        entry = self.get_work(plan_id)
        if entry is None:
            return None

        result = dict(entry)
        result["artifacts"] = {}

        # Load spec and plan artifacts
        for phase in ("specify", "plan"):
            for name in ("spec.md", "specification.md", "plan.md"):
                content = self.read_artifact(plan_id, phase, name)
                if content:
                    result["artifacts"][f"{phase}/{name}"] = content[:5000]  # Truncate for UI
                    break

        return result

    async def approve_plan(
        self,
        plan_id: str,
        action: str = "approve",
        feedback: str | None = None,
    ) -> dict[str, Any]:
        """Approve a planning work item and optionally spawn execution tasks.

        Awaits approve_and_spawn and returns its result directly.
        If the operation fails, returns an error dict.

        Args:
            plan_id: The planning work item ID.
            action: One of "approve", "request_revision", "reject".
            feedback: Optional feedback text.

        Returns:
            Dict with plan_id, status, spawned_ids (if approved), and
            error key on failure.
        """
        from spine.work.dispatcher import approve_and_spawn

        try:
            result = await approve_and_spawn(plan_id, action, feedback, self._config)
            return result
        except Exception as e:
            logger.exception(f"approve_plan failed for {plan_id}")
            return {
                "plan_id": plan_id,
                "status": "error",
                "action": action,
                "error": str(e),
            }
