"""SPINE UI API — sole read/write interface for Streamlit.

UI pages MUST use UIApi for all data access. Never import directly from
workflow/, phases/, or work.dispatcher. This maintains the zero-duplication
principle: CLI and UI share the same backend code paths.
"""

from __future__ import annotations

import json
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
        from spine.persistence.project_store import ProjectStore

        self._projects = ProjectStore(base_path=self._config.project_path)

    # ── Project operations (read-only) ──

    def get_project(self, project_id: str) -> dict[str, Any] | None:
        """Return a project spec as a dict, or None if it does not exist."""
        spec = self._projects.load_project(project_id)
        return spec.model_dump() if spec else None

    def list_projects(self) -> list[dict[str, Any]]:
        """List all projects with id, title, and member count."""
        out: list[dict[str, Any]] = []
        for pid in self._projects.list_projects():
            spec = self._projects.load_project(pid)
            if spec:
                out.append(
                    {"id": spec.id, "title": spec.title, "members": len(spec.member_work_ids)}
                )
        return out

    def get_project_coverage(self, project_id: str) -> dict[str, Any] | None:
        """Compute deterministic requirement coverage for a project, or None.

        Wraps the async aggregator with ``asyncio.run`` so Streamlit's sync
        pages can call it directly.
        """
        import asyncio

        spec = self._projects.load_project(project_id)
        if spec is None:
            return None
        from spine.project.aggregator import aggregate_project_coverage

        return asyncio.run(aggregate_project_coverage(spec, self._config))

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

    def enqueue_onboarding(
        self,
        workspace_root: str,
        mode: str,
        tech_stack: list[str] | None = None,
    ) -> dict[str, Any]:
        """Enqueue a repository-onboarding job via the RalphLoopWorker.

        Non-blocking, mirroring :meth:`enqueue_work`: serialises the onboarding
        parameters into the queue item's JSON description and enqueues it with
        ``work_type="onboarding"``. The dispatcher routes that work_type to the
        onboarding engine (analyse/scaffold/synthesise) instead of the LangGraph
        workflow. The Onboarding page polls ``get_queue_overview()`` for
        progress, exactly like the queue page.

        Args:
            workspace_root: Absolute path to the project to onboard.
            mode: ``"greenfield"`` or ``"brownfield"``.
            tech_stack: Optional stack tags (seed for greenfield).

        Returns:
            Dict with ``queue_id``, ``status``, and ``work_type``.
        """
        worker = get_worker(self._config)
        description = json.dumps(
            {
                "workspace_root": workspace_root,
                "mode": mode,
                "tech_stack": list(tech_stack or []),
            }
        )
        queue_id = worker.enqueue(description=description, work_type="onboarding")
        worker.start()  # no-op if already running
        logger.info(f"Enqueued onboarding via RalphLoopWorker: queue_id={queue_id}")
        return {
            "queue_id": queue_id,
            "status": "pending",
            "work_type": "onboarding",
        }

    # ── Artifact operations ──

    def _artifact_store_for(self, work_id: str) -> ArtifactStore:
        """Resolve the :class:`ArtifactStore` that holds *work_id*'s artifacts.

        Onboarding jobs write their manifest + four docs under
        ``<workspace_root>/.spine/artifacts`` where ``workspace_root`` is the
        *target* repo being onboarded — which, for an EXTERNAL repo, differs
        from spine's own ``artifact_path``. The engine records that
        ``workspace_root`` in the work entry's result payload, so we read it and
        point the store at the matching base. Any non-onboarding item (or an
        onboarding item without a recorded ``workspace_root``) falls back to the
        global store, leaving all other artifact reads unchanged.
        """
        from pathlib import Path

        entry = get_work_status(work_id, self._config)
        if entry and entry.get("work_type") == "onboarding":
            result = entry.get("result")
            if isinstance(result, dict):
                workspace_root = result.get("workspace_root")
                if isinstance(workspace_root, str) and workspace_root:
                    base = str(Path(workspace_root) / ".spine" / "artifacts")
                    if base != self._config.artifact_path:
                        return ArtifactStore(base_path=base)
        return self._artifacts

    def get_artifacts(self, work_id: str) -> list[dict[str, Any]]:
        """List all artifacts for a work item.

        Args:
            work_id: The work item ID.

        Returns:
            List of artifact metadata dicts.
        """
        return self._artifact_store_for(work_id).list_artifacts(work_id)

    def read_artifact(self, work_id: str, phase: str, name: str) -> str | None:
        """Read the content of a specific artifact.

        Args:
            work_id: The work item ID.
            phase: The phase name.
            name: The artifact filename.

        Returns:
            The artifact content, or None.
        """
        return self._artifact_store_for(work_id).load_artifact(work_id, phase, name)

    def read_onboarding_doc(
        self, name: str, workspace_root: str | None = None
    ) -> str | None:
        """Read one onboarding document from the stable docs directory.

        Onboarding documents are a single source of truth written to
        ``<workspace_root>/.spine/onboarding/<NAME>.md`` — there is exactly one
        version per workspace, independent of any onboarding job's ``work_id``.
        Reads come straight from that location, so no work item is needed.

        Args:
            name: The document filename (e.g. ``PROJECT_DEFINITION.md``).
            workspace_root: The project to read docs from. Defaults to the
                configured workspace root when omitted/empty (the common case);
                pass an explicit path to read an externally-onboarded repo.

        Returns:
            The document content, or ``None`` if unavailable.
        """
        from spine.work.onboarding.synthesis_tools import onboarding_docs_dir

        root = workspace_root or self._config.workspace_root
        if not root:
            return None
        try:
            return (onboarding_docs_dir(root) / name).read_text(encoding="utf-8")
        except OSError:
            return None

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

    def get_critic_review(self, work_id: str) -> dict[str, Any] | None:
        """Get the critic's final verdict for a work item.

        Unlike ``get_feedback`` (which only surfaces needs_review entries),
        this returns the ``last_critic_review`` state field — the critic's
        most recent verdict — which is available for any flagged item,
        including ``awaiting_approval`` plans whose critic review PASSED.

        Args:
            work_id: The work item ID.

        Returns:
            The review dict (keys: phase, status, tier, reason, suggestions,
            attempt), or None if no critic review is available.
        """
        entry = self.get_work(work_id)
        if entry is None:
            return None
        result = entry.get("result", {})
        if isinstance(result, dict):
            review = result.get("last_critic_review")
            if isinstance(review, dict) and review:
                return review
        return None

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
            "providers": self._config.providers,
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

    # ── Provider CRUD ──

    def _save_config(self, all_config: dict) -> bool:
        """Write *all_config* to ``.spine/config.yaml`` and reload.

        Common helper for all provider/MCP configuration mutations.

        Args:
            all_config: The full configuration dict to persist.

        Returns:
            ``True`` on success, ``False`` on exception.
        """
        import yaml

        config_path = ".spine/config.yaml"
        try:
            with open(config_path, "w") as f:
                yaml.dump(all_config, f, default_flow_style=False, sort_keys=False)
            self._config = SpineConfig.load()
            return True
        except Exception:
            logger.exception("Failed to save config")
            return False

    def get_providers(self) -> dict:
        """Return the current provider configuration.

        Returns:
            Dict with ``llm`` (list of provider dicts) and ``phases`` (dict).
        """
        return self._config.providers

    def add_llm_provider(self, name: str, provider_config: dict) -> bool:
        """Add a new LLM provider to the configuration.

        Args:
            name: Provider name (also stored in the provider dict).
            provider_config: Provider config dict (model, base_url, api_key, etc.).

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
            llm_list = all_config.get("providers", {}).get("llm", [])
            if llm_list is None:
                llm_list = []
            entry = {**provider_config, "name": name}
            llm_list.append(entry)
            all_config.setdefault("providers", {})["llm"] = llm_list
            return self._save_config(all_config)
        except Exception:
            logger.exception("Failed to add LLM provider '%s'", name)
            return False

    def update_llm_provider(self, name: str, provider_config: dict) -> bool:
        """Update an existing LLM provider's configuration.

        Fields in *provider_config* are merged into the existing provider
        entry; other fields on the existing entry are preserved.

        Args:
            name: Provider name to find.
            provider_config: Config dict with fields to update.

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
            llm_list = all_config.get("providers", {}).get("llm", [])
            if llm_list is None:
                llm_list = []
            for entry in llm_list:
                if isinstance(entry, dict) and entry.get("name") == name:
                    entry.update(provider_config)
                    break
            else:
                # Provider not found — create it
                llm_list.append({**provider_config, "name": name})
            all_config.setdefault("providers", {})["llm"] = llm_list
            return self._save_config(all_config)
        except Exception:
            logger.exception("Failed to update LLM provider '%s'", name)
            return False

    def remove_llm_provider(self, name: str) -> bool:
        """Remove an LLM provider and clean up phase references.

        Args:
            name: Provider name to remove.

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
            llm_list = all_config.get("providers", {}).get("llm", [])
            if llm_list is None:
                llm_list = []
            all_config.setdefault("providers", {})["llm"] = [
                entry for entry in llm_list
                if not (isinstance(entry, dict) and entry.get("name") == name)
            ]
            # Clean up phase references that point to this provider
            phases = all_config.get("providers", {}).get("phases", {})
            if phases:
                for phase_cfg in phases.values():
                    if isinstance(phase_cfg, dict) and phase_cfg.get("provider") == name:
                        del phase_cfg["provider"]
            all_config.setdefault("providers", {})["phases"] = phases
            return self._save_config(all_config)
        except Exception:
            logger.exception("Failed to remove LLM provider '%s'", name)
            return False

    def set_phase_provider(self, phase: str, config: dict) -> bool:
        """Set (or update) the provider configuration for a phase.

        Args:
            phase: Phase name (e.g. ``"implement"``).
            config: Phase config dict. May contain ``provider`` (reference
                name) or direct keys like ``model``, ``base_url``,
                ``api_key``, ``temperature``, etc.

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
            phases = all_config.get("providers", {}).get("phases", {})
            if phases is None:
                phases = {}
            phases[phase] = config
            all_config.setdefault("providers", {})["phases"] = phases
            return self._save_config(all_config)
        except Exception:
            logger.exception("Failed to set phase provider '%s'", phase)
            return False

    def get_phase_providers(self) -> dict:
        """Return the phase-provider mapping.

        Returns:
            Dict of phase names to their provider configuration dicts.
        """
        return self._config.providers.get("phases", {})

    # ── Worker ──

    def get_worker_status(self) -> dict[str, Any]:
        """Get the RalphLoopWorker queue status.

        ``running`` reflects the actual liveness of the worker's
        processing-loop thread (not a remembered flag), so the queue page
        never reports "not running" while the loop is in fact alive.

        Returns:
            Dict with queue item counts by status.
        """
        worker = get_worker(self._config)
        return {
            "running": worker.is_alive(),
            "queue": worker.queue_status(),
        }

    def ensure_worker_running(self) -> None:
        """Start the background worker loop if it isn't already alive.

        Idempotent — safe to call on every Streamlit re-run. Called at
        app boot so the queue is always being serviced and the worker
        status the UI shows is truthful.
        """
        worker = get_worker(self._config)
        if not worker.is_alive():
            worker.start()

    # ── Queue operations ──

    def get_queue_overview(self) -> dict[str, Any]:
        """Get a combined view of the queue: pending items, every active
        job with phase and timing, and recent history.

        Active jobs are sourced from the **work_entries** table, which is
        the one record updated by *every* execution path — the normal
        queue loop, onboarding, restart, and resume all mark a work entry
        ``running``. The **queue** table only knows about jobs that went
        through the queue loop, so sourcing the active set from it alone
        hid restart/resume jobs (and reported the worker idle while they
        ran). Each running work entry is enriched with its queue row
        (queue id, enqueued-at) when one exists.

        Returns:
            Dict with keys: pending, active_jobs (list), active (the first
            active job, for back-compat), recent, status_summary.
        """
        worker = get_worker(self._config)
        pending = worker.list_pending()
        recent = worker.list_recent_completed()
        active_jobs = self._active_jobs(worker)

        return {
            "pending": pending,
            "active_jobs": active_jobs,
            "active": active_jobs[0] if active_jobs else None,
            "recent": recent,
            "status_summary": worker.queue_status(),
        }

    def _active_jobs(self, worker: Any) -> list[dict[str, Any]]:
        """Return every work item currently executing, whatever launched it.

        Two sources, unioned by work_id:

        1. **Running queue rows** — jobs the worker loop is actively
           processing. Each is enriched with its work entry fetched *by
           work_id* (whatever its status), so a job whose work entry was
           marked ``stalled`` mid-run still surfaces that signal on its
           active card. Rows with no work entry yet (the hand-off window)
           are kept so the list never flickers empty.
        2. **Running work entries with no running queue row** — restart,
           resume, and any other path that executes off the queue loop and
           only updates ``work_entries``.

        A reset/finalised job leaves both sources (its queue row goes
        ``pending`` and its work entry leaves ``running``), so it correctly
        drops out of the active set.
        """
        active: list[dict[str, Any]] = []
        seen_work_ids: set[str] = set()

        # 1. Queue rows the worker is processing right now.
        for row in worker.list_running():
            work_id = row.get("work_id")
            entry = get_work_status(work_id, self._config) if work_id else None
            if work_id:
                seen_work_ids.add(work_id)
            active.append(self._build_active_job(entry, row))

        # 2. Off-queue in-flight work (restart/resume) the queue table can't
        #    see. Include "stalled" as well as "running" so a stuck off-queue
        #    job stays visible with its stalled banner instead of silently
        #    vanishing the moment the stall-timeout fires.
        for entry in self._off_queue_active_entries():
            work_id = entry.get("id")
            if work_id and work_id in seen_work_ids:
                continue
            if work_id:
                seen_work_ids.add(work_id)
            active.append(self._build_active_job(entry, None))

        return active

    def _off_queue_active_entries(self) -> list[dict[str, Any]]:
        """Work entries that count as in-flight: ``running`` or ``stalled``."""
        return list_work(
            status="running", limit=50, config=self._config
        ) + list_work(status="stalled", limit=50, config=self._config)

    def _build_active_job(
        self,
        entry: dict[str, Any] | None,
        queue_row: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Normalise a running job into the shape the queue/onboarding
        pages expect, merging the work entry (phase, timing, work status)
        with its queue row (queue id, enqueued-at) when available.
        """
        active: dict[str, Any] = {"status": "running"}

        if queue_row is not None:
            active.update(
                {
                    "id": queue_row.get("id"),
                    "work_id": queue_row.get("work_id") or "",
                    "work_type": queue_row.get("work_type"),
                    "enqueued_at": queue_row.get("enqueued_at"),
                    "started_at": queue_row.get("started_at"),
                    "created_at": queue_row.get("started_at"),
                    "description": queue_row.get("description", ""),
                }
            )

        if entry is not None:
            active["work_id"] = entry.get("id") or active.get("work_id", "")
            for key in (
                "work_type",
                "current_phase",
                "created_at",
                "updated_at",
                "description",
            ):
                value = entry.get(key)
                if value:
                    active[key] = value
            # Work-entry status drives stalled detection; queue lifecycle
            # status stays "running" for the active card.
            if entry.get("status"):
                active["work_status"] = entry["status"]
            # Onboarding jobs record their mode in the result payload; surface
            # it so the Onboarding page picks the right phase sequence.
            result = entry.get("result")
            if isinstance(result, dict):
                mode = result.get("mode")
                if isinstance(mode, str) and mode:
                    active["mode"] = mode

        active.setdefault("work_type", "spec")
        active.setdefault("description", "")
        active.setdefault("work_id", "")
        return active

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

        # Run resume in the background via the worker's shared executor.
        import asyncio

        def _run():
            asyncio.run(_async_resume(work_id, human_feedback, action, self._config))

        # Run on the worker's shared, persistent out-of-band pool so the
        # job is tracked alongside the worker rather than on a throwaway
        # executor that is never shut down.
        get_worker(self._config).get_executor().submit(_run)

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

        # Run resume in the background via the worker's shared executor.
        import asyncio

        def _run():
            asyncio.run(_async_resume(work_id, action, feedback, self._config))

        # Run on the worker's shared, persistent out-of-band pool so the
        # job is tracked alongside the worker rather than on a throwaway
        # executor that is never shut down.
        get_worker(self._config).get_executor().submit(_run)

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

        def _run() -> None:
            import asyncio

            asyncio.run(restart_work(work_id, self._config, clear_artifacts=clear_artifacts))

        # Run on the worker's shared, persistent out-of-band pool so the
        # job is tracked alongside the worker rather than on a throwaway
        # executor that is never shut down.
        get_worker(self._config).get_executor().submit(_run)

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

        # Mark the work entry as running immediately so the UI
        # shows progress right away.
        self._mark_running(work_id)

        def _run() -> None:
            import asyncio

            asyncio.run(
                _async_restart(work_id, phase_name, self._config, clear_artifacts=clear_artifacts)
            )

        # Run on the worker's shared, persistent out-of-band pool so the
        # job is tracked alongside the worker rather than on a throwaway
        # executor that is never shut down.
        get_worker(self._config).get_executor().submit(_run)

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
        def _run() -> None:
            worker = get_worker(self._config)
            # Try to cancel a running queue item first.
            active = worker.get_active()
            if active and active.get("work_id") == work_id:
                worker.cancel_running(work_id)
                return

            # Then a pending queue item (work_id is stored after submission).
            db = worker._get_db()
            item = db["queue"].rows_where(
                "work_id = ? AND status = ?",
                [work_id, "pending"],
                limit=1,
            )
            item = item[0] if item else None
            if item:
                worker.cancel_item(item["id"])
                return

            # No queue row — this is a restart/resume job running on the
            # out-of-band pool. Mark the work entry cancelled and purge its
            # checkpoint so it drops out of the active set and can be
            # restarted cleanly. (The detached thread isn't force-killed,
            # but the graph honours the cancelled status at its next step.)
            from spine.persistence.checkpoint import CheckpointStore
            from spine.work.dispatcher import update_work_status

            # "cancelled" matches the literal the queue table uses; there is
            # no TaskStatus.CANCELLED member.
            update_work_status(work_id, "cancelled", config=self._config)
            try:
                import asyncio

                store = CheckpointStore(db_path=self._config.checkpoint_path)
                saver = asyncio.run(store.get_checkpointer())
                asyncio.run(saver.adelete_thread(work_id))
            except Exception:
                pass

        # Run on the worker's shared, persistent out-of-band pool so the
        # job is tracked alongside the worker rather than on a throwaway
        # executor that is never shut down.
        get_worker(self._config).get_executor().submit(_run)

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

        Also finalises lingering ``work_entries`` rows. Because the active
        set is now sourced from in-flight work entries (not just queue
        rows), a job whose thread died leaves its entry stuck
        ``running``/``stalled`` and would otherwise show as a permanent
        phantom active card — including off-queue restart/resume jobs that
        never had a queue row at all, which the queue-only reset could
        never clear. Every in-flight entry not backed by a *live* running
        queue row is marked terminal (``failed``) and its checkpoint
        purged, so it leaves the active set.

        Returns:
            The number of items that were reset/finalised.
        """
        from spine.work.dispatcher import update_work_status

        worker = get_worker(self._config)

        # Snapshot BEFORE resetting so we don't catch a job the worker loop
        # reprocesses in the meantime. A "live" work_id is one currently
        # backed by a running queue row; everything else that is in-flight
        # is a ghost to finalise.
        live_work_ids = {
            row.get("work_id")
            for row in worker.list_running()
            if row.get("work_id")
        }
        ghosts = [
            entry
            for entry in self._off_queue_active_entries()
            if entry.get("id") and entry["id"] not in live_work_ids
        ]

        # Reset stuck queue rows (running/stalled -> pending) for reprocessing.
        reset_count = worker.reset_stuck_items()

        # Finalise the ghost work entries so they drop out of the active set.
        # The queue-correlated ones (live_work_ids) get a fresh work_id on
        # reprocess, so finalising their stale entry here is safe too.
        finalised = self._finalise_ghost_entries(
            ghosts + [{"id": wid} for wid in live_work_ids]
        )

        return reset_count + finalised

    def _finalise_ghost_entries(self, entries: list[dict[str, Any]]) -> int:
        """Mark each lingering work entry ``failed`` and purge its checkpoint.

        ``failed`` is terminal and not part of the in-flight set, so the
        entry leaves the active list. Returns the count actually finalised
        (entries already terminal are skipped).
        """
        from spine.persistence.checkpoint import CheckpointStore
        from spine.work.dispatcher import update_work_status

        store = CheckpointStore(db_path=self._config.checkpoint_path)
        seen: set[str] = set()
        count = 0
        for entry in entries:
            work_id = entry.get("id")
            if not work_id or work_id in seen:
                continue
            seen.add(work_id)
            current = get_work_status(work_id, self._config)
            if not current or current.get("status") not in ("running", "stalled"):
                continue
            update_work_status(work_id, "failed", config=self._config)
            try:
                import asyncio

                saver = asyncio.run(store.get_checkpointer())
                asyncio.run(saver.adelete_thread(work_id))
            except Exception:
                logger.warning(
                    "Checkpoint purge failed for ghost work %s (continuing)",
                    work_id,
                    exc_info=True,
                )
            count += 1
        return count

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
