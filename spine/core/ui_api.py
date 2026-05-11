"""Backend API for the SPINE UI.

Thin API layer that the Streamlit UI calls to interact with SPINE core.
Separates UI concerns from core business logic.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional


class UIApi:
    """Thin API layer between the Streamlit UI and SPINE core.

    Provides thread-safe read/write operations for the UI.
    All write operations use a lock to prevent concurrent access.
    """

    def __init__(self, checkpoint_path: str = ".spine/spine.db"):
        """Initialize the UI API.

        Args:
            checkpoint_path: Path to the checkpoint SQLite database.
        """
        self._checkpoint_path = checkpoint_path
        self._lock = threading.Lock()

    # ── Read Operations ───────────────────────────────────────

    def get_active_work_items(self) -> list[dict]:
        """Return all work items with their latest status.

        Returns:
            List of work item dicts with phase, progress, etc.
        """
        from ..ui.utils import get_active_work_items
        return get_active_work_items(self._checkpoint_path)

    def get_work_item_detail(self, thread_id: str) -> Optional[dict]:
        """Load full state from the latest checkpoint.

        Args:
            thread_id: Thread ID to query.

        Returns:
            Full state dict or None if not found.
        """
        from ..ui.utils import get_work_item_detail
        return get_work_item_detail(thread_id, self._checkpoint_path)

    def get_checkpoints(self, thread_id: str) -> list[dict]:
        """Return all checkpoints for a work item.

        Args:
            thread_id: Thread ID to query.

        Returns:
            List of checkpoint records.
        """
        from ..ui.utils import get_checkpoints
        return get_checkpoints(thread_id, self._checkpoint_path)

    def get_llm_providers(self) -> list[dict]:
        """Get configured LLM providers.

        Returns:
            List of provider config dicts.
        """
        from ..ui.utils import get_llm_providers
        return get_llm_providers()

    # ── Write Operations ──────────────────────────────────────

    def start_work(
        self,
        requirement: str,
        method: str = "Quick Work",
        project_type: str = "Greenfield",
        llm_provider: str = "ollama",
        parallel_agents: int = 3,
        idempotency_key: Optional[str] = None,
    ) -> dict:
        """Start a new work item. Returns thread_id on success.

        Args:
            requirement: The requirement text for the work item.
            method: Automation level ("Quick Work", "Full Spec Work", etc.).
            project_type: Environment type ("Greenfield" or "Brownfield").
            llm_provider: LLM provider name.
            parallel_agents: Maximum parallel agents within a phase.
            idempotency_key: UUIDv4 for duplicate detection.

        Returns:
            Dict with 'thread_id' on success, or with 'error' on failure.
        """
        from ..ui.utils import start_work

        with self._lock:
            result = start_work(
                requirement=requirement,
                method=method,
                project_type=project_type,
                llm_provider=llm_provider,
                parallel_agents=parallel_agents,
                checkpoint_path=self._checkpoint_path,
                idempotency_key=idempotency_key,
            )
            return result or {"error": "start_work returned no result"}

    def approve_gate(self, thread_id: str) -> bool:
        """Approve the critic gate for a work item.

        Args:
            thread_id: Thread ID for the work item.

        Returns:
            True on success, False on failure.
        """
        return UIApi.approve_gate_file(thread_id, True)

    def reject_gate(self, thread_id: str, feedback: str) -> bool:
        """Reject the critic gate with feedback for rework.

        Args:
            thread_id: Thread ID for the work item.
            feedback: Feedback text for the planner to address.

        Returns:
            True on success, False on failure.
        """
        return UIApi.approve_gate_file(
            thread_id, False, feedback=feedback
        )

    @staticmethod
    def approve_gate_file(thread_id: str, approved: bool, feedback: str = "") -> bool:
        """Write gate approval/rejection to a file.

        Args:
            thread_id: Thread ID for the work item.
            approved: True for approval, False for rejection.
            feedback: Optional feedback text.

        Returns:
            True on success, False on failure.
        """
        gate_file = Path(f".spine/state/gate_result_{thread_id}.json")
        gate_file.parent.mkdir(parents=True, exist_ok=True)
        gate_file.write_text(json.dumps({
            "approved": approved,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat(),
        }))
        return True

    def resume_work(self, thread_id: str) -> bool:
        """Resume a paused work item.

        Args:
            thread_id: Thread ID to resume.

        Returns:
            True on success, False on failure.
        """
        from ..ui.utils import resume_work
        return resume_work(thread_id, self._checkpoint_path)

    def delete_work(self, thread_id: str) -> bool:
        """Delete a work item's checkpoint data.

        Args:
            thread_id: Thread ID to delete.

        Returns:
            True on success, False on failure.
        """
        from ..ui.utils import delete_work
        return delete_work(thread_id)

    def update_provider_config(self, config: dict) -> bool:
        """Update the provider configuration.

        Args:
            config: Full provider configuration dict.

        Returns:
            True on success, False on failure.
        """
        from ..ui.utils import save_config
        try:
            save_config(config, ".spine/config.yaml")
            return True
        except Exception:
            return False

    def get_provider_config(self) -> dict:
        """Get the current provider configuration.

        Returns:
            Full provider configuration dict.
        """
        from ..ui.utils import load_config
        return load_config(".spine/config.yaml")

    # ── Queue Operations ───────────────────────────────────────

    def get_queue_status(self) -> dict[str, int]:
        """Get summary counts from the task queue.

        Returns:
            Dict with pending, running, success, failed, cancelled counts.
        """
        from ..ui.utils import get_queue_status
        return get_queue_status()

    def get_queue_items(self, status: Optional[str] = None) -> list[dict]:
        """Get queue items, optionally filtered by status.

        Args:
            status: Optional status filter (pending, running, success, failed).

        Returns:
            List of queue item dicts.
        """
        from ..ui.utils import get_queue_items
        return get_queue_items(status)

    def enqueue_task(self, requirement: str, method: str = "Quick Work",
                     priority: int = 0) -> Optional[str]:
        """Enqueue a task for Ralph Loop processing.

        Args:
            requirement: The work requirement text.
            method: Automation level.
            priority: Task priority.

        Returns:
            Task ID on success, None on failure.
        """
        from ..ui.utils import enqueue_task
        return enqueue_task(requirement, method, priority)

    def retry_queue_task(self, task_id: str) -> bool:
        """Re-enqueue a failed task with the same payload.

        Args:
            task_id: The failed task ID.

        Returns:
            True on success, False on failure.
        """
        from ..ui.utils import retry_queue_task
        return retry_queue_task(task_id)

    def clear_completed_queue_tasks(self) -> int:
        """Remove all acknowledged items from the queue.

        Returns:
            Number of items removed.
        """
        from ..ui.utils import clear_completed_queue_tasks
        return clear_completed_queue_tasks()

    def get_worker_status(self) -> dict:
        """Get the current Ralph Loop worker status.

        Returns:
            Worker status dict.
        """
        from ..work.ralph_worker import get_worker
        worker = get_worker()
        return worker.status

    def start_worker(self) -> None:
        """Start the Ralph Loop worker."""
        from ..work.ralph_worker import get_worker
        worker = get_worker()
        worker.start()

    def pause_worker(self) -> None:
        """Pause the Ralph Loop worker."""
        from ..work.ralph_worker import get_worker
        worker = get_worker()
        worker.pause()

    # ── Artifact Operations ────────────────────────────────────

    def get_work_item_artifacts(self, thread_id: str) -> list[dict]:
        """Get all artifact files for a work item.

        Args:
            thread_id: Work item thread ID.

        Returns:
            List of artifact dicts.
        """
        from ..ui.utils import get_work_item_artifacts
        return get_work_item_artifacts(thread_id)

    def get_feature_slice_outcomes(self, detail: dict) -> list[dict]:
        """Extract FeatureSlice outcome data from work item detail.

        Args:
            detail: Work item detail dict.

        Returns:
            List of slice outcome dicts.
        """
        from ..ui.utils import get_feature_slice_outcomes
        return get_feature_slice_outcomes(detail)

    # ── Agent Resource Operations ──────────────────────────────

    def get_agent_resources(self) -> list[dict]:
        """Read all agent resource files.

        Returns:
            List of resource dicts.
        """
        from ..ui.utils import get_agent_resources
        return get_agent_resources()

    def save_agent_resource(self, key: str, content: str) -> bool:
        """Save content to an agent resource file.

        Args:
            key: Resource key.
            content: New content.

        Returns:
            True on success.
        """
        from ..ui.utils import save_agent_resource
        return save_agent_resource(key, content)

    def regenerate_agent_resource(self, key: str) -> Optional[str]:
        """Regenerate an agent resource from project analysis.

        Args:
            key: Resource key.

        Returns:
            Generated content or None.
        """
        from ..ui.utils import regenerate_agent_resource
        return regenerate_agent_resource(key)

    # ── SDD Operations ─────────────────────────────────────────

    def get_sdd_projects(self) -> list[dict]:
        """Get all SDD projects.

        Returns:
            List of project dicts.
        """
        from ..ui.utils import get_sdd_projects
        return get_sdd_projects()

    def start_sdd_project(self, name: str, requirement: str,
                          method: str = "Full Spec Project",
                          project_type: str = "Greenfield",
                          llm_provider: str = "",
                          use_worktrees: bool = False) -> Optional[dict]:
        """Start a new SDD project.

        Args:
            name: Project name.
            requirement: The work requirement.
            method: Automation level.
            project_type: Environment type.
            llm_provider: LLM provider name.
            use_worktrees: Whether to use git worktrees.

        Returns:
            Dict with project_id on success.
        """
        from ..ui.utils import start_sdd_project
        return start_sdd_project(
            name, requirement, method, project_type,
            llm_provider, use_worktrees,
        )

    def update_sdd_project_phase(self, project_id: str, phase: str,
                                  status: str) -> None:
        """Update the phase status of an SDD project.

        Args:
            project_id: Project ID.
            phase: Phase name.
            status: Phase status.
        """
        from ..ui.utils import update_sdd_project_phase
        update_sdd_project_phase(project_id, phase, status)
