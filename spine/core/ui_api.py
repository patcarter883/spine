"""Backend API for the SPINE UI.

Thin API layer that the Streamlit UI calls to interact with SPINE core.
Separates UI concerns from core business logic.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..core.state_machine import SpineStateMachine


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
    ) -> dict:
        """Start a new work item. Returns thread_id on success.

        Args:
            requirement: The requirement text for the work item.
            method: Automation level ("Quick Work", "Full Spec Work", etc.).
            project_type: Environment type ("Greenfield" or "Brownfield").
            llm_provider: LLM provider name.
            parallel_agents: Maximum parallel agents within a phase.

        Returns:
            Dict with 'thread_id' on success, empty dict on failure.
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
            )
            return result or {}

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
