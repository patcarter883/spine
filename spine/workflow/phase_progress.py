"""Phase progress helpers — update work_entries when a phase starts.

Shared module used by both ``subgraph_wrapper.py`` and ``compose.py``
so every phase node (subgraph-wrapped, legacy, or critic) marks its
phase as started before doing any work.

Failures are logged, never raised — progress tracking is best-effort
and must not crash a running workflow.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def mark_phase_started(parent_state: dict[str, Any], phase_name: str) -> None:
    """Update work_entries to reflect that a phase has started.

    Opens the work database, sets ``current_phase`` and ``status`` to
    ``"running"``, and publishes a ``"phase_started"`` WebSocket event.

    Safe to call from any node function — failures are logged, not raised.

    Args:
        parent_state: The parent state dict (must contain ``work_id``).
        phase_name: The phase identifier (e.g. ``"specify"``, ``"plan"``).
    """
    work_id = parent_state.get("work_id", "")
    if not work_id:
        return  # Nothing to update if no work_id

    try:
        from spine.config import SpineConfig
        from spine.work.dispatcher import get_work_db, update_work_phase_started

        config = SpineConfig.load()
        db = get_work_db(config)
        update_work_phase_started(db, work_id, phase_name)
    except Exception:
        logger.warning(
            f"Failed to mark phase {phase_name} as started for {work_id}",
            exc_info=True,
        )
