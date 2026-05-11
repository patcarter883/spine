"""Ralph Loop Worker — dequeues tasks from the queue and processes them.

Uses submit_work_from_config() for execution — same code path as CLI.
Runs in a background thread, managed by the UI.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from ..config.queue import TaskQueue, QueueTask, QueueStatus


class RalphLoopWorker:
    """Dequeues tasks from the queue and processes them autonomously.

    The worker loop:
    1. Dequeues the next pending task from the TaskQueue
    2. Processes it using submit_work_from_config() — same code as CLI
    3. Acknowledges success or records failure
    4. Repeats until paused or no more tasks

    The worker runs in a daemon background thread. It can be started
    and paused from the UI. Pausing stops dequeuing new items but
    allows the current task to finish.

    Usage:
        worker = RalphLoopWorker()
        worker.start()   # Begin processing
        worker.pause()   # Stop after current task
        worker.status    # Get current state
    """

    def __init__(
        self,
        queue: Optional[TaskQueue] = None,
        config_path: str = ".spine/config.yaml",
        poll_interval: float = 2.0,
    ):
        """Initialize the Ralph Loop Worker.

        Args:
            queue: Optional TaskQueue instance. If None, creates one.
            config_path: Path to the SPINE config file for provider loading.
            poll_interval: Seconds to wait between dequeue attempts when idle.
        """
        self._queue = queue or TaskQueue()
        self._config_path = config_path
        self._poll_interval = poll_interval
        self._running = False
        self._current_task: Optional[QueueTask] = None
        self._thread: Optional[threading.Thread] = None
        self._processed_count = 0
        self._failed_count = 0
        self._last_error: Optional[str] = None

    def start(self) -> None:
        """Start the worker loop in a background thread.

        If already running, this is a no-op.
        """
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        """Stop dequeuing new tasks after the current one finishes.

        The worker will complete the currently processing task (if any)
        and then stop. Call start() to resume.
        """
        self._running = False

    @property
    def is_running(self) -> bool:
        """Whether the worker is actively processing."""
        return self._running

    @property
    def status(self) -> dict:
        """Current worker status for UI display.

        Returns:
            Dict with running, current_task, processed_count,
            failed_count, and last_error.
        """
        return {
            "running": self._running,
            "current_task": (
                {
                    "id": self._current_task.id,
                    "type": self._current_task.task_type,
                    "payload": self._current_task.payload,
                    "started_at": self._current_task.started_at,
                }
                if self._current_task
                else None
            ),
            "processed_count": self._processed_count,
            "failed_count": self._failed_count,
            "last_error": self._last_error,
        }

    def _loop(self) -> None:
        """Main worker loop — dequeues and processes tasks."""
        while self._running:
            task = self._queue.dequeue(task_types=["spine_work"])
            if task is None:
                # No work available — wait and retry
                time.sleep(self._poll_interval)
                continue

            self._current_task = task
            try:
                self._process_task(task)
                self._queue.acknowledge(task.id, result={
                    "status": "success",
                    "thread_id": task.id,
                })
                self._processed_count += 1
                self._last_error = None
            except Exception as e:
                self._queue.fail(task.id, error=str(e))
                self._failed_count += 1
                self._last_error = str(e)
            finally:
                self._current_task = None

    def _process_task(self, task: QueueTask) -> None:
        """Process a single task using the unified submission path.

        This is the same code path as the CLI — submit_work_from_config()
        handles provider loading, checkpoint recording, and workflow
        execution.

        Args:
            task: The QueueTask to process.
        """
        from ..work.dispatcher import submit_work_from_config

        payload = task.payload
        requirement = payload.get("requirement", "")

        if not requirement:
            raise ValueError(f"Task {task.id} has no requirement in payload")

        # Use the same submission path as CLI — this ensures:
        # 1. Providers are loaded from config (same as CLI)
        # 2. Work item is recorded to work_items table
        # 3. State machine runs with proper checkpoint persistence
        # 4. Provider objects are passed through config, not state
        result = submit_work_from_config(
            requirement=requirement,
            config_path=self._config_path,
            thread_id=task.id,
            background=False,  # Synchronous within worker thread
        )

        # Check for submission errors
        if result.get("error"):
            raise RuntimeError(
                f"Work submission failed: {result['error']}"
            )


# ── Singleton accessor for UI ──────────────────────────────

_WORKER_INSTANCE: Optional[RalphLoopWorker] = None
_WORKER_LOCK = threading.Lock()


def get_worker() -> RalphLoopWorker:
    """Get or create the singleton RalphLoopWorker instance.

    The UI calls this to access the worker for start/pause/status.

    Returns:
        The shared RalphLoopWorker instance.
    """
    global _WORKER_INSTANCE
    with _WORKER_LOCK:
        if _WORKER_INSTANCE is None:
            _WORKER_INSTANCE = RalphLoopWorker()
        return _WORKER_INSTANCE


__all__ = [
    "RalphLoopWorker",
    "get_worker",
]
