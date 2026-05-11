"""Background task worker for executing SPINE workflows asynchronously.

Decouples work execution from the request thread to prevent blocking
and timeouts in the Streamlit UI. Runs as a standalone process or
thread, polling the queue for pending work items.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any

from ..config.queue import TaskQueue, QueueStatus

POLL_INTERVAL = 1.0


def get_provider_from_config(config: dict, provider_type: str = "llm"):
    """Instantiate a provider from config dict."""
    providers = config.get("providers", {}).get(provider_type, [])
    if not providers:
        return None
    instance = providers[0]
    ptype = instance.get("type", "").lower()
    pconfig = instance.get("config", instance)

    if ptype == "ollama":
        from ..providers.llm import OllamaProvider
        return OllamaProvider(
            model=pconfig.get("model", "qwen3:32b"),
            base_url=pconfig.get("base_url", "http://localhost:11434"),
        )
    elif ptype == "openai":
        from ..providers.llm import OpenAIProvider
        return OpenAIProvider(
            api_key=pconfig.get("api_key", ""),
            model=pconfig.get("model", "gpt-4"),
        )
    elif ptype == "local-openai":
        from ..providers.llm import LocalOpenAIProvider
        return LocalOpenAIProvider(
            api_key=pconfig.get("api_key", "not-required"),
            model=pconfig.get("model", "local-model"),
            base_url=pconfig.get("base_url", "http://localhost:8000/v1"),
        )
    elif ptype == "openrouter":
        from ..providers.llm import OpenRouterProvider
        provider = OpenRouterProvider(
            api_key=pconfig.get("api_key", ""),
            model=pconfig.get("model", "openai/gpt-4"),
            base_url=pconfig.get("base_url", OpenRouterProvider.DEFAULT_BASE_URL),
        )
        return provider
    return None


def execute_work_task(payload: dict) -> dict:
    """Execute a SPINE workflow task.

    Creates a SpineStateMachine, invokes the LangGraph workflow,
    and returns the result. Designed to run in a background worker.

    Args:
        payload: Dict containing:
            - requirement: str
            - thread_id: str
            - checkpoint_path: str
            - providers: dict (optional)
            - variables: dict (optional)
            - agent_provider: optional agent provider
            - project_context: dict (optional)

    Returns:
        Result dict with status, phase, completed_tasks, errors.
    """
    requirement = payload.get("requirement", "")
    thread_id = payload.get("thread_id", str(uuid.uuid4()))
    checkpoint_path = payload.get("checkpoint_path", ".spine/spine.db")
    variables = payload.get("variables", {})
    providers_dict = payload.get("providers", {})
    agent_provider = payload.get("agent_provider")
    project_context = payload.get("project_context", {})

    from ..core.state_machine import SpineStateMachine

    config = {}
    try:
        from ..ui.utils import load_config
        config = load_config()
    except Exception:
        pass

    llm_provider = get_provider_from_config(config)
    machine = SpineStateMachine(
        checkpoint_path=checkpoint_path,
        llm_provider=llm_provider,
    )

    try:
        # Build real providers dict with actual provider objects (not the
        # deserialized dicts from the payload).  Passing them through
        # config["configurable"] avoids LangGraph's checkpoint serialization
        # which would turn them back into plain dicts.
        real_providers = {
            "llm": llm_provider,
        }

        result = machine.app.invoke(
            {
                "phase": "INIT",
                "previous_phase": None,
                "requirement": requirement,
                "plan": None,
                "tasks": {},
                "completed_tasks": [],
                "failed_tasks": [],
                "swarm_state": {},
                "hive_cells": {},
                "swarm_events": [],
                "variables": {
                    "work_item_id": thread_id,
                    "thread_id": thread_id,
                    "checkpoint_path": checkpoint_path,
                    **variables,
                },
                "errors": [],
                "providers": providers_dict,
                "agent_provider": agent_provider,
                "critic_gate_result": None,
                "error_state": None,
                "error_history": [],
                "project_context": project_context,
            },
            {"configurable": {"thread_id": thread_id, "providers": real_providers}},
        )

        final_phase = result.get("phase", "UNKNOWN")
        success = final_phase == "COMPLETE"

        return {
            "status": "success" if success else "completed_with_issues",
            "phase": final_phase,
            "completed_tasks": len(result.get("completed_tasks", [])),
            "errors": result.get("errors", []),
            "thread_id": thread_id,
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "status": "failed",
            "error": f"{type(e).__name__}: {e}",
            "thread_id": thread_id,
        }


def worker_loop(queue: TaskQueue, poll_interval: float = POLL_INTERVAL) -> None:
    """Main worker loop — processes tasks from the queue.

    Args:
        queue: TaskQueue instance to poll.
        poll_interval: Seconds between polls.
    """
    print(f"[worker] Starting worker loop (backend: {queue.backend_type})")
    print(f"[worker] Polling every {poll_interval}s")

    while True:
        try:
            task = queue.dequeue(["work"])
            if task:
                print(f"[worker] Processing task {task.id} ({task.task_type})")
                result = execute_work_task(task.payload)

                if result.get("status") == "success":
                    queue.acknowledge(task.id, result)
                    print(f"[worker] Task {task.id} completed successfully")
                else:
                    queue.fail(task.id, result.get("error", "Unknown error"))
                    print(f"[worker] Task {task.id} failed: {result.get('error')}")
            else:
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            print("\n[worker] Shutting down...")
            break
        except Exception as e:
            print(f"[worker] Error in worker loop: {e}")
            traceback.print_exc()
            time.sleep(poll_interval)


def run_worker(
    redis_url: str | None = None,
    db_path: str = ".spine/queue.db",
    poll_interval: float = POLL_INTERVAL,
) -> None:
    """Run the task worker as a standalone process.

    Args:
        redis_url: Optional Redis URL for queue backend.
        db_path: SQLite fallback path.
        poll_interval: Seconds between polls.
    """
    queue = TaskQueue(redis_url=redis_url, db_path=db_path)
    worker_loop(queue, poll_interval)


__all__ = [
    "execute_work_task",
    "worker_loop",
    "run_worker",
]
