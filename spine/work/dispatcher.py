"""Unified work submission and execution.

This module provides a single entry point for submitting work items,
used by both the CLI and UI. It handles:
- Recording work items to the database
- Running the state machine (sync or async)
- Updating status on completion
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

# ── Database Operations ────────────────────────────────────────


def _init_work_items_table(db_path: str) -> None:
    """Ensure the work_items tracking table exists."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS work_items (
                thread_id TEXT PRIMARY KEY,
                requirement TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                phase TEXT NOT NULL DEFAULT 'INIT',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_tasks INTEGER DEFAULT 0,
                total_tasks INTEGER DEFAULT 0,
                error_message TEXT
            )
        """)
        conn.commit()


def record_work_item(
    db_path: str,
    thread_id: str,
    requirement: str,
    status: str = "queued",
    phase: str = "INIT",
    completed_tasks: int = 0,
    total_tasks: int = 0,
    error_message: Optional[str] = None,
) -> None:
    """Insert or update a work item record in the database."""
    now = datetime.now(timezone.utc).isoformat()
    _init_work_items_table(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """INSERT INTO work_items
               (thread_id, requirement, status, phase, created_at, updated_at,
                completed_tasks, total_tasks, error_message)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(thread_id) DO UPDATE SET
               status=excluded.status, phase=excluded.phase,
               updated_at=excluded.updated_at,
               completed_tasks=excluded.completed_tasks,
               total_tasks=excluded.total_tasks,
               error_message=excluded.error_message""",
            (thread_id, requirement, status, phase, now, now,
             completed_tasks, total_tasks, error_message),
        )
        conn.commit()


# ── State Machine Execution ────────────────────────────────────


def run_workflow(
    requirement: str,
    thread_id: str,
    checkpoint_path: str,
    providers: dict[str, Any],
    project_context: Optional[dict] = None,
    variables: Optional[dict] = None,
    agent_provider: Optional[Any] = None,
    debug_prompts: bool = False,
    stream_callback: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Run the SPINE state machine workflow.

    This is the core execution function used by both CLI and UI.
    Providers must be real provider instances (not config dicts),
    passed through the LangGraph config to avoid serialization.

    Args:
        requirement: The work requirement text.
        thread_id: Unique identifier for this work item.
        checkpoint_path: Path to the checkpoint database.
        providers: Dict of provider instances (e.g., {"llm": LLMProvider}).
        project_context: Optional project context dict.
        variables: Optional additional variables.
        agent_provider: Optional agent provider instance.
        debug_prompts: Whether to print prompts to console.
        stream_callback: Optional callback for streaming updates.

    Returns:
        Result dict with status, phase, completed_tasks, errors.
    """
    from ..core.state_machine import SpineStateMachine

    llm_provider = providers.get("llm")
    # Agent provider can be supplied explicitly OR included in providers dict
    # under the "agent" key. The latter is what submit_work() does so that all
    # provider categories travel together through config (avoiding LangGraph
    # checkpoint serialization that turns objects into plain dicts).
    if agent_provider is None:
        agent_provider = providers.get("agent")
    elif "agent" not in providers:
        # Mirror it into providers so config["configurable"]["providers"]
        # carries it through to phase nodes that look it up there.
        providers = {**providers, "agent": agent_provider}

    machine = SpineStateMachine(
        checkpoint_path=checkpoint_path,
        llm_provider=llm_provider,
    )
    machine._debug_prompts = debug_prompts

    project_context = project_context or {}
    variables = variables or {}

    initial_state = {
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
            "debug_prompts": debug_prompts,
            **variables,
        },
        "errors": [],
        "providers": {},  # Empty in state - providers go through config
        # NOTE: do NOT put agent_provider in state — LangGraph's checkpointer
        # serializes state between nodes and converts provider instances into
        # plain dicts, breaking downstream .execute() calls. Pass it via
        # config["configurable"]["providers"]["agent"] instead.
        "agent_provider": None,
        "critic_gate_result": None,
        "error_state": None,
        "error_history": [],
        "project_context": project_context,
        "pending_messages": [],
        "model_call_count": 0,
    }

    config = {
        "configurable": {
            "thread_id": thread_id,
            "providers": providers,
        }
    }

    try:
        if stream_callback:
            # Streaming mode (for CLI)
            final_state = None
            for chunk in machine.app.stream(initial_state, config):
                for node_name, state in chunk.items():
                    stream_callback(state)
                    final_state = state
            result_state = final_state or {}
        else:
            # Invoke mode (for UI background)
            result_state = machine.app.invoke(initial_state, config)

        final_phase = result_state.get("phase", "UNKNOWN")
        success = final_phase == "COMPLETE"

        # Get total tasks from plan if available. The state's
        # ``completed_tasks`` list grows across PLANNING + EXECUTION +
        # VERIFICATION subphases, so it can exceed the plan's task count.
        # We want a stable "X of Y" denominator that won't display "10/1"
        # in the UI. Use the larger of (plan tasks, completed) so the bar
        # reads sensibly when subphase tasks dominate.
        plan = result_state.get("plan") or {}
        plan_tasks = plan.get("tasks", []) if isinstance(plan, dict) else []
        completed_count = len(result_state.get("completed_tasks", []))
        if plan_tasks:
            total_tasks = max(len(plan_tasks), completed_count)
        else:
            total_tasks = max(completed_count, 1)

        return {
            "status": "success" if success else "completed_with_issues",
            "phase": final_phase,
            "completed_tasks": completed_count,
            "total_tasks": total_tasks,
            "errors": result_state.get("errors", []),
            "thread_id": thread_id,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "failed",
            "error": f"{type(e).__name__}: {e}",
            "thread_id": thread_id,
        }


# ── Submission API ─────────────────────────────────────────────


def submit_work(
    requirement: str,
    thread_id: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    providers: Optional[dict[str, Any]] = None,
    project_context: Optional[dict] = None,
    agent_provider: Optional[Any] = None,
    debug_prompts: bool = False,
    background: bool = False,
    stream_callback: Optional[Callable[[dict], None]] = None,
    on_complete: Optional[Callable[[dict], None]] = None,
    warnings: Optional[list[str]] = None,
) -> dict:
    """Submit a new work item for execution.

    This is the unified entry point for both CLI and UI.
    Records the work item to the database and runs the workflow.

    Args:
        requirement: The work requirement text.
        thread_id: Optional thread ID (auto-generated if not provided).
        checkpoint_path: Optional checkpoint path (default: .spine/spine.db).
        providers: Optional dict of provider instances.
        project_context: Optional project context dict.
        agent_provider: Optional agent provider instance.
        debug_prompts: Whether to print prompts to console.
        background: If True, run in a background thread (for UI).
        stream_callback: Optional callback for streaming updates (CLI).
        on_complete: Optional callback when background task completes.

    Returns:
        Dict with thread_id and status.
    """
    thread_id = thread_id or str(uuid.uuid4())
    checkpoint_path = checkpoint_path or ".spine/spine.db"
    providers = providers or {}
    project_context = project_context or {}
    warnings = warnings or []

    # Record the work item as queued. If we have warnings (e.g. agent
    # provider configured but binary missing), surface them in the
    # error_message field so the UI shows the user *why* execution may
    # behave oddly — this is the difference between "silent stub" and
    # "user knows what's going on".
    queued_warning = "; ".join(warnings) if warnings else None
    record_work_item(
        checkpoint_path, thread_id, requirement,
        status="queued", phase="INIT",
        error_message=queued_warning,
    )

    def _execute_and_record():
        """Run workflow and update status."""
        # Preserve any queued warnings (e.g. agent provider unavailable)
        # in the running record so the UI can keep showing them.
        record_work_item(
            checkpoint_path, thread_id, requirement,
            status="running", phase="INIT",
            error_message=queued_warning,
        )

        result = run_workflow(
            requirement=requirement,
            thread_id=thread_id,
            checkpoint_path=checkpoint_path,
            providers=providers,
            project_context=project_context,
            agent_provider=agent_provider,
            debug_prompts=debug_prompts,
            stream_callback=stream_callback,
        )

        # Update final status. Always carry the queued_warning forward so
        # users keep seeing "agent unavailable" even after success — that's
        # the whole point: success without the configured agent means stub-y
        # success, which is the user-confusing behavior we're documenting.
        if result.get("status") == "success":
            final_msg = queued_warning
            record_work_item(
                checkpoint_path, thread_id, requirement,
                status="completed", phase="COMPLETE",
                completed_tasks=result.get("completed_tasks", 0),
                total_tasks=result.get("total_tasks", 0),
                error_message=final_msg,
            )
        else:
            err_text = result.get("error", "Unknown error")
            if queued_warning:
                err_text = f"{queued_warning} | {err_text}"
            record_work_item(
                checkpoint_path, thread_id, requirement,
                status="failed", phase=result.get("phase", "ERROR"),
                error_message=err_text,
            )

        if on_complete:
            on_complete(result)

        return result

    if background:
        # Run in background thread
        thread = threading.Thread(target=_execute_and_record, daemon=True)
        thread.start()
        return {
            "thread_id": thread_id,
            "status": "queued",
            "message": "Work submitted to background queue",
            "warnings": list(warnings or []),
        }
    else:
        # Run synchronously (CLI mode)
        result = _execute_and_record()
        return {
            "thread_id": thread_id,
            "status": result.get("status", "unknown"),
            "phase": result.get("phase"),
            "completed_tasks": result.get("completed_tasks", 0),
            "errors": result.get("errors", []),
        }


# ── Convenience Functions ──────────────────────────────────────


def submit_work_from_config(
    requirement: str,
    config_path: str = ".spine/config.yaml",
    thread_id: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    debug_prompts: bool = False,
    background: bool = False,
    stream_callback: Optional[Callable[[dict], None]] = None,
    on_complete: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Submit work using providers loaded from config file.

    Convenience function that loads providers from config and
    submits the work item.

    Args:
        requirement: The work requirement text.
        config_path: Path to the config file.
        thread_id: Optional thread ID.
        checkpoint_path: Optional checkpoint path.
        debug_prompts: Whether to print prompts.
        background: If True, run in background thread.
        stream_callback: Optional callback for streaming updates.
        on_complete: Optional callback when background task completes.

    Returns:
        Dict with thread_id and status.
    """
    from ..cli import load_providers, get_primary_provider, load_config

    # Load providers
    providers_by_type = load_providers(config_path)
    providers = {}
    agent_warnings: list[str] = []
    for category in providers_by_type:
        primary = get_primary_provider(providers_by_type, category)
        if primary is not None:
            # For agent providers, verify the underlying CLI binary is actually
            # installed. A configured-but-unavailable agent would otherwise
            # silently fall back to LLM-only execution that produces text but
            # no real code changes — making work appear to "complete instantly".
            if category == "agent" and not primary.enabled:
                provider_name = getattr(primary, "name", type(primary).__name__)
                agent_warnings.append(
                    f"Agent provider '{provider_name}' is configured but not "
                    f"available (binary missing or disabled). Install it or "
                    f"remove the agent provider from config to silence this warning."
                )
            providers[category] = primary

    # Load project context from config
    config = load_config(config_path)
    import os
    project_root = os.getcwd()
    project_name = Path(project_root).name
    project_config = config.get("project", {})
    project_context = {
        "name": project_config.get("name", project_name),
        "root": project_config.get("root", project_root),
        "description": project_config.get("description", ""),
        "tech_stack": project_config.get("tech_stack", []),
    }

    # Get agent provider if configured
    agent_provider = providers.get("agent")

    # Determine checkpoint path from config if not specified
    if not checkpoint_path:
        checkpoint_path = config.get("spine", {}).get("checkpoint_path", ".spine/spine.db")

    return submit_work(
        requirement=requirement,
        thread_id=thread_id,
        checkpoint_path=checkpoint_path,
        providers=providers,
        project_context=project_context,
        agent_provider=agent_provider,
        debug_prompts=debug_prompts,
        background=background,
        stream_callback=stream_callback,
        on_complete=on_complete,
        warnings=agent_warnings,
    )


__all__ = [
    "record_work_item",
    "run_workflow",
    "submit_work",
    "submit_work_from_config",
]
