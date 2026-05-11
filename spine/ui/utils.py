"""Core utilities for config loading, checkpoint reading, and formatting."""

import os
import json
import sqlite3
import re
import threading
import uuid
from pathlib import Path
from typing import Any, Optional

import yaml
from datetime import datetime, timezone


# ── Phase display constants ──────────────────────────────────

PHASE_ICONS: dict[str, str] = {
    "INIT": "⚙️",
    "PLANNING": "📋",
    "EXECUTION": "🔨",
    "VERIFICATION": "✅",
    "COMPLETE": "🏁",
    "REWORK": "🔄",
    "ERROR": "❌",
    "BLOCKED": "🚧",
    "HUMAN_REVIEW": "👤",
}

PHASE_COLORS: dict[str, str] = {
    "INIT": "cyan",
    "PLANNING": "blue",
    "EXECUTION": "yellow",
    "VERIFICATION": "green",
    "COMPLETE": "green",
    "REWORK": "magenta",
    "ERROR": "red",
    "BLOCKED": "red",
    "HUMAN_REVIEW": "yellow",
}


# ── Config Loading ───────────────────────────────────────────

def _expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in string values.

    Supports ${VAR} and $VAR patterns. Missing vars expand to empty string.

    Args:
        value: Any value (string, dict, list, or primitive).

    Returns:
        Value with all ${VAR} and $VAR patterns replaced by env var values.
    """
    if isinstance(value, str):
        # Expand ${VAR} patterns
        def expand_braced(match: re.Match) -> str:
            return os.environ.get(match.group(1), "")
        result = re.sub(r"\$\{([^}]+)\}", expand_braced, value)
        # Expand $VAR patterns (word chars after $)
        result = re.sub(
            r"\$([A-Za-z_][A-Za-z0-9_]*)",
            lambda m: os.environ.get(m.group(1), ""),
            result,
        )
        return result
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def load_config(config_path: str = ".spine/config.yaml") -> dict:
    """Load configuration from YAML file, expanding env vars.

    Args:
        config_path: Path to the config file.

    Returns:
        Parsed configuration dict, or empty dict if file not found.
    """
    path = Path(config_path)
    if not path.exists():
        return {}

    with open(path) as f:
        config = yaml.safe_load(f) or {}

    return _expand_env_vars(config)


def save_config(config: dict, config_path: str = ".spine/config.yaml") -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration dict to save.
        config_path: Path to the config file.
    """
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# ── Checkpoint / State Reading ───────────────────────────────

def get_checkpoint_path(checkpoint_path: Optional[str] = None) -> Path:
    """Resolve the checkpoint DB path from config or a provided path.

    Args:
        checkpoint_path: Explicit path, or None to read from config.

    Returns:
        Resolved Path to the checkpoint SQLite database.
    """
    if checkpoint_path:
        return Path(checkpoint_path)
    config = load_config()
    cp_path = config.get("spine", {}).get("checkpoint_path", ".spine/spine.db")
    return Path(cp_path)


def _get_langgraph_thread_ids(checkpoint_path: Path) -> list[str]:
    """Extract all known thread IDs from LangGraph's checkpoint store.

    LangGraph's MemorySaver (and SqliteSaver) store state keyed by thread_id.
    This function queries the checkpoint tables to find all thread IDs.

    Args:
        checkpoint_path: Path to the checkpoint SQLite database.

    Returns:
        List of thread ID strings (empty list if none found).
    """
    if not checkpoint_path.exists():
        return []

    thread_ids: list[str] = []
    try:
        conn = sqlite3.connect(str(checkpoint_path))
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            try:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [col[1] for col in cursor.fetchall()]
                if "thread_id" in columns:
                    cursor.execute(f"SELECT DISTINCT thread_id FROM {table}")
                    for row in cursor.fetchall():
                        tid = row[0]
                        if tid and tid not in thread_ids:
                            thread_ids.append(tid)
            except Exception:
                continue

        conn.close()
    except Exception:
        pass

    return thread_ids


def get_latest_checkpoint(
    thread_id: str,
    checkpoint_path: Optional[str] = None,
) -> Optional[dict]:
    """Read the latest checkpoint for a thread from the LangGraph checkpoint store.

    Args:
        thread_id: Thread ID to read (required).
        checkpoint_path: Explicit checkpoint path, or None to read from config.

    Returns:
        State dict from the latest checkpoint, or None if not found.
    """
    cp_path = get_checkpoint_path(checkpoint_path)
    if not cp_path.exists():
        return None

    try:
        conn = sqlite3.connect(str(cp_path))
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        # Try common LangGraph checkpoint table names
        for table in ["checkpoint_blobs", "checkpoints"]:
            if table not in tables:
                continue

            cursor.execute(
                f"SELECT parent_id, ts, blob "
                f"FROM {table} WHERE thread_id=? ORDER BY ts DESC LIMIT 1",
                (thread_id,),
            )
            row = cursor.fetchone()
            if row:
                blob_data = row[2]
                if blob_data:
                    try:
                        # LangGraph stores JSON blobs
                        data = json.loads(blob_data) if isinstance(blob_data, str) else blob_data
                        return data
                    except (json.JSONDecodeError, TypeError):
                        # Try orjson-style format
                        try:
                            import orjson
                            return orjson.loads(blob_data)
                        except Exception:
                            return {"raw": str(blob_data)}

        # Fallback: search all tables
        for table in tables:
            try:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [col[1] for col in cursor.fetchall()]
                if "thread_id" in columns:
                    cursor.execute(
                        f"SELECT blob FROM {table} WHERE thread_id=? ORDER BY rowid DESC LIMIT 1",
                        (thread_id,),
                    )
                    row = cursor.fetchone()
                    if row:
                        blob_data = row[0]
                        if blob_data:
                            try:
                                return json.loads(blob_data) if isinstance(blob_data, str) else blob_data
                            except Exception:
                                try:
                                    import orjson
                                    return orjson.loads(blob_data)
                                except Exception:
                                    return {"table": table, "raw": str(blob_data)}
            except Exception:
                continue

        conn.close()
        return None

    except Exception:
        return None


def get_active_work_items(checkpoint_path: Optional[str] = None) -> list[dict]:
    """Return all work items with their latest status.

    Reads from the work_items table (the single source of truth)
    rather than trying to parse LangGraph's internal checkpoint
    format, which uses msgpack serialization that is not reliably
    decodable from raw SQL.

    Args:
        checkpoint_path: Explicit checkpoint path, or None to read from config.

    Returns:
        List of work item dicts with thread_id, requirement, phase, progress, etc.
    """
    cp_path = get_checkpoint_path(checkpoint_path)
    if not cp_path.exists():
        return []

    items: list[dict] = []
    try:
        conn = sqlite3.connect(str(cp_path))
        cursor = conn.cursor()

        # Check if work_items table exists (it may not if only CLI was used)
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='work_items'"
        )
        has_work_items = cursor.fetchone() is not None

        if has_work_items:
            cursor.execute(
                "SELECT thread_id, requirement, status, phase, "
                "completed_tasks, total_tasks, error_message "
                "FROM work_items ORDER BY updated_at DESC"
            )
            rows = cursor.fetchall()
            if rows:
                # Get real-time phase from checkpoints for running items
                from spine.cli.commands.status import get_threads
                conn.close()
                threads = get_threads(str(cp_path))
                thread_phases = {t["thread_id"]: t.get("phase") for t in threads}
                
                for row in rows:
                    tid, req, status, phase, comp, total, err = row
                    # Use real-time phase from checkpoint for running items
                    if status == "running" and tid in thread_phases:
                        phase = thread_phases[tid] or phase
                    items.append({
                        "thread_id": tid,
                        "requirement": req or "Untitled",
                        "phase": phase or status.upper(),
                        "status": status,
                        "progress": (comp or 0) / max(1, total or 1),
                        "completed_tasks": comp or 0,
                        "failed_tasks": 0,
                        "total_tasks": total or 1,
                        "started_at": "",
                        "errors": [err] if err else [],
                        "error_state": None,
                        "critic_gate_result": None,
                    })
                return items
            # Table exists but is empty — fall through to checkpoint fallback

        # Fallback: read from LangGraph checkpoint tables directly
        # Use the same logic as the CLI status command
        from spine.cli.commands.status import get_threads
        conn.close()  # Close our connection before get_threads opens its own
        threads = get_threads(str(cp_path))
        for t in threads:
            phase = t.get("phase", "UNKNOWN")
            items.append({
                "thread_id": t["thread_id"],
                "requirement": t.get("requirement") or "Unknown (no work_items table)",
                "phase": phase,
                "status": phase.lower(),
                "progress": t.get("completed_tasks", 0) / max(1, t.get("completed_tasks", 0) or 1),
                "completed_tasks": t.get("completed_tasks", 0),
                "failed_tasks": 0,
                "total_tasks": max(1, t.get("completed_tasks", 0) or 1),
                "started_at": "",
                "errors": [],
                "error_state": None,
                "critic_gate_result": None,
            })
    except Exception:
        pass

    return items


def get_work_item_detail(
    thread_id: str,
    checkpoint_path: Optional[str] = None,
) -> Optional[dict]:
    """Load full state from the latest checkpoint for a work item.

    Uses the SpineStateMachine's checkpointer (LangGraph SqliteSaver)
    to deserialize the checkpoint properly, rather than trying to
    parse the raw SQLite blob (which is msgpack-encoded).

    Args:
        thread_id: Thread ID to read (required).
        checkpoint_path: Explicit checkpoint path, or None to read from config.

    Returns:
        Full state dict or None if not found.
    """
    cp_path = get_checkpoint_path(checkpoint_path)
    if not cp_path.exists():
        return None

    # Try reading from the work_items table first (single source of truth)
    try:
        conn = sqlite3.connect(str(cp_path))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='work_items'"
        )
        has_work_items = cursor.fetchone() is not None
        if has_work_items:
            cursor.execute(
                "SELECT requirement, status, phase, completed_tasks, "
                "total_tasks, error_message FROM work_items WHERE thread_id = ?",
                (thread_id,),
            )
            row = cursor.fetchone()
            conn.close()
            if row:
                req, status, phase, comp, total, err = row
                return {
                    "thread_id": thread_id,
                    "phase": phase or "INIT",
                    "previous_phase": None,
                    "requirement": req or "",
                    "plan": None,
                    "tasks": {},
                    "completed_tasks": list(range(comp or 0)),
                    "failed_tasks": [],
                    "swarm_state": {},
                    "hive_cells": {},
                    "swarm_events": [],
                    "critic_gate_result": None,
                    "error_state": None,
                    "error_history": [],
                    "variables": {},
                    "errors": [err] if err else [],
                    "started_at": "",
                }
    except Exception:
        pass

    # Fallback: use SpineStateMachine to read the checkpoint via LangGraph
    try:
        from ..core.state_machine import SpineStateMachine
        machine = SpineStateMachine(checkpoint_path=str(cp_path))
        state = machine.resume(thread_id)
        if state is not None:
            return {
                "thread_id": thread_id,
                "phase": state.get("phase", "INIT"),
                "previous_phase": state.get("previous_phase"),
                "requirement": state.get("requirement", ""),
                "plan": state.get("plan"),
                "tasks": state.get("tasks", {}),
                "completed_tasks": state.get("completed_tasks", []),
                "failed_tasks": state.get("failed_tasks", []),
                "swarm_state": state.get("swarm_state", {}),
                "hive_cells": state.get("hive_cells", {}),
                "swarm_events": state.get("swarm_events", []),
                "critic_gate_result": state.get("critic_gate_result"),
                "error_state": state.get("error_state"),
                "error_history": state.get("error_history", []),
                "variables": state.get("variables", {}),
                "errors": state.get("errors", []),
                "started_at": state.get("variables", {}).get("timestamp", ""),
            }
    except Exception:
        pass

    return None


def get_checkpoints(
    thread_id: str,
    checkpoint_path: Optional[str] = None,
) -> list[dict]:
    """Return all checkpoints for a work item, newest first.

    Args:
        thread_id: Thread ID to query.
        checkpoint_path: Explicit checkpoint path, or None to read from config.

    Returns:
        List of checkpoint records.
    """
    cp_path = get_checkpoint_path(checkpoint_path)
    if not cp_path.exists():
        return []

    checkpoints: list[dict] = []
    try:
        conn = sqlite3.connect(str(cp_path))
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            try:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [col[1] for col in cursor.fetchall()]
                if "thread_id" not in columns:
                    continue

                cursor.execute(
                    f"SELECT rowid, ts, blob FROM {table} WHERE thread_id=? ORDER BY rowid DESC",
                    (thread_id,),
                )
                for row in cursor.fetchall():
                    blob_data = row[2]
                    data = {}
                    if blob_data:
                        try:
                            data = json.loads(blob_data) if isinstance(blob_data, str) else blob_data
                        except Exception:
                            try:
                                import orjson
                                data = orjson.loads(blob_data)
                            except Exception:
                                data = {"raw": str(blob_data)}

                    checkpoints.append({
                        "row_id": row[0],
                        "table": table,
                        "timestamp": row[1],
                        "data": data,
                    })
            except Exception:
                continue

        conn.close()
    except Exception:
        pass

    return checkpoints


# ── Work Item Actions ────────────────────────────────────────

# ── Work Items Table (single source of truth) ─────────────────


def _init_work_items_table(db_path: str) -> None:
    """Ensure the work_items tracking table exists in spine.db.

    This table is the single source of truth that both the status
    command and the UI read from.  It records every work item that
    has been created so that even queued/pending items are visible.
    """
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


def _upsert_work_item(
    db_path: str,
    thread_id: str,
    requirement: str,
    status: str = "queued",
    phase: str = "INIT",
    completed_tasks: int = 0,
    total_tasks: int = 0,
    error_message: Optional[str] = None,
) -> None:
    """Insert or update a work item record."""
    now = datetime.now(timezone.utc).isoformat()
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


def _dispatch_work_background(
    requirement: str,
    thread_id: str,
    checkpoint_path: str,
    providers_dict: dict,
    project_context: dict,
    idempotency_key: str,
) -> None:
    """Run the state machine workflow in a daemon background thread.

    Writes checkpoints to spine.db as it progresses so that
    'spine status' reflects real-time state.

    providers_dict must contain **real provider instances** (e.g.
    LLMProvider), NOT raw config dicts.  Raw dicts cause
    AttributeError: 'dict' object has no attribute 'enabled' when
    the state machine tries to use them after deserialization.
    """
    from ..jobs.task_worker import execute_work_task

    def _run():
        payload = {
            "requirement": requirement,
            "thread_id": thread_id,
            "checkpoint_path": checkpoint_path,
            # Pass empty dict for providers in the payload — the real
            # provider objects are delivered through the LangGraph
            # config below, bypassing checkpoint serialization.
            "providers": {},
            "variables": {
                "thread_id": thread_id,
                "work_item_id": thread_id,
                "checkpoint_path": checkpoint_path,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "idempotency_key": idempotency_key,
            },
            "project_context": project_context,
        }
        _upsert_work_item(checkpoint_path, thread_id, requirement,
                          status="running", phase="INIT")
        result = execute_work_task(payload)
        if result.get("status") == "success":
            _upsert_work_item(checkpoint_path, thread_id, requirement,
                              status="completed", phase="COMPLETE",
                              completed_tasks=result.get("completed_tasks", 0),
                              total_tasks=result.get("total_tasks", 0))
        else:
            _upsert_work_item(checkpoint_path, thread_id, requirement,
                              status="failed", phase="ERROR",
                              error_message=result.get("error", "Unknown error"))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


def start_work(
    requirement: str,
    method: str = "Quick Work",
    project_type: str = "Greenfield",
    llm_provider: str = "qwen3:32b (Ollama)",
    parallel_agents: int = 3,
    checkpoint_path: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> Optional[dict]:
    """Start a new work item and dispatch it for background processing.

    Uses the unified submit_work_from_config function which records
    to the work_items table and runs the workflow in a background thread.

    Args:
        requirement: The requirement text for the work item.
        method: Automation level ("Quick Work", "Full Spec Work", etc.).
        project_type: Environment type ("Greenfield" or "Brownfield").
        llm_provider: LLM provider name.
        parallel_agents: Maximum parallel agents within a phase.
        checkpoint_path: Explicit checkpoint path.
        idempotency_key: UUIDv4 for duplicate detection. Generated if not provided.

    Returns:
        Dict with 'thread_id', 'task_id', and 'idempotency_key' on success,
        or dict with 'error' message on failure.
    """
    from ..work import submit_work_from_config
    
    key = idempotency_key or str(uuid.uuid4())

    # Check if this key was already processed (dedup)
    _idempotency_path = Path(".spine/idempotency")
    _idempotency_path.mkdir(parents=True, exist_ok=True)
    dedup_file = _idempotency_path / f"{key}.json"
    if dedup_file.exists():
        try:
            existing = json.loads(dedup_file.read_text())
            return existing
        except (json.JSONDecodeError, OSError):
            pass

    try:
        # Use unified submission - this handles everything:
        # - Loading providers from config
        # - Recording to work_items table
        # - Running in background thread
        result = submit_work_from_config(
            requirement=requirement,
            checkpoint_path=checkpoint_path,
            background=True,  # Run in background thread
        )
        
        if result.get("status") == "queued":
            thread_id = result["thread_id"]
            payload = {
                "thread_id": thread_id,
                "task_id": thread_id,
                "idempotency_key": key,
            }
            dedup_file.write_text(json.dumps(payload))
            return payload
        else:
            return {"error": result.get("error", "Unknown error"), "idempotency_key": key}

    except Exception as e:
        error_msg = str(e)
        print(f"[spine.ui.utils] Failed to start work: {error_msg}")
        return {"error": error_msg, "idempotency_key": key}


def _rollback_work(
    thread_id: str,
    checkpoint_path: Optional[str] = None,
) -> None:
    """Roll back a failed work submission by removing its checkpoint data.

    Args:
        thread_id: The thread ID to clean up.
        checkpoint_path: Explicit checkpoint path.
    """
    cp_path = get_checkpoint_path(checkpoint_path)
    if not cp_path.exists():
        return
    try:
        conn = sqlite3.connect(str(cp_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        for table in tables:
            try:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [col[1] for col in cursor.fetchall()]
                if "thread_id" in columns:
                    cursor.execute(f"DELETE FROM {table} WHERE thread_id=?", (thread_id,))
            except Exception:
                continue
        conn.commit()
        conn.close()
    except Exception:
        pass


def approve_gate(thread_id: str) -> bool:
    """Approve the critic gate for a work item.

    Writes an approval flag that the state machine reads to allow
    transition from PLANNING to EXECUTION.

    Args:
        thread_id: Thread ID for the work item.

    Returns:
        True on success.
    """
    gate_file = Path(f".spine/state/gate_result_{thread_id}.json")
    gate_file.parent.mkdir(parents=True, exist_ok=True)
    gate_file.write_text(json.dumps({
        "approved": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }))
    return True


def reject_gate(thread_id: str, feedback: str) -> bool:
    """Reject the critic gate with feedback for rework.

    Args:
        thread_id: Thread ID for the work item.
        feedback: Feedback text for the planner to address.

    Returns:
        True on success.
    """
    gate_file = Path(f".spine/state/gate_result_{thread_id}.json")
    gate_file.parent.mkdir(parents=True, exist_ok=True)
    gate_file.write_text(json.dumps({
        "approved": False,
        "feedback": feedback,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }))
    return True


def resume_work(thread_id: str, checkpoint_path: Optional[str] = None) -> bool:
    """Resume a paused work item from its last checkpoint.

    Args:
        thread_id: Thread ID to resume.
        checkpoint_path: Explicit checkpoint path.

    Returns:
        True on success, False if no state found.
    """
    from ..core.state_machine import SpineStateMachine

    machine = SpineStateMachine(checkpoint_path=checkpoint_path or ".spine/spine.db")
    try:
        state = machine.resume(thread_id)
        return state is not None
    except Exception:
        return False


def delete_work(thread_id: str) -> bool:
    """Delete a work item's checkpoint data.

    Removes all checkpoint entries for the given thread ID.

    Args:
        thread_id: Thread ID to delete.

    Returns:
        True on success, False on failure.
    """
    cp_path = get_checkpoint_path()
    if not cp_path.exists():
        return True

    try:
        conn = sqlite3.connect(str(cp_path))
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        deleted = 0
        for table in tables:
            try:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [col[1] for col in cursor.fetchall()]
                if "thread_id" in columns:
                    cursor.execute(f"DELETE FROM {table} WHERE thread_id=?", (thread_id,))
                    deleted += cursor.rowcount
            except Exception:
                continue

        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


# ── Provider Config Helpers ──────────────────────────────────

def _load_providers_from_config(config: dict) -> dict[str, list]:
    """Load provider configurations from a parsed config dict.

    Returns dict mapping category to list of (name, config_dict) tuples.
    Used to build providers dict for the state machine.
    """
    providers_by_category: dict[str, list] = {}

    for category, provider_list in config.get("providers", {}).items():
        if not provider_list:
            continue
        for instance in provider_list:
            name = instance.get("name", "unnamed")
            enabled = instance.get("enabled", True)
            if not enabled:
                continue
            providers_by_category.setdefault(category, []).append((name, instance))

    return providers_by_category


def get_llm_providers() -> list[dict]:
    """Get configured LLM providers from the config.

    Returns:
        List of LLM provider config dicts.
    """
    config = load_config()
    return config.get("providers", {}).get("llm", [])


def set_llm_providers(providers: list[dict]) -> None:
    """Save LLM providers list to the config file.

    Args:
        providers: List of provider config dicts.
    """
    config = load_config()
    config.setdefault("providers", {})["llm"] = providers
    save_config(config)


# ── Formatting Helpers ───────────────────────────────────────

def format_phase_icon(phase: str) -> str:
    """Return emoji for a phase name."""
    return PHASE_ICONS.get(phase, "•")


def format_phase_color(phase: str) -> str:
    """Return rich color name for a phase."""
    return PHASE_COLORS.get(phase, "white")


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration.

    Args:
        seconds: Number of seconds.

    Returns:
        Human-readable duration string.
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


def format_bytes(size: int) -> str:
    """Format bytes into human-readable size.

    Args:
        size: Number of bytes.

    Returns:
        Human-readable size string.
    """
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# ── Navigation ───────────────────────────────────────────────

def navigate_to_work(thread_id: str) -> None:
    """Set session state for navigation to a work item detail."""
    st = __import__("streamlit").st
    st.session_state.selected_work_id = thread_id
    st.session_state.page = "Work Detail"


def go_back() -> None:
    """Navigate back to dashboard."""
    st = __import__("streamlit").st
    st.session_state.page = "Dashboard"
    st.session_state.selected_work_id = None
