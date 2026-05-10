"""Core utilities for config loading, checkpoint reading, and formatting."""

import os
import json
import sqlite3
import re
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
        List of thread ID strings.
    """
    if not checkpoint_path.exists():
        return ["default"]

    thread_ids: list[str] = []
    try:
        conn = sqlite3.connect(str(checkpoint_path))
        cursor = conn.cursor()

        # LangGraph stores data in tables like:
        # - checkpoint_blobs: stores state blobs keyed by thread_id, ts
        # - checkpoint_writes: stores intermediate writes
        # - checkpoints: stores checkpoint metadata
        # - metadata: stores table metadata
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            try:
                # Try to find thread_id column
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

    return thread_ids if thread_ids else ["default"]


def get_latest_checkpoint(
    thread_id: str = "default",
    checkpoint_path: Optional[str] = None,
) -> Optional[dict]:
    """Read the latest checkpoint for a thread from the LangGraph checkpoint store.

    Args:
        thread_id: Thread ID to read.
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
    """Read checkpoint store and return all work items with their latest status.

    Args:
        checkpoint_path: Explicit checkpoint path, or None to read from config.

    Returns:
        List of work item dicts with thread_id, requirement, phase, progress, etc.
    """
    cp_path = get_checkpoint_path(checkpoint_path)
    thread_ids = _get_langgraph_thread_ids(cp_path)
    items = []

    for tid in thread_ids:
        detail = get_work_item_detail(tid, checkpoint_path)
        if detail:
            phase = detail.get("phase", "INIT")
            completed = len(detail.get("completed_tasks", []))
            failed = len(detail.get("failed_tasks", []))
            total = max(1, completed + failed)

            items.append({
                "thread_id": tid,
                "requirement": detail.get("requirement", "Untitled"),
                "phase": phase,
                "status": phase,
                "progress": completed / total,
                "completed_tasks": completed,
                "failed_tasks": failed,
                "total_tasks": total,
                "started_at": detail.get("variables", {}).get("timestamp", ""),
                "errors": detail.get("errors", []),
                "error_state": detail.get("error_state"),
                "critic_gate_result": detail.get("critic_gate_result"),
            })

    return items


def get_work_item_detail(
    thread_id: str = "default",
    checkpoint_path: Optional[str] = None,
) -> Optional[dict]:
    """Load full state from the latest checkpoint for a work item.

    Args:
        thread_id: Thread ID to read.
        checkpoint_path: Explicit checkpoint path, or None to read from config.

    Returns:
        Full state dict or None if not found.
    """
    checkpoint = get_latest_checkpoint(thread_id, checkpoint_path)
    if not checkpoint:
        return None

    return {
        "thread_id": thread_id,
        "phase": checkpoint.get("phase", "INIT"),
        "previous_phase": checkpoint.get("previous_phase"),
        "requirement": checkpoint.get("requirement", ""),
        "plan": checkpoint.get("plan"),
        "tasks": checkpoint.get("tasks", {}),
        "completed_tasks": checkpoint.get("completed_tasks", []),
        "failed_tasks": checkpoint.get("failed_tasks", []),
        "swarm_state": checkpoint.get("swarm_state", {}),
        "hive_cells": checkpoint.get("hive_cells", {}),
        "swarm_events": checkpoint.get("swarm_events", []),
        "critic_gate_result": checkpoint.get("critic_gate_result"),
        "error_state": checkpoint.get("error_state"),
        "error_history": checkpoint.get("error_history", []),
        "variables": checkpoint.get("variables", {}),
        "errors": checkpoint.get("errors", []),
        "started_at": checkpoint.get("variables", {}).get("timestamp", ""),
    }


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

def start_work(
    requirement: str,
    method: str = "Quick Work",
    project_type: str = "Greenfield",
    llm_provider: str = "qwen3:32b (Ollama)",
    parallel_agents: int = 3,
    checkpoint_path: Optional[str] = None,
    idempotency_key: Optional[str] = None,
) -> Optional[dict]:
    """Start a new work item via the task queue.

    Enqueues the work for background processing and returns immediately
    with the thread_id and task_id. A separate worker process picks up
    the task and executes the LangGraph workflow asynchronously.
    Uses an idempotency key to prevent duplicate submissions.

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
    from ..config.queue import TaskQueue

    # Generate idempotency key if not provided
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

    # Build providers dict from config
    config = load_config()
    providers_by_type = _load_providers_from_config(config)
    providers_dict = {}
    for category, provider_list in providers_by_type.items():
        if provider_list:
            providers_dict[category] = provider_list[0][1]

    thread_id = str(uuid.uuid4())
    checkpoint_path_str = checkpoint_path or ".spine/spine.db"

    # Detect project context
    project_root = os.getcwd()
    project_name = Path(project_root).name
    project_config = config.get("project", {})
    project_context = {
        "name": project_config.get("name", project_name),
        "root": project_config.get("root", project_root),
        "description": project_config.get("description", ""),
        "tech_stack": project_config.get("tech_stack", []),
    }

    # Enqueue work for background processing
    try:
        queue = TaskQueue(db_path=".spine/queue.db")
        task_id = queue.enqueue("work", {
            "requirement": requirement,
            "thread_id": thread_id,
            "checkpoint_path": checkpoint_path_str,
            "method": method,
            "project_type": project_type,
            "providers": providers_dict,
            "variables": {
                "thread_id": thread_id,
                "work_item_id": thread_id,
                "checkpoint_path": checkpoint_path_str,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "idempotency_key": key,
            },
            "project_context": project_context,
        })
        payload = {
            "thread_id": thread_id,
            "task_id": task_id,
            "idempotency_key": key,
        }
        dedup_file.write_text(json.dumps(payload))
        return payload
    except Exception as e:
        error_msg = str(e)
        print(f"[spine.ui.utils] Failed to enqueue work: {error_msg}")
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
