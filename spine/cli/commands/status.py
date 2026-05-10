"""Status command - queries checkpoint storage for thread information."""

import sqlite3
import os
from typing import Any


def _decode_phase(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        repr_val = raw.get("repr", "")
        if "PhaseName." in repr_val:
            return repr_val.split("PhaseName.")[1].split(":")[0].strip("'")
        return repr_val
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _decode_value(raw: Any) -> Any:
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return raw


def _get_latest_checkpoint(conn: sqlite3.Connection, thread_id: str) -> dict | None:
    cur = conn.execute(
        "SELECT checkpoint FROM checkpoints WHERE thread_id = ? "
        "ORDER BY checkpoint_id DESC LIMIT 1",
        (thread_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None

    import ormsgpack

    try:
        ckpt = ormsgpack.unpackb(row[0])
    except Exception:
        return None

    cv = ckpt.get("channel_values", {})
    if not isinstance(cv, dict):
        return None

    phase_raw = cv.get("phase", "")
    phase = _decode_phase(phase_raw)

    completed_tasks_raw = cv.get("completed_tasks", [])
    if isinstance(completed_tasks_raw, (list, tuple)):
        completed_count = len(completed_tasks_raw)
    elif isinstance(completed_tasks_raw, dict):
        completed_count = len(completed_tasks_raw)
    else:
        completed_count = 0

    requirement_raw = cv.get("requirement", "")
    requirement = _decode_value(requirement_raw) if requirement_raw else ""

    plan = cv.get("plan")
    plan_exists = plan is not None

    return {
        "thread_id": thread_id,
        "phase": phase,
        "completed_tasks": completed_count,
        "requirement": requirement[:80] if requirement else "",
        "plan_exists": plan_exists,
    }


def get_threads(checkpoint_path: str) -> list[dict]:
    """Fetch all threads from the checkpoint database.

    Args:
        checkpoint_path: Path to the SQLite checkpoint database.

    Returns:
        List of dicts with keys: thread_id, phase, completed_tasks, requirement, plan_exists.
    """
    if not os.path.isfile(checkpoint_path):
        return []

    import ormsgpack

    try:
        conn = sqlite3.connect(str(checkpoint_path))
        conn.row_factory = sqlite3.Row
    except (sqlite3.Error, Exception):
        return []

    try:
        cur = conn.execute("SELECT DISTINCT thread_id FROM checkpoints")
        thread_ids = [row["thread_id"] for row in cur.fetchall()]
    except sqlite3.Error:
        conn.close()
        return []

    threads = []
    for tid in thread_ids:
        info = _get_latest_checkpoint(conn, tid)
        if info:
            threads.append(info)

    conn.close()
    return threads
