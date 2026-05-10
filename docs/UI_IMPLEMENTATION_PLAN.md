# SPINE UI Implementation Plan

## Executive Summary

This document specifies the concrete implementation plan for the SPINE UI v1 — a Streamlit-based web dashboard. It maps directly to the research in `UI_RESEARCH.md` and provides file-by-file specifications, data flow diagrams, and a phased development schedule.

**Goal**: Build a fully functional UI that can monitor and interact with SPINE workflows, with critic gate approval, provider configuration, and checkpoint history — all in a single day.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│               User Browser                           │
│         (http://localhost:8501)                      │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP (Streamlit protocol)
                       ▼
┌─────────────────────────────────────────────────────┐
│              Streamlit App                           │
│              (spine/ui/app.py)                       │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │Dashboard │  │WorkDetail│  │Providers │          │
│  │Page      │  │Page      │  │Config    │          │
│  └──────────┘  └──────────┘  └──────────┘          │
│  ┌──────────┐  ┌──────────┐                         │
│  │NewWork   │  │Settings  │                         │
│  │Form      │  │Page      │                         │
│  └──────────┘  └──────────┘                         │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│           SPINE Core Backend                         │
│  (spine/ui/ui_api.py — thin read/write layer)        │
│                                                      │
│  StateMachine │ Hive │ Providers │ Persistence       │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│           Persistence (.spine/)                      │
│  spine.db (SQLite checkpoint)                        │
│  config.yaml                                         │
│  events/swarm.log                                    │
│  state/hive/cells.json                               │
└─────────────────────────────────────────────────────┘
```

### Data Flow

```
User action (click "Start Work")
  → Streamlit calls ui_api.start_work(requirement, config)
    → ui_api calls SpineStateMachine.app.create_new_thread()
    → UI polls state.db for updates every 2-3s
      → Streamlit re-renders with new phase info
        → User sees progress bars, agent outputs
```

---

## 2. File Specifications

### 2.1 `spine/ui/__init__.py`

```python
"""SPINE UI — Streamlit dashboard for the agent harness."""
__version__ = "0.1.0"
```

### 2.2 `spine/ui/app.py` — Main Entry Point

**Purpose**: Streamlit app entry point. Sets up page config, navigation, and sidebar.

```python
import streamlit as st
from .dashboard import render_dashboard
from .work_detail import render_work_detail
from .work_new import render_new_work
from .providers_config import render_providers_config
from .settings import render_settings

PAGE_ICONS = {
    "Dashboard": "📊",
    "New Work": "➕",
    "Work Detail": "🔍",
    "Providers": "🔌",
    "Settings": "⚙️",
}

PAGE_NAMES = ["Dashboard", "New Work", "Work Detail", "Providers", "Settings"]


def main():
    st.set_page_config(
        page_title="SPINE Dashboard",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Navigation state persisted across reruns
    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"
    if "selected_work_id" not in st.session_state:
        st.session_state.selected_work_id = None

    with st.sidebar:
        st.title("⚡ SPINE")
        st.markdown("### Navigation")
        selected = st.radio(
            "Navigation",
            PAGE_NAMES,
            key="nav_page",
            index=PAGE_NAMES.index(st.session_state.page),
        )
        st.session_state.page = selected

    # Route to page handler
    page = st.session_state.page
    if page == "Dashboard":
        render_dashboard()
    elif page == "New Work":
        render_new_work()
    elif page == "Work Detail":
        render_work_detail()
    elif page == "Providers":
        render_providers_config()
    elif page == "Settings":
        render_settings()


if __name__ == "__main__":
    main()
```

**CLI entry point** (add to `cli.py`):

```python
@cli.command()
def ui():
    """Start the SPINE web dashboard."""
    import subprocess
    import sys
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "-c", "server.headless=true",
        "-b", "localhost:8501",
        "-m", "spine.ui.app",
    ])
```

### 2.3 `spine/ui/utils.py` — Shared Helpers

**Purpose**: Common utilities for reading SPINE state, formatting data, navigating.

```python
import os
import json
import time
import sqlite3
from pathlib import Path
from typing import Any, Optional

import yaml
from datetime import datetime


# ── Config Loading ──────────────────────────────────────────

def _expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in string values."""
    if isinstance(value, str):
        import re
        result = re.sub(r'\$\{([^}]+)\}', lambda m: os.environ.get(m.group(1), ''), value)
        result = re.sub(r'\$([A-Za-z_][A-Za-z0-9_]*)',
                        lambda m: os.environ.get(m.group(1), ''), result)
        return result
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def load_config(config_path: str = ".spine/config.yaml") -> dict:
    """Load .spine/config.yaml, expanding env vars."""
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path) as f:
        config = yaml.safe_load(f) or {}
    return _expand_env_vars(config)


def save_config(config: dict, config_path: str = ".spine/config.yaml"):
    """Save .spine/config.yaml."""
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# ── Checkpoint / State Reading ──────────────────────────────

def get_checkpoint_path(thread_id: str = "default") -> Path:
    """Resolve the checkpoint DB path from config or default."""
    config = load_config()
    return Path(config.get("spine", {}).get("checkpoint_path", ".spine/spine.db"))


def get_latest_checkpoint(thread_id: str = "default") -> Optional[dict]:
    """Read the latest checkpoint for a thread from the SQLite checkpoint DB.

    LangGraph's SqliteSaver stores checkpoints in a 'checkpoint_blobs' table.
    This function reads the most recent state for the given thread.
    """
    cp_path = get_checkpoint_path(thread_id)
    if not cp_path.exists():
        return None

    try:
        conn = sqlite3.connect(str(cp_path))
        cursor = conn.cursor()

        # List all tables to find checkpoint store
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        # Try common LangGraph checkpoint table names
        for table in ["checkpoint_blobs", "checkpoint_writes", "states"]:
            if table in tables:
                cursor.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT 1")
                row = cursor.fetchone()
                if row:
                    # Attempt to parse JSON blob
                    try:
                        return json.loads(row[-1]) if isinstance(row[-1], str) else row[-1]
                    except (json.JSONDecodeError, TypeError):
                        return {"raw": str(row)}

        # Fallback: list all tables and find any JSON data
        for table in tables:
            if table.startswith("checkpoint") or table.startswith("state"):
                cursor.execute(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT 1")
                row = cursor.fetchone()
                if row:
                    return {"table": table, "data": [str(v) for v in row]}

        conn.close()
        return None

    except Exception:
        return None


def get_all_threads_from_checkpoint() -> list[str]:
    """Extract all known thread IDs from the checkpoint DB."""
    cp_path = get_checkpoint_path()
    if not cp_path.exists():
        return ["default"]

    try:
        conn = sqlite3.connect(str(cp_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        # LangGraph stores thread_id in the 'thread_id' column
        # We look for a table containing thread_id info
        for table in tables:
            if "thread" in table.lower():
                cursor = sqlite3.connect(str(cp_path)).cursor()
                cursor.execute(f"SELECT DISTINCT thread_id FROM {table}")
                threads = [row[0] for row in cursor.fetchall()]
                cursor.close()
                return threads or ["default"]

        return ["default"]
    except Exception:
        return ["default"]


def get_work_item_detail(thread_id: str = "default") -> Optional[dict]:
    """Load full state from the latest checkpoint for a work item.

    Returns all state fields including phase, requirement, plan, tasks,
    swarm_state, hive_cells, swarm_events, critic_gate_result, etc.
    """
    checkpoint = get_latest_checkpoint(thread_id)
    if not checkpoint:
        return None

    # The checkpoint contains the full graph state
    # Extract common fields
    return {
        "thread_id": thread_id,
        "raw": checkpoint,
        "phase": checkpoint.get("phase", "INIT"),
        "requirement": checkpoint.get("requirement", ""),
        "plan": checkpoint.get("plan"),
        "completed_tasks": checkpoint.get("completed_tasks", []),
        "failed_tasks": checkpoint.get("failed_tasks", []),
        "swarm_state": checkpoint.get("swarm_state", {}),
        "hive_cells": checkpoint.get("hive_cells", {}),
        "swarm_events": checkpoint.get("swarm_events", []),
        "critic_gate_result": checkpoint.get("critic_gate_result"),
        "error_state": checkpoint.get("error_state"),
        "variables": checkpoint.get("variables", {}),
        "errors": checkpoint.get("errors", []),
        "started_at": datetime.now().isoformat(),  # TODO: read from checkpoint timestamp
    }


def get_active_work_items() -> list[dict]:
    """Read checkpoint DB and return all work items with their latest status."""
    thread_ids = get_all_threads_from_checkpoint()
    items = []

    for tid in thread_ids:
        detail = get_work_item_detail(tid)
        if detail:
            phase = detail.get("phase", "INIT")
            completed = len(detail.get("completed_tasks", []))
            total = completed + len(detail.get("failed_tasks", []))

            items.append({
                "thread_id": tid,
                "requirement": detail.get("requirement", "Untitled"),
                "phase": phase,
                "status": phase,
                "progress": completed / max(1, total),
                "completed_tasks": completed,
                "total_tasks": total,
                "started_at": detail.get("started_at", ""),
                "errors": detail.get("errors", []),
            })

    return items


def get_checkpoints(thread_id: str) -> list[dict]:
    """Return all checkpoints for a work item, newest first."""
    # TODO: enumerate all checkpoints, not just the latest
    # This requires reading checkpoint metadata table
    cp_path = get_checkpoint_path(thread_id)
    if not cp_path.exists():
        return []

    checkpoints = []
    try:
        conn = sqlite3.connect(str(cp_path))
        cursor = conn.cursor()

        # List all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            if "checkpoint" in table or "state" in table:
                cursor.execute(f"SELECT rowid, * FROM {table} ORDER BY rowid DESC")
                for row in cursor.fetchall():
                    try:
                        data = json.loads(row[-1]) if isinstance(row[-1], str) else row[-1]
                        checkpoints.append({
                            "row_id": row[0],
                            "table": table,
                            "timestamp": datetime.now().isoformat(),
                            "data": data,
                        })
                    except (json.JSONDecodeError, TypeError):
                        pass

        conn.close()
    except Exception:
        pass

    return checkpoints


# ── State Machine / Work Item Actions ───────────────────────

def start_work(requirement: str, method: str = "Quick Work",
               project_type: str = "Greenfield",
               llm_provider: str = "qwen3:32b (Ollama)",
               parallel_agents: int = 3) -> Optional[dict]:
    """Start a new work item via the backend state machine.

    Returns thread_id on success, None on failure.
    """
    from ..core import SpineStateMachine

    machine = SpineStateMachine()
    try:
        result = machine.create_new_thread(
            requirement=requirement,
            method=method,
            project_type=project_type,
            llm_provider=llm_provider,
            parallel_agents=parallel_agents,
        )
        return {"thread_id": result.get("thread_id", "new")}
    except Exception as e:
        print(f"Failed to start work: {e}")
        return None


def approve_gate(thread_id: str) -> bool:
    """Approve the critic gate for a work item."""
    # Write approval to a gate_result file that the state machine reads
    gate_file = Path(f".spine/state/gate_result_{thread_id}.json")
    gate_file.write_text(json.dumps({
        "approved": True,
        "timestamp": datetime.now().isoformat(),
    }))
    return True


def reject_gate(thread_id: str, feedback: str) -> bool:
    """Reject the critic gate with feedback for rework."""
    gate_file = Path(f".spine/state/gate_result_{thread_id}.json")
    gate_file.write_text(json.dumps({
        "approved": False,
        "feedback": feedback,
        "timestamp": datetime.now().isoformat(),
    }))
    return True


def resume_work(thread_id: str) -> bool:
    """Resume a paused work item."""
    from ..core import SpineStateMachine
    machine = SpineStateMachine()
    try:
        machine.resume(thread_id)
        return True
    except Exception:
        return False


# ── Formatting Helpers ──────────────────────────────────────

PHASE_ICONS = {
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

PHASE_COLORS = {
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


def format_phase_icon(phase: str) -> str:
    """Return emoji for phase."""
    return PHASE_ICONS.get(phase, "•")


def format_phase_color(phase: str) -> str:
    """Return rich color name for phase."""
    return PHASE_COLORS.get(phase, "white")


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s"


def format_bytes(size: int) -> str:
    """Format bytes into human-readable size."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# ── Navigation ──────────────────────────────────────────────

def navigate_to_work(thread_id: str):
    """Set session state for navigation to a work item."""
    st.session_state.selected_work_id = thread_id
    st.session_state.page = "Work Detail"


def go_back():
    """Navigate back to dashboard."""
    st.session_state.page = "Dashboard"
    st.session_state.selected_work_id = None
```

### 2.4 `spine/ui/dashboard.py` — Dashboard Page

**Purpose**: List all work items with status, progress bars, and quick actions.

```python
import streamlit as st
from .utils import (
    get_active_work_items,
    format_phase_icon,
    format_phase_color,
    go_back,
)


def render_dashboard():
    st.title("📊 SPINE Dashboard")

    # Auto-refresh toggle
    auto_refresh = st.toggle("Auto-refresh", value=True)

    # Fetch work items
    work_items = get_active_work_items()

    # Summary metrics
    active = sum(1 for w in work_items if w["phase"] not in ("COMPLETE", "CANCELLED"))
    complete = sum(1 for w in work_items if w["phase"] == "COMPLETE")
    blocked = sum(1 for w in work_items if w["phase"] == "BLOCKED")

    col1, col2, col3 = st.columns(3)
    col1.metric("Active", active)
    col2.metric("Complete", complete)
    col3.metric("Blocked", blocked)

    if not work_items:
        st.info("No work items yet. Click **New Work** to get started.")
        return

    # Work item cards
    st.subheader("Active Work Items")

    for item in work_items:
        progress = item["completed_tasks"] / max(1, item.get("total_tasks", 1))
        icon = format_phase_icon(item["phase"])
        color = format_phase_color(item["phase"])

        with st.container():
            col_icon, col_title, col_phase, col_progress, col_actions = st.columns(
                [1, 3, 2, 3, 2]
            )

            col_icon.write(icon)
            col_title.write(f"**{item['requirement']}**")
            col_phase.write(f"[{color}]{item['phase']}[/{color}]")
            col_progress.progress(progress)

            if col_actions.button("View", key=f"view_{item['thread_id']}"):
                st.session_state.selected_work_id = item["thread_id"]
                st.session_state.page = "Work Detail"
                st.rerun()

    # Footer
    st.caption("Click 'View' on any work item to see detailed progress.")
```

### 2.5 `spine/ui/work_new.py` — Work Creation Form

**Purpose**: Form for creating new work items with all options from FEATURE_LIST.MD.

```python
import streamlit as st
from .utils import start_work, go_back


def render_new_work():
    st.title("➕ New Work Item")

    with st.form("new_work_form"):
        st.subheader("Work Details")
        title = st.text_input(
            "Working Title",
            placeholder="e.g., Build authentication system",
        )
        description = st.text_area(
            "Description",
            placeholder="Detailed description of what needs to be done...",
            height=100,
        )

        st.subheader("Method")
        method = st.radio(
            "Automation Level",
            ["Quick Work", "Full Spec Work", "Full Spec Project"],
            index=1,
            help=(
                "Quick Work: Plan → Implement → Verify. "
                "Full Spec: adds Design and Spec phases."
            ),
        )

        st.subheader("Project Type")
        project_type = st.radio(
            "Environment",
            ["Greenfield", "Brownfield"],
            index=0,
        )

        st.subheader("Execution")

        # LLM provider selection
        providers = ["qwen3:32b (Ollama)", "gpt-4.1 (OpenAI)", "local-model (Local)"]
        llm_provider = st.selectbox("LLM Provider", providers, index=0)

        parallel_agents = st.slider(
            "Max Parallel Agents",
            min_value=1,
            max_value=10,
            value=3,
            help="Maximum agents to run in parallel within a phase",
        )

        submitted = st.form_submit_button("▶ Start Work → ", type="primary")

        if submitted:
            if not title:
                st.error("Please enter a working title.")
            else:
                result = start_work(
                    requirement=title,
                    method=method,
                    project_type=project_type,
                    llm_provider=llm_provider,
                    parallel_agents=parallel_agents,
                )
                if result:
                    st.session_state.selected_work_id = result["thread_id"]
                    st.session_state.page = "Work Detail"
                    st.success("Work item created! Redirecting...")
                    st.rerun()
                else:
                    st.error("Failed to start work item. Check the console for details.")
```

### 2.6 `spine/ui/work_detail.py` — Work Item Detail Page

**Purpose**: Main page for monitoring a single work item. Shows state machine, phase progress, agent outputs, swarm events.

```python
import streamlit as st
from .utils import (
    get_work_item_detail,
    format_phase_icon,
    format_phase_color,
    get_checkpoints,
    go_back,
    approve_gate,
    reject_gate,
    resume_work,
)


def render_work_detail():
    if not st.session_state.get("selected_work_id"):
        st.warning(
            "No work item selected. "
            "Go to **Dashboard** and click 'View' on a work item."
        )
        return

    thread_id = st.session_state.selected_work_id
    detail = get_work_item_detail(thread_id)

    if not detail:
        st.error("Work item not found.")
        return

    # ── Header ──
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Phase", detail.get("phase", "INIT"))
    col2.metric("Completed", len(detail.get("completed_tasks", [])))
    col3.metric("Failed", len(detail.get("failed_tasks", [])))
    col4.metric("Errors", len(detail.get("errors", [])))

    st.write(f"**Requirement:** {detail.get('requirement', 'Unknown')}")

    # Action buttons
    col1, col2, col3 = st.columns(3)
    if col1.button("▶ Resume", key=f"resume_{thread_id}"):
        if resume_work(thread_id):
            st.success("Resumed work item.")
            st.rerun()
        else:
            st.error("Failed to resume.")
    if col2.button("🗑 Delete", key=f"delete_{thread_id}"):
        st.warning("Delete not yet implemented. Use CLI.")

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs(
        ["State Machine", "Sub-Phases", "Agent Outputs", "Swarm Events"]
    )

    with tab1:
        render_state_machine(detail)

    with tab2:
        render_subphases(detail)

    with tab3:
        render_agent_outputs(detail)

    with tab4:
        render_swarm_events(detail)


def render_state_machine(detail: dict):
    """Render the state machine visualization."""
    phases = [
        ("INIT", "⚙️"),
        ("PLANNING", "📋"),
        ("EXECUTION", "🔨"),
        ("VERIFICATION", "✅"),
        ("COMPLETE", "🏁"),
    ]

    current = detail.get("phase", "INIT")
    phase_names = [p[0] for p in phases]
    current_idx = phase_names.index(current) if current in phase_names else 0

    st.subheader("State Machine Progress")

    # Build visual progress bar with emojis
    parts = []
    for i, (name, icon) in enumerate(phases):
        if i < current_idx:
            parts.append(f"{icon} **{name}** ✓")
        elif i == current_idx:
            parts.append(f"{icon} **{name}** ● current")
        else:
            parts.append(f"{icon} {name} ○")

    st.markdown(" → ".join(parts))

    # Critic gate indicator
    critic_result = detail.get("critic_gate_result")
    if critic_result:
        status = critic_result.get("status", "unknown")
        if status == "approved":
            st.success("✅ Critic gate passed — can proceed to EXECUTION")
        elif status == "rejected":
            st.error("✗ Critic gate rejected — rework required")
        else:
            st.warning("⏳ Critic gate pending review")
    else:
        st.info("No critic gate result yet.")

    # Plan preview
    plan = detail.get("plan")
    if plan and isinstance(plan, dict):
        tasks = plan.get("tasks", [])
        if tasks:
            st.subheader("Plan Tasks")
            for t in tasks:
                status_icon = "✓" if t.get("completed") else "○"
                st.write(f"  {status_icon} **{t.get('id', '?')}**: {t.get('description', '')}")


def render_subphases(detail: dict):
    """Render sub-phase progress for the current phase."""
    swarm_state = detail.get("swarm_state", {})
    subphases = swarm_state.get("active_subphases", [])

    st.subheader(f"Phase: {detail.get('phase', 'UNKNOWN')}")

    if not subphases:
        st.info("No sub-phases defined for this phase yet.")
        return

    for sp in subphases:
        name = sp.get("name", "Unknown")
        agent = sp.get("agent", "unknown")
        status = sp.get("status", "pending")
        progress = sp.get("progress", 0)

        icon = {"success": "🟢", "running": "🟡", "pending": "⚪", "failed": "🔴"}.get(
            status, "⚪"
        )

        st.progress(progress)
        st.write(f"  {icon} **{name}** ({agent}) — {status} ({progress:.0%})")


def render_agent_outputs(detail: dict):
    """Render agent outputs, tabbed by sub-phase."""
    swarm_events = detail.get("swarm_events", [])

    if not swarm_events:
        st.info("No agent outputs yet. This will populate as the workflow runs.")
        return

    # Group events by source agent
    agent_groups: dict[str, list] = {}
    for event in swarm_events:
        source = event.get("from", "unknown")
        if source not in agent_groups:
            agent_groups[source] = []
        agent_groups[source].append(event)

    tabs = st.tabs(list(agent_groups.keys()))
    for (agent_name, events), tab in zip(agent_groups.items(), tabs):
        with tab:
            for event in events:
                with st.expander(f"{event.get('subject', 'Event')} — {event.get('timestamp', '')}"):
                    st.code(str(event.get("body", "")), language="json")


def render_swarm_events(detail: dict):
    """Render swarm event log."""
    events = detail.get("swarm_events", [])

    if not events:
        st.info("No swarm events recorded yet.")
        return

    # Table view
    rows = []
    for e in events:
        rows.append({
            "Timestamp": e.get("timestamp", ""),
            "From": e.get("from", ""),
            "To": e.get("to", ""),
            "Subject": e.get("subject", ""),
            "Preview": str(e.get("body", ""))[:80],
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)
```

### 2.7 `spine/ui/providers_config.py` — Provider Configuration UI

**Purpose**: CRUD UI for provider configurations in `.spine/config.yaml`.

```python
import streamlit as st
from .utils import load_config, save_config


LLM_TYPES = ["openai", "ollama", "openrouter", "local-openai"]


def render_providers_config():
    st.title("🔌 Provider Configuration")

    config = load_config()

    llm_config = config.get("providers", {}).get("llm", [])

    with st.form("llm_providers_form"):
        st.subheader("LLM Providers")

        # Current providers
        for i, provider in enumerate(llm_config):
            with st.container():
                st.write(f"**Provider {i + 1}**")
                name = st.text_input("Name", value=provider.get("name", ""), key=f"llm_name_{i}")
                ltype = st.selectbox("Type", LLM_TYPES, index=LLM_TYPES.index(provider.get("type", "ollama")), key=f"llm_type_{i}")
                model = st.text_input("Model", value=provider.get("model", "qwen3:32b"), key=f"llm_model_{i}")
                api_key = st.text_input("API Key", value=provider.get("api_key", ""), type="password", key=f"llm_key_{i}")
                base_url = st.text_input("Base URL", value=provider.get("base_url", ""), key=f"llm_url_{i}")
                priority = st.number_input("Priority", min_value=0, max_value=10, value=provider.get("priority", 1), key=f"llm_pri_{i}")

                if st.button(f"Remove Provider {i + 1}", key=f"llm_rm_{i}"):
                    llm_config.pop(i)
                    st.rerun()

        st.markdown("---")

        if st.form_submit_button("+ Add Provider"):
            llm_config.append({"name": "new", "type": LLM_TYPES[0], "model": "", "priority": len(llm_config)})
            st.rerun()

    # Save button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.form_submit_button("💾 Save Configuration"):
            config.setdefault("providers", {})["llm"] = llm_config
            save_config(config)
            st.success("Configuration saved!")
```

### 2.8 `spine/ui/components.py` — Shared Components

**Purpose**: Reusable Streamlit components used across pages.

```python
import streamlit as st
from .utils import format_phase_icon, format_phase_color


def phase_badge(phase: str) -> str:
    """Return a colored badge for a phase."""
    icon = format_phase_icon(phase)
    color = format_phase_color(phase)
    return f"[{color}]{icon} {phase}[/{color}]"


def phase_progress_bar(progress: float, label: str = "") -> None:
    """Render a Streamlit progress bar with label."""
    if label:
        st.write(f"**{label}**")
    st.progress(progress)


def status_badge(status: str) -> str:
    """Return a colored status badge."""
    color_map = {
        "success": "green",
        "running": "blue",
        "pending": "gray",
        "failed": "red",
        "blocked": "red",
    }
    color = color_map.get(status, "gray")
    return f"[{color}]{status}[/{color}]"


def empty_state(message: str, icon: str = "ℹ️") -> None:
    """Show an empty state with icon and message."""
    col1, col2 = st.columns([1, 3])
    col1.write(icon)
    col2.info(message)
```

### 2.9 `spine/ui/settings.py` — Settings Page

**Purpose**: Global settings for the UI.

```python
import streamlit as st
from .utils import go_back


def render_settings():
    st.title("⚙️ Settings")

    st.subheader("Display")
    st.toggle("Dark mode", value=False, help="Toggle dark/light theme")
    refresh_interval = st.slider(
        "Auto-refresh interval (seconds)",
        min_value=1,
        max_value=30,
        value=3,
        help="How often to poll for updates"
    )

    st.subheader("Data")
    if st.button("Clear session state"):
        st.session_state.clear()
        st.success("Session state cleared.")

    st.subheader("About")
    st.write("**SPINE UI** v0.1.0")
    st.write("Streamlit dashboard for the SPINE agent harness.")
    st.caption("Built with Streamlit • Python-native • No build step")
```

### 2.10 `spine/core/ui_api.py` — Backend API for UI

**Purpose**: Thin API layer that the UI pages call to interact with SPINE core. Separates UI concerns from core logic.

```python
"""Backend API for the SPINE UI."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

from .state_machine import SpineStateMachine


class UIApi:
    """Thin API layer between the Streamlit UI and SPINE core."""

    def __init__(self, checkpoint_path: str = ".spine/spine.db"):
        self._checkpoint_path = checkpoint_path
        self._lock = threading.Lock()

    # ── Read Operations ───────────────────────────────────────

    def get_active_work_items(self) -> list[dict]:
        """Return all work items with their latest status."""
        raise NotImplementedError("Read via utils.get_active_work_items()")

    def get_work_item_detail(self, thread_id: str) -> Optional[dict]:
        """Load full state from the latest checkpoint."""
        raise NotImplementedError("Read via utils.get_work_item_detail()")

    def get_checkpoints(self, thread_id: str) -> list[dict]:
        """Return all checkpoints for a work item."""
        raise NotImplementedError("Read via utils.get_checkpoints()")

    # ── Write Operations ──────────────────────────────────────

    def start_work(
        self,
        requirement: str,
        method: str = "Quick Work",
        project_type: str = "Greenfield",
        llm_provider: str = "ollama",
        parallel_agents: int = 3,
    ) -> dict:
        """Start a new work item. Returns thread_id on success."""
        with self._lock:
            machine = SpineStateMachine(checkpoint_path=self._checkpoint_path)
            result = machine.create_new_thread(
                requirement=requirement,
                method=method,
                project_type=project_type,
                llm_provider=llm_provider,
                parallel_agents=parallel_agents,
            )
            return {"thread_id": result.get("thread_id", "new")}

    def approve_gate(self, thread_id: str) -> bool:
        """Approve the critic gate for a work item."""
        gate_file = Path(f".spine/state/gate_result_{thread_id}.json")
        gate_file.parent.mkdir(parents=True, exist_ok=True)
        gate_file.write_text(json.dumps({
            "approved": True,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }))
        return True

    def reject_gate(self, thread_id: str, feedback: str) -> bool:
        """Reject the critic gate with feedback for rework."""
        gate_file = Path(f".spine/state/gate_result_{thread_id}.json")
        gate_file.parent.mkdir(parents=True, exist_ok=True)
        gate_file.write_text(json.dumps({
            "approved": False,
            "feedback": feedback,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }))
        return True

    def resume_work(self, thread_id: str) -> bool:
        """Resume a paused work item."""
        machine = SpineStateMachine(checkpoint_path=self._checkpoint_path)
        try:
            machine.resume(thread_id)
            return True
        except Exception:
            return False
```

---

## 3. pyproject.toml Changes

```toml
[project]
dependencies = [
    "langgraph>=0.4.0",
    "langgraph-checkpoint-sqlite>=3.0.0",
    "langgraph-supervisor>=0.0.13",
    "openai>=1.0.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    "click>=8.0",
    "rich>=13.0",
    "sqlite-utils>=3.36",
    "streamlit>=1.30.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "black>=24.0",
    "ruff>=0.4.0",
    "mypy>=1.10",
]
ui = [
    "fastapi>=0.100.0",     # v2: SPA frontend backend
    "uvicorn>=0.23.0",
    "websockets>=12.0",
]

[project.scripts]
spine = "spine.cli:main"
```

---

## 4. Development Schedule

### Sprint 1: Foundation (4 hours)

| Task | File | Description |
|------|------|-------------|
| Create `spine/ui/` package | All files | `__init__.py`, `app.py` with sidebar navigation |
| Implement `utils.py` | `utils.py` | Config loading, checkpoint reading, formatting helpers |
| Add CLI entry point | `cli.py` | `spine ui` command |
| Add streamlit dependency | `pyproject.toml` | Update dependencies |

### Sprint 2: Core Pages (6 hours)

| Task | File | Description |
|------|------|-------------|
| Dashboard page | `dashboard.py` | Work items list with progress bars, summary metrics |
| Work creation form | `work_new.py` | Form with title, description, method, project type, provider |
| Work detail page | `work_detail.py` | State machine viz, sub-phase progress, agent outputs, swarm events |
| State machine visualizer | `work_detail.py` | Inline phase progress bar with emoji indicators |

### Sprint 3: Configuration & Polish (4 hours)

| Task | File | Description |
|------|------|-------------|
| Provider config page | `providers_config.py` | CRUD form for LLM providers in config.yaml |
| Settings page | `settings.py` | Auto-refresh interval, theme, about |
| Shared components | `components.py` | Phase badges, progress bars, empty states |
| Backend API | `ui_api.py` | Thin API layer wrapping state machine |

### Total: ~14 hours for a fully functional v1

---

## 5. Testing Strategy

Since this is a UI, testing focuses on:

1. **Unit tests** for `utils.py` helpers (config loading, env var expansion, formatting)
2. **Integration tests** for `ui_api.py` (start work, approve gate, resume)
3. **Smoke tests** for Streamlit pages (render without errors)

```python
# tests/test_ui_utils.py
def test_expand_env_vars():
    import os
    os.environ["TEST_KEY"] = "test_value"
    from spine.ui.utils import _expand_env_vars
    result = _expand_env_vars("${TEST_KEY}")
    assert result == "test_value"

def test_format_phase_icon():
    from spine.ui.utils import format_phase_icon
    assert format_phase_icon("COMPLETE") == "🏁"
    assert format_phase_icon("UNKNOWN") == "•"
```

---

## 6. Deployment

### Local development

```bash
# Install with UI dependencies
pip install -e ".[dev]"

# Run the dashboard
spine ui

# Opens http://localhost:8501
```

### With Docker (future)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e ".[ui]"
EXPOSE 8501
ENTRYPOINT ["spine", "ui"]
```

---

## 7. Migration Path to v2 (FastAPI + React)

When Streamlit becomes limiting:

1. **Extract `ui_api.py`** → FastAPI backend with REST endpoints
2. **Create React SPA** with:
   - React Router for navigation
   - React Query for data fetching (replaces polling)
   - WebSocket client for real-time events
   - TanStack Table for data grids
3. **Keep Streamlit** as a "quick look" mode (`spine ui --streamlit`)
4. **Gradually phase out** Streamlit pages as React equivalents ship

```
spine ui            → Streamlit (quick view)
spine api           → FastAPI (full API + React SPA)
spine work "..."    → CLI (automation, CI/CD)
```

---

## 8. Summary of Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Framework | **Streamlit** | Python-native, rapid dev, fits SPINE's stack |
| Navigation | **Sidebar radio** | Simple, intuitive, no routing complexity |
| Real-time | **Polling 2-3s** | Simple, reliable, sufficient for v1 |
| Write safety | **Explicit buttons** | No auto-save, human must confirm |
| Data access | **Read SQLite + utils** | Works with LangGraph's SqliteSaver |
| Config | **Inline YAML editor** | Both CLI and UI share `.spine/config.yaml` |
| State mgmt | **st.session_state** | Built-in, no extra deps |
| Thread safety | **Thread locks on writes** | Reads from SQLite are safe concurrently |
| Next UI version | **FastAPI + React** | Documented migration path |
