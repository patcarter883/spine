"""SPINE Work Detail page — combined view of work status and artifacts."""

from __future__ import annotations

import streamlit as st

from spine.ui.pages import get as get_page
from spine.ui_api import UIApi
from spine.ui.utils import format_duration, format_timestamp, status_icon, truncate

# ── Fragment refresh interval (seconds) ──
_POLL_INTERVAL = 10


# ── Helpers ──


def _execution_duration(api: UIApi, work_id: str, entry: dict[str, object], status: str) -> str:
    """Compute execution duration from audit log events.

    For completed/failed items, returns the wall-clock time between the first
    ``work_submitted`` (or ``work_started``) event and the last
    ``work_completed`` (or ``work_failed``) event.

    For running items, returns the time elapsed since the first start
    event.

    Args:
        api: The UIApi instance for querying audit logs.
        work_id: The work item ID.
        entry: The work entry dict (used for fallback timestamps).
        status: Current status of the work item.

    Returns:
        Human-readable duration string, or ``"—`` if insufficient data.
    """
    terminal_statuses = ("completed", "failed", "needs_review")
    audit_events = api.get_audit_log(work_id=work_id)
    if not audit_events:
        return format_duration(entry.get("created_at"), entry.get("updated_at"))

    # Determine start event: first work_submitted or work_started
    start_event = next(
        (e for e in audit_events if e.get("event_type") in ("work_submitted", "work_started")),
        None,
    )
    start_ts = start_event.get("timestamp") if start_event else None

    if status in terminal_statuses:
        # Find the last terminal event for completed/failed items
        end_event = next(
            (
                e
                for e in reversed(audit_events)
                if e.get("event_type") in ("work_completed", "work_failed")
            ),
            None,
        )
        end_ts = end_event.get("timestamp") if end_event else None
        return format_duration(start_ts, end_ts)
    elif status == "running":
        # Live elapsed time since start
        return format_duration(start_ts)
    else:
        # Unknown or pending — fall back to entry timestamps
        return format_duration(entry.get("created_at"), entry.get("updated_at"))


# ── Fragment sections ──


@st.fragment(run_every=_POLL_INTERVAL)
def _render_status_header(api: UIApi, work_id: str) -> None:
    """Status header with metrics — auto-refreshing fragment."""
    entry = api.get_work(work_id)
    if entry is None:
        st.error(f"Work item '{work_id}' not found.")
        return

    status = entry.get("status", "unknown")
    icon = status_icon(status)

    st.header(f"{icon} {work_id}")

    col1, col2 = st.columns(2)
    col1.write(f"**Status:** {status}")
    col1.write(f"**Type:** {entry.get('work_type', 'N/A')}")
    col1.write(f"**Phase:** {entry.get('current_phase', 'N/A')}")
    col2.write(f"**Execution Time:** {_execution_duration(api, work_id, entry, status)}")
    col2.write(f"**Updated:** {format_timestamp(entry.get('updated_at'))}")


@st.fragment(run_every=_POLL_INTERVAL)
def _render_artifacts(api: UIApi, work_id: str) -> None:
    """Artifacts section — auto-refreshing fragment."""
    # Fetch fresh entry to get latest work_type
    entry = api.get_work(work_id)
    if entry is None:
        return

    artifacts = api.get_artifacts(work_id)
    if artifacts:
        # Build phase order from the work item's workflow sequence,
        # filtering out critic nodes (artifacts are keyed by phase name).
        from spine.workflow.compose import WORKFLOW_SEQUENCES

        work_type = entry.get("work_type", "quick")
        sequence = WORKFLOW_SEQUENCES.get(work_type, WORKFLOW_SEQUENCES.get("quick", []))
        # Only non-critic nodes produce artifact directories
        node_index = {
            name: i for i, (name, _) in enumerate(sequence) if not name.startswith("critic")
        }

        sorted_artifacts = sorted(
            artifacts,
            key=lambda a: (node_index.get(a.get("phase", ""), 99), a.get("name", "")),
        )
        for artifact in sorted_artifacts:
            phase = artifact.get("phase", "unknown")
            name = artifact.get("name", "unknown")
            size = artifact.get("size", 0)
            modified = artifact.get("modified", "N/A")

            with st.expander(f"📄 {phase}/{name} ({size} bytes)"):
                st.write(f"**Phase:** {phase}")
                st.write(f"**Name:** {name}")
                st.write(f"**Size:** {size} bytes")
                st.write(f"**Modified:** {modified}")

                # Load and display content
                content = api.read_artifact(work_id, phase, name)
                if content:
                    if name.endswith(".md") or name.endswith(".txt"):
                        st.markdown(content)
                    elif name.endswith(".json"):
                        st.json(content)
                    else:
                        st.code(content, language="text")
    else:
        st.info("No artifacts found for this work item.")


@st.fragment(run_every=_POLL_INTERVAL)
def _render_audit_log(api: UIApi, work_id: str) -> None:
    """Audit log section — auto-refreshing fragment."""
    audit_events = api.get_audit_log(work_id=work_id, limit=20)
    if audit_events:
        for event in audit_events:
            col1, col2, col3, col4 = st.columns([2, 3, 1, 1])
            col1.write(f"**{event.get('event_type', 'N/A')}**")
            phase = event.get("phase", "N/A")
            col2.write(
                f"{phase}: {event.get('message', 'N/A')}"
                if phase != "N/A"
                else event.get("message", "N/A")
            )
            col3.write(
                event.get("details", "").get("status", "")
                if isinstance(event.get("details"), dict)
                else ""
            )
            col4.write(format_timestamp(event.get("timestamp")))
    else:
        st.info("No audit events found for this work item.")


# ── Page ──


def render(api: UIApi) -> None:
    """Render the work detail page with combined status and artifacts."""
    st.title("🔍 Work Details")

    # Get work ID from URL query parameters or user input
    work_id = st.query_params.get("work_id", "")

    if not work_id:
        # Show input form when no work_id is provided via URL params
        work_id = st.text_input("Work ID", placeholder="Enter work item ID")

        if not work_id:
            st.info("Enter a work item ID or click a work item from the dashboard to view details.")

            # Show recent work items as clickable buttons
            recent_work = api.list_work(limit=20)
            if recent_work:
                st.subheader("Recent Work Items (newest first)")
                for item in recent_work:
                    item_id = item.get("id", "")
                    status = item.get("status", "unknown")
                    icon = status_icon(status)
                    desc = truncate(item.get("description", ""), 60)
                    created = format_timestamp(item.get("created_at"))

                    col1, col2, col3 = st.columns([1, 4, 2])
                    col1.write(f"{icon}")
                    if col2.button(
                        f"**{item_id}** — {desc}",
                        key=f"recent_{item_id}",
                        use_container_width=True,
                    ):
                        st.switch_page(
                            get_page("work-detail"),
                            query_params={"work_id": item_id},
                        )
                    col3.caption(created)

            return

    # Get work details
    entry = api.get_work(work_id)
    if entry is None:
        st.error(f"Work item '{work_id}' not found.")
        if st.button("← Back to Dashboard"):
            st.switch_page(get_page("dashboard"))
        return

    # ── Status display (auto-refreshing) ──
    _render_status_header(api, work_id)

    # ── Description ──
    st.divider()
    st.subheader("Description")
    st.write(entry.get("description", "N/A"))

    # ── Result ──
    result = entry.get("result", {})
    if isinstance(result, dict):
        if result.get("artifacts"):
            st.subheader("Artifacts Summary")
            for phase, names in result["artifacts"].items():
                st.write(f"**{phase}**: {', '.join(names) if isinstance(names, list) else names}")

        if result.get("error"):
            st.error(f"Error: {result['error']}")

    # ── Detailed Artifacts Section (auto-refreshing) ──
    st.divider()
    st.subheader("📁 Artifacts")
    _render_artifacts(api, work_id)

    # ── Status-specific actions ──
    # NOTE: These are NOT inside a fragment so that text_area input
    # is preserved and not cleared on fragment re-renders.
    status = entry.get("status", "unknown")
    if status == "needs_review":
        st.warning("This work item needs human review.")
        action = st.radio(
            "Resume action",
            ["Rework from flagged phase", "Approve and proceed"],
            horizontal=True,
            key=f"action_{work_id}",
        )
        resume_action = "rework" if action.startswith("Rework") else "approve"
        human_input = st.text_area(
            "Your feedback / instructions",
            placeholder="Describe what needs to change, or confirm approval...",
            key=f"feedback_{work_id}",
        )
        if st.button("▶ Resume with feedback", type="primary", key=f"resume_{work_id}"):
            if not human_input.strip():
                st.error("Please provide feedback before resuming.")
            else:
                result = api.resume_work(work_id, human_input.strip(), resume_action)
                st.success(f"Resumed! Status: {result['status']} | Action: {result['action']}")
                st.rerun()
    elif status == "running":
        st.info("Work is currently in progress. Updates will appear automatically.")

    # ── Audit log section (auto-refreshing) ──
    st.divider()
    st.subheader("📋 Audit Log")
    _render_audit_log(api, work_id)
