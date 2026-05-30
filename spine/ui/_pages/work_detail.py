"""SPINE Work Detail page — combined view of work status and artifacts.

This is the single surface for human review: it shows the critic's feedback
and provides the review/feedback/approval actions for items flagged as
``needs_review`` or ``awaiting_approval``. The Human Review page is only an
index that links here.
"""

from __future__ import annotations

import asyncio

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
    terminal_statuses = (
        "completed", "failed", "needs_review", "stalled",
        "awaiting_approval", "approved", "rejected",
    )
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


def _render_critic_review(api: UIApi, work_id: str) -> None:
    """Show the critic's verdict so the reviewer knows why it was flagged."""
    review = api.get_critic_review(work_id)
    if not review:
        return

    phase = review.get("phase", "")
    status = review.get("status", "")
    header = " · ".join(p for p in (phase, status) if p) or "Critic Review"
    with st.expander(f"🔍 Critic Review — {header}", expanded=True):
        st.caption(f"Review tier: {review.get('tier', 'unknown')}")
        reason = review.get("reason")
        if reason:
            st.markdown(f"**Reason:** {reason}")
        suggestions = review.get("suggestions") or []
        if suggestions:
            st.markdown("**Suggestions:**")
            for suggestion in suggestions:
                st.markdown(f"- {suggestion}")


def _run_plan_action(api: UIApi, work_id: str, action: str, feedback: str | None) -> None:
    """Submit an approve / request_revision / reject decision for a plan."""
    with st.spinner(f"Submitting {action}..."):
        result = asyncio.run(api.approve_plan(work_id, action=action, feedback=feedback))

    if "error" in result or result.get("status") == "error":
        st.error(f"Failed: {result.get('error', 'Unknown error')}")
        return

    if action == "approve":
        spawned = result.get("spawned_ids", [])
        if spawned:
            st.success(f"Approved! Spawned {len(spawned)} task(s).")
        else:
            # Same work item continued from IMPLEMENT (no fresh spawn).
            st.success(
                f"Plan approved — continued from implement "
                f"(status: {result.get('status', 'ok')})."
            )
    elif action == "request_revision":
        st.info(f"Revision requested. Status: {result.get('status', 'submitted')}")
    else:
        st.info(f"Plan rejected. Status: {result.get('status', 'rejected')}")
    st.rerun()


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

        work_type = entry.get("work_type", "task")
        sequence = WORKFLOW_SEQUENCES.get(work_type, WORKFLOW_SEQUENCES.get("task", []))
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
    work_type = entry.get("work_type", "task")
    current_phase = entry.get("current_phase", "")

    # Helper: render "Restart from Phase" section
    def _render_restart_from_phase(work_id: str, work_type: str, current_phase: str) -> None:
        """Render a phase selector and restart-from-phase button."""
        valid_phases = api.get_restart_phases(work_type)
        if not valid_phases:
            return

        # Default to the current phase if it's valid, otherwise the first phase
        default_idx = 0
        if current_phase in valid_phases:
            default_idx = valid_phases.index(current_phase)

        selected_phase = st.selectbox(
            "Restart from phase",
            valid_phases,
            index=default_idx,
            key=f"restart_phase_{work_id}",
            help="Select which phase to re-run from. Earlier phases and their artifacts are preserved.",
        )
        clear = st.checkbox(
            "Clear artifacts from this phase onward",
            key=f"clear_phase_artifacts_{work_id}",
            help="Delete on-disk artifacts for the selected phase and later. "
            "Earlier artifacts are always preserved.",
        )
        if st.button(
            f"▶ Restart from {selected_phase}",
            key=f"restart_phase_btn_{work_id}",
        ):
            result = api.restart_from_phase(work_id, selected_phase, clear_artifacts=clear)
            if result.get("status") == "skipped":
                # Work is already running - show info message with details
                message = result.get("message", "This task is already running.")
                st.info(f"Restart skipped: {message}")
            else:
                st.success(
                    f"Restarted from **{selected_phase}**! "
                    f"Status: {result['status']} | Action: {result['action']}. "
                    "The workflow will continue from the selected phase."
                )
            st.rerun()

    if status == "needs_review":
        st.warning("This work item needs human review.")

        # ── Review feedback display ──
        feedback = api.get_feedback(work_id)
        if feedback:
            st.subheader("Review needed")
            for fb in feedback:
                if isinstance(fb, dict):
                    reason = fb.get("reason", "No reason provided")
                    tier = fb.get("tier", "unknown")
                    suggestions = fb.get("suggestions", [])

                    st.markdown(f"**Reason:** {reason}")
                    st.caption(f"Review tier: {tier}")

                    if suggestions:
                        st.markdown("**Suggestions:**")
                        for suggestion in suggestions:
                            st.markdown(f"- {suggestion}")

                    st.divider()

        # Show two resume options: interrupt-based (preferred) and legacy
        st.subheader("Resume Options")
        st.caption(
            "**Resume from review** continues from the exact point the workflow "
            "paused (recommended).  **Resume from scratch** restarts the entire "
            "graph with your feedback appended."
        )

        action = st.radio(
            "Resume action",
            ["Rework from flagged phase", "Approve and proceed", "Abort"],
            horizontal=True,
            key=f"action_{work_id}",
        )
        resume_action = (
            "rework"
            if action.startswith("Rework")
            else "approve"
            if action.startswith("Approve")
            else "abort"
        )
        human_input = st.text_area(
            "Your feedback / instructions",
            placeholder="Describe what needs to change, or confirm approval...",
            key=f"feedback_{work_id}",
        )

        # Primary: interrupt-based resume (preserves checkpoint position)
        if st.button("▶ Resume from review", type="primary", key=f"resume_interrupt_{work_id}"):
            if not human_input.strip() and resume_action != "abort":
                st.error("Please provide feedback before resuming.")
            else:
                result = api.resume_interrupted_work(work_id, resume_action, human_input.strip())
                st.success(f"Resumed! Status: {result['status']} | Action: {result['action']}")
                st.rerun()

        # Secondary: legacy resume (restarts from scratch)
        with st.expander("Legacy: Resume from scratch"):
            st.warning("This restarts the entire workflow from phase 0 with your feedback.")
            if st.button("▶ Legacy Resume with feedback", key=f"resume_legacy_{work_id}"):
                if not human_input.strip():
                    st.error("Please provide feedback before resuming.")
                else:
                    result = api.resume_work(work_id, human_input.strip(), resume_action)
                    st.success(f"Resumed! Status: {result['status']} | Action: {result['action']}")
                    st.rerun()

        st.divider()
        st.caption("Or restart from a specific phase (discarding accumulated feedback):")
        _render_restart_from_phase(work_id, work_type, current_phase)

        st.divider()
        st.caption("Or restart from phase 0 (discarding accumulated feedback):")
        if st.button("🔄 Restart from phase 0", key=f"restart_{work_id}"):
            clear = st.checkbox(
                "Clear all artifacts (force full regeneration)", key=f"clear_artifacts_{work_id}"
            )
            result = api.restart_work(work_id, clear_artifacts=clear)
            st.success(
                f"Restarted! Status: {result['status']} | Action: {result['action']}. "
                "The workflow will re-run from the beginning."
            )
            st.rerun()

    elif status == "awaiting_approval":
        st.warning(
            "This plan is awaiting your approval before execution tasks are spawned."
        )

        # ── Critic verdict ──
        _render_critic_review(api, work_id)

        st.subheader("Approval")
        st.caption(
            "The specification and plan are shown in the Artifacts section above. "
            "Approve to continue execution, request a revision with feedback, or reject."
        )
        plan_feedback = st.text_area(
            "Feedback",
            placeholder="Optional for approve/reject; required when requesting a revision.",
            key=f"plan_feedback_{work_id}",
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✅ Approve & Run", type="primary", key=f"plan_approve_{work_id}"):
                _run_plan_action(api, work_id, "approve", plan_feedback or None)
        with col2:
            if st.button("🔄 Request Revision", key=f"plan_revise_{work_id}"):
                if not plan_feedback.strip():
                    st.error("Please add feedback explaining the requested revisions.")
                else:
                    _run_plan_action(api, work_id, "request_revision", plan_feedback)
        with col3:
            if st.button("❌ Reject", key=f"plan_reject_{work_id}"):
                _run_plan_action(api, work_id, "reject", plan_feedback or None)

    elif status == "running":
        st.info("Work is currently in progress. Updates will appear automatically.")

        with st.expander("🔄 Restart options"):
            _render_restart_from_phase(work_id, work_type, current_phase)

            st.divider()
            if st.button("🔄 Restart from phase 0", key=f"restart_{work_id}"):
                clear = st.checkbox(
                    "Clear all artifacts (force full regeneration)",
                    key=f"clear_artifacts_{work_id}",
                )
                result = api.restart_work(work_id, clear_artifacts=clear)
                st.success(
                    f"Restarted! Status: {result['status']} | Action: {result['action']}. "
                    "The workflow will re-run from the beginning."
                )
                st.rerun()

    elif status == "stalled":
        st.warning("This work item has stalled (no progress within the timeout).")

        st.subheader("Restart Options")
        st.caption(
            "Restart from a specific phase to resume from where the stall occurred, "
            "or restart from phase 0 to re-run the entire workflow."
        )
        _render_restart_from_phase(work_id, work_type, current_phase)

        st.divider()
        if st.button("🔄 Restart from phase 0", key=f"restart_{work_id}"):
            clear = st.checkbox(
                "Clear all artifacts (force full regeneration)", key=f"clear_artifacts_{work_id}"
            )
            result = api.restart_work(work_id, clear_artifacts=clear)
            st.success(
                f"Restarted! Status: {result['status']} | Action: {result['action']}. "
                "The workflow will re-run from the beginning."
            )
            st.rerun()

    # ── Audit log section (auto-refreshing) ──
    st.divider()
    st.subheader("📋 Audit Log")
    _render_audit_log(api, work_id)
