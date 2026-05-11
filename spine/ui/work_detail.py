"""Work item detail page — state machine, sub-phases, agent outputs, swarm events."""

import streamlit as st

from spine.ui.utils import (
    get_work_item_detail,
    get_checkpoints,
    format_phase_icon,
    format_phase_color,
    approve_gate,
    reject_gate,
    resume_work,
    rerun_work,
    delete_work,
    go_back,
)


def render_work_detail() -> None:
    """Render the work item detail page."""
    if not st.session_state.get("selected_work_id"):
        st.warning(
            "No work item selected. "
            "Go to **Dashboard** and click 'View' on a work item."
        )
        return

    thread_id = st.session_state.selected_work_id
    detail = get_work_item_detail(thread_id)

    if not detail:
        st.error(f"Work item not found: {thread_id}")
        st.info("The checkpoint may have been deleted or corrupted.")
        return

    # ── Header ──
    phase = detail.get("phase", "INIT")
    icon = format_phase_icon(phase)
    color = format_phase_color(phase)
    st.markdown(f"**{icon} {phase}**")
    st.caption(f"Thread: {thread_id}")

    # Phase progress metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Phase", detail.get("phase", "INIT"))
    col2.metric("Completed", len(detail.get("completed_tasks", [])))
    col3.metric("Failed", len(detail.get("failed_tasks", [])))
    col4.metric("Errors", len(detail.get("errors", [])))
    total_tasks = max(1, len(detail.get("completed_tasks", [])) + len(detail.get("failed_tasks", [])))
    col5.metric(
        "Progress",
        f"{len(detail.get('completed_tasks', []))}/{total_tasks}",
    )

    st.write(f"**Requirement:** {detail.get('requirement', 'Unknown')}")

    # Surface configuration warnings (e.g. agent provider configured but
    # binary missing). Without this, users see "completed" instantly with
    # no work done and no clue why.
    warning_msg = detail.get("error_message")
    if warning_msg and detail.get("status") in ("queued", "running", "completed"):
        if "configured but not available" in str(warning_msg) or "Agent provider" in str(warning_msg):
            st.warning(f"⚠ {warning_msg}")

    # Action buttons
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("▶ Resume", key=f"resume_{thread_id}"):
        if resume_work(thread_id):
            st.success("Work item resumed!")
            st.rerun()
        else:
            st.error("Failed to resume work item.")
    if col2.button("🔄 Re-run", key=f"rerun_{thread_id}"):
        result = rerun_work(thread_id)
        if result and not result.get("error"):
            new_id = result.get("thread_id")
            st.success(f"Re-run submitted as new work item: {new_id}")
            if new_id:
                st.session_state.selected_work_id = new_id
            st.rerun()
        else:
            err = (result or {}).get("error", "Re-run failed (unknown error).")
            st.error(f"Re-run failed: {err}")
    if col3.button("🗑 Delete", key=f"delete_{thread_id}"):
        st.session_state.delete_confirm = thread_id
    if col4.button("← Back", key=f"back_{thread_id}"):
        go_back()
        st.rerun()

    # Delete confirmation
    if st.session_state.get("delete_confirm") == thread_id:
        with st.container():
            st.warning("Are you sure you want to delete this work item?")
            col_a, col_b = st.columns(2)
            if col_a.button("Confirm Delete", type="primary"):
                if delete_work(thread_id):
                    st.success("Work item deleted.")
                    st.session_state.selected_work_id = None
                    st.session_state.delete_confirm = None
                    st.session_state.page = "Dashboard"
                    st.rerun()
            if col_b.button("Cancel"):
                st.session_state.delete_confirm = None
                st.rerun()

    st.divider()

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


def render_state_machine(detail: dict) -> None:
    """Render the state machine visualization."""
    st.subheader("State Machine Progress")

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

    # Build visual progress with emojis and text
    parts = []
    for i, (name, icon) in enumerate(phases):
        if i < current_idx:
            parts.append(f"{icon} **{name}** ✓")
        elif i == current_idx:
            parts.append(f"{icon} **{name}** ● current")
        else:
            parts.append(f"{icon} {name} ○")

    st.markdown(" → ".join(parts))

    # Full state machine transitions
    st.markdown("---")
    st.caption("Full state transitions:")
    transitions = [
        ("INIT", "→", "PLANNING"),
        ("PLANNING", "⟲", "PLANNING (revision)"),
        ("PLANNING", "→", "EXECUTION (approved)"),
        ("EXECUTION", "→", "VERIFICATION"),
        ("VERIFICATION", "⟲", "REWORK"),
        ("VERIFICATION", "→", "COMPLETE"),
        ("REWORK", "→", "EXECUTION"),
        ("ERROR", "→", "REWORK (transient)"),
        ("ERROR", "→", "HUMAN_REVIEW (fatal)"),
        ("ERROR", "→", "BLOCKED (timeout)"),
        ("BLOCKED", "⟳", "BLOCKED (manual resume)"),
        ("HUMAN_REVIEW", "→", "REWORK"),
        ("HUMAN_REVIEW", "→", "PLANNING"),
        ("HUMAN_REVIEW", "→", "EXECUTION"),
    ]
    for a, arrow, b in transitions:
        st.caption(f"  {a} {arrow} {b}")

    # Critic gate indicator
    st.divider()
    st.subheader("Critic Gate")
    critic_result = detail.get("critic_gate_result")
    if critic_result == "APPROVED":
        st.success("✅ Critic gate passed — workflow proceeded to next phase")
    elif critic_result == "REJECTED":
        st.error("✗ Critic gate rejected — plan requires revision")
        st.info("The critic gate blocked transition to EXECUTION. Review the plan and submit for re-review.")
    elif critic_result:
        st.warning(f"⏳ Critic gate status: {critic_result}")
    else:
        st.info("No critic gate result yet. Gate will appear after PLANNING phase completes.")

    # Human review indicator
    if detail.get("phase") == "HUMAN_REVIEW":
        st.warning("👤 **Human review required.** The workflow is paused waiting for your input.")
        with st.form("human_review_form"):
            feedback = st.text_area("Feedback / Instructions:", height=80)
            if st.form_submit_button("Submit Review"):
                st.info("Human review submitted. The agent will continue based on your feedback.")
                # Write human review result
                review_file = Path(f".spine/state/human_review_{thread_id}.json")
                review_file.parent.mkdir(parents=True, exist_ok=True)
                import json
                from datetime import datetime
                review_file.write_text(json.dumps({
                    "feedback": feedback,
                    "timestamp": datetime.now().isoformat(),
                }))

    # Plan preview
    plan = detail.get("plan")
    if plan and isinstance(plan, dict):
        st.divider()
        tasks = plan.get("tasks", [])
        if tasks:
            st.subheader("Plan Tasks")
            completed = set(detail.get("completed_tasks", []))
            for t in tasks:
                task_id = t.get("id", "?")
                task_desc = t.get("description", "")
                status_icon = "✓" if task_id in completed else "○"
                st.write(f"  {status_icon} **{task_id}**: {task_desc}")
        else:
            st.info("No tasks in plan yet. Tasks will appear after planning completes.")

    # Error state
    if detail.get("error_state"):
        st.divider()
        st.error(f"**Error State:** {detail['error_state']}")
        for i, err in enumerate(detail.get("error_history", [])):
            with st.expander(f"Error #{i+1}"):
                st.json(err)


def render_subphases(detail: dict) -> None:
    """Render sub-phase progress for the current phase."""
    st.subheader(f"Phase: {format_phase_icon(detail.get('phase', 'UNKNOWN'))} {detail.get('phase', 'UNKNOWN')}")

    swarm_state = detail.get("swarm_state", {})
    active_subphases = swarm_state.get("active_subphases", [])

    if not active_subphases:
        st.info("No sub-phases defined for this phase yet. Sub-phases will appear as the workflow runs.")

        # Show sub-phase definitions based on current phase
        phase = detail.get("phase", "")
        if phase == "PLANNING":
            st.info("PLANNING sub-phases: ANALYZE, TECH_RESEARCH, RISK_ASSESSMENT, SYNTHESIZE")
        elif phase == "EXECUTION":
            st.info("EXECUTION sub-phases: BACKEND, FRONTEND (or project-specific tasks)")

        return

    # Render each sub-phase
    for sp_name in active_subphases:
        # Estimate progress based on task completion
        completed = set(detail.get("completed_tasks", []))
        task_ids = {
            "ANALYZE": "analyze_requirement",
            "TECH_RESEARCH": "research_stack",
            "RISK_ASSESSMENT": "assess_risks",
            "SYNTHESIZE": "draft_plan",
            "BACKEND": "backend_impl",
            "FRONTEND": "frontend_impl",
        }
        task_id = task_ids.get(sp_name, sp_name.lower().replace(" ", "_"))
        completed_sp = task_id in completed

        icon = "🟢" if completed_sp else "🟡" if completed else "⚪"
        status = "Complete" if completed_sp else "Running" if completed else "Pending"
        progress = 1.0 if completed_sp else (0.5 if completed else 0.0)

        with st.container():
            col_icon, col_name, col_status, col_progress = st.columns([1, 3, 2, 3])
            col_icon.write(icon)
            col_name.write(f"**{sp_name}**")
            col_status.write(status)
            col_progress.progress(progress)

        # Show sub-phase result if available
        subphase_results = detail.get("plan", {}).get("subphase_results", {})
        if sp_name in subphase_results:
            result = subphase_results[sp_name]
            if isinstance(result, dict):
                with st.expander(f"{sp_name} Details"):
                    for key, value in result.items():
                        st.write(f"**{key}**: {value}")
            else:
                with st.expander(f"{sp_name} Details"):
                    st.write(result)


def render_agent_outputs(detail: dict) -> None:
    """Render agent outputs, grouped by source agent."""
    swarm_events = detail.get("swarm_events", [])

    if not swarm_events:
        st.info("No agent outputs yet. This will populate as the workflow runs.")
        st.markdown(
            "Agent outputs are generated during phase execution. "
            "Each agent produces outputs based on its role "
            "(explorer, sme, analyst, planner, critic, etc.)."
        )
        return

    # Group events by source agent
    agent_groups: dict[str, list] = {}
    for event in swarm_events:
        source = event.get("from", "unknown")
        if source not in agent_groups:
            agent_groups[source] = []
        agent_groups[source].append(event)

    if not agent_groups:
        st.info("No agent outputs recorded yet.")
        return

    tabs = st.tabs(list(agent_groups.keys()))
    for (agent_name, events), tab in zip(agent_groups.items(), tabs):
        with tab:
            for event in events:
                subject = event.get("subject", "Event")
                timestamp = event.get("timestamp", "unknown")
                preview = str(event.get("body", ""))[:120]

                with st.expander(f"{subject} — {timestamp}", expanded=False):
                    body = event.get("body", {})
                    if isinstance(body, dict):
                        for key, value in body.items():
                            if key != "type":
                                st.write(f"**{key}**: {value}")
                    else:
                        st.code(str(body), language="json")


def render_swarm_events(detail: dict) -> None:
    """Render swarm event log as a table."""
    events = detail.get("swarm_events", [])

    if not events:
        st.info("No swarm events recorded yet. Events are logged as agents coordinate.")
        return

    # Build table rows
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

    # Event detail expansion
    st.divider()
    st.subheader("Event Details")
    for i, e in enumerate(events):
        with st.expander(
            f"#{i+1}: {e.get('subject', 'Event')} — {e.get('from', 'unknown')} → {e.get('to', 'unknown')}",
            expanded=False,
        ):
            st.json(e)
