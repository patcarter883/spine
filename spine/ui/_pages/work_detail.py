"""SPINE Work Detail page — combined view of work status and artifacts."""

from __future__ import annotations

import streamlit as st

from spine.ui_api import UIApi
from spine.ui.utils import format_timestamp, status_icon, truncate


def render(api: UIApi) -> None:
    """Render the work detail page with combined status and artifacts."""
    st.title("🔍 Work Details")
    
    # Get work ID from URL parameters or user input
    work_id = st.query_params.get("work_id", "")
    
    if not work_id:
        # Show input form when no work_id is provided via URL params
        work_id = st.text_input("Work ID", placeholder="Enter work item ID")
        
        if not work_id:
            st.info("Enter a work item ID or click a work item from the dashboard to view details.")
            
            # Show recent work items as clickable links
            recent_work = api.list_work(limit=10)
            if recent_work:
                st.subheader("Recent Work Items")
                for item in recent_work:
                    item_id = item.get("id", "")
                    status = item.get("status", "unknown")
                    icon = status_icon(status)
                    desc = truncate(item.get("description", ""), 60)
                    
                    # Create clickable element
                    col1, col2 = st.columns([1, 4])
                    col1.write(f"{icon}")
                    
                    # Use markdown to make it clickable
                    col2.markdown(
                        f"[**{item_id}**](?work_id={item_id}) — {desc}",
                        help=f"Click to view details for {item_id}"
                    )
            
            return
    
    # Get work details
    entry = api.get_work(work_id)
    if entry is None:
        st.error(f"Work item '{work_id}' not found.")
        
        # Show option to go back to dashboard
        if st.button("← Back to Dashboard"):
            st.query_params.clear()
            st.rerun()
        return
    
    # Auto-refresh while work is running so the user sees phase progress
    if entry.get("status") == "running":
        st.markdown(
            '<meta http-equiv="refresh" content="5">',
            unsafe_allow_html=True,
        )
    
    # ── Status display ──
    status = entry.get("status", "unknown")
    icon = status_icon(status)
    
    st.header(f"{icon} {work_id}")
    
    col1, col2 = st.columns(2)
    col1.write(f"**Status:** {status}")
    col1.write(f"**Type:** {entry.get('work_type', 'N/A')}")
    col1.write(f"**Phase:** {entry.get('current_phase', 'N/A')}")
    col2.write(f"**Created:** {format_timestamp(entry.get('created_at'))}")
    col2.write(f"**Updated:** {format_timestamp(entry.get('updated_at'))}")
    
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
    
    # ── Detailed Artifacts Section ──
    st.divider()
    st.subheader("📁 Artifacts")
    
    artifacts = api.get_artifacts(work_id)
    if artifacts:
        for artifact in artifacts:
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
    
    # ── Action buttons ──
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Back to Dashboard"):
            st.query_params.clear()
            st.rerun()
    
    with col2:
        if st.button("📜 View Work History"):
            st.query_params.clear()
            # Navigate to work history page
            st.switch_page("spine.ui._pages.work_history")
    
    # ── Status-specific actions ──
    if status == "needs_review":
        st.warning("This work item needs human review.")
        _human_input = st.text_input("Your input / decision")
        if st.button("Resume with input"):
            st.info("Resume functionality coming soon.")
    elif status == "running":
        st.info("Work is currently in progress. This page will auto-refresh.")
    
    # ── Audit log section ──
    st.divider()
    st.subheader("📋 Audit Log")
    
    audit_events = api.get_audit_log(work_id=work_id, limit=20)
    if audit_events:
        for event in audit_events:
            col1, col2, col3 = st.columns([2, 3, 1])
            col1.write(f"**{event.get('event_type', 'N/A')}**")
            col2.write(event.get('message', 'N/A'))
            col3.write(format_timestamp(event.get('timestamp')))
    else:
        st.info("No audit events found for this work item.")