"""SPINE Submit Work page — create new work items."""

from __future__ import annotations

import streamlit as st

from spine.ui_api import UIApi


def render(api: UIApi) -> None:
    """Render the submit work page."""
    st.title("📝 Submit Work")

    st.markdown(
        "Enter a work description and choose a workflow type. "
        "SPINE will orchestrate the AI agent through the appropriate phases."
    )

    # ── Input form ──
    description = st.text_area(
        "Work Description",
        placeholder="Describe what you want the agent to accomplish...",
        height=150,
    )

    work_type = st.selectbox(
        "Workflow Type",
        options=["spec", "critical_spec", "quick", "critical_quick"],
        format_func=lambda x: {
            "spec": "📐 Spec Work (SPECIFY → PLAN → CRITIC → TASKS → IMPLEMENT → VERIFY)",
            "critical_spec": "🔒 Critical Spec (SPECIFY → CRITIC → PLAN → CRITIC → TASKS → CRITIC → IMPLEMENT → VERIFY)",
            "quick": "⚡ Quick Work (TASKS → IMPLEMENT → VERIFY)",
            "critical_quick": "🔒 Critical Quick (TASKS → CRITIC → IMPLEMENT → VERIFY)",
        }.get(x, x),
    )

    if st.button("🚀 Submit", type="primary", disabled=not description.strip()):
        result = api.enqueue_work(description, work_type)

        if "error" in result:
            st.error(f"Failed: {result['error']}")
        else:
            queue_id = result["queue_id"]
            st.success(
                f"Work enqueued! **Queue ID: {queue_id}**  \n"
                f"Status: `{result['status']}` · Type: `{result['work_type']}`"
            )
            st.info(
                "Your work has been queued for background processing. "
                "Use the **Work Status** page to track progress."
            )
            st.json(result)

    # ── Workflow type reference ──
    st.divider()
    st.subheader("Workflow Types Reference")

    workflow_data = {
        "Type": ["Quick", "Critical Quick", "Spec", "Critical Spec"],
        "Phases": [
            "TASKS → IMPLEMENT → VERIFY",
            "TASKS → CRITIC → IMPLEMENT → VERIFY",
            "SPECIFY → PLAN → CRITIC → TASKS → IMPLEMENT → VERIFY",
            "SPECIFY → CRITIC → PLAN → CRITIC → TASKS → CRITIC → IMPLEMENT → VERIFY",
        ],
        "Best For": [
            "Simple tasks, quick fixes",
            "Important quick tasks needing review",
            "New features, greenfield work",
            "Critical features, production changes",
        ],
    }
    st.table(workflow_data)
