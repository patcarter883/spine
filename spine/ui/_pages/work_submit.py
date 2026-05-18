"""SPINE Submit Work page — create new execution work items (quick types only)."""

from __future__ import annotations

import streamlit as st

from spine.ui_api import UIApi


def render(api: UIApi) -> None:
    """Render the submit work page — quick workflows only."""
    st.title("📝 Submit Work")

    st.markdown(
        "Submit direct execution work (no planning phase — agents start coding immediately). "
        "For work that needs a plan first, use the **Plans** section."
    )

    # ── Input form ──
    description = st.text_area(
        "Work Description",
        placeholder="Describe what you want the agent to accomplish...",
        height=150,
    )

    work_type = st.selectbox(
        "Workflow Type",
        options=["quick", "critical_quick"],
        format_func=lambda x: {
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
                "Updates will appear automatically on the dashboard."
            )
            st.json(result)

    # ── Workflow type reference ──
    st.divider()
    st.subheader("Quick Workflow Types")

    workflow_data = {
        "Type": ["Quick", "Critical Quick"],
        "Phases": [
            "TASKS → IMPLEMENT → VERIFY",
            "TASKS → CRITIC → IMPLEMENT → VERIFY",
        ],
        "Best For": [
            "Simple tasks, quick fixes",
            "Important quick tasks needing review",
        ],
    }
    st.table(workflow_data)

    st.info(
        "\u2728 **Need planning?** Head to the Plans section to create "
        "a specification and plan before execution."
    )
