"""SPINE Submit Work page — create new work items.

For Spec & Planning work, use the Spec & Planning page instead.
The submit work page is restricted to quick work types for direct execution.
"""

from __future__ import annotations

import streamlit as st

from spine.ui_api import UIApi


def render(api: UIApi) -> None:
    """Render the submit work page."""
    st.title("📝 Submit Work")

    st.markdown(
        "Enter a work description and choose a workflow type. "
        "SPINE will orchestrate the AI agent through the appropriate phases.\n\n"
        "**Note:** This page is for **task workflow types** (direct execution). "
        "For Spec & Planning workflows (new feature planning, architecture), "
        "use the **Spec & Planning** page instead."
    )

    st.info(
        "🌳 **Runs in an isolated git worktree sandbox.** Both task types reach "
        "the IMPLEMENT phase, so the agent's code changes are made in a throwaway "
        "git worktree — never your live working tree. The patch is fast-forward "
        "merged into the main branch **only when the run completes successfully**; "
        "any other outcome (needs-review, stalled, failed) is rolled back so main "
        "is never left dirty."
    )

    # ── Input form ──
    description = st.text_area(
        "Work Description",
        placeholder="Describe what you want the agent to accomplish...",
        height=150,
    )

    work_type = st.selectbox(
        "Workflow Type",
        options=["task", "critical_task"],
        format_func=lambda x: {
            "task": "⚡ Task (SPECIFY → PLAN → CRITIC_PLAN → IMPLEMENT → VERIFY)",
            "critical_task": "🔒 Critical Task (SPECIFY → PLAN → CRITIC_PLAN → ADVERSARIAL_PLAN → IMPLEMENT → VERIFY)",
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
    st.subheader("Workflow Types Reference")

    workflow_data = {
        "Type": ["Task", "Critical Task"],
        "Phases": [
            "SPECIFY → PLAN → CRITIC_PLAN → IMPLEMENT → VERIFY",
            "SPECIFY → PLAN → CRITIC_PLAN → ADVERSARIAL_PLAN → IMPLEMENT → VERIFY",
        ],
        "Best For": [
            "Standard tasks with full workflow",
            "Important tasks needing adversarial red-team review of the plan",
        ],
    }
    st.table(workflow_data)

    st.divider()
    st.info(
        "🔧 **For new feature planning, use the Spec & Planning page** from the "
        "navigation menu. This allows you to review and approve specifications "
        "and plans before spawning execution tasks."
    )
