"""SPINE Plan Submit page — create new planning work items."""

from __future__ import annotations

import streamlit as st

from spine.ui_api import UIApi


def render(api: UIApi) -> None:
    """Render the submit plan page."""
    st.title("📋 Submit Plan")

    st.markdown(
        "Create a specification and plan for your project. "
        "Once the plan is reviewed and approved, you can split it into individual execution work items."
    )

    # ── Input form ──
    description = st.text_area(
        "Project Description",
        placeholder="Describe what you want to build at a high level...",
        height=150,
    )

    work_type = st.selectbox(
        "Planning Type",
        options=["plan", "plan_spec"],
        format_func=lambda x: {
            "plan_spec": "🔒 Critical Plan (SPECIFY → CRITIC → PLAN → CRITIC)",
            "plan": "📐 Plan (SPECIFY → PLAN → CRITIC)",
        }.get(x, x),
    )

    if st.button("📋 Submit Plan", type="primary", disabled=not description.strip()):
        result = api.enqueue_work(description, work_type)

        if "error" in result:
            st.error(f"Failed: {result['error']}")
        else:
            queue_id = result["queue_id"]
            st.success(
                f"Plan enqueued! **Queue ID: {queue_id}**  \n"
                f"Status: `{result['status']}` · Type: `{result['work_type']}`"
            )
            st.info(
                "Your plan is being generated. "
                "Once complete and approved, you can split it into execution tasks."
            )
            st.json(result)

    # ── Reference ──
    st.divider()
    st.subheader("Planning Workflow")

    st.markdown(
        "Planning work items produce a **specification** and **implementation plan**, "
        "but no code. After a plan passes critic review, you can:\n\n"
        "1. View the generated spec and plan artifacts\n"
        "2. Use **Split Plan** to create individual execution work items\n"
        "3. Each spawned item inherits from the approved plan"
    )
