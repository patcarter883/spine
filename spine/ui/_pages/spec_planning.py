"""SPINE Spec & Planning page — review and approve specifications and plans.

This page is the interface for the planning workflow. Users can:
1. Submit new planning work (spec+plan with approval gate after critic_plan,
   plus an adversarial red-team stage for critical workflows)
2. View existing planning work items awaiting review
3. Review and approve specifications and plans
4. Spawn execution tasks from approved plans
"""

from __future__ import annotations

import asyncio

import streamlit as st

from spine.ui_api import UIApi


def render(api: UIApi) -> None:
    """Render the spec & planning page."""
    st.title("📐 Spec & Planning")

    st.markdown(
        "Create and review specifications and plans before spawning execution tasks. "
        "This workflow lets you iterate on the specification and plan with AI assistance "
        "before committing to implementation."
    )

    st.caption(
        "🌳 Planning itself writes no code. When you **approve** a plan, the work "
        "continues into the IMPLEMENT phase inside an isolated git worktree sandbox — "
        "the resulting patch merges to main only on success and rolls back otherwise."
    )

    # ── Tabs ──
    tab1, tab2 = st.tabs(["📝 Submit Planning Work", "📋 Review Planning Work"])

    # ── Tab 1: Submit Planning Work ──
    with tab1:
        _render_submit_tab(api)

    # ── Tab 2: Review Planning Work ──
    with tab2:
        _render_review_tab(api)


def _render_submit_tab(api: UIApi) -> None:
    """Render the submit planning work tab."""
    st.subheader("Start New Planning Work")

    st.markdown(
        "Submit a description for planning. The AI will produce a specification "
        "and technical plan that you can review before spawning execution tasks."
    )

    description = st.text_area(
        "Planning Description",
        placeholder="Describe the feature, system, or change you want to plan...",
        height=150,
        key="planning_description",
    )

    work_type = st.selectbox(
        "Planning Workflow",
        options=["reviewed_task", "critical_reviewed_task"],
        format_func=lambda x: {
            "reviewed_task": "📋 Reviewed Task (SPECIFY → PLAN → CRITIC_PLAN → await approval → spawn TASKs)",
            "critical_reviewed_task": "📐 Critical Reviewed Task (SPECIFY → PLAN → CRITIC_PLAN → ADVERSARIAL_PLAN → await approval → spawn TASKs)",
        }.get(x, x),
        help=(
            "Reviewed Task stops after critic_plan; Critical Reviewed Task adds an "
            "adversarial red-team stage and stops after adversarial_plan. On approval, "
            "fresh task work items are spawned for execution."
        ),
    )

    if st.button("🚀 Submit Planning Work", type="primary", disabled=not description.strip()):
        result = api.enqueue_work(description, work_type)

        if "error" in result:
            st.error(f"Failed: {result['error']}")
        else:
            queue_id = result["queue_id"]
            st.success(
                f"Planning work enqueued! **Queue ID: {queue_id}**  \n"
                f"Status: `{result['status']}` · Type: `{result['work_type']}`"
            )
            st.info(
                "Your planning work has been queued. Check the 'Review Planning Work' tab "
                "to see progress and review results."
            )
            st.json(result)

    # ── Workflow type reference ──
    st.divider()
    st.markdown("### Planning Workflow Types")

    st.markdown("""
    | Type | Phases | Use Case |
    |------|--------|----------|
    | **Reviewed Task** | SPECIFY → PLAN → CRITIC_PLAN → await approval → spawn TASKs | Standard planning with plan review |
    | **Critical Reviewed Task** | SPECIFY → PLAN → CRITIC_PLAN → ADVERSARIAL_PLAN → await approval → spawn TASKs | Full planning with an adversarial red-team review of the approved plan |

    On approval, fresh ``task`` work items are spawned for each unit and run
    through their own implement/verify cycle. The reviewed-task graph itself
    never runs implement/verify directly — that's what the human gate guards.
    """)


def _render_review_tab(api: UIApi) -> None:
    """Render the review planning work tab."""
    st.subheader("Review Planning Work")

    # Filter options
    col1, col2 = st.columns([2, 1])
    with col1:
        status_filter = st.selectbox(
            "Status Filter",
            options=["awaiting_approval", "needs_review", "all"],
            format_func=lambda x: {
                "awaiting_approval": "Awaiting Approval",
                "needs_review": "Needs Review",
                "all": "All Planning Work",
            }.get(x, x),
        )
    with col2:
        limit = st.number_input("Limit", min_value=1, max_value=100, value=20)

    # Fetch planning items
    status = None if status_filter == "all" else status_filter
    items = api.list_planning_sessions(status=status, limit=limit)

    if not items:
        st.info("No planning work items found.")
        return

    st.markdown(f"### Found {len(items)} planning work item(s)")

    for item in items:
        _render_planning_item(api, item)


def _render_planning_item(api: UIApi, item: dict) -> None:
    """Render a single planning work item card."""
    work_id = item.get("id", "unknown")
    status = item.get("status", "unknown")
    description = item.get("description", "No description")

    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{work_id[:8]}** · {description[:60]}...")
        with col2:
            status_emoji = {
                "awaiting_approval": "⏳",
                "approved": "✅",
                "needs_review": "👁️",
            }.get(status, "❓")
            st.markdown(f"{status_emoji} `{status}`")

        # Load artifacts
        detail = api.get_planning_detail(work_id)

        # Show spec if available
        spec_key = None
        for key in ("specify/spec.md", "specify/specification.md"):
            if detail and key in detail.get("artifacts", {}):
                spec_key = key
                break

        if spec_key:
            with st.expander("📄 Specification", expanded=False):
                st.markdown(detail["artifacts"][spec_key])

        # Show plan if available
        plan_key = None
        for key in ("plan/plan.md",):
            if detail and key in detail.get("artifacts", {}):
                plan_key = key
                break

        if plan_key:
            with st.expander("📋 Plan", expanded=False):
                st.markdown(detail["artifacts"][plan_key])

        # Approval actions
        if status in ("awaiting_approval", "needs_review", "approved"):
            st.markdown("---")
            _render_approval_actions(api, work_id, status)


def _render_approval_actions(api: UIApi, work_id: str, status: str) -> None:
    """Render approval action buttons for a planning item."""
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✅ Approve & Spawn", key=f"approve_{work_id}"):
            with st.spinner("Approving and spawning execution tasks..."):
                result = asyncio.run(api.approve_plan(work_id, action="approve"))
                if "error" in result:
                    st.error(f"Failed: {result['error']}")
                elif result.get("status") == "error":
                    st.error(f"Failed: {result.get('error', 'Unknown error')}")
                else:
                    spawned = result.get("spawned_ids", [])
                    if spawned:
                        st.success(f"Approved! Spawned tasks: {len(spawned)}")
                    else:
                        st.success(f"Plan approved (status: {result.get('status', 'ok')})")
                    st.rerun()

    with col2:
        if st.button("🔄 Request Revision", key=f"revise_{work_id}"):
            feedback = st.text_area("Feedback", key=f"feedback_{work_id}")
            if feedback and st.button("Submit Revision", key=f"submit_rev_{work_id}"):
                result = asyncio.run(
                    api.approve_plan(work_id, action="request_revision", feedback=feedback)
                )
                if "error" in result:
                    st.error(f"Failed: {result['error']}")
                elif result.get("status") == "error":
                    st.error(f"Failed: {result.get('error', 'Unknown error')}")
                else:
                    st.info(f"Revision requested. Status: {result.get('status', 'submitted')}")
                    st.rerun()

    with col3:
        if st.button("❌ Reject", key=f"reject_{work_id}"):
            if st.button("Confirm Reject", key=f"confirm_reject_{work_id}"):
                result = asyncio.run(api.approve_plan(work_id, action="reject"))
                if "error" in result:
                    st.error(f"Failed: {result['error']}")
                elif result.get("status") == "error":
                    st.error(f"Failed: {result.get('error', 'Unknown error')}")
                else:
                    st.info(f"Work rejected. Status: {result.get('status', 'rejected')}")
                    st.rerun()
