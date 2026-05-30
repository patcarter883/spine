"""SPINE Human Review page — manage work items needing attention."""

from __future__ import annotations

import asyncio
from typing import Any

import streamlit as st

from spine.ui_api import UIApi
from spine.ui.utils import format_timestamp, status_icon, truncate


_REVIEW_STATUSES = ("needs_review", "awaiting_approval")


def render(api: UIApi) -> None:
    """Render the human review page."""
    st.title("👤 Human Review")

    items = _fetch_review_items(api, limit=50)

    if not items:
        st.success("No work items need human review!")
        return

    st.warning(f"{len(items)} work item(s) need your attention.")

    for item in items:
        _render_review_item(api, item)


def _fetch_review_items(api: UIApi, limit: int) -> list[dict[str, Any]]:
    """Collect work items across every status that requires human review."""
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for status in _REVIEW_STATUSES:
        for item in api.list_work(status=status, limit=limit):
            work_id = item.get("id")
            if not work_id or work_id in seen:
                continue
            seen.add(work_id)
            merged.append(item)

    merged.sort(
        key=lambda r: (r.get("updated_at") or r.get("created_at") or ""),
        reverse=True,
    )
    return merged[:limit]


def _render_review_item(api: UIApi, item: dict[str, Any]) -> None:
    work_id = item.get("id", "unknown")
    status = item.get("status", "unknown")
    icon = status_icon(status)
    desc = truncate(item.get("description", ""), 80)

    with st.expander(f"{icon} {work_id} — {desc}"):
        st.write(f"**Status:** `{status}`")
        st.write(f"**Type:** {item.get('work_type', 'N/A')}")
        st.write(f"**Phase:** {item.get('current_phase', 'N/A')}")
        st.write(f"**Created:** {format_timestamp(item.get('created_at'))}")
        st.write(f"**Updated:** {format_timestamp(item.get('updated_at'))}")

        st.write("**Description:**")
        st.write(item.get("description", ""))

        if status == "awaiting_approval":
            _render_awaiting_approval_actions(api, work_id)
        else:
            _render_needs_review_actions(api, work_id)

        if st.button("View Details", key=f"hr_view_{work_id}"):
            from spine.ui.pages import get as get_page

            st.switch_page(
                get_page("work-detail"),
                query_params={"work_id": work_id},
            )


def _render_needs_review_actions(api: UIApi, work_id: str) -> None:
    """Resume an interrupted run via Command(resume=...)."""
    action = st.radio(
        "Action",
        ["Rework", "Approve"],
        horizontal=True,
        key=f"hr_action_{work_id}",
    )
    resume_action = "rework" if action == "Rework" else "approve"
    feedback = st.text_input(
        "Feedback",
        placeholder="Optional: add instructions...",
        key=f"hr_feedback_{work_id}",
    )
    if st.button("▶ Resume", key=f"hr_resume_{work_id}"):
        api.resume_interrupted_work(work_id, resume_action, feedback)
        st.success(f"Resumed {work_id} with action: {resume_action}")
        st.rerun()


def _render_awaiting_approval_actions(api: UIApi, work_id: str) -> None:
    """Approve a reviewed-task plan (or request revision / reject)."""
    detail = api.get_planning_detail(work_id) or {}
    artifacts = detail.get("artifacts", {})

    spec_key = next(
        (k for k in ("specify/spec.md", "specify/specification.md") if k in artifacts),
        None,
    )
    if spec_key:
        with st.expander("📄 Specification", expanded=False):
            st.markdown(artifacts[spec_key])

    if "plan/plan.md" in artifacts:
        with st.expander("📋 Plan", expanded=False):
            st.markdown(artifacts["plan/plan.md"])

    feedback = st.text_area(
        "Feedback (required for revision)",
        key=f"hr_plan_feedback_{work_id}",
        placeholder="Optional for approve/reject; required when requesting revision.",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Approve & Run", key=f"hr_plan_approve_{work_id}"):
            _run_plan_action(api, work_id, "approve", feedback or None)
    with col2:
        if st.button("🔄 Request Revision", key=f"hr_plan_revise_{work_id}"):
            if not feedback.strip():
                st.error("Please add feedback explaining the requested revisions.")
            else:
                _run_plan_action(api, work_id, "request_revision", feedback)
    with col3:
        if st.button("❌ Reject", key=f"hr_plan_reject_{work_id}"):
            _run_plan_action(api, work_id, "reject", feedback or None)


def _run_plan_action(
    api: UIApi,
    work_id: str,
    action: str,
    feedback: str | None,
) -> None:
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
