"""SPINE Git-Gated Execution page — run work behind a transactional gate.

Submits work through the :class:`SpineGitOrchestrator` transactional
lifecycle: an isolated sandbox is created, the validation pipeline runs,
and the patch is ff-merged on success or rolled back on failure.
"""

from __future__ import annotations

import streamlit as st

from spine.ui_api import UIApi


def render(api: UIApi) -> None:
    """Render the git-gated execution page.

    Args:
        api: The shared :class:`UIApi` instance for backend access.
    """
    st.title("🔒 Git-Gated Execution")

    st.markdown(
        "Run work behind a **transactional git gate**. SPINE isolates the run "
        "in a sandbox, executes the workflow, then runs the validation pipeline "
        "(lint → typecheck → test). On success the patch is fast-forward merged "
        "into the main branch; on any failure the workspace is rolled back so "
        "the main tree is never left dirty.\n\n"
        "**Note:** This runs in the background — submit and watch the Sandbox "
        "Status section below for progress."
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
            "critical_task": (
                "🔒 Critical Task (SPECIFY → CRITIC_SPECIFY → PLAN → "
                "CRITIC_PLAN → IMPLEMENT → VERIFY)"
            ),
        }.get(x, x),
    )

    if st.button("🚀 Submit Gated Run", type="primary", disabled=not description.strip()):
        result = api.submit_gated_work(description, work_type)

        if "error" in result:
            st.error(f"Failed: {result['error']}")
        else:
            st.success(
                f"Gated run started!  \n"
                f"Status: `{result['status']}` · Type: `{result['work_type']}`"
            )
            st.info(
                "The gated lifecycle is running in the background. Refresh the "
                "Sandbox Status section to follow its progress."
            )
            st.json(result)

    # ── Sandbox status ──
    st.divider()
    st.subheader("Sandbox Status")

    if st.button("🔄 Refresh Status"):
        st.rerun()

    status = api.get_gate_status()
    if status.get("active"):
        st.warning("A sandbox is currently active.")
    else:
        st.success("No active sandbox.")
    st.json(status)

    # ── Lifecycle reference ──
    st.divider()
    st.info(
        "🔒 **Gated lifecycle:** isolate (sandbox worktree/branch) → "
        "validate (run the configured validation pipeline) → "
        "merge-or-rollback (fast-forward merge on success, nuclear rollback "
        "on any gate failure)."
    )
